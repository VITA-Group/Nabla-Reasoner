import argparse
import json
import math
import os
import sys
from typing import List, Dict, Any, Optional
from contextlib import redirect_stdout, redirect_stderr
import traceback
import pdb
import datetime

import torch.multiprocessing as mp
from multiprocessing.managers import BaseManager
from multiprocessing import Queue

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

from templates import format_with_template
from decoding import NablaDecoding

from data import read_prompts_from_benchmark, read_labels_from_benchmark, available_datasets

import warnings
warnings.filterwarnings("ignore")

MASTER_AUTHKEY = b"awesome_nabla_decoding"

MIXED_PRECISION_NAME_MAP = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

_SERVER_QUEUE = None

def _server_get_queue():
    # runs in the manager server process
    global _SERVER_QUEUE
    if _SERVER_QUEUE is None:
        _SERVER_QUEUE = mp.Queue(maxsize=1024)
    return _SERVER_QUEUE

class QueueManager(BaseManager):
    pass

QueueManager.register("get_queue", callable=_server_get_queue)

def start_queue_server(addr, port, authkey):
    mgr = QueueManager(address=(addr, port), authkey=authkey)
    mgr.start()
    return mgr

def connect_queue_client(addr, port, authkey):
    QueueManager.register("get_queue")
    mgr = QueueManager(address=(addr, port), authkey=authkey)
    mgr.connect()
    return mgr

def write_json(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=4)

def merge_jsons(per_proc_files, merged_path):
    all_rows = []
    for p in per_proc_files:
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            all_rows += json.load(f)
    all_rows = sorted(all_rows, key=lambda x: x["global_index"])
    write_json(merged_path, all_rows)

    # Also compute and save averages of numeric Nabla stats across prompts.
    sums = {}
    counts = {}
    for row in all_rows:
        row_stats = row.get("stats", [])
        if isinstance(row_stats, dict):
            row_stats = [row_stats]
        if not isinstance(row_stats, list):
            continue
        for stats in row_stats:
            if not isinstance(stats, dict):
                continue
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    sums[key] = sums.get(key, 0.0) + float(value)
                    counts[key] = counts.get(key, 0) + 1

    avg_stats = {k: (sums[k] / counts[k]) for k in sums if counts[k] > 0}
    stats_path = os.path.join(os.path.dirname(merged_path), "generation_stats.json")
    write_json(stats_path, avg_stats)


def is_torchrun():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def init_models(lm_model_name, rm_model_name, device, attn_impl="flash_attention_2", torch_dtype=torch.bfloat16):
    print(f"[{device}] Loading LM: {lm_model_name}", flush=True)
    lm_tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
    if lm_tokenizer.chat_template is None:
        raise ValueError("Only supports language model tokenizer with chat template!")
    lm_model = AutoModelForCausalLM.from_pretrained(
        lm_model_name,
        torch_dtype=torch_dtype,
        device_map=device,
        attn_implementation=attn_impl,
    )
    lm_model.resize_token_embeddings(len(lm_tokenizer))
    for p in lm_model.parameters():
        p.requires_grad_(False)

    rm_model, rm_tokenizer = None, None
    if rm_model_name is not None:
        print(f"[{device}] Loading RM: {rm_model_name}", flush=True)
        rm_tokenizer = AutoTokenizer.from_pretrained(rm_model_name)
        if rm_tokenizer.chat_template is None:
            rm_tokenizer.chat_template = lm_tokenizer.chat_template

        rm_model = AutoModelForSequenceClassification.from_pretrained(
            rm_model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            attn_implementation=attn_impl,
            num_labels=1,
        )
        rm_model.resize_token_embeddings(len(rm_tokenizer))
        for p in rm_model.parameters():
            p.requires_grad_(False)

    if lm_tokenizer.pad_token is None:
        lm_tokenizer.pad_token = lm_tokenizer.eos_token
    if rm_tokenizer is not None and rm_tokenizer.pad_token is None and rm_tokenizer.eos_token is not None:
        rm_tokenizer.pad_token = rm_tokenizer.eos_token

    if rm_tokenizer is not None and lm_tokenizer.get_vocab() != rm_tokenizer.get_vocab():
        print(f"[{device}] WARNING: LM and RM have different vocabularies.", flush=True)

    return lm_model, lm_tokenizer, rm_model, rm_tokenizer

def worker_process(rank, world_size, device, all_prompt_list, args, per_proc_outfile, addr, port, vllm_url):

    # Per-rank base seed to decouple RNG streams across workers
    base_seed = args.seed + rank

    mgr = connect_queue_client(addr, port, MASTER_AUTHKEY)
    q = mgr.get_queue()

    if device.startswith("cuda"):
        torch.cuda.set_device(int(device.split(":")[1]))

    lm_model, lm_tokenizer, rm_model, rm_tokenizer = init_models(
        args.lm_model_name,
        args.rm_model_name,
        device=device,
        attn_impl=args.attn_implementation,
        torch_dtype=MIXED_PRECISION_NAME_MAP[args.mixed_precision]
    )

    train_args = dict(
        max_iters=args.max_iters,
        warmup_iters_ratio=args.warmup_iters_ratio,
        learning_rate=args.learning_rate,
        min_lr_ratio=args.min_lr_ratio,
        weight_decay=args.weight_decay,
        reward_coeff=args.reward_coeff,
        mixed_precision=MIXED_PRECISION_NAME_MAP[args.mixed_precision],
        grad_caching=not args.no_grad_caching,
        update_postfix=args.update_postfix,
        embedder_type=args.embedder_type,
    )
    
    nabla_decoder = NablaDecoding(
        lm_model, lm_tokenizer,
        rm_model, rm_tokenizer,
        train_args, device=device,
        max_length=args.max_generation_len,
        verbose=args.verbose,
        rejection_sampling=args.rej_sampling,
        max_n_generations=args.n_generations,
        rollout_tau=args.rollout_tau,
        rollout_top_k=args.rollout_top_k,
        rollout_top_p=args.rollout_top_p,
        resample_tau=args.resample_tau,
        resample_top_k=args.resample_top_k,
        resample_top_p=args.resample_top_p,
        backend="vllm",
        vllm_url=vllm_url,
        vllm_model_name=args.vllm_model_name,
        confidence_selector_threshold=args.confidence_threshold,
        grad_selector_threshold=args.grad_threshold,
    )


    all_outputs = []
    while True:
        queue_item = q.get()
        if queue_item is None:
            break

        global_idx, prompt, label = queue_item
        # Reseed per prompt for full control (stable across retries)
        prompt_seed = base_seed + global_idx
        one_output = {
            "process_rank": rank,
            "device": device,
            "global_index": global_idx,
            "prompt": prompt,
            "label": label,
        }

        print(f"[{device}] Processing prompt {global_idx}: {prompt}", flush=True)

        responses, stats = [], []

        for repeat_idx in range(args.n_samples):
            stdout_file = os.path.join(args.output_dir, f"logs/{global_idx}_{repeat_idx}.txt")
            with open(stdout_file, "a", encoding="utf-8") as f, redirect_stdout(f), redirect_stderr(f):
                repeat_seed = prompt_seed + repeat_idx
                try:
                    token_ids = nabla_decoder.generate(prompt, return_prompt=False, seed=repeat_seed)
                    text = lm_tokenizer.decode(token_ids[0], skip_special_tokens=True)
                    responses.append(text)
                    stats.append(nabla_decoder.get_stats())
                except Exception as e:
                    responses.append(repr(e))
                    stats.append({})
                    print("Traceback:", traceback.format_exc(), flush=True)

        one_output["responses"] = responses
        one_output["stats"] = stats
        all_outputs.append(one_output)

        print(f"[{device}] Processed {global_idx+1}/{len(all_prompt_list)} prompts: {prompt}", flush=True)

        write_json(per_proc_outfile, all_outputs)

def launch_prompts_queue_server(prompts, labels, world_size, addr, port):

    mgr = start_queue_server(addr, port, MASTER_AUTHKEY)
    shared_q = mgr.get_queue()
    for i, prompt in enumerate(prompts):
        shared_q.put((i, prompt, labels[i] if labels is not None else None))
    for _ in range(world_size):
        shared_q.put(None)

    print(f"Prompts queue server started at {addr}:{port}", flush=True)
    return mgr


def main():

    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Multi-process Nabla decoding over a list of prompts with vLLM ahead.")

    parser.add_argument("--lm_model_name", "--lm", type=str, default=None, help="LM model path/name.")
    parser.add_argument("--rm_model_name", "--rm", type=str, default=None, help="RM model path/name.")
    parser.add_argument("--vllm_url", type=str, default="", help="vLLM server base URL")
    parser.add_argument("--vllm_model_name", type=str, default=None, help="Model name to send to vLLM server")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of repeated generations per prompt.")
    parser.add_argument("--port", type=int, default=29500, help="Port for prompts queue server.")
    parser.add_argument("--verbose", type=int, default=2, help="Verbosity level for Nabla decoding.")


    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", help="HF attention implementation.")
    parser.add_argument("--embedder_type", type=str, choices=["logits", "latents"], default="logits", help="Optimization backend.")
    parser.add_argument("--max_iters", type=int, default=100, help="Maximum DTO optimization iterations per prompt.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for DTO optimization.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for DTO optimization.")
    parser.add_argument("--reward_coeff", type=float, default=0.1, help="Reward loss coefficient in DTO objective.")
    parser.add_argument("--max_generation_len", type=int, default=1024, help="Maximum number of generated tokens.")
    parser.add_argument("--mixed_precision", type=str, choices=["bf16", "fp32"], default="bf16", help="Torch dtype for model.")
    parser.add_argument("--no_grad_caching", action="store_true", help="Disable grad caching in DTO")
    parser.add_argument("--warmup_iters_ratio", type=float, default=0, help="Warmup ratio for DTO LR scheduling.")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Minimum LR ratio after decay for DTO.")
    parser.add_argument("--rollout_tau", type=float, default=0., help="Temperature for rollout sampling.")
    parser.add_argument("--rollout_top_k", type=int, default=None, help="Top-k for rollout sampling; None disables top-k.")
    parser.add_argument("--rollout_top_p", type=float, default=0.8, help="Top-p for rollout sampling.")
    parser.add_argument("--resample_tau", type=float, default=0.6, help="Temperature for resampling.")
    parser.add_argument("--resample_top_k", type=int, default=None, help="Top-k for resampling; None disables top-k.")
    parser.add_argument("--resample_top_p", type=float, default=0.95, help="Top-p for resampling.")
    parser.add_argument("--n_generations", type=int, default=4, help="Number of candidate rollouts in rejection sampling.")
    parser.add_argument("--rej_sampling", action="store_true", help="Enable rejection sampling in Nabla decoding")
    parser.add_argument("--no_rej_sampling", action="store_false", dest="rej_sampling", help="Disable rejection sampling in Nabla decoding")
    parser.set_defaults(rej_sampling=True)
    parser.add_argument("--update_postfix", action="store_true", help="Enable postfix token updates during optimization.")

    # Skip optimization criteria
    parser.add_argument("--confidence_threshold", type=float, default=0.95, help="Threshold for confidence based token selection criteria.")
    parser.add_argument("--grad_threshold", type=float, default=None, help="Threshold for gradient L2-norm based token selection criteria.")

    # Device partitioning
    parser.add_argument("--num_procs", type=int, required=True, help="Number of worker processes to launch.")

    parser.add_argument("--prompts", type=str, required=True, help="Path to prompts file (.txt/.json/.jsonl) or benchmark name (e.g., AIME-2025)")
    parser.add_argument("--max_prompts", type=int, default=None, help="If set, only use the first N prompts from the dataset/file.")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to store results")
    parser.add_argument("--output_name", type=str, default="responses", help="Base name for JSON outputs")

    args = parser.parse_args()
    if args.n_samples < 1:
        raise ValueError("--n_samples must be >= 1")

    # Set `args.vllm_model_name` default to `args.lm_model_name` if not specified
    if args.vllm_model_name is None:
        args.vllm_model_name = args.lm_model_name

    # Read prompts
    if args.prompts.upper() in available_datasets():
        prompts = read_prompts_from_benchmark(args.prompts.upper())
        labels = read_labels_from_benchmark(args.prompts.upper())
    else:
        from data import read_prompts_from_file
        prompts = read_prompts_from_file(args.prompts)
        labels = None

    # Optionally restrict to the first N prompts for quick tests.
    if args.max_prompts is not None and args.max_prompts > 0:
        prompts = prompts[:args.max_prompts]
        if labels is not None:
            labels = labels[:args.max_prompts]
        print(f"Using only the first {len(prompts)} prompts (max_prompts={args.max_prompts}).", flush=True)
    if not prompts:
        print("No prompts found.", flush=True)
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "procs"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    merged_file = os.path.join(args.output_dir, f"{args.output_name}.json")

    # Check CUDA compatibility
    assert torch.cuda.is_available(), "CUDA is not available"
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if args.num_procs > len(devices):
        print("WARNING: Number of processes exceed the number of GPUs. Multiple processes will share GPUs.")

    # Launch prompt queue server
    addr = "0.0.0.0"
    port = args.port

    mgr = launch_prompts_queue_server(prompts, labels, args.num_procs, addr, port)

    # Spawn processes
    per_proc_files = [
        os.path.join(args.output_dir, "procs", f"{args.output_name}.proc{rank}.json")
        for rank in range(args.num_procs)
    ]
    worker_devices = devices[:args.num_procs]
    ctx = torch.multiprocessing.get_context("spawn")
    procs = []
    for rank in range(args.num_procs):
        device = worker_devices[rank % len(worker_devices)]
        p = ctx.Process(
            target=worker_process,
            args=(rank, args.num_procs, device, prompts, args, per_proc_files[rank], addr, port, args.vllm_url),
        )
        p.daemon = False
        p.start()
        procs.append(p)

    # Join
    for p in procs:
        p.join()
        if p.exitcode != 0:
            print(f"WARNING: worker exited with code {p.exitcode}", file=sys.stderr)

    # Merge
    merge_jsons(per_proc_files, merged_file)
    print(f"Done. Merged results written to: {merged_file}")

    mgr.shutdown()


if __name__ == "__main__":
    main()
