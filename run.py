import argparse
import json
import os
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from data import (
    available_datasets,
    read_labels_from_benchmark,
    read_prompts_from_benchmark,
    read_prompts_from_file,
)
from decoding import NablaDecoding
from utils import seed_everything


MIXED_PRECISION_NAME_MAP = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def init_models(
    lm_model_name: str,
    rm_model_name: Optional[str],
    device: str,
    attn_impl: str,
    torch_dtype: torch.dtype,
):
    print(f"[{device}] Loading LM: {lm_model_name}", flush=True)
    lm_tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
    if lm_tokenizer.chat_template is None:
        raise ValueError("Only supports language model tokenizer with chat template.")
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


def resolve_prompt(args):
    if args.prompt is not None:
        return args.prompt, args.label

    if args.prompt_file is not None:
        return open(args.prompt_file, "r", encoding="utf-8").read(), args.label

    if args.prompt_ref is not None:
        if ":" not in args.prompt_ref:
            raise ValueError("--prompt_ref must be in '<dataset|file>:<index>' format.")
        source, idx_str = args.prompt_ref.split(":", 1)
        idx = int(idx_str)

        if source.upper() in available_datasets():
            prompts = read_prompts_from_benchmark(source.upper())
            labels = read_labels_from_benchmark(source.upper())
            return prompts[idx], labels[idx]

        prompts = read_prompts_from_file(source)
        return prompts[idx], args.label

    raise ValueError("Provide one of --prompt, --prompt_file, or --prompt_ref.")


def main():
    parser = argparse.ArgumentParser(description="Single-prompt Nabla decoding runner.")

    parser.add_argument("--lm_model_name", "--lm", type=str, required=True, help="LM model path/name.")
    parser.add_argument("--rm_model_name", "--rm", type=str, required=True, help="RM model path/name.")
    parser.add_argument("--vllm_url", type=str, default="", help="vLLM server base URL.")
    parser.add_argument("--vllm_model_name", type=str, default=None, help="Model name sent to vLLM server.")

    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--verbose", type=int, default=2, help="Verbosity level for Nabla decoding.")
    parser.add_argument("--device", type=str, default=None, help="Device, e.g. 'cuda:0' or 'cpu'.")

    parser.add_argument("--prompt", type=str, default=None, help="Prompt text.")
    parser.add_argument("--prompt_file", type=str, default=None, help="Read prompt text from a file.")
    parser.add_argument("--prompt_ref", type=str, default=None, help="Dataset/file ref in '<source>:<index>' format.")
    parser.add_argument("--label", type=str, default=None, help="Optional label override.")
    parser.add_argument("--output_file", type=str, default=None, help="Optional JSON output path.")

    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", help="HF attention implementation.")
    parser.add_argument("--embedder_type", type=str, choices=["logits", "latents"], default="logits", help="Optimization backend.")
    parser.add_argument("--max_iters", type=int, default=100, help="Maximum DTO iterations.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for DTO optimization.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for DTO optimization.")
    parser.add_argument("--reward_coeff", type=float, default=0.1, help="Reward loss coefficient.")
    parser.add_argument("--max_generation_len", type=int, default=1024, help="Maximum generated tokens.")
    parser.add_argument("--mixed_precision", type=str, choices=["bf16", "fp32"], default="bf16", help="Torch dtype.")
    parser.add_argument("--no_grad_caching", action="store_true", help="Disable grad caching in DTO.")
    parser.add_argument("--warmup_iters_ratio", type=float, default=0, help="Warmup ratio for LR schedule in DTO.")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Minimum LR ratio after decay in DTO.")
    parser.add_argument("--rollout_tau", type=float, default=0.0, help="Temperature for rollout sampling.")
    parser.add_argument("--rollout_top_k", type=int, default=None, help="Top-k for rollout sampling.")
    parser.add_argument("--rollout_top_p", type=float, default=0.8, help="Top-p for rollout sampling.")
    parser.add_argument("--resample_tau", type=float, default=0.6, help="Temperature for resampling.")
    parser.add_argument("--resample_top_k", type=int, default=None, help="Top-k for resampling.")
    parser.add_argument("--resample_top_p", type=float, default=0.95, help="Top-p for resampling.")
    parser.add_argument("--n_generations", type=int, default=4, help="Max candidate rollouts in rejection sampling.")
    parser.add_argument("--rej_sampling", action="store_true", help="Enable rejection sampling.")
    parser.add_argument("--no_rej_sampling", action="store_false", dest="rej_sampling", help="Disable rejection sampling.")
    parser.set_defaults(rej_sampling=True)
    parser.add_argument("--update_postfix", action="store_true", help="Enable postfix token updates.")
    parser.add_argument("--confidence_threshold", type=float, default=0.95, help="Confidence selector threshold.")
    parser.add_argument("--grad_threshold", type=float, default=None, help="Gradient L2 selector threshold.")

    args = parser.parse_args()

    prompt_input_count = sum(x is not None for x in [args.prompt, args.prompt_file, args.prompt_ref])
    if prompt_input_count != 1:
        raise ValueError("Provide exactly one of --prompt, --prompt_file, --prompt_ref.")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    prompt, label = resolve_prompt(args)

    # Set `args.vllm_model_name` default to `args.lm_model_name` if not specified
    if args.vllm_model_name is None:
        args.vllm_model_name = args.lm_model_name

    lm_model, lm_tokenizer, rm_model, rm_tokenizer = init_models(
        args.lm_model_name,
        args.rm_model_name,
        device=device,
        attn_impl=args.attn_implementation,
        torch_dtype=MIXED_PRECISION_NAME_MAP[args.mixed_precision],
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
        lm_model,
        lm_tokenizer,
        rm_model,
        rm_tokenizer,
        train_args,
        device=device,
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
        vllm_url=args.vllm_url,
        vllm_model_name=args.vllm_model_name,
        confidence_selector_threshold=args.confidence_threshold,
        grad_selector_threshold=args.grad_threshold,
    )

    seed_everything(int(args.seed))
    token_ids = nabla_decoder.generate(prompt, return_prompt=False, seed=int(args.seed))
    response = lm_tokenizer.decode(token_ids[0], skip_special_tokens=True)
    stats = nabla_decoder.get_stats()

    result = {
        "prompt": prompt,
        "label": label,
        "response": response,
        "stats": stats,
    }

    print(f"\n=== Response ===\n{response}\n", flush=True)
    print("=== Stats ===", flush=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    if label is not None:
        print(f"Label: {label}", flush=True)

    if args.output_file:
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved to {args.output_file}", flush=True)


if __name__ == "__main__":
    main()
