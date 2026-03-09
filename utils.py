import os
import math
import time
import random
from datetime import datetime
from functools import partial
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

import transformers

import re

def seed_everything(seed: int = 42):
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Disable TF32 to reduce non-deterministic numeric variance
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass

    transformers.set_seed(seed)  # Hugging Face helper


def compute_vocabulary_correspondence(src_tokenizer, ref_tokenizer):
    src_vocab, ref_vocab = src_tokenizer.get_vocab(), ref_tokenizer.get_vocab()

    ref_vocab = sorted(ref_vocab.items(), key=lambda t: t[1])
    idx_mapping = [src_vocab.get(t[0], src_tokenizer.pad_token_id) for t in ref_vocab]

    return idx_mapping

# Align the vocabulary of the source to reference tokenizers.
def align_vocab(src_embed, src_tokenizer, ref_tokenizer=None, vocab_dim=0):
    if ref_tokenizer is None or ref_tokenizer.get_vocab() == src_tokenizer.get_vocab():
        if src_embed.shape[vocab_dim] > len(src_tokenizer):
            slice_idx = [slice(None)] * len(src_embed.shape)
            slice_idx[vocab_dim] = slice(0, len(src_tokenizer))
            return src_embed[tuple(slice_idx)]
        elif src_embed.shape[vocab_dim] == len(src_tokenizer):
            return src_embed
        else:
            raise ValueError(f"No solution for the number of embeddings is less than the tokenizer's vocabulary: {src_embed.shape[vocab_dim]} > {len(src_tokenizer)}")
    else:
        idx_mapping = compute_vocabulary_correspondence(src_tokenizer, ref_tokenizer)
        return torch.index_select(src_embed, vocab_dim, torch.as_tensor(idx_mapping, device=src_embed.device))

def infer_device_from_model(model):
    if hasattr(model, 'hf_device_map'):
        # This attribute exists if the model was loaded with device_map="auto"
        device_set = list(set(model.hf_device_map.values()))
        
    else:
        device_set = list(set([p.device for p in model.parameters()]))

    assert len(device_set) == 1, f"ERROR: Model is split across multiple devices: {device_set}."

    return device_set[0]

def get_default_system_prompt(tokenizer):
    if not getattr(tokenizer, "chat_template", None):
        return None  # No template → no reliable default

    # A unique placeholder that will never appear naturally
    SYS_MARK = "__CAPTURE_SYS_BLOCK_1f7c2b__"
    USER_MARK = "$$PLACEHOLDER_USER_CONTENT_1f7c2b$$"
    ASSISTANT_MARK = "$$PLACEHOLDER_ASSISTANT_CONTENT_1f7c2b$$"

    # (A) Render WITH an explicit system placeholder, plus a tiny user probe to anchor the format
    msgs_with_marker = [
        {"role": "system", "content": SYS_MARK},
        {"role": "user", "content": USER_MARK},
        {"role": "assistant", "content": ASSISTANT_MARK},
    ]
    rendered_marker = tokenizer.apply_chat_template(
        msgs_with_marker, tokenize=False, add_generation_prompt=False
    )

    # Build a regex: escape everything, then replace the placeholder with a lazy capture group.
    # Because both strings come from the SAME template, we usually don't need whitespace normalization.
    pattern = re.escape(rendered_marker)
    pattern = pattern.replace(re.escape(SYS_MARK), r"(?P<system>[\s\S]*?)")

    # Optional: Make the pattern a bit tolerant to whitespace runs.
    # (safe because we only touch whitespace outside of our capture)
    pattern = re.sub(r"(?:\\n|\\t|\\r| )+", r"\\s+", pattern)

    # (B) Render WITHOUT a system message to trigger the default (if any)
    msgs_no_system = [
        {"role": "user", "content": USER_MARK},
        {"role": "assistant", "content": ASSISTANT_MARK},
    ]
    rendered_default = tokenizer.apply_chat_template(
        msgs_no_system, tokenize=False, add_generation_prompt=False
    )

    # (C) Match and extract
    m = re.search(pattern, rendered_default, flags=re.DOTALL)
    if not m:
        return ""

    # Clean up common leading/trailing whitespace artifacts
    sys_text = m.group("system").strip()
    return sys_text

def _get_linear_schedule_with_warmup_and_min_lr_lambda(current_step, *, num_warmup_steps, num_training_steps, min_lr_ratio):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(min_lr_ratio, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))


def get_linear_schedule_with_warmup_and_min_lr(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1, min_lr_ratio=0.1):
    lr_lambda = partial(
        _get_linear_schedule_with_warmup_and_min_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_ratio=min_lr_ratio
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_cosine_schedule_with_warmup_and_min_lr_lambda(
    current_step, *, num_warmup_steps, num_training_steps, num_cycles, min_lr_ratio
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_cosine_schedule_with_warmup_and_min_lr(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1, min_lr_ratio=0.1):

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_and_min_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_ratio=min_lr_ratio
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_scheduler(scheduler_type, optimizer, **kwargs):

    if scheduler_type == 'linear':
        return get_linear_schedule_with_warmup_and_min_lr(
            optimizer, kwargs['num_warmup_steps'], kwargs['num_training_steps'],
            last_epoch=kwargs.get('last_epoch', -1), min_lr_ratio=kwargs.get('min_lr_ratio', 0.1)
        )
    elif scheduler_type == 'cosine':
        return get_cosine_schedule_with_warmup_and_min_lr(
            optimizer, kwargs['num_warmup_steps'], kwargs['num_training_steps'],
            num_cycles=kwargs.get('last_epoch', 0.5), last_epoch=kwargs.get('last_epoch', -1),
            min_lr_ratio=kwargs.get('min_lr_ratio', 0.1)
        )
    else:
        return transformers.get_scheduler(transformers.SchedulerType(scheduler_type), **kwargs)

@torch.no_grad()
def manipulate_kv_cache(past_kvs, pred_k, pred_v=None):
    if pred_v is None:
        pred_v = pred_k

    new_kvs = []
    for l, layer_cache in enumerate(past_kvs):
        k, v = layer_cache
        new_kvs.append((pred_k(l, k), pred_v(l, v)))
    return tuple(new_kvs)

@torch.no_grad()
def slice_kv_cache(past_kvs, slice_obj):
    slicing = lambda l, x: x[:, :, slice_obj]
    return manipulate_kv_cache(past_kvs, slicing)

@torch.no_grad()
def update_kv_cache(past_kvs, new_kvs):
    upd_k = lambda l, k: torch.cat([k, new_kvs[l][0]], dim=-2)
    upd_v = lambda l, v: torch.cat([v, new_kvs[l][1]], dim=-2)
    return manipulate_kv_cache(past_kvs, upd_k, upd_v)

def get_print_by_verbosity(verbose):

    def print0(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
        else:
            pass

    return print0

def print_str_diff(a, b):

    from diff_match_patch import diff_match_patch
    from rich.console import Console
    from rich.text import Text

    dmp = diff_match_patch()
    diffs = dmp.diff_main(a, b)
    dmp.diff_cleanupSemantic(diffs)

    t = Text()
    for op, seg in diffs:
        if op == dmp.DIFF_EQUAL:
            t.append(seg)
        elif op == dmp.DIFF_INSERT:
            t.append(seg, style="bold green")        # present only in b
        elif op == dmp.DIFF_DELETE:
            t.append(seg, style="bold red strike")   # deleted from a

    Console(force_terminal=True).print(t)

def markup_to_ansi(markup_text, style=None):
    from rich.console import Console
    from rich.markup import escape

    console = Console(force_terminal=True)
    with console.capture() as capture:
        console.print(escape(markup_text), style=style, end="")
    return capture.get()

def merge_json_files(json_files):
    all_rows = []
    for p in json_files:
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            all_rows += json.load(f)
    return all_rows
