import os
import math
import numpy as np

from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F


from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

from optimize import LogitTrainer
import templates
import utils
from utils import slice_kv_cache, update_kv_cache

import requests

@dataclass
class GenerationStates:
    past_token_ids: torch.Tensor = None
    ahead_token_ids: torch.Tensor = None
    ahead_latents: torch.Tensor = None
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ahead_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None


class NablaDecoding:

    def __init__(self, lm_model, lm_tokenizer, rm_model, rm_tokenizer, train_args, device=None,
        max_length = -1, stop_strings=None, verbose=0,
        rollout_tau=None, rollout_top_k=None, rollout_top_p=None,
        resample_tau=None, resample_top_k=None, resample_top_p=None,
        backend="vllm", vllm_url=None, vllm_model_name=None, vllm_output_type="token_ids",
        entropy_selector_threshold=None, confidence_selector_threshold=None, grad_selector_threshold=None,
        rejection_sampling=True, max_n_generations=16,
    ):

        self.model = lm_model
        self.tokenizer = lm_tokenizer

        self.device = device
        if self.device is None:
            self.device = utils.infer_device_from_model(self.model)

        self.max_length = max_length
        self.verbose = verbose

        self.gen_ahead_config = dict(
            temperature = rollout_tau if rollout_tau is not None else self.model.generation_config.temperature,
            top_k = rollout_top_k if rollout_top_k is not None else self.model.generation_config.top_k,
            top_p = rollout_top_p if rollout_top_p is not None else self.model.generation_config.top_p,
        )

        self.resample_config = dict(
            temperature = resample_tau if resample_tau is not None else self.model.generation_config.temperature,
            top_k = resample_top_k if resample_top_k is not None else self.model.generation_config.top_k,
            top_p = resample_top_p if resample_top_p is not None else self.model.generation_config.top_p,
        )

        self.stop_strings = stop_strings
        if isinstance(stop_strings, str):
            self.stop_strings = [stop_strings]

        self.stop_token_ids = []
        if self.stop_strings is not None:
            self.stop_token_ids = [self.tokenizer.encode(s, add_special_tokens=False, return_tensors="pt") for s in self.stop_strings]
        eos_token_ids = self.model.generation_config.eos_token_id if isinstance(self.model.generation_config.eos_token_id, list) else [self.model.generation_config.eos_token_id]
        self.stop_token_ids += [torch.tensor([[t]], dtype=torch.int64) for t in eos_token_ids]

        self.prompt_text = None

        self.generation_states = GenerationStates()

        self.logits_optimizer = LogitTrainer(lm_model, lm_tokenizer, rm_model, rm_tokenizer,
            show_train_pbar=(verbose >= 3), show_train_logs=(verbose >= 4), **train_args
        )
        self.embedder_type = str(train_args.get("embedder_type", "logits")).lower()
        if self.embedder_type not in ("logits", "latents"):
            raise ValueError(f"Unsupported embedder_type: {self.embedder_type}.")

        self.backend = backend
        assert self.backend in ["vllm", "huggingface"], f"Unsupported backend: {self.backend}. Supported backends: ['vllm', 'huggingface']"

        self.vllm_url = vllm_url
        self.vllm_model_name = vllm_model_name if vllm_model_name is not None else getattr(lm_tokenizer, "name_or_path", None)
        self.vllm_output_type = vllm_output_type
        assert self.vllm_output_type in ["token_ids", "text"], f"vLLM output type {self.vllm_output_type} is not supported. Supported types: [`token_ids`, `text`]"

        assert (self.backend != "vllm") or (self.vllm_url and self.vllm_model_name), "vLLM backend requires `vllm_url` and `vllm_model_name` to be specified."

        # Per-call seed; set by generate(..., seed=...)
        self.seed = None

        # token selection strategies
        self.entropy_selector_threshold = entropy_selector_threshold
        self.confidence_selector_threshold = confidence_selector_threshold
        self.grad_selector_threshold = grad_selector_threshold

        self.rejection_sampling = rejection_sampling

        self.max_n_generations = max_n_generations
        self.n_generations = 0
        self.n_rejections = 0

        # Stats
        self.prompt_len = 0
        self.generation_len = 0
        # Cumulative calls of LLMs during the whole process
        self.num_llm_calls = 0

        self.num_generated_tokens = 0
        self.num_optimized_tokens = 0
        self.num_rewarded_tokens = 0

    def print_logs(self, *args, **kwargs):
        if self.verbose >= 1:
            print(f"===[Info]===", *args, **kwargs)

    def print_details(self, *args, **kwargs):
        if self.verbose >= 2:
            print(*args, **kwargs)

    @property
    def past_token_ids(self):
        return self.generation_states.past_token_ids
    @past_token_ids.setter
    def past_token_ids(self, value):
        self.generation_states.past_token_ids = value

    @property
    def past_key_values(self):
        return self.generation_states.past_key_values
    @past_key_values.setter
    def past_key_values(self, value):
        self.generation_states.past_key_values = value
    
    @property
    def ahead_token_ids(self):
        return self.generation_states.ahead_token_ids
    @ahead_token_ids.setter
    def ahead_token_ids(self, value):
        self.generation_states.ahead_token_ids = value

    @property
    def ahead_key_values(self):
        return self.generation_states.ahead_key_values
    @ahead_key_values.setter
    def ahead_key_values(self, value):
        self.generation_states.ahead_key_values = value
    
    @property
    def ahead_latents(self):
        return self.generation_states.ahead_latents
    @ahead_latents.setter
    def ahead_latents(self, value):
        self.generation_states.ahead_latents = value

    @property
    def current_seed(self):
        # base seed + generation length
        if self.seed is not None:
            return self.seed + self.past_token_ids.shape[1] - self.prompt_len
        else:
            return None

    def seed_if_needed(self):
        if self.current_seed is not None:
            torch.manual_seed(self.current_seed)
            torch.cuda.manual_seed_all(self.current_seed)
            torch.random.manual_seed(self.current_seed)

    def to_logits(self, ahead_latents):
        if self.embedder_type == "logits":
            return ahead_latents
        elif self.embedder_type == "latents":
            return self.model.get_output_embeddings()(ahead_latents)
        else:
            raise ValueError(f"Unknown embedder type: {self.embedder_type}")

    def prepare_decoding(self, prompt, system_prompt=None, seed=None):

        self.seed = seed

        del self.generation_states
        self.generation_states = GenerationStates()

        self.prompt_text = prompt
        self.sys_prompt_text = system_prompt

        prefix_with_template = templates.format_with_template(self.tokenizer, prompt, system_prompt=system_prompt)
        self.past_token_ids = self.tokenizer.encode(prefix_with_template, add_special_tokens=False, return_tensors="pt").to(self.device)

        self.prompt_len = int(self.past_token_ids.shape[1])
        self.generation_len = 0

        self.n_rejections = 0
        self.n_generations = 0
        self.num_llm_calls = 0

        self.num_generated_tokens = 0
        self.num_optimized_tokens = 0
        self.num_rewarded_tokens = 0

    def commit_n_tokens(self, n_tokens):
        if n_tokens <= 0:
            return # no tokens to commit

        self.past_token_ids = torch.cat([self.past_token_ids, self.ahead_token_ids[:, :n_tokens]], -1)
        self.ahead_token_ids = self.ahead_token_ids[:, n_tokens:]
        self.ahead_latents = self.ahead_latents[:, n_tokens:]

        self.generation_len += n_tokens

        # update KV cache for huggingface backend
        if self.backend == "huggingface":
            self.past_key_values = update_kv_cache(self.past_key_values, slice_kv_cache(self.ahead_key_values, slice(0, n_tokens)))
            self.ahead_key_values = slice_kv_cache(self.ahead_key_values, slice(n_tokens, None))

    def optimize_ahead_latents(self):

        assert self.ahead_token_ids is not None and self.ahead_latents is not None

        with torch.random.fork_rng(devices=[self.device]):
            self.seed_if_needed()

            generated_token_ids = self.past_token_ids[:, self.prompt_len:]
            optimized_results = self.logits_optimizer.optimize(self.prompt_text, generated_token_ids, self.ahead_latents)
            optimized_logits = optimized_results["logits"]

        self.num_llm_calls += optimized_results["num_llm_calls"]
        self.num_optimized_tokens += int(self.ahead_token_ids.shape[1]) * optimized_results["num_grad_steps"]

        return optimized_logits

    def move_to_next_optimizable_token(self):

        selector = torch.ones(self.ahead_latents.shape[1], device=self.ahead_latents.device, dtype=torch.bool)
        ahead_logits = self.to_logits(self.ahead_latents)

        # entropy selection criteria
        if self.entropy_selector_threshold is not None:
            probs = F.softmax(ahead_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs), dim=-1).squeeze(0) # [# ahead tokens]
            entropy_selector = (entropy >= self.entropy_selector_threshold)
            selector &= entropy_selector

        # confidence selection criteria
        if self.confidence_selector_threshold is not None:
            probs = F.softmax(ahead_logits, dim=-1)
            top_prob = probs.max(dim=-1).values.squeeze(0) # [# ahead tokens]
            confidence_selector = (top_prob <= self.confidence_selector_threshold)
            selector &= confidence_selector

        # gradient selection criteria
        if self.grad_selector_threshold is not None:
            generated_token_ids = self.past_token_ids[:, self.prompt_len:]
            grad_onehots = self.logits_optimizer.compute_gradient_for_onehots(self.prompt_text, generated_token_ids, self.ahead_latents)
            grad_norm = torch.linalg.norm(grad_onehots, ord=2, dim=-1).squeeze(0)  # [# ahead tokens]
            grad_selector = (grad_norm >= self.grad_selector_threshold)
            selector &= grad_selector

        selected_position = None
        optimizable_positions = selector.nonzero(as_tuple=False)
        if optimizable_positions.numel() > 0:
            selected_position = int(optimizable_positions[0].item())
        else:
            selected_position = self.ahead_token_ids.shape[1]

        # commit all tokens before the selected position to the past
        if selected_position > 0:
            self.commit_n_tokens(selected_position)

        return selected_position

    def acceptance_criteria(self, proposed_generation):
        proposed_token_ids = torch.cat([proposed_generation.past_token_ids[:, self.prompt_len:], proposed_generation.ahead_token_ids], dim=-1)
        current_token_ids = torch.cat([self.past_token_ids, self.ahead_token_ids], dim=-1)
        reward_proposed = self.logits_optimizer.get_reward_for_token_ids(self.prompt_text, proposed_token_ids)
        reward_current = self.logits_optimizer.get_reward_for_token_ids(self.prompt_text, current_token_ids)
        self.num_rewarded_tokens += int(
            proposed_generation.past_token_ids.shape[1] + proposed_generation.ahead_token_ids.shape[1]
            + self.past_token_ids.shape[1] + self.ahead_token_ids.shape[1]
        )
        self.num_llm_calls += 2

        self.print_details(f"Current reward: {reward_current}, proposed reward: {reward_proposed}")

        return reward_proposed > reward_current


    def generate(self, prompt, system_prompt=None, return_prompt=True, seed=None):

        self.prepare_decoding(prompt, system_prompt, seed=seed)
        
        self.generation_states = self.generate_ahead(None)

        while self.ahead_latents.shape[1] > 0:
            assert self.generation_len == self.past_token_ids.shape[1] - self.prompt_len

            selected_position = self.move_to_next_optimizable_token()

            self.print_logs(f"Move {selected_position} tokens forward, and start optimizing.")

            # `self.ahead_token_ids` has length zero means we have committed all ahead tokens -> stop generation
            if self.ahead_token_ids.shape[1] == 0:
                self.print_logs(f"No optimizable tokens remaining, abort.")
                break

            optimized_logits = self.optimize_ahead_latents()

            logits_for_sampling = optimized_logits[:, 0]

            with torch.no_grad():
                next_token = self.sample_token(logits_for_sampling) # [bs, 1]

                accepted = False
                do_regeneration = False
                if next_token[0, 0] != self.ahead_token_ids[0, 0]:
                    do_regeneration = True

                    self.print_logs(f"Token revision: {self.ahead_token_ids[0, :1].item()} -> {next_token[0].item()}, "
                        f"{self.tokenizer.decode(self.ahead_token_ids[0, :1], skip_special_tokens=False)} -> {self.tokenizer.decode(next_token[0], skip_special_tokens=False)}.")

                    proposed_generation = self.generate_ahead(next_token)

                    self.print_details("Current: ",
                        utils.markup_to_ansi(self.tokenizer.decode(self.past_token_ids[0, :], skip_special_tokens=False), "green"),
                        utils.markup_to_ansi(self.tokenizer.decode(self.ahead_token_ids[0, :1], skip_special_tokens=False), "red"),
                        utils.markup_to_ansi(self.tokenizer.decode(self.ahead_token_ids[0, 1:], skip_special_tokens=False), "blue")
                    )
                    self.print_details("Proposed: ",
                        utils.markup_to_ansi(self.tokenizer.decode(proposed_generation.past_token_ids[0, :-1], skip_special_tokens=False), "italic green"),
                        utils.markup_to_ansi(self.tokenizer.decode(proposed_generation.past_token_ids[0, -1:], skip_special_tokens=False), "italic red"),
                        utils.markup_to_ansi(self.tokenizer.decode(proposed_generation.ahead_token_ids[0, :], skip_special_tokens=False), "italic blue")
                    )

                    if self.rejection_sampling:
                        accepted = self.acceptance_criteria(proposed_generation)
                    else:
                        accepted = True

                if do_regeneration:
                    if not accepted:
                        self.n_rejections += 1
                        self.print_logs(f"Revision rejected. Regeneration: {self.n_generations} / {self.max_n_generations}, Rejection: {self.n_rejections} / {self.n_generations}")
                    else:
                        self.print_logs(f"Revision accepted. Regeneration: {self.n_generations} / {self.max_n_generations}, Rejection: {self.n_rejections} / {self.n_generations}")

                if accepted:
                    self.generation_states = proposed_generation
                    self.generation_len += 1
                else:
                    self.commit_n_tokens(1)

                self.print_logs(f"Generated {self.generation_len} tokens.")

                # Stop if the stop string is generated
                stop = self.should_stop()
                if stop["stop"]:
                    self.print_logs(f"Meet stop string, abort.")

                    if stop["trim_len"] > 0:
                        self.past_token_ids = self.past_token_ids[:, :-stop["trim_len"]]                    
                    break


                # Early stop if we have used up all regeneration limit (calls of completion)
                if self.n_generations >= self.max_n_generations:
                    self.print_logs(f"Hit regeneration limit: {self.n_generations} / {self.max_n_generations}, abort.")

                    self.commit_n_tokens(self.ahead_token_ids.shape[1])
                    break

        if not return_prompt:
            self.past_token_ids = self.past_token_ids[:, self.prompt_len:]

        return self.past_token_ids

    def get_stats(self):
        return {
            "n_generations": self.n_generations,
            "n_rejections": self.n_rejections,

            "prompt_tokens_total": self.prompt_len,
            "n_llm_calls": self.num_llm_calls,

            "n_generated_tokens": self.num_generated_tokens,
            "n_optimized_tokens": self.num_optimized_tokens,
            "n_rewarded_tokens": self.num_rewarded_tokens,
        }

    def generate_ahead(self, new_tokens):

        if new_tokens is None:
            proposed_past_token_ids = self.past_token_ids
        else:
            proposed_past_token_ids = torch.cat([self.past_token_ids, new_tokens], dim=-1)

        if self.backend == "huggingface":
            proposal = self.generate_ahead_hf(proposed_past_token_ids, past_key_values=self.past_key_values)
        elif self.backend == "vllm":
            proposal = self.generate_ahead_vllm(proposed_past_token_ids)
        else:
            raise ValueError(f"Backend {self.backend} is not supported.")

        self.n_generations += 1

        return proposal

    @torch.no_grad()
    def generate_ahead_vllm(self, proposed_past_token_ids):

        # 1) Use vLLM server /v1/completions to generate continuation text
        # Build full textual prefix including the chat template and already generated tokens
        generation_len = proposed_past_token_ids.shape[1] - self.prompt_len

        if self.vllm_output_type == "token_ids":
            prefix_ids = proposed_past_token_ids[0].tolist()
        elif self.vllm_output_type == "text":
            prefix_text = self.tokenizer.decode(proposed_past_token_ids[0], skip_special_tokens=False)
        else:
            raise ValueError(f"Unknown vLLM output type: {self.vllm_output_type}")

        max_new_tokens = (self.max_length - generation_len) if self.max_length > 0 else None
        temperature = self.gen_ahead_config.get("temperature", 0.0) or 0.0
        top_p = self.gen_ahead_config.get("top_p", 1.0)
        stop = self.stop_strings if self.stop_strings is not None else None

        url = self.vllm_url.rstrip("/") + "/v1/completions"
        body = {
            "model": self.vllm_model_name or "unknown",
            "prompt": prefix_ids if self.vllm_output_type == "token_ids" else prefix_text,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(self.gen_ahead_config.get("top_k")) if self.gen_ahead_config.get("top_k") is not None else None,
            "max_tokens": int(max_new_tokens) if max_new_tokens is not None else None,
            "stop": stop,
            "return_token_ids": (self.vllm_output_type == "token_ids"),
            "stream": False,
            # Seed controls vLLM sampling deterministically per request
            "seed": self.current_seed
        }
        # Remove None fields if any
        body = {k: v for k, v in body.items() if v is not None}

        try:
            resp = requests.post(url, json=body, timeout=600)
            resp.raise_for_status()
            data = resp.json()

            if self.vllm_output_type == "token_ids":
                out_ids = torch.tensor([data["choices"][0].get("token_ids", [])], dtype=torch.int64, device=self.device)
                input_ids = torch.cat([proposed_past_token_ids, out_ids], dim=-1)

            elif self.vllm_output_type == "text":
                out_text = data["choices"][0].get("text", "")
                final_text = prefix_text + out_text
                input_ids = self.tokenizer.encode(final_text, add_special_tokens=False, return_tensors="pt").to(self.device)

            else:
                raise ValueError(f"Unknown vLLM output type: {self.vllm_output_type}")

        except Exception as e:
            raise RuntimeError(f"vLLM request failed: {repr(e)}")

        # 2) Prefill with HF model to obtain logits and KV cache for the entire final sequence

        total_len = input_ids.shape[1]
        ctx_len = self.prompt_len + generation_len
        n_llm_generated_tokens = max(0, total_len - ctx_len)

        with torch.no_grad():
            outputs = self.model(
                input_ids,
                use_cache=False,
                return_dict=True,
                output_hidden_states=(self.embedder_type == "latents"),
                # the predicted logits are shifted to right by 1 token. so we keep the last (total_len - ctx_len + 1) tokens.
                # the first (ctx_len - 1) tokens are the needed context whlie the last logit is the prediction for the next token of the last given token.
                logits_to_keep=total_len - ctx_len + 1
            )

        # 3) Partition tokens and caches into past and ahead parts
        proposed_past_token_ids = input_ids[:, :ctx_len]
        proposed_ahead_token_ids = input_ids[:, ctx_len:]

        # Drop the last step: it predicts the next token beyond the generated tail.
        if self.embedder_type == "logits":
            proposed_ahead_latents = outputs.logits[:, :-1]
        elif self.embedder_type == "latents":
            proposed_ahead_latents = outputs.hidden_states[-1][:, :-1, :]
        else:
            raise ValueError(f"Unknown embedder type: {self.embedder_type}")

        self.num_generated_tokens += n_llm_generated_tokens
        # num of llm calls = num of auto-regressively generated tokens
        self.num_llm_calls += n_llm_generated_tokens

        return GenerationStates(
            past_token_ids=proposed_past_token_ids,
            ahead_token_ids=proposed_ahead_token_ids,
            ahead_latents=proposed_ahead_latents,
        )


    @torch.no_grad()
    def generate_ahead_hf(self, proposed_past_token_ids, past_key_values=None):

        cxt_len = int(proposed_past_token_ids.shape[1])
        generation_len = cxt_len - self.prompt_len

        with torch.random.fork_rng(devices=[self.device]):
            # Optional deterministic seeding for HF generate. We cannot pass a `generator`
            # kwarg to `model.generate` with this transformers version, so we seed globally.
            self.seed_if_needed()

            outputs = self.model.generate(
                # NOTE: HF generation() requires giving the full context even with kv cache (instead of just the previously generated token)
                proposed_past_token_ids,
                past_key_values=past_key_values,
                use_cache=True,

                tokenizer=self.tokenizer,
                max_new_tokens=(self.max_length - generation_len) if self.max_length > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
                stop_strings=self.stop_strings,

                return_dict_in_generate=True,
                output_logits=(self.embedder_type == "logits"),
                output_hidden_states=(self.embedder_type == "latents"),

                do_sample=self.gen_ahead_config['temperature'] > 0,
                **self.gen_ahead_config
            )
        past_token_ids = outputs.sequences[:, :cxt_len]
        ahead_token_ids = outputs.sequences[:, cxt_len:]

        all_past_key_values = outputs.past_key_values
        past_key_values = slice_kv_cache(all_past_key_values, slice(0, cxt_len))
        ahead_key_values = slice_kv_cache(all_past_key_values, slice(cxt_len, None))

        if self.embedder_type == "logits":
            ahead_latents = torch.cat(outputs.logits, dim=0).unsqueeze(0)

        elif self.embedder_type == "latents":
            step_latents = [step_hidden[-1][:, -1:, :] for step_hidden in outputs.hidden_states]
            if len(step_latents) > 0:
                ahead_latents = torch.cat(step_latents, dim=1)
            else:
                hidden_size = int(self.model.config.hidden_size)
                ahead_latents = torch.empty(
                    (past_token_ids.shape[0], 0, hidden_size),
                    dtype=self.model.get_input_embeddings().weight.dtype,
                    device=past_token_ids.device,
                )
        else:
            raise ValueError(f"Unknown embedder type: {self.embedder_type}")

        # num of llm calls = num of auto-regressively generated tokens
        self.num_generated_tokens += int(ahead_token_ids.shape[1])
        self.num_llm_calls += int(ahead_token_ids.shape[1])

        return GenerationStates(
            past_token_ids=past_token_ids,
            ahead_token_ids=ahead_token_ids,
            ahead_latents=ahead_latents,
            past_key_values=past_key_values,
            ahead_key_values=ahead_key_values,
        )

    
    @torch.no_grad()
    def sample_token(self, logits):

        # seed for randomness control
        generator = None
        if self.current_seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.current_seed)

        temperature = self.resample_config["temperature"]
        top_k = self.resample_config["top_k"]
        top_p = self.resample_config["top_p"]

        # 1. Apply temperature scaling
        if temperature <= 0:
            temperature = 1.0
        logits = logits / temperature

        # 2. Apply top-k filtering
        if top_k > 0:
            # Get the top k logits and their values
            top_k_values, _ = torch.topk(logits, top_k)
            # Create a mask for tokens that are not in the top-k
            # by finding the value of the k-th element
            min_value_to_keep = top_k_values[:, -1, None]
            # Set all logits below this value to -inf
            logits[logits < min_value_to_keep] = -float('inf')

        # 3. Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Create a mask for tokens to remove
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the mask to the right to keep the first token that crosses the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Scatter the mask back to the original vocabulary order
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('inf')

        # 4. Sample the next token from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1, generator=generator)

        return next_token


    @torch.no_grad()
    def should_stop(self):
        # check stop string
        for stop_token_ids in self.stop_token_ids:
            stop_len = stop_token_ids.shape[-1]
            tail_ids = self.past_token_ids[:, -stop_len:]
            if torch.equal(tail_ids, stop_token_ids.to(tail_ids.device)):
                return {"stop": True, "trim_len": stop_len}
        
        # check generation length
        if self.max_length > 0 and self.generation_len >= self.max_length:
            return {"stop": True, "trim_len": 0}

        return {"stop": False, "trim_len": 0}

