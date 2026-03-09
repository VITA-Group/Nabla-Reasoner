import os
import math
import numpy as np

import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

import pdb

from templates import GenerationTemplate, ORMTemplate
import utils

def straight_through_softmax(logits, tau=1.0, hard=False, gumbel_noise=1.0, dim=-1):

    if gumbel_noise > 0:
        # Deterministic Gumbel(0,1): -log(-log(U))
        # Clamp U for numerical stability
        U = torch.rand_like(logits, memory_format=torch.legacy_contiguous_format)
        U = U.clamp_(min=1e-6, max=1. - 1e-6)
        gumbels = -torch.log(-torch.log(U)) * gumbel_noise
        y_soft = F.softmax((logits + gumbels) / tau, dim=dim)
    else:
        # Standard Softmax / Straight-Through: Deterministic
        y_soft = F.softmax(logits / tau, dim=dim)

    if hard:
        # Straight-Through Estimator trick
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        # Combine hard forward pass with soft backward pass
        return (y_hard - y_soft).detach() + y_soft
    else:
        # If not hard, just return the soft probabilities
        return y_soft

class DiffAnyToEmbedding(nn.Module):

    def __init__(self, lm_model, lm_tokenizer, rm_model, rm_tokenizer):
        super().__init__()

        self.lm_tokenizer = lm_tokenizer
        self.rm_tokenizer = rm_tokenizer

    def is_initialized(self):
        raise NotImplementedError("No implementation for base model.")

    def initialize(self, init_data):
        raise NotImplementedError("No implementation for base model.")

    def deconstruct(self):
        raise NotImplementedError("No implementation for base model.")

    def forward(self, *args, **kwargs):
        raise NotImplementedError("No implementation for base model.")

    def get_logits(self):
        raise NotImplementedError("No implementation for base model.")

    def argmax_decode(self):
        raise NotImplementedError("No implementation for base model.")


class DiffLogitsToEmbedding(DiffAnyToEmbedding):

    def __init__(self, lm_model, lm_tokenizer, rm_model, rm_tokenizer, hard=True, temperature=1., gumbel_noise=-1.):

        super().__init__(lm_model, lm_tokenizer, rm_model, rm_tokenizer)

        self.hard = hard
        self.tau = temperature
        self.gumbel_noise = gumbel_noise

        self.latents_to_tune = None

        self.register_buffer("lm_embed_in", lm_model.get_input_embeddings().weight)
        # Reward model may be optional; set RM embeddings only if provided
        
        if rm_model is not None and rm_tokenizer is not None:
            aligned_rm_embed = utils.align_vocab(
                rm_model.get_input_embeddings().weight,
                self.rm_tokenizer,
                self.lm_tokenizer,
            )
            # Register as buffer for proper device/dtype moves
            self.register_buffer("rm_embed_in", aligned_rm_embed)
        else:
            self.rm_embed_in = None

    def is_initialized(self):
        return self.latents_to_tune is not None

    def initialize(self, init_logits):
        if self.latents_to_tune is not None:
            del self.latents_to_tune
        self.latents_to_tune = nn.Parameter(init_logits.clone())

    def deconstruct(self):
        if self.latents_to_tune is not None:
            del self.latents_to_tune
        self.latents_to_tune = None

    def forward(self, onehot_only=False):
        soft_one_hot_y = straight_through_softmax(self.latents_to_tune,
            tau=self.tau,
            hard=self.hard,
            gumbel_noise=self.gumbel_noise,
            dim=-1,
        )

        if not onehot_only:
            lm_soft_embeds = torch.matmul(soft_one_hot_y.to(self.lm_embed_in.dtype), self.lm_embed_in)
            ret = dict(soft_onehot=soft_one_hot_y, lm_embeds=lm_soft_embeds)
            if getattr(self, "rm_embed_in", None) is not None:
                rm_soft_embeds = torch.matmul(soft_one_hot_y.to(self.rm_embed_in.dtype), self.rm_embed_in)
                ret["rm_embeds"] = rm_soft_embeds
            return ret
        else:
            return dict(soft_onehot=soft_one_hot_y)

    @torch.no_grad()
    def get_logits(self):
        return self.latents_to_tune

    @torch.no_grad()
    def argmax_decode(self):
        final_token_ids = torch.argmax(self.latents_to_tune, dim=-1)
        return final_token_ids


# instead of starting from logits to differentiable one-hot emb, starting from hidden states, the hidden states of the LLM's last layer before LM head, and optimizing in that space. 
class DiffLatentsToEmbedding(DiffAnyToEmbedding):

    def __init__(self, lm_model, lm_tokenizer, rm_model, rm_tokenizer, hard=True, temperature=1., gumbel_noise=-1.):
        super().__init__(lm_model, lm_tokenizer, rm_model, rm_tokenizer)

        self.hard = hard
        self.tau = temperature
        self.gumbel_noise = gumbel_noise

        self.latents_to_tune = None

        self.register_buffer("lm_embed_in", lm_model.get_input_embeddings().weight)

        lm_head = lm_model.get_output_embeddings()
        if lm_head is None or not hasattr(lm_head, "weight"):
            raise ValueError("LM output head is required for DiffLatentsToEmbedding.")
        self.lm_head = lm_head

        if rm_model is not None and rm_tokenizer is not None:
            aligned_rm_embed = utils.align_vocab(
                rm_model.get_input_embeddings().weight,
                self.rm_tokenizer,
                self.lm_tokenizer,
            )
            self.register_buffer("rm_embed_in", aligned_rm_embed)
        else:
            self.rm_embed_in = None

    def is_initialized(self):
        return self.latents_to_tune is not None

    def initialize(self, init_latents):
        hidden_dim = self.lm_head.weight.shape[-1]
        if init_latents.shape[-1] != hidden_dim:
            raise ValueError(
                f"`DiffLatentsToEmbedding` expects soft_token_init with last dim = hidden size ({hidden_dim}), "
                f"but got {init_latents.shape[-1]}."
            )
        if self.latents_to_tune is not None:
            del self.latents_to_tune
        self.latents_to_tune = nn.Parameter(init_latents.clone())

    def deconstruct(self):
        if self.latents_to_tune is not None:
            del self.latents_to_tune
        self.latents_to_tune = None

    def project_to_logits(self):
        return self.lm_head(self.latents_to_tune)

    def forward(self, onehot_only=False):
        token_logits = self.project_to_logits()
        soft_one_hot_y = straight_through_softmax(
            token_logits,
            tau=self.tau,
            hard=self.hard,
            gumbel_noise=self.gumbel_noise,
            dim=-1,
        )

        if onehot_only:
            return dict(soft_onehot=soft_one_hot_y)

        lm_soft_embeds = torch.matmul(soft_one_hot_y.to(self.lm_embed_in.dtype), self.lm_embed_in)
        ret = dict(soft_onehot=soft_one_hot_y, lm_embeds=lm_soft_embeds)
        if getattr(self, "rm_embed_in", None) is not None:
            rm_soft_embeds = torch.matmul(soft_one_hot_y.to(self.rm_embed_in.dtype), self.rm_embed_in)
            ret["rm_embeds"] = rm_soft_embeds
        return ret

    @torch.no_grad()
    def get_logits(self):
        return self.project_to_logits()

    @torch.no_grad()
    def argmax_decode(self):
        final_token_ids = torch.argmax(self.project_to_logits(), dim=-1)
        return final_token_ids


class LatentTrainer:

    def __init__(self, lm_model, lm_tokenizer, rm_model, rm_tokenizer,
        max_iters = 1000, learning_rate = 1e-3, min_lr = None, min_lr_ratio = 0.1, weight_decay = 0.,
        warmup_iters = None, warmup_iters_ratio = 0., reward_coeff = 0.1, nll_coeff=1e-3, lr_scheduler_type='cosine',
        device=None, mixed_precision=torch.bfloat16, grad_caching=False, update_postfix=False, embed_args=dict(),
        embedder_type="logits", show_train_pbar=False, show_train_logs=False,
    ):

        self.lm_model = lm_model
        self.rm_model = rm_model

        self.lm_tokenizer = lm_tokenizer
        self.rm_tokenizer = rm_tokenizer

        self.max_iters = max_iters
        self.warmup_iters = warmup_iters if warmup_iters is not None else math.floor(self.max_iters * warmup_iters_ratio)
        self.lr = learning_rate
        self.min_lr_ratio = (min_lr / self.lr) if min_lr is not None else min_lr_ratio
        self.wd = weight_decay
        self.reward_coeff = reward_coeff
        self.nll_coeff = nll_coeff
        self.lr_scheduler_type = lr_scheduler_type

        self.mixed_precision = mixed_precision
        assert self.mixed_precision in (torch.float32, torch.bfloat16), f"Mixed precision {self.mixed_precision} not supported! Only (torch.float32, torch.bfloat16) are supported."

        self.device = device
        if self.device is None:
            self.device = utils.infer_device_from_model(self.lm_model)

        self.grad_caching = grad_caching

        embedder_type = str(embedder_type).lower()
        if embedder_type == "logits":
            embedder_cls = DiffLogitsToEmbedding
        elif embedder_type == "latents":
            embedder_cls = DiffLatentsToEmbedding
        else:
            raise ValueError(f"Unknown embedder_type: {embedder_type}. Supported: logits, latents.")

        self.soft_embedder = embedder_cls(self.lm_model, self.lm_tokenizer, self.rm_model, self.rm_tokenizer, **embed_args)
        self.soft_embedder = self.soft_embedder.to(self.device)

        self.show_train_pbar = show_train_pbar
        self.show_train_logs = show_train_logs
        assert (not self.show_train_logs) or self.show_train_pbar, "`show_train_logs` requires `show_train_pbar` to be True."

        self.update_postfix = update_postfix

    def print_train_logs(self, *args, **kwargs):
        if self.show_train_logs:
            print(*args, **kwargs)

    def optimize(self, prompt_text, generated_token_ids, soft_token_init):

        num_llm_calls = 0
        num_grad_steps = 0

        lm_template = GenerationTemplate(prompt_text, generated_token_ids, self.lm_model, self.lm_tokenizer)
        use_reward = (self.rm_model is not None and self.reward_coeff is not None and self.reward_coeff != 0)
        rm_template = None
        if use_reward:
            rm_template = ORMTemplate(prompt_text, generated_token_ids, self.rm_model, self.rm_tokenizer)

        self.soft_embedder.initialize(soft_token_init)

        # Set up the optimizer to tune only these logits.
        optimizer = torch.optim.Adam(self.soft_embedder.parameters(), lr=self.lr, weight_decay=self.wd)
        lr_scheduler = utils.get_scheduler(
            self.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.warmup_iters,
            num_training_steps=self.max_iters,
            min_lr_ratio=self.min_lr_ratio,
        )

        current_token_ids, current_response_text = self.argmax_decode()

        self.print_train_logs("Generation template: ", self.lm_tokenizer.decode(lm_template.apply_to_token_ids(current_token_ids)[0]))
        if use_reward:
            self.print_train_logs("Reward template: ", self.rm_tokenizer.decode(rm_template.apply_to_token_ids(current_token_ids)[0]))
        self.print_train_logs(f"==== Iter 0/{self.max_iters} | Loss: N/A | Log P(y): N/A | Reward: {self.get_reward_for_text(prompt_text, current_response_text):.4f}")
        self.print_train_logs(current_response_text)

        # enforce gradient computation for the first step
        cached_grad = None

        # if `show_train_logs` is enabled, print text progress. otherwise, use progress bar
        if (not self.show_train_pbar) or self.show_train_logs:
            train_iter = range(self.max_iters)
        else:
            train_iter = tqdm.tqdm(range(self.max_iters), desc="Optimizing")

        for it in train_iter:
            optimizer.zero_grad()

            skip_grad_compute = self.grad_caching and (cached_grad is not None)

            with torch.autocast("cuda", dtype=self.mixed_precision):

                if skip_grad_compute:
                    assert cached_grad is not None

                    soft_onehot = self.soft_embedder(onehot_only=True)['soft_onehot']
                    loss = torch.dot(cached_grad.view(-1), soft_onehot.view(-1))
                else:
                    outputs = self.soft_embedder(onehot_only = False)
                    soft_onehot = outputs['soft_onehot']

                    # Retain grad only if needed for grad caching
                    if self.grad_caching and soft_onehot.requires_grad:
                        soft_onehot.retain_grad()

                    if not use_reward:
                        del outputs['rm_embeds']

                    loss_dict = self.compute_loss(outputs, lm_template, rm_template)
                    loss = loss_dict['loss']
                    loglikelihood = loss_dict['loglikelihood']
                    reward = loss_dict['reward']

                    # One for LM forward, one for RM forward, one for LM backward, one for RM backward
                    num_llm_calls += 4
                    num_grad_steps += 1

            loss.backward()

            # Mask gradient for postfix if `update_postfix` is False
            if not self.update_postfix:
                param = next(iter(self.soft_embedder.parameters()))
                if param.grad is not None:
                    mask = torch.zeros_like(param.grad)
                    mask[:, :1, :] = 1.0
                    param.grad.mul_(mask)

            optimizer.step()
            lr_scheduler.step()

            # if gradient has been recomputed, then cache it now
            # cache gradient only if `self.grad_caching` is enabled
            if not skip_grad_compute and self.grad_caching:
                cached_grad = soft_onehot.grad.detach().clone()

            response_token_ids, response_text = self.argmax_decode()
            if not torch.equal(response_token_ids, current_token_ids):
                current_token_ids = response_token_ids
                cached_grad = None
            if response_text != current_response_text:
                if reward is not None:
                    log_str = f"Loss: {loss.item():.4f} | Log P(y): {loglikelihood.item():.4f} | Reward: {reward.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.3e}"
                else:
                    log_str = f"Loss: {loss.item():.4f} | Log P(y): {loglikelihood.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.3e}"
                if self.show_train_logs:
                    self.print_train_logs(f"==== Iter {it+1: >3}/{self.max_iters} | {log_str}")
                    utils.print_str_diff(current_response_text, response_text)
                elif self.show_train_pbar:
                    train_iter.set_description(log_str)
                current_response_text = response_text


        optimized_soft_tokens = self.soft_embedder.get_logits().detach().clone()

        # release memory
        self.soft_embedder.deconstruct()
        del optimizer, lr_scheduler
        torch.cuda.empty_cache()

        return {
            "logits": optimized_soft_tokens,
            "num_llm_calls": num_llm_calls,
            "num_grad_steps": num_grad_steps,
        }

    def compute_gradient_for_onehots(self, prompt_text, generated_token_ids, soft_token_init):

        lm_template = GenerationTemplate(prompt_text, generated_token_ids, self.lm_model, self.lm_tokenizer)
        use_reward = (self.rm_model is not None and self.reward_coeff is not None and self.reward_coeff != 0)
        rm_template = None
        if use_reward:
            rm_template = ORMTemplate(prompt_text, generated_token_ids, self.rm_model, self.rm_tokenizer)

        self.soft_embedder.initialize(soft_token_init)

        probe_outputs = self.soft_embedder(onehot_only=False)
        probe_outputs['soft_onehot'].retain_grad()
        if not use_reward:
            del probe_outputs['rm_embeds']

        probe_loss_dict = self.compute_loss(probe_outputs, lm_template, rm_template)
        probe_loss_dict['loss'].backward()
        grad_onehots = probe_outputs['soft_onehot'].grad.detach().clone()

        # clean up
        self.soft_embedder.deconstruct()
        torch.cuda.empty_cache()

        return grad_onehots

    def compute_loss(self, soft_embed_outputs, lm_template, rm_template=None):
        prompt_len = lm_template.prompt_ids.shape[1]

        soft_onehot = soft_embed_outputs['soft_onehot']
        lm_soft_embeds = soft_embed_outputs['lm_embeds']

        # Calculate likelihood log P(y)
        lm_outputs = self.lm_model(inputs_embeds=lm_template.apply(lm_soft_embeds))
        pred_logits = lm_outputs.logits[..., prompt_len-1:-1, :]
        log_pred_probs = F.log_softmax(pred_logits, dim=-1)
        loglikelihood = (log_pred_probs * soft_onehot).sum() * self.nll_coeff

        # Calculate reward r(y) if available
        if 'rm_embeds' in soft_embed_outputs:
            reward = self.rm_model(inputs_embeds=rm_template.apply(soft_embed_outputs['rm_embeds'])).logits[0][0]
            loss = -(loglikelihood + self.reward_coeff * reward)
        else:
            reward = None
            loss = -loglikelihood

        return {
            "loss": loss,
            "loglikelihood": loglikelihood,
            "reward": reward,
        }

    def argmax_decode(self):
        assert self.soft_embedder.is_initialized(), "Can be only called during optimization!"
        final_token_ids = self.soft_embedder.argmax_decode()
        final_response_text = self.lm_tokenizer.decode(final_token_ids.squeeze(0), skip_special_tokens=True)
        return final_token_ids, final_response_text

    def get_reward_for_token_ids(self, prompt_text, generated_token_ids):
        # specify None toreward model to avoid applying the reward model and save memory
        rm_template = ORMTemplate(prompt_text, None, rm_model=None, rm_tokenizer=self.rm_tokenizer)
        with torch.no_grad():
            score = self.rm_model(input_ids=rm_template.apply_to_token_ids(generated_token_ids)).logits[0][0].item()

        return score

    def get_reward_for_text(self, prompt, response):
        if self.rm_model is None or self.reward_coeff is None or self.reward_coeff == 0:
            return 0.0
        conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]

        # Format and tokenize the conversations
        conv_formatted = self.rm_tokenizer.apply_chat_template(conv, tokenize=False)
        conv_tokenized = self.rm_tokenizer(conv_formatted, return_tensors="pt").to(self.device)

        # Get the reward scores
        with torch.no_grad():
            score = self.rm_model(**conv_tokenized).logits[0][0].item()

        return score


# Backward compatibility for existing imports/usages.
LogitTrainer = LatentTrainer
