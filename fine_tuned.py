#!/usr/bin/env python3
import os
import random
import time
import pickle
from dataclasses import dataclass
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn

from tokenizer2 import BPE_RLE_Tokenizer as Tokenizer

# ------------------------------------------------------------------
# 1) Model & Tokenizer Setup
# ------------------------------------------------------------------
tokenizer = Tokenizer()
tokenizer.load_merges("neo_tokenizer/merges.json")
tokenizer.load_vocab("neo_tokenizer/vocab.json")

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # Flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 4140
    n_layer: int = 18
    n_head: int = 12
    n_embd: int = 768
    num_channels: int = 2

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            wce = nn.Embedding(config.num_channels, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # scale init for certain linear layers
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, channel_idx=None, targets=None):
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}")

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)       # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)       # (T, n_embd)
        if channel_idx is not None:
            cha_emb = self.transformer.wce(channel_idx)
            x = tok_emb + pos_emb + cha_emb
        else:
            x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        return logits, loss

# ------------------------------------------------------------------
# Utility: compute_completion_loss_with_channels
# ------------------------------------------------------------------
def compute_completion_loss_with_channels(
    model,
    prompt_tokens, prompt_channels,
    completion_tokens, completion_channels,
    device="cuda"
):
    """
    Returns the average cross-entropy loss on 'completion_tokens',
    given 'prompt_tokens' + 'prompt_channels'.
    """
    model.eval()
    with torch.no_grad():
        # 1) Concatenate prompt + completion
        full_tokens = torch.cat([prompt_tokens, completion_tokens], dim=0).unsqueeze(0).to(device)
        full_channels = torch.cat([prompt_channels, completion_channels], dim=0).unsqueeze(0).to(device)

        total_len = full_tokens.size(1)
        prompt_len = prompt_tokens.size(0)
        completion_len = completion_tokens.size(0)

        # 2) Build a mask that is 0 for prompt tokens, 1 for completion tokens
        mask_vals = [0]*prompt_len + [1]*completion_len
        mask = torch.tensor(mask_vals, device=device).unsqueeze(0)

        # 3) Forward
        logits, _ = model(idx=full_tokens, channel_idx=full_channels)  # (1, total_len, vocab_size)

        # 4) shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_tokens = full_tokens[:, 1:].contiguous()
        shift_mask   = mask[:, 1:].contiguous()

        # 5) Flatten
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_tokens = shift_tokens.view(-1)
        flat_mask   = shift_mask.view(-1)

        # 6) Cross entropy per token
        ce_per_token = F.cross_entropy(flat_logits, flat_tokens, reduction='none')

        # 7) Zero out the prompt region
        ce_completion = ce_per_token * flat_mask

        # 8) Average loss over completion tokens
        sum_loss = ce_completion.sum()
        num_completion_tokens = flat_mask.sum()
        avg_loss = sum_loss / (num_completion_tokens + 1e-9)

    return avg_loss.item()

# ------------------------------------------------------------------
# 1) Gather fine-tuning data first (same as before)
# ------------------------------------------------------------------
def gather_finetuning_data(
    shards_dir="validation_datasets_imageNet/shards",
    segment_size=512,
    num_samples_per_subject=5,
    device="cpu"
):
    """
    Loads shards from `shards_dir`.
    For each subject, tries to pick 'num_samples_per_subject' random (prompt, correct) pairs
    to fine-tune on (these are the "correct completions").
    Returns a list of dicts with {prompt_tokens, prompt_chans, completion_tokens, completion_chans}.
    """
    all_pt_files = [
        f for f in os.listdir(shards_dir)
        if f.endswith(".pt") and "shard_train_" in f
    ]
    if not all_pt_files:
        print(f"No .pt files found in {shards_dir}")
        return []

    data_by_subject = {}
    for pt_file in all_pt_files:
        full_path = os.path.join(shards_dir, pt_file)
        shard_data = torch.load(full_path)

        tokens = shard_data['tokens']
        channels = shard_data['channels']
        pair_info = shard_data['original_pair']

        # parse subject + image from the file info
        coeffs_filename = pair_info[0]
        basename = coeffs_filename.replace('_coeffs.txt', '')
        parts = basename.split('_image_')
        if len(parts) != 2:
            continue
        subject_str = parts[0]   # e.g. "subject_0"
        image_str   = parts[1]   # e.g. "n02492035_gran_coarse"

        if subject_str not in data_by_subject:
            data_by_subject[subject_str] = []
        data_by_subject[subject_str].append((tokens, channels))

    finetune_data = []
    for subject, shards_list in data_by_subject.items():
        if not shards_list:
            continue
        count = 0
        attempts = 0
        max_attempts = 50 * num_samples_per_subject
        while count < num_samples_per_subject and attempts < max_attempts:
            attempts += 1
            tokens_shard, chans_shard = random.choice(shards_list)
            if tokens_shard.size(0) < 2*segment_size:
                # not enough tokens in this shard
                continue
            total_len = tokens_shard.size(0)

            # randomly pick prompt chunk
            start_idx_prompt = random.randint(0, total_len - segment_size)
            prompt_tokens = tokens_shard[start_idx_prompt : start_idx_prompt + segment_size]
            prompt_chans  = chans_shard[start_idx_prompt : start_idx_prompt + segment_size]

            # randomly pick correct chunk
            start_idx_correct = random.randint(0, total_len - segment_size)
            correct_tokens = tokens_shard[start_idx_correct : start_idx_correct + segment_size]
            correct_chans  = chans_shard[start_idx_correct : start_idx_correct + segment_size]

            finetune_data.append({
                'prompt_tokens': prompt_tokens.clone(),
                'prompt_chans': prompt_chans.clone(),
                'completion_tokens': correct_tokens.clone(),
                'completion_chans': correct_chans.clone(),
                'subject': subject
            })
            count += 1

    random.shuffle(finetune_data)
    return finetune_data

# ------------------------------------------------------------------
# 2) Minimal Fine-Tuning Loop (replaces the old function)
# ------------------------------------------------------------------
def minimal_finetune_loop(
    model,
    data_list,
    device='cpu',
    max_steps=1000,
    grad_accum_steps=1,
    lr=1e-5,
    clip_grad=1.0
):
    """
    A minimal training loop that imitates your original structure:
      - total of 'max_steps' updates
      - optional gradient accumulation
      - random sampling from data_list each step (batch size = 1 for simplicity)
      - no checkpoint saving or logging
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    data_index = 0
    random.shuffle(data_list)

    for step in range(max_steps):
        print(step)
        # 1) If we've reached the end of data_list, reshuffle and reset
        if data_index >= len(data_list):
            random.shuffle(data_list)
            data_index = 0

        # 2) Grab the current example and increment data_index
        example = data_list[data_index]
        data_index += 1

        # Zero the gradients once per outer step
        optimizer.zero_grad()

        # We'll accumulate gradients over grad_accum_steps
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            # 3) Build the training sequence (prompt+completion)
            prompt_tokens = example['prompt_tokens'].to(device)
            prompt_chans = example['prompt_chans'].to(device)
            completion_tokens = example['completion_tokens'].to(device)
            completion_chans = example['completion_chans'].to(device)

            full_tokens = torch.cat([prompt_tokens, completion_tokens], dim=0).unsqueeze(0)
            full_chans = torch.cat([prompt_chans, completion_chans], dim=0).unsqueeze(0)

            # 4) Forward pass
            logits, loss = model(
                idx=full_tokens[:, :-1],
                channel_idx=full_chans[:, :-1],
                targets=full_tokens[:, 1:]
            )

            # If gradient accumulation is used, scale the loss
            loss = loss / grad_accum_steps
            loss_accum += loss.item()

            # Backward
            loss.backward()

        # 5) Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        # 6) Optimizer step
        optimizer.step()

        # 7) Print minimal info each step (optional)
        if (step % 100) == 0:
            print(f"[Step {step}] Train Loss (avg over micro-steps): {loss_accum:.4f}")

    # End of fine-tuning
    model.eval()

# ------------------------------------------------------------------
# 3) Multi-Class Forced Choice Evaluation (same as before)
# ------------------------------------------------------------------
def evaluate_multi_class_forced_choice(
    model,
    shards_dir="validation_datasets_imageNet/shards",
    segment_size=512,
    num_trials_per_subject=5,
    device="cpu"
):
    all_pt_files = [
        f for f in os.listdir(shards_dir)
        if f.endswith(".pt") and "shard_train_" in f
    ]
    if not all_pt_files:
        print(f"No .pt files found in {shards_dir}")
        return 0.0

    data_by_subject = {}
    for pt_file in all_pt_files:
        full_path = os.path.join(shards_dir, pt_file)
        shard_data = torch.load(full_path)

        tokens = shard_data['tokens']
        channels = shard_data['channels']
        pair_info = shard_data['original_pair']

        # parse subject + image
        coeffs_filename = pair_info[0]
        basename = coeffs_filename.replace('_coeffs.txt', '')
        parts = basename.split('_image_')
        if len(parts) != 2:
            continue
        subject_str = parts[0]
        image_str   = parts[1]

        if subject_str not in data_by_subject:
            data_by_subject[subject_str] = {}
        if image_str not in data_by_subject[subject_str]:
            data_by_subject[subject_str][image_str] = []
        data_by_subject[subject_str][image_str].append({
            'tokens': tokens,
            'channels': channels
        })

    total_trials = 0
    correct_count = 0

    subjects = list(data_by_subject.keys())
    for subject in subjects[0:1]:
        images_dict = data_by_subject[subject]
        image_ids = list(images_dict.keys())

        # We require at least 2 images to do multi-class forced choice
        if len(image_ids) < 2:
            continue

        for trial_i in range(num_trials_per_subject):
            print(trial_i)
            correct_image_id = random.choice(image_ids)
            shards_for_correct_image = images_dict[correct_image_id]
            correct_shard = random.choice(shards_for_correct_image)
            tokens_correct_shard = correct_shard['tokens']
            chans_correct_shard  = correct_shard['channels']

            total_len = tokens_correct_shard.size(0)
            if total_len < 2 * segment_size:
                continue

            # Prompt
            import random

            start_idx_prompt = random.randrange(0, total_len - segment_size + 1, 130)
            prompt_tokens = tokens_correct_shard[start_idx_prompt: start_idx_prompt + segment_size]
            prompt_chans  = chans_correct_shard[start_idx_prompt: start_idx_prompt + segment_size]

            # Correct completion
            start_idx_correct = random.randrange(0, total_len - segment_size + 1, 130)
            correct_tokens = tokens_correct_shard[start_idx_correct: start_idx_correct + segment_size]
            correct_chans  = chans_correct_shard[start_idx_correct: start_idx_correct + segment_size]

            # Wrong completions
            wrong_losses = []
            for other_image_id in image_ids:
                if other_image_id == correct_image_id:
                    continue
                shards_for_other_image = images_dict[other_image_id]
                other_shard = random.choice(shards_for_other_image)
                tokens_other_shard = other_shard['tokens']
                chans_other_shard  = other_shard['channels']

                total_len_other = tokens_other_shard.size(0)
                if total_len_other < segment_size:
                    continue
                max_start_other = total_len_other - segment_size
                start_idx_other = random.randint(0, max_start_other)
                wrong_tokens = tokens_other_shard[start_idx_other : start_idx_other + segment_size]
                wrong_chans  = chans_other_shard[start_idx_other : start_idx_other + segment_size]

                lw = compute_completion_loss_with_channels(
                    model,
                    prompt_tokens,
                    prompt_chans,
                    wrong_tokens,
                    wrong_chans,
                    device=device
                )
                wrong_losses.append(lw)

            if len(wrong_losses) == 0:
                continue

            # correct_loss
            correct_loss = compute_completion_loss_with_channels(
                model,
                prompt_tokens,
                prompt_chans,
                correct_tokens,
                correct_chans,
                device=device
            )

            total_trials += 1
            if correct_loss < min(wrong_losses):
                correct_count += 1
            print(correct_loss, (wrong_losses))
    if total_trials == 0:
        return 0.0
    accuracy = correct_count / total_trials
    print(f"\nMulti-class forced-choice accuracy: {correct_count}/{total_trials} = {accuracy:.4f}")
    return accuracy

# ------------------------------------------------------------------
# 4) Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device('mps')
    model = GPT(GPTConfig()).to(device)

    # Load your checkpoint
    checkpoint_path = 'log/model_14000_150M_small.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # retrieve the state_dict
    orig_sd = checkpoint['model']
    # remove "_orig_mod." from each key
    fixed_sd = {}
    for k, v in orig_sd.items():
        new_key = k.replace("_orig_mod.", "")
        fixed_sd[new_key] = v
    model.load_state_dict(fixed_sd, strict=True)

    # Optional: if your checkpoint has config updates
    # model.config(checkpoint['config'])

    # 1) Gather random (prompt, correct) pairs from your dataset
    finetune_data = gather_finetuning_data(
        shards_dir="validation_datasets_imageNet/shards",
        segment_size=512,
        num_samples_per_subject=1000,
        device=device
    )
    print(f"Gathered {len(finetune_data)} total fine-tuning examples.")

    # # 2) Fine-tune the model in memory using our minimal loop
    # if len(finetune_data) > 0:
    #     minimal_finetune_loop(
    #         model=model,
    #         data_list=finetune_data,
    #         device=device,
    #         max_steps=1000,       # total steps
    #         grad_accum_steps=2,  # example of gradient accumulation
    #         lr=1e-6,
    #         clip_grad=1.0
    #     )

    # 3) Evaluate with multi-class forced choice *using the finetuned model*
    accuracy = evaluate_multi_class_forced_choice(
        model=model,
        shards_dir="validation_datasets_imageNet/shards",
        segment_size=512,
        num_trials_per_subject=5,
        device=device
    )
    print(f"\nFinal multi-class forced-choice accuracy (fine-tuned) = {accuracy:.4f}")
