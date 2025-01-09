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
# 2) Utility: compute_completion_loss_with_channels
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
# 3) Multi-Class Forced Choice Evaluation
# ------------------------------------------------------------------
def evaluate_multi_class_forced_choice(
    model,
    shards_dir="validation_datasets_imageNet/shards",
    segment_size=512,
    num_trials_per_subject=5,
    device="cpu"
):
    """
    1) Loads all .pt shards from `shards_dir`.
    2) Extracts subject, image_id from 'original_pair'.
    3) Groups data by (subject -> image_id -> list_of_shards).
    4) For each subject:
       - Repeatedly pick a random (subject, image_id_correct, shard) as the "correct" image.
       - From that shard, pick prompt chunk + correct chunk (non-overlapping).
       - Gather "wrong" completions from *other images* of the same subject (one random chunk each).
       - Compute loss for correct completion and for each wrong completion.
       - If correct_loss < min(all_wrong_losses), we count +1 to the correct_count.
    5) Return overall accuracy across all subjects.

    NOTE: You can also compute per-subject accuracy if desired.
    """
    # 3.1) Load shards
    all_pt_files = [
        f for f in os.listdir(shards_dir)
        if f.endswith(".pt") and "shard_train_" in f
    ]
    if not all_pt_files:
        print(f"No .pt files found in {shards_dir}")
        return 0.0

    print(f"Found {len(all_pt_files)} shard files in '{shards_dir}'. Loading...")

    data_by_subject = {}

    for pt_file in all_pt_files:
        full_path = os.path.join(shards_dir, pt_file)
        shard_data = torch.load(full_path)

        tokens = shard_data['tokens']
        channels = shard_data['channels']
        pair_info = shard_data['original_pair']
        # example: pair_info = (
        #    "subject_0_image_n02492035_gran_coarse_coeffs.txt",
        #    "subject_0_image_n02492035_gran_coarse_channels.txt"
        # )
        # We parse the first part, typically "..._coeffs.txt"

        # 3.2) Parse subject + image from the filename string
        #     "subject_0_image_n02492035_gran_coarse_coeffs.txt"
        #      subject_0
        #      image_n02492035_gran_coarse
        coeffs_filename = pair_info[0]  # e.g. subject_0_image_n02492035_gran_coarse_coeffs.txt
        # remove the trailing '_coeffs.txt'
        basename = coeffs_filename.replace('_coeffs.txt', '')
        # now we have something like "subject_0_image_n02492035_gran_coarse"
        parts = basename.split('_image_')  # e.g. ["subject_0", "n02492035_gran_coarse"]
        if len(parts) != 2:
            print(f"Warning: unexpected file format: {basename}")
            continue
        subject_str = parts[0]   # e.g. "subject_0"
        image_str   = parts[1]   # e.g. "n02492035_gran_coarse"

        # We'll store it in a dict: data_by_subject[subject_str][image_str] -> list of dicts
        if subject_str not in data_by_subject:
            data_by_subject[subject_str] = {}
        if image_str not in data_by_subject[subject_str]:
            data_by_subject[subject_str][image_str] = []
        data_by_subject[subject_str][image_str].append({
            'tokens': tokens,
            'channels': channels
        })

    # ------------------------------------------------------------------
    # 4) For each subject, do multi-class forced choice
    # ------------------------------------------------------------------
    total_trials = 0
    correct_count = 0

    subjects = list(data_by_subject.keys())
    print(f"Subjects found: {subjects}")

    for subject in subjects:
        images_dict = data_by_subject[subject]  # { image_id: [list_of_shards], ... }
        image_ids = list(images_dict.keys())

        # If there's only 1 image for that subject, we can't do multi-class with "other images"
        if len(image_ids) < 2:
            print(f"Subject {subject} has only {len(image_ids)} images. Skipping multi-class for this subject.")
            continue

        print(f"\n=== Subject {subject} has {len(image_ids)} images: {image_ids}")

        # We'll do num_trials_per_subject attempts
        for trial_i in range(num_trials_per_subject):
            print(trial_i)
            # 4.1) Pick a random correct image
            correct_image_id = random.choice(image_ids)
            shards_for_correct_image = images_dict[correct_image_id]

            # 4.2) Pick a random shard from the correct_image
            correct_shard = random.choice(shards_for_correct_image)
            tokens_correct_shard = correct_shard['tokens']
            chans_correct_shard  = correct_shard['channels']

            # 4.3) We need to pick a random chunk for the prompt + a chunk for the correct completion
            #     Each chunk has length = segment_size. So we need at least 2*segment_size tokens available.
            total_len = tokens_correct_shard.size(0)
            if total_len < 2 * segment_size:
                # not enough tokens for prompt+completion
                # skip this trial
                continue

            # pick a random start that allows 2*segment_size tokens
            max_start = total_len - 2*segment_size
            start_idx = random.randint(0, max_start)

            # prompt chunk
            prompt_tokens = tokens_correct_shard[start_idx : start_idx + segment_size]
            prompt_chans  = chans_correct_shard[start_idx : start_idx + segment_size]

            # correct chunk
            correct_tokens = tokens_correct_shard[start_idx + segment_size : start_idx + 2*segment_size]
            correct_chans  = chans_correct_shard[start_idx + segment_size : start_idx + 2*segment_size]

            # 4.4) Get "wrong" completions from the other images of the same subject
            #      We'll gather one random chunk from each other image
            wrong_losses = []
            for other_image_id in image_ids:
                if other_image_id == correct_image_id:
                    continue  # skip the correct image

                # pick a random shard from that other image
                shards_for_other_image = images_dict[other_image_id]
                other_shard = random.choice(shards_for_other_image)
                tokens_other_shard = other_shard['tokens']
                chans_other_shard  = other_shard['channels']

                total_len_other = tokens_other_shard.size(0)
                if total_len_other < segment_size:
                    # can't pick a chunk of length segment_size
                    # skip
                    continue

                max_start_other = total_len_other - segment_size
                start_idx_other = random.randint(0, max_start_other)

                # "wrong" chunk from other image
                wrong_tokens = tokens_other_shard[start_idx_other : start_idx_other + segment_size]
                wrong_chans  = chans_other_shard[start_idx_other : start_idx_other + segment_size]

                # compute loss
                lw = compute_completion_loss_with_channels(
                    model,
                    prompt_tokens,
                    prompt_chans,
                    wrong_tokens,
                    wrong_chans,
                    device=device
                )
                wrong_losses.append(lw)

            # If we ended up with no wrong completions (e.g., all other images had insufficient tokens),
            # we skip
            if len(wrong_losses) == 0:
                continue

            # 4.5) Now compute the correct completion loss
            correct_loss = compute_completion_loss_with_channels(
                model,
                prompt_tokens,
                prompt_chans,
                correct_tokens,
                correct_chans,
                device=device
            )

            # 4.6) Multi-class decision
            # We count as correct if correct_loss < min(wrong_losses)
            total_trials += 1
            if correct_loss < min(wrong_losses):
                correct_count += 1
            print(correct_count/total_trials)
    # 5) Final results
    if total_trials == 0:
        print("No valid multi-class trials were performed.")
        return 0.0
    accuracy = correct_count / total_trials
    print(f"\nMulti-class forced-choice accuracy: {correct_count}/{total_trials} = {accuracy:.4f}")
    return accuracy

# ------------------------------------------------------------------
# 4) Main: load model, run evaluation
# ------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device('cpu')
    model = GPT(GPTConfig).to(device)

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

    model.config(checkpoint['config'])

    model.eval()

    # Finally, run multi-class forced choice
    acc = evaluate_multi_class_forced_choice(
        model=model,
        shards_dir="validation_datasets_imageNet/shards",
        segment_size=512,         # or another chunk size
        num_trials_per_subject=5, # how many random attempts per subject
        device=device
    )
    print(f"\nFinal multi-class forced-choice accuracy = {acc:.4f}")
