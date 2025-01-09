#!/usr/bin/env python3
import time

generate = True
import pickle
from dataclasses import dataclass
import numpy as np
import torch
import math
import os
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch import nn

import tokenizer2
from sandbox3 import tokenizer
from tokenizer2 import BPE_RLE_Tokenizer as Tokenizer
from utils import wavelet_reconstruct_window, dequantize_number

tokenizer = Tokenizer()
tokenizer.load_merges("neo_tokenizer/merges.json")
tokenizer.load_vocab("neo_tokenizer/vocab.json")

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
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
            wce = nn.Embedding(config.num_channels, config.n_embd),  # <-- new
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # NOTE: We now take an extra argument: channel_idx
    def forward(self, idx, channel_idx=None, targets=None):
        """
        idx: (B, T) tokens
        channel_idx: (B, T) channel IDs (e.g., 0 or 1)
        targets: (B, T) next-token predictions
        """
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, "
            f"block size is only {self.config.block_size}"
        )

        # forward the token, position, and (optionally) channel embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T,)
        pos_emb = self.transformer.wpe(pos)   # (T, n_embd)
        tok_emb = self.transformer.wte(idx)   # (B, T, n_embd)

        if channel_idx is not None:
            # Make sure channel_idx is the same shape as idx
            # channel_idx must be in [0..num_channels-1]
            cha_emb = self.transformer.wce(channel_idx)  # (B, T, n_embd)
            x = tok_emb + pos_emb + cha_emb
        else:
            # fallback if no channel_idx is provided
            x = tok_emb + pos_emb

        # pass through transformer
        for block in self.transformer.h:
            x = block(x)

        # final layernorm + linear head
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        return logits, loss
device='cpu'

import torch

import torch
import torch.nn.functional as F
import torch
import torch.nn.functional as F


def compute_completion_loss_with_channels(
        model,
        prompt_tokens, prompt_channels,
        completion_tokens, completion_channels,
        device="cuda"
):
    """
    Compute the average cross-entropy loss on 'completion_tokens' portion,
    given that 'prompt_tokens' is the initial context. The model expects:
        logits, _ = model(idx=tokens, channel_idx=channels)

    prompt_tokens     : Tensor of shape (prompt_len,)
    prompt_channels   : Tensor of shape (prompt_len,)
    completion_tokens : Tensor of shape (completion_len,)
    completion_channels : Tensor of shape (completion_len,)

    Returns: float (average CE loss on the completion tokens)
    """

    # 1) Concatenate prompt + completion for tokens and channels
    full_tokens = torch.cat([prompt_tokens, completion_tokens], dim=0).unsqueeze(0).to(device)
    full_channels = torch.cat([prompt_channels, completion_channels], dim=0).unsqueeze(0).to(device)

    total_len = full_tokens.size(1)
    prompt_len = prompt_tokens.size(0)
    completion_len = completion_tokens.size(0)

    # 2) Build a mask that is 0 for prompt tokens, 1 for completion tokens
    #    shape: (1, total_len)
    mask_vals = [0] * prompt_len + [1] * completion_len
    mask = torch.tensor(mask_vals, device=device).unsqueeze(0)

    # 3) Forward pass (no gradient) to get logits
    with torch.no_grad():
        # model returns logits shaped (B, T, vocab_size), possibly also a loss
        logits, _ = model(idx=full_tokens, channel_idx=full_channels)  # (1, total_len, vocab_size)

    # 4) We want to predict the next token at each position, so we shift
    #    by one in the time dimension.
    shift_logits = logits[:, :-1, :].contiguous()  # shape (1, total_len - 1, vocab_size)
    shift_tokens = full_tokens[:, 1:].contiguous()  # shape (1, total_len - 1)
    shift_mask = mask[:, 1:].contiguous()  # shape (1, total_len - 1)

    # 5) Flatten
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))  # ( (total_len-1), vocab_size )
    flat_tokens = shift_tokens.view(-1)  # ( (total_len-1) )
    flat_mask = shift_mask.view(-1)  # ( (total_len-1) )

    # 6) Cross entropy per token
    ce_per_token = F.cross_entropy(flat_logits, flat_tokens, reduction='none')  # shape (total_len-1,)

    # 7) Zero out the prompt region by multiplying with the mask
    ce_completion = ce_per_token * flat_mask

    # 8) Average loss over completion tokens
    sum_loss = ce_completion.sum()
    num_completion_tokens = flat_mask.sum()  # the count of '1's in mask
    avg_loss = sum_loss / (num_completion_tokens + 1e-9)

    return avg_loss.item()

def load_shard_group(shard_paths):
    """
    Given a list of shard paths, load them into memory and return
    a list of (tokens, channels) tuples.
    """
    group_data = []
    for path in shard_paths:
        shard = torch.load(path)  # e.g. {'tokens': ..., 'channels': ...}
        tokens = shard['tokens'].cpu()
        channels = shard['channels'].cpu()
        group_data.append((tokens, channels))
    return group_data
import random

def pick_random_segment(group_data, segment_size):
    """
    group_data : list of (tokens, channels) for each shard in this group
    segment_size: how many tokens to slice
    Returns: (seg_tokens, seg_channels)
    """
    # Pick a random shard
    shard_index = random.randint(0, len(group_data) - 1)
    tokens, channels = group_data[shard_index]

    # Check we can actually pick segment_size from this shard
    if tokens.size(0) < segment_size:
        raise ValueError(f"Shard {shard_index} has only {tokens.size(0)} tokens, need >= {segment_size}.")

    # The last valid offset is len(tokens) - segment_size
    max_offset = tokens.size(0) - segment_size
    offset = random.randint(0, max_offset)

    seg_tokens = tokens[offset : offset + segment_size]
    seg_channels = channels[offset : offset + segment_size]
    return seg_tokens, seg_channels, shard_index, offset
def pick_random_nonoverlapping_segment_in_same_shard(
    tokens, channels,
    used_range_start, used_range_end,
    segment_size
):
    """
    tokens, channels : Tensors (N, )
    used_range_start, used_range_end : the [start, end) region already used for the prompt
    segment_size : length of segment to sample
    Returns: (seg_tokens, seg_channels)
    """
    import random

    N = tokens.size(0)
    if N < segment_size:
        raise ValueError(f"Shard too small, only {N} tokens total.")

    # We'll build a list of valid offsets that do NOT overlap [used_range_start, used_range_end).
    # i.e. the segment [offset, offset+segment_size) must not overlap
    # the prompt region [used_range_start, used_range_end).
    valid_offsets = []
    last_valid = N - segment_size

    for candidate_offset in range(0, last_valid + 1):
        segment_start = candidate_offset
        segment_end = candidate_offset + segment_size
        # check if there's overlap with the prompt range
        # overlap if segment_start < used_range_end and segment_end > used_range_start
        if not (segment_start < used_range_end and segment_end > used_range_start):
            valid_offsets.append(candidate_offset)

    if not valid_offsets:
        raise ValueError("No valid non-overlapping offset found.")

    offset = random.choice(valid_offsets)
    seg_tokens = tokens[offset : offset + segment_size]
    seg_channels = channels[offset : offset + segment_size]
    return seg_tokens, seg_channels

import torch
import random
from time import time

def evaluate_shard_groups_with_channels(
    model,
    group_a_shards,       # list of paths for group A
    group_b_shards,       # list of paths for group B
    segment_size=500,
    num_comparisons=1000,
    device="cuda",
    seed=42
):
    """
    Compare group A vs. group B in a forced-choice setup:
      - Prompt is randomly taken from either group A or B
      - 'Correct' completion is from the same group (non-overlapping with the prompt)
      - 'Wrong' completion is from the other group
    We'll do `num_comparisons` times and measure how often the model prefers
    the correct completion over the wrong one.

    Returns: float accuracy
    """

    # 0) Set random seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)

    # 1) Load the shards for each group
    groupA = load_shard_group(group_a_shards)  # list of (tokens, channels)
    groupB = load_shard_group(group_b_shards)  # list of (tokens, channels)

    total_evals = 0
    correct_count = 0

    for i in range(num_comparisons):
        print(i)
        # for example, flip a coin to decide if the prompt is from groupA or groupB
        use_group_A_for_prompt = (random.random() < 0.5)

        if use_group_A_for_prompt:
            # pick random prompt from group A
            prompt_tokens, prompt_channels, shard_index, offset = pick_random_segment(groupA, segment_size)

            # pick correct from same shard (non-overlapping)
            tokensA, chansA = groupA[shard_index]
            correct_tokens, correct_channels = pick_random_nonoverlapping_segment_in_same_shard(
                tokensA, chansA,
                used_range_start=offset,
                used_range_end=offset + segment_size,
                segment_size=segment_size
            )

            # pick wrong from group B
            wrong_tokens, wrong_channels, _, _ = pick_random_segment(groupB, segment_size)

        else:
            # pick random prompt from group B
            prompt_tokens, prompt_channels, shard_index, offset = pick_random_segment(groupB, segment_size)

            # correct from same shard (non-overlapping)
            tokensB, chansB = groupB[shard_index]
            correct_tokens, correct_channels = pick_random_nonoverlapping_segment_in_same_shard(
                tokensB, chansB,
                used_range_start=offset,
                used_range_end=offset + segment_size,
                segment_size=segment_size
            )

            # pick wrong from group A
            wrong_tokens, wrong_channels, _, _ = pick_random_segment(groupA, segment_size)

        # 2) Compute cross-entropy loss for correct vs. wrong
        loss_correct = compute_completion_loss_with_channels(
            model,
            prompt_tokens, prompt_channels,
            correct_tokens, correct_channels,
            device=device
        )
        loss_wrong = compute_completion_loss_with_channels(
            model,
            prompt_tokens, prompt_channels,
            wrong_tokens,  wrong_channels,
            device=device
        )

        # 3) Compare
        total_evals += 1
        if loss_correct < loss_wrong:
            correct_count += 1

        # Optional progress print
        if (i+1) % 100 == 0 or (i+1) == num_comparisons:
            print(f"[{i+1}/{num_comparisons}] correct={correct_count}, total={total_evals}, accuracy={correct_count/total_evals:.3f}")

    # Final accuracy
    if total_evals == 0:
        print("No evaluations performed.")
        return 0.0

    accuracy = correct_count / total_evals
    print(f"\n[evaluate_shard_groups_with_channels] Final Accuracy = {accuracy:.4f} "
          f"({correct_count}/{total_evals})")
    return accuracy



model = GPT(GPTConfig)

checkpoint = torch.load('log/model_14000_150M_small.pt', map_location=torch.device('cpu'), weights_only=False)
# retrieve the state_dict
orig_sd = checkpoint['model']

# remove "_orig_mod." from each key
fixed_sd = {}
for k, v in orig_sd.items():
    new_key = k.replace("_orig_mod.", "")
    fixed_sd[new_key] = v

# now load with the fixed state dict
model.load_state_dict(fixed_sd, strict=True)
model.config(checkpoint['config'])
model.eval()

group_a_shards = [
    "validation_datasets_imageNet/shards/shard_train_610.pt",
    "validation_datasets_imageNet/shards/shard_train_634.pt",
    # etc...
]
group_b_shards = [
    "validation_datasets_imageNet/shards/shard_train_600.pt",
    "validation_datasets_imageNet/shards/shard_train_243.pt",]


accuracy = evaluate_shard_groups_with_channels(
    model=model,
    group_a_shards=group_a_shards,
    group_b_shards=group_b_shards,
    segment_size=512,
    num_comparisons=100,  # for example
    device="cpu",
)

print("Final forced-choice accuracy:", accuracy)