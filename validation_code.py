#!/usr/bin/env python3
import random
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
    # # for 14000 steps 150M model (1.7GB)
    # block_size: int = 1024
    # vocab_size: int = 4140
    # n_layer: int = 18
    # n_head: int = 12
    # n_embd: int = 768
    # num_channels: int = 2

    # for steps 1.3B Large model 14.4GB
    block_size: int = 2048
    vocab_size: int = 4140
    n_layer: int = 20
    n_head: int = 36  # Increased from 16 to allow more parallel attention patterns
    n_embd: int = 2304  # Increased to maintain head_dim with more heads
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


def evaluate_shards_with_channels(
        model,  # your trained model
        shard0_path,  # path to "shard_train_0.pt"
        shard1_path,  # path to "shard_train_1.pt"
        device="cuda",
        segment_size=500
):
    """
    For each shard (0 and 1), we iterate in blocks of 2*segment_size tokens:
        - prompt = tokens[i : i + segment_size],  channels[i : i + segment_size]
        - correct continuation = tokens[i + segment_size : i + 2*segment_size], channels[i+segment_size : i+2*segment_size]
    Then we pick an 'incorrect' completion from the other shard, e.g. tokens[i : i+segment_size] if feasible.

    We measure which completion yields lower average cross-entropy.
    Returns overall accuracy = fraction of times the model prefers the correct continuation.
    """

    import torch
    import torch.nn.functional as F
    from time import time

    # 1) Load tokens & channels from each shard
    shard0 = torch.load(shard0_path)  # e.g. {'tokens': ..., 'channels': ...}
    tokens0 = shard0['tokens']  # shape (N0,)
    chan0 = shard0['channels']  # shape (N0,)
    shard1 = torch.load(shard1_path)
    tokens1 = shard1['tokens']  # shape (N1,)
    chan1 = shard1['channels']  # shape (N1,)

    # Move them to CPU just in case
    tokens0, chan0 = tokens0.cpu(), chan0.cpu()
    tokens1, chan1 = tokens1.cpu(), chan1.cpu()

    len0 = tokens0.size(0)
    len1 = tokens1.size(0)

    total_evals = 0
    correct_count = 0

    # -------------------------------------------------
    # Evaluate shard0
    # -------------------------------------------------
    i = 0
    block_index = 0
    # how many full 1000-token blocks we can process in shard0
    max_blocks_0 = len0 // (2 * segment_size)

    while i + 2 * segment_size <= len0:
        block_index += 1
        print(f"\n[Shard0] Processing block {block_index}/{max_blocks_0} at i={i} ...")

        t0 = time()

        # Prompt and correct continuation from shard0
        prompt_0_tokens = tokens0[i: i + segment_size]
        prompt_0_chans = chan0[i: i + segment_size]
        max_gap_multiplier = 0 # Adjust this to control the max gap size in terms of segment_size
        gap_multiplier = random.randint(0, max_gap_multiplier)  # Random multiplier for the gap
        gap_size = gap_multiplier * segment_size

        # gap_size = 0  # Uncomment to make consecutive
        correct_0_tokens = tokens0[i + segment_size + gap_size: i + 2 * segment_size + gap_size]
        correct_0_chans = chan0[i + segment_size + gap_size: i + 2 * segment_size + gap_size]

        # "Wrong" completion from shard1
        if i + segment_size <= len1:
            wrong_1_tokens = tokens1[i: i + segment_size]
            wrong_1_chans = chan1[i: i + segment_size]
        else:
            # If shard1 doesn't have enough tokens at index i
            if len1 >= segment_size:
                wrong_1_tokens = tokens1[0: segment_size]
                wrong_1_chans = chan1[0: segment_size]
            else:
                break

        # 2) Compute the average cross-entropy loss for correct vs. wrong
        loss_correct = compute_completion_loss_with_channels(
            model,
            prompt_0_tokens, prompt_0_chans,
            correct_0_tokens, correct_0_chans,
            device=device
        )
        loss_wrong = compute_completion_loss_with_channels(
            model,
            prompt_0_tokens, prompt_0_chans,
            wrong_1_tokens, wrong_1_chans,
            device=device
        )

        # 3) Compare which is lower
        total_evals += 1
        if loss_correct < loss_wrong:
            correct_count += 1

        i += 2 * segment_size

        t1 = time()
        dt = (t1 - t0) * 1000
        print(
            f"[Shard0] Block {block_index} took {dt:.2f} ms | loss_correct={loss_correct:.4f} | loss_wrong={loss_wrong:.4f}")

    # -------------------------------------------------
    # Evaluate shard1
    # -------------------------------------------------
    j = 0
    block_index = 0
    max_blocks_1 = len1 // (2 * segment_size)

    while j + 2 * segment_size <= len1:
        block_index += 1
        print(f"\n[Shard1] Processing block {block_index}/{max_blocks_1} at j={j} ...")

        t0 = time()

        # Prompt and correct continuation from shard1
        prompt_1_tokens = tokens1[j: j + segment_size]
        prompt_1_chans = chan1[j: j + segment_size]
        correct_1_tokens = tokens1[j + segment_size: j + 2 * segment_size]
        correct_1_chans = chan1[j + segment_size: j + 2 * segment_size]

        # "Wrong" from shard0
        if j + segment_size <= len0:
            wrong_0_tokens = tokens0[j: j + segment_size]
            wrong_0_chans = chan0[j: j + segment_size]
        else:
            if len0 >= segment_size:
                wrong_0_tokens = tokens0[0: segment_size]
                wrong_0_chans = chan0[0: segment_size]
            else:
                break

        loss_correct = compute_completion_loss_with_channels(
            model,
            prompt_1_tokens, prompt_1_chans,
            correct_1_tokens, correct_1_chans,
            device=device
        )
        loss_wrong = compute_completion_loss_with_channels(
            model,
            prompt_1_tokens, prompt_1_chans,
            wrong_0_tokens, wrong_0_chans,
            device=device
        )

        total_evals += 1
        if loss_correct < loss_wrong:
            correct_count += 1

        j += 2 * segment_size

        t1 = time()
        dt = (t1 - t0) * 1000
        print(
            f"[Shard1] Block {block_index} took {dt:.2f} ms | loss_correct={loss_correct:.4f} | loss_wrong={loss_wrong:.4f}")

    # Final results
    if total_evals == 0:
        print("No evaluations were performed (possibly not enough tokens).")
        return 0.0

    accuracy = correct_count / total_evals
    print(f"\n[evaluate_shards_with_channels] Final Accuracy = {correct_count}/{total_evals} = {accuracy:.4f}")
    return accuracy


model = GPT(GPTConfig)

checkpoint = torch.load('log/model_07359.pt', map_location=torch.device('cuda'), weights_only=False)
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
acc = evaluate_shards_with_channels(
    model=model,
    shard0_path="validation_datasets/shards/shard_train_1.pt",
    shard1_path="validation_datasets/shards/shard_train_0.pt",
    device="cuda",
    segment_size=512
)