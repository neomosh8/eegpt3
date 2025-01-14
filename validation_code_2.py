#!/usr/bin/env python3
import inspect
import random
import time
import os
import math
import pickle
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt

from tokenizer2 import BPE_RLE_Tokenizer as Tokenizer
tokenizer = Tokenizer()
tokenizer.load_merges("neo_tokenizer/merges.json")
tokenizer.load_vocab("neo_tokenizer/vocab.json")

##############################################################################
# Model Definition (unchanged)
##############################################################################

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.attn_dropout = nn.Dropout(p=getattr(config, 'attn_dropout', 0.05))
        self.resid_dropout = nn.Dropout(p=getattr(config, 'resid_dropout', 0.05))

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # Flash attention (PyTorch 2.0+)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = self.attn_dropout(y)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.resid_dropout(y)
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
    small_model: bool = False
    block_size: int = 1024
    vocab_size: int = 6460

    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

    num_channels: int = 2
    mlp_dropout: float = 0.05
    attn_dropout: float = 0.05
    resid_dropout: float = 0.05

    def __post_init__(self):
        if not self.small_model:
            self.n_layer = 36
            self.n_head  = 20
            self.n_embd  = 1280

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.channel_dim = 32
        self.wce = nn.Embedding(config.num_channels, self.channel_dim)
        self.channel_proj = nn.Linear(self.channel_dim, config.n_embd)
        self.channel_scale = nn.Parameter(torch.tensor(1.0))

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "wpe": nn.Embedding(config.block_size, config.n_embd),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": nn.LayerNorm(config.n_embd)
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, channel_idx=None, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=idx.dtype, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)

        if channel_idx is not None:
            cha_emb_small = self.wce(channel_idx)
            cha_emb_large = self.channel_proj(cha_emb_small)
            cha_emb_scaled = self.channel_scale * cha_emb_large
            x = tok_emb + pos_emb + cha_emb_scaled
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

    def configure_optimizer(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        decay_params = []
        nodecay_params = []
        channel_params = []

        channel_lr = learning_rate * 0.1

        for pn, p in param_dict.items():
            if 'wce' in pn or 'channel_proj' in pn or 'channel_scale' in pn:
                channel_params.append(p)
            elif p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate},
            {'params': nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate},
            {'params': channel_params, 'weight_decay': 0.0, 'lr': channel_lr},
        ]

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and ('cuda' in device)

        print(f"num decayed parameter tensors: {len(decay_params)} with {sum(p.numel() for p in decay_params):,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)} with {sum(p.numel() for p in nodecay_params):,} parameters")
        print(f"num channel parameter tensors: {len(channel_params)} with {sum(p.numel() for p in channel_params):,} parameters")
        print(f"Using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(
            optim_groups,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused
        )
        return optimizer

##############################################################################
# Evaluation Helpers (unchanged)
##############################################################################

def compute_completion_loss_with_channels(
        model,
        prompt_tokens, prompt_channels,
        completion_tokens, completion_channels,
        device="cuda"
):
    full_tokens = torch.cat([prompt_tokens, completion_tokens], dim=0).unsqueeze(0).to(device)
    full_channels = torch.cat([prompt_channels, completion_channels], dim=0).unsqueeze(0).to(device)

    prompt_len = prompt_tokens.size(0)
    completion_len = completion_tokens.size(0)
    total_len = full_tokens.size(1)

    mask_vals = [0] * prompt_len + [1] * completion_len
    mask = torch.tensor(mask_vals, device=device).unsqueeze(0)

    with torch.no_grad():
        logits, _ = model(idx=full_tokens, channel_idx=full_channels)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_tokens = full_tokens[:, 1:].contiguous()
    shift_mask = mask[:, 1:].contiguous()

    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_tokens = shift_tokens.view(-1)
    flat_mask = shift_mask.view(-1)

    ce_per_token = F.cross_entropy(flat_logits, flat_tokens, reduction='none')
    ce_completion = ce_per_token * flat_mask
    sum_loss = ce_completion.sum()
    num_completion_tokens = flat_mask.sum()
    avg_loss = sum_loss / (num_completion_tokens + 1e-9)

    return avg_loss.item()

def evaluate_shards_with_channels(
        model,
        shard0_path,
        shard1_path,
        device="cuda",
        segment_size=500
):
    import torch
    import torch.nn.functional as F
    from time import time

    shard0 = torch.load(shard0_path, weights_only=False)
    tokens0 = shard0['tokens']
    chan0 = shard0['channels']

    shard1 = torch.load(shard1_path, weights_only=False)
    tokens1 = shard1['tokens']
    chan1 = shard1['channels']

    tokens0, chan0 = tokens0.cpu(), chan0.cpu()
    tokens1, chan1 = tokens1.cpu(), chan1.cpu()

    len0 = tokens0.size(0)
    len1 = tokens1.size(0)

    total_evals = 0
    correct_count = 0

    # ---- Evaluate shard0
    i = 0
    block_index = 0
    max_blocks_0 = len0 // (2 * segment_size)

    while i + 2 * segment_size <= len0:
        block_index += 1
        print(f"\n[Shard0] Processing block {block_index}/{max_blocks_0} at i={i} ...")
        t0 = time()

        if segment_size >= len0:
            raise ValueError("Segment size must be smaller than the sequence length.")

        # We'll pick a valid prompt and correct offset at random:
        while True:
            prompt_offset_candidates = list(range(0, len0 - segment_size + 1, 256))
            if not prompt_offset_candidates:
                raise ValueError("No valid prompt offsets possible with given parameters.")
            prompt_offset = random.choice(prompt_offset_candidates)
            print(f"Prompt Offset: {prompt_offset}")
            prompt_0_tokens = tokens0[prompt_offset: prompt_offset + segment_size]
            prompt_0_chans = chan0[prompt_offset: prompt_offset + segment_size]

            correct_offset_candidates = list(range(prompt_offset + segment_size, len0 - segment_size + 1, 256))
            if not correct_offset_candidates:
                print("No valid correct offsets, retrying with new prompt offset")
                continue
            correct_offset = random.choice(correct_offset_candidates)
            print(f"Correct Offset: {correct_offset}")
            correct_0_tokens = tokens0[correct_offset: correct_offset + segment_size]
            correct_0_chans = chan0[correct_offset: correct_offset + segment_size]

            break

        # "Wrong" from shard1
        if len1 >= segment_size:
            wrong_offset = random.randint(0, len1 - segment_size)
            wrong_1_tokens = tokens1[wrong_offset: wrong_offset + segment_size]
            wrong_1_chans = chan1[wrong_offset: wrong_offset + segment_size]
        else:
            break

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

        total_evals += 1
        if loss_correct < loss_wrong:
            correct_count += 1

        i += 2 * segment_size
        t1 = time()
        dt = (t1 - t0) * 1000
        print(f"[Shard0] Block {block_index} took {dt:.2f} ms | loss_correct={loss_correct:.4f} | loss_wrong={loss_wrong:.4f}")

    # ---- Evaluate shard1
    j = 0
    block_index = 0
    max_blocks_1 = len1 // (2 * segment_size)

    while j + 2 * segment_size <= len1:
        block_index += 1
        print(f"\n[Shard1] Processing block {block_index}/{max_blocks_1} at j={j} ...")
        t0 = time()

        if segment_size >= len1:
            raise ValueError("Segment size must be smaller than the sequence length.")

        while True:
            prompt_offset_candidates = list(range(0, len1 - segment_size + 1, 256))
            if not prompt_offset_candidates:
                raise ValueError("No valid prompt offsets possible with given parameters.")
            prompt_offset = random.choice(prompt_offset_candidates)
            print(f"Prompt Offset 1: {prompt_offset}")
            prompt_1_tokens = tokens1[prompt_offset: prompt_offset + segment_size]
            prompt_1_chans = chan1[prompt_offset: prompt_offset + segment_size]

            correct_offset_candidates = list(range(prompt_offset + segment_size, len1 - segment_size + 1, 256))
            if not correct_offset_candidates:
                print("No valid correct offsets, retrying with new prompt offset")
                continue
            correct_offset = random.choice(correct_offset_candidates)
            print(f"Correct Offset 1: {correct_offset}")
            correct_1_tokens = tokens1[correct_offset: correct_offset + segment_size]
            correct_1_chans = chan1[correct_offset: correct_offset + segment_size]

            break

        if len0 >= segment_size:
            wrong_offset = random.randint(0, len0 - segment_size)
            wrong_0_tokens = tokens0[wrong_offset: wrong_offset + segment_size]
            wrong_0_chans = chan0[wrong_offset: wrong_offset + segment_size]
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
        print(f"[Shard1] Block {block_index} took {dt:.2f} ms | loss_correct={loss_correct:.4f} | loss_wrong={loss_wrong:.4f}")

    if total_evals == 0:
        print("No evaluations were performed (possibly not enough tokens).")
        return 0.0

    accuracy = correct_count / total_evals
    print(f"\n[evaluate_shards_with_channels] Final Accuracy = {correct_count}/{total_evals} = {accuracy:.4f}")
    return accuracy

##############################################################################
# Parallelized Evaluation
##############################################################################
import multiprocessing
from tqdm import tqdm

def run_evaluation_single_epoch(epoch_idx, small_model, shard0_path, shard1_path, device):
    """
    Each process calls this function to:
      1) Load the model (naive approach).
      2) Evaluate shards with channels.
      3) Return the accuracy.
    """
    # For demonstration, we re-load the model here in every process.
    # This is straightforward but can be slow if the model is large.
    print(f"[Process {multiprocessing.current_process().name}] Starting epoch {epoch_idx+1} ...")

    # Pick the checkpoint by model size
    if small_model:
        checkpoint_path = "log/model_15000.pt"
    else:
        checkpoint_path = "log/model_30000.pt"

    device_torch = torch.device(device)
    model = GPT(GPTConfig(small_model=small_model)).to(device_torch)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device_torch)
    orig_sd = checkpoint['model']
    fixed_sd = {}
    for k, v in orig_sd.items():
        new_key = k.replace("_orig_mod.", "")
        fixed_sd[new_key] = v
    model.load_state_dict(fixed_sd, strict=True)

    model.eval()
    # Evaluate
    acc = evaluate_shards_with_channels(
        model=model,
        shard0_path=shard0_path,
        shard1_path=shard1_path,
        device=device_torch,
        segment_size=512
    )
    print(f"[Process {multiprocessing.current_process().name}] Finished epoch {epoch_idx+1} with acc={acc:.4f}")

    return acc

def evaluate_model_for_condition(
        small_model: bool,
        shard0_path: str,
        shard1_path: str,
        epochs: int = 30,
        device: str = "cpu",
        num_workers: int = 4
):
    """
    Spawns multiple processes to evaluate each epoch in parallel.
    Returns a list of accuracies (one per epoch).
    """

    # We'll spin up a pool of `num_workers` processes
    pool_args = [
        (epoch_idx, small_model, shard0_path, shard1_path, device)
        for epoch_idx in range(epochs)
    ]

    # Use a context manager to ensure pool terminates cleanly
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Map over the arguments in parallel
        results = list(
            tqdm(
                pool.starmap(run_evaluation_single_epoch, pool_args),
                total=epochs,
                desc=f"Evaluating {'small' if small_model else 'large'} model"
            )
        )
    return results

##############################################################################
# Plotting
##############################################################################

def plot_results(group_name: str, accs_small: list, accs_large: list):
    accs_small = np.array(accs_small)
    accs_large = np.array(accs_large)

    mean_small = np.mean(accs_small)
    std_small = np.std(accs_small)
    mean_large = np.mean(accs_large)
    std_large = np.std(accs_large)

    # 95% CI approximation
    ci_small = 1.96 * std_small / np.sqrt(len(accs_small))
    ci_large = 1.96 * std_large / np.sqrt(len(accs_large))

    x_vals = [0, 1]
    means = [mean_small, mean_large]
    errors = [ci_small, ci_large]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x_vals, means, yerr=errors, color=['lightblue', 'salmon'], capsize=5)
    ax.set_xticks(x_vals)
    ax.set_xticklabels(['Small Model', 'Large Model'])
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy Distribution for {group_name} (Mean ± 95% CI)")

    # Scatter raw epoch points
    jitter = 0.05
    ax.scatter(
        x=np.zeros_like(accs_small) + x_vals[0] + np.random.uniform(-jitter, jitter, size=len(accs_small)),
        y=accs_small,
        color='blue', alpha=0.5, marker='o'
    )
    ax.scatter(
        x=np.zeros_like(accs_large) + x_vals[1] + np.random.uniform(-jitter, jitter, size=len(accs_large)),
        y=accs_large,
        color='red', alpha=0.5, marker='o'
    )

    plt.tight_layout()
    plt.savefig(f"{group_name}.png")
    plt.close()

##############################################################################
# Main
##############################################################################

if __name__ == "__main__":
    # On Windows, macOS, or notebooks, you MUST protect multiprocessing code with this check.
    device = "cpu"  # or "cuda"

    # Conditions you want to evaluate
    conditions = {
        "lat/pal": ("output/shards/shard_train_0.pt", "output/shards/shard_train_2.pt"),
        "lat/rest": ("output/shards/shard_train_1.pt", "output/shards/shard_train_2.pt"),
        "palm/rest": ("output/shards/shard_train_1.pt", "output/shards/shard_train_0.pt")
    }

    # Dictionary to hold results
    results = {}

    for group_name, (shard0_path, shard1_path) in conditions.items():
        # Evaluate small model in parallel
        accs_small = evaluate_model_for_condition(
            small_model=True,
            shard0_path=shard0_path,
            shard1_path=shard1_path,
            epochs=100,       # set how many epochs you want
            device=device,
            num_workers=multiprocessing.cpu_count()-2   # number of parallel processes
        )
        # Evaluate large model in parallel
        accs_large = evaluate_model_for_condition(
            small_model=False,
            shard0_path=shard0_path,
            shard1_path=shard1_path,
            epochs=5,
            device=device,
            num_workers=multiprocessing.cpu_count()-2
        )
        results[group_name] = (accs_small, accs_large)

    # Plot
    for group_name, (accs_small, accs_large) in results.items():
        plot_results(group_name, accs_small, accs_large)
        print(f"Saved plot for {group_name} as {group_name}.png")
