#!/usr/bin/env python3
import inspect
import random
import time
import pickle
from dataclasses import dataclass
import numpy as np
import torch
import math
import os
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch import nn

from tokenizer2 import BPE_RLE_Tokenizer as Tokenizer
small_model = False
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
    block_size: int = 1024
    vocab_size: int = 6460
    if small_model:
        n_layer: int = 12   # number of layers
        n_head: int = 12    # number of heads
        n_embd: int = int(768)  # embedding dimension
    else:
        n_layer: int = 36
        n_head: int = 20
        n_embd: int = 1280
    num_channels: int = 2
    mlp_dropout: float = 0.05
    attn_dropout: float = 0.05
    resid_dropout: float = 0.05

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
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, channel_idx=None, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
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
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        num_channel_params = sum(p.numel() for p in channel_params)
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and ('cuda' in device)
        print(f"num decayed parameter tensors: {len(decay_params)} with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)} with {num_nodecay_params:,} parameters")
        print(f"num channel parameter tensors: {len(channel_params)} with {num_channel_params:,} parameters")
        print(f"Using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused
        )
        return optimizer

device = 'cpu'

# === Helper function for computing completion loss ===
def compute_completion_loss_with_channels(
        model,
        prompt_tokens, prompt_channels,
        completion_tokens, completion_channels,
        device="cuda"
):
    # Concatenate prompt and candidate completion.
    full_tokens = torch.cat([prompt_tokens, completion_tokens], dim=0).unsqueeze(0).to(device)
    full_channels = torch.cat([prompt_channels, completion_channels], dim=0).unsqueeze(0).to(device)
    prompt_len = prompt_tokens.size(0)
    mask_vals = [0] * prompt_len + [1] * completion_tokens.size(0)
    mask = torch.tensor(mask_vals, device=device).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(idx=full_tokens, channel_idx=full_channels)
    # Compute loss only on the candidate (completion) portion.
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

# === Few-shot multi-class forced choice evaluation ===
def evaluate_multiclass_with_channels(
        model,           # trained model
        shard_paths,     # list of shard file paths
        device="cuda",
        num_evals=5,     # number of evaluations per shard
        prompt_stride=256
):
    """
    For each shard:
      - Sample two demonstration pairs. Each demonstration pair is a contiguous 256-token segment:
           * The first 128 tokens are the input.
           * The following 128 tokens are the associated correct continuation.
      - Concatenate the two demonstration pairs to form a 512-token few-shot prompt.
      - Then sample a candidate continuation (512 tokens) from the same shard—ensuring its token window does not overlap either demonstration pair.
        This is the "correct" candidate.
      - For every other shard, sample a candidate continuation of 512 tokens.
      - For each candidate, compute the loss (when appended to the few-shot prompt) over only the candidate portion.
      - The candidate with the lowest loss is selected.
      - A confusion matrix is built and overall accuracy is reported.
    """
    # Candidate continuation length (in tokens)
    candidate_length = 512

    # Load all shards into memory.
    shards = []
    for path in shard_paths:
        shard = torch.load(path, map_location="cpu")  # expected to contain {'tokens': ..., 'channels': ...}
        tokens = shard['tokens'].cpu()
        channels = shard['channels'].cpu()
        shards.append({
            'tokens': tokens,
            'channels': channels,
            'length': tokens.size(0),
            'path': path
        })
    num_shards = len(shards)
    # Initialize confusion matrix: rows = true (prompt) shard, columns = predicted candidate shard.
    confusion_matrix = np.zeros((num_shards, num_shards), dtype=int)
    total_evals = 0
    correct_count = 0

    # A simple helper to check if two intervals [a_start, a_end) and [b_start, b_end) overlap.
    def overlaps(a_start, a_end, b_start, b_end):
        return not (a_end <= b_start or a_start >= b_end)

    # For each shard, use it as the source for the few-shot prompt and the correct candidate.
    for i, shard in enumerate(shards):
        tokens_i = shard['tokens']
        chans_i = shard['channels']
        len_i = tokens_i.size(0)
        # We require enough tokens to sample demonstration pairs and candidate.
        if len_i < 256 or len_i < candidate_length:
            print(f"Shard {i} is too short for evaluation. Skipping...")
            continue

        for eval_idx in range(num_evals):
            print(f"\n[Shard {i}] Evaluation {eval_idx+1} ...")
            # --- Sample Demonstration Pair 1 ---
            # Choose a random index such that a contiguous 256-token block can be taken.
            demo1_start = random.randint(0, len_i - 256)
            demo1_tokens = tokens_i[demo1_start : demo1_start + 256]
            demo1_chans = chans_i[demo1_start : demo1_start + 256]

            # --- Sample Demonstration Pair 2 ---
            demo2_start = random.randint(0, len_i - 256)
            demo2_tokens = tokens_i[demo2_start : demo2_start + 256]
            demo2_chans = chans_i[demo2_start : demo2_start + 256]

            # Construct the 512-token few-shot prompt.
            few_shot_prompt_tokens = torch.cat([demo1_tokens, demo2_tokens], dim=0)
            few_shot_prompt_chans  = torch.cat([demo1_chans, demo2_chans], dim=0)

            # --- Prepare the correct candidate continuation (512 tokens) from the same shard ---
            # Its window must not overlap with either demonstration pair.
            candidate_offsets = [c for c in range(0, len_i - candidate_length + 1, prompt_stride)
                                 if (not overlaps(c, c + candidate_length, demo1_start, demo1_start + 256)
                                     and not overlaps(c, c + candidate_length, demo2_start, demo2_start + 256))]
            if not candidate_offsets:
                print(f"No valid candidate window in shard {i} for evaluation {eval_idx+1}. Skipping this evaluation.")
                continue
            correct_offset = random.choice(candidate_offsets)
            correct_tokens = tokens_i[correct_offset : correct_offset + candidate_length]
            correct_chans = chans_i[correct_offset : correct_offset + candidate_length]

            candidate_info = []
            candidate_info.append({
                'tokens': correct_tokens,
                'channels': correct_chans,
                'label': 'correct',
                'source_shard': i
            })

            # --- Prepare candidate continuations from every other shard ---
            for j, other_shard in enumerate(shards):
                if j == i:
                    continue
                tokens_j = other_shard['tokens']
                chans_j = other_shard['channels']
                len_j = tokens_j.size(0)
                if len_j < candidate_length:
                    print(f"Shard {j} is too short for candidate sampling. Skipping candidate from this shard.")
                    continue
                # Sample a candidate block using a random valid offset.
                wrong_offset = random.randint(0, len_j - candidate_length)
                wrong_tokens = tokens_j[wrong_offset : wrong_offset + candidate_length]
                wrong_chans = chans_j[wrong_offset : wrong_offset + candidate_length]
                candidate_info.append({
                    'tokens': wrong_tokens,
                    'channels': wrong_chans,
                    'label': 'wrong',
                    'source_shard': j
                })

            # --- Evaluate each candidate ---
            candidate_losses = []
            for candidate in candidate_info:
                loss = compute_completion_loss_with_channels(
                    model,
                    few_shot_prompt_tokens, few_shot_prompt_chans,
                    candidate['tokens'], candidate['channels'],
                    device=device
                )
                candidate_losses.append(loss)
            # Pick the candidate with the lowest loss.
            min_loss_index = np.argmin(candidate_losses)
            chosen = candidate_info[min_loss_index]
            predicted_shard = chosen['source_shard']
            confusion_matrix[i, predicted_shard] += 1
            if chosen['label'] == 'correct':
                correct_count += 1
            total_evals += 1

            print(f"[Shard {i}] Eval {eval_idx+1} candidate losses: {candidate_losses}")
            print(f" -> Correct candidate loss: {candidate_losses[0]:.4f} vs. others: {[f'{l:.4f}' for l in candidate_losses[1:]]}")
            print(f" -> Model selected candidate from shard {predicted_shard} (label: {chosen['label']})")

    if total_evals == 0:
        print("No evaluations were performed (possibly not enough tokens in the shards).")
        return 0.0

    accuracy = correct_count / total_evals
    print(f"\n[Few-shot Evaluation] Final Accuracy = {correct_count}/{total_evals} = {accuracy:.4f}")

    # Print the confusion matrix.
    print("\nConfusion Matrix (rows: true prompt shard, columns: predicted candidate shard):")
    header = "      " + " ".join([f"Shd{j}" for j in range(num_shards)])
    print(header)
    for i in range(num_shards):
        row_counts = " ".join([f"{confusion_matrix[i, j]:5d}" for j in range(num_shards)])
        print(f"Shd{i} : {row_counts}")
    return accuracy

# === Main script ===
d = 'cuda'
device = torch.device(d)
model = GPT(GPTConfig).to(device)
if small_model:
    checkpoint = torch.load('log/model_15000.pt', map_location=device)
else:
    checkpoint = torch.load('log/model_30000.pt', map_location=device)
orig_sd = checkpoint['model']
fixed_sd = {}
for k, v in orig_sd.items():
    new_key = k.replace("_orig_mod.", "")
    fixed_sd[new_key] = v
model.load_state_dict(fixed_sd, strict=True)
model.config(checkpoint['config'])
model.eval()

# Example: Evaluate over 10 epochs using three shards.
accs = []
epochs = 20
for epoch in range(epochs):
    print(f"\n=== Epoch {epoch+1}/{epochs} ===")
    acc = evaluate_multiclass_with_channels(
        model=model,
        shard_paths=[
            "output_MEMA/shards/shard_train_0.pt",
            "output_MEMA/shards/shard_train_1.pt",
            "output_MEMA/shards/shard_train_2.pt"
        ],
        device=d,
        num_evals=5,
        prompt_stride=256
    )
    accs.append(acc)
mean = np.mean(accs)
print(f"\nMean Accuracy over {epochs} epochs: {mean}")
print(f"Accuracies per epoch: {accs}")
