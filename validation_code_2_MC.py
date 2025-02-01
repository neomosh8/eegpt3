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
        n_layer: int = 12  # number of layers
        n_head: int = 12   # number of heads
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

# === Helper function for computing completion loss (unchanged) ===
def compute_completion_loss_with_channels(
        model,
        prompt_tokens, prompt_channels,
        completion_tokens, completion_channels,
        device="cuda"
):
    # 1) Concatenate prompt + completion for tokens and channels
    full_tokens = torch.cat([prompt_tokens, completion_tokens], dim=0).unsqueeze(0).to(device)
    full_channels = torch.cat([prompt_channels, completion_channels], dim=0).unsqueeze(0).to(device)
    total_len = full_tokens.size(1)
    prompt_len = prompt_tokens.size(0)
    completion_len = completion_tokens.size(0)
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

# === New multi-class force choice evaluation function with confusion matrix ===
def evaluate_multiclass_with_channels(
        model,           # the trained model
        shard_paths,     # list of shard file paths (e.g., ["shard_train_0.pt", "shard_train_1.pt", "shard_train_2.pt"])
        device="cuda",
        segment_size=500,
        prompt_stride=256
):
    """
    For each shard in shard_paths, we perform an evaluation block where:
      - We sample a prompt from the shard.
      - We choose a correct continuation (from the same shard) that does not overlap the prompt.
      - For every other shard, we sample one candidate continuation.
      - We compute the average cross-entropy loss for each candidate given the prompt.
      - The candidate with the lowest loss is considered the model's prediction.
    We record the ground truth (the prompt's shard) and the predicted candidate's shard
    in order to build a confusion matrix.
    Returns overall accuracy and prints the confusion matrix.
    """
    # Load all shards into memory.
    shards = []
    for path in shard_paths:
        shard = torch.load(path, weights_only=False)  # expected to contain {'tokens': ..., 'channels': ...}
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

    # For each shard, use it as the source of the prompt and the correct candidate.
    for i, shard in enumerate(shards):
        tokens_i = shard['tokens']
        chans_i = shard['channels']
        len_i = shard['length']
        block_index = 0
        pos = 0
        # Process blocks until we run out of tokens (using a step of 2*segment_size per block)
        while pos + 2 * segment_size <= len_i:
            block_index += 1
            print(f"\n[Shard {i}] Processing block {block_index} at pos={pos} ...")
            # Sample a prompt offset from the current shard.
            prompt_offset_candidates = list(range(0, len_i - segment_size + 1, prompt_stride))
            if not prompt_offset_candidates:
                print(f"Not enough tokens in shard {i} for a prompt. Skipping...")
                break
            # Loop until a valid correct candidate (non-overlapping with prompt) is found.
            while True:
                prompt_offset = random.choice(prompt_offset_candidates)
                prompt_tokens = tokens_i[prompt_offset : prompt_offset + segment_size]
                prompt_chans  = chans_i[prompt_offset : prompt_offset + segment_size]
                all_candidates = list(range(0, len_i - segment_size + 1, prompt_stride))
                correct_offset_candidates = [c for c in all_candidates if (c + segment_size <= prompt_offset or c >= prompt_offset + segment_size)]
                if not correct_offset_candidates:
                    print(f"No valid correct offsets in shard {i} for prompt offset {prompt_offset}. Retrying...")
                    continue
                correct_offset = random.choice(correct_offset_candidates)
                correct_tokens = tokens_i[correct_offset : correct_offset + segment_size]
                correct_chans  = chans_i[correct_offset : correct_offset + segment_size]
                break

            # Build the list of candidate completions:
            # First candidate: correct (from the same shard)
            candidate_info = []
            candidate_info.append({
                'tokens': correct_tokens,
                'channels': correct_chans,
                'label': 'correct',
                'source_shard': i  # correct candidate is from the same shard as the prompt
            })
            # For every other shard, sample one candidate continuation.
            for j, other_shard in enumerate(shards):
                if j == i:
                    continue
                len_j = other_shard['length']
                tokens_j = other_shard['tokens']
                chans_j  = other_shard['channels']
                if len_j < segment_size:
                    print(f"Shard {j} is too short. Skipping candidate from this shard.")
                    continue
                wrong_offset = random.randint(0, len_j - segment_size)
                wrong_tokens = tokens_j[wrong_offset : wrong_offset + segment_size]
                wrong_chans  = chans_j[wrong_offset : wrong_offset + segment_size]
                candidate_info.append({
                    'tokens': wrong_tokens,
                    'channels': wrong_chans,
                    'label': 'wrong',
                    'source_shard': j
                })

            # Evaluate each candidate using the helper function.
            candidate_losses = []
            for candidate in candidate_info:
                loss = compute_completion_loss_with_channels(
                    model,
                    prompt_tokens, prompt_chans,
                    candidate['tokens'], candidate['channels'],
                    device=device
                )
                candidate_losses.append(loss)
            # Find the candidate with the lowest loss.
            min_loss_index = np.argmin(candidate_losses)
            chosen = candidate_info[min_loss_index]
            predicted_shard = chosen['source_shard']
            true_shard = i  # The prompt always comes from shard i
            confusion_matrix[true_shard, predicted_shard] += 1

            if chosen['label'] == 'correct':
                correct_count += 1
            total_evals += 1

            print(f"[Shard {i}] Block {block_index} losses: {candidate_losses}")
            print(f" -> Correct candidate loss: {candidate_losses[0]:.4f} vs. others: {[f'{l:.4f}' for l in candidate_losses[1:]]}")
            print(f" -> Model selected candidate from shard {predicted_shard} (label: {chosen['label']})")
            pos += 2 * segment_size  # advance the block position

    if total_evals == 0:
        print("No evaluations were performed (possibly not enough tokens in the shards).")
        return 0.0

    accuracy = correct_count / total_evals
    print(f"\n[Multi-class Evaluation] Final Accuracy = {correct_count}/{total_evals} = {accuracy:.4f}")

    # Print the confusion matrix.
    print("\nConfusion Matrix (rows: true prompt shard, columns: predicted candidate shard):")
    header = "      " + " ".join([f"Shd{j}" for j in range(num_shards)])
    print(header)
    for i in range(num_shards):
        row_counts = " ".join([f"{confusion_matrix[i, j]:5d}" for j in range(num_shards)])
        print(f"Shd{i} : {row_counts}")
    return accuracy

# === Main script ===
device = torch.device('cpu')
model = GPT(GPTConfig).to(device)
if small_model:
    checkpoint = torch.load('log/model_15000.pt', map_location=device, weights_only=False)
else:
    checkpoint = torch.load('log/model_30000.pt', map_location=device, weights_only=False)
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
epochs = 10
for epoch in range(epochs):
    print(f"\n=== Epoch {epoch+1}/{epochs} ===")
    acc = evaluate_multiclass_with_channels(
        model=model,
        shard_paths=[
            "output_MEMA/shards/shard_train_0.pt",
            "output_MEMA/shards/shard_train_1.pt",
            "output_MEMA/shards/shard_train_2.pt"
        ],
        device="cpu",
        segment_size=512
    )
    accs.append(acc)
mean = np.mean(accs)
print(f"\nMean Accuracy over {epochs} epochs: {mean}")
print(f"Accuracies per epoch: {accs}")
