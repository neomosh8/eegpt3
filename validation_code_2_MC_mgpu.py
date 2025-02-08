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
from torch import nn

# For debugging or optional usage:
import matplotlib.pyplot as plt

# =============================================================================
# 1. TOKENIZER - Replace with actual tokenizer code or your import
# =============================================================================

from tokenizer2 import BPE_RLE_Tokenizer as Tokenizer
tokenizer = Tokenizer()
tokenizer.load_merges("neo_tokenizer/merges.json")
tokenizer.load_vocab("neo_tokenizer/vocab.json")

# Flag to pick model size
small_model = False


# =============================================================================
# 2. MODEL COMPONENTS
# =============================================================================

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
        # scaled_dot_product_attention in PyTorch 2.0+ automatically applies
        # attention masking if is_causal=True.
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


# =============================================================================
# 3. GPT CONFIG AND MODEL
# =============================================================================
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 6460
    # Parameter sizes:
    # Switch depending on small_model global flag
    if small_model:
        n_layer: int = 12
        n_head: int = 12
        n_embd: int = 768
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

        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # scale for deeper layers
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, channel_idx=None, targets=None):
        B, T = idx.size()
        # position indices
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        # forward the GPT model
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
        # typical AdamW setup
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

        optimizer = torch.optim.AdamW(
            optim_groups,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused
        )
        return optimizer

# =============================================================================
# 4. EVALUATION HELPER
# =============================================================================

def compute_completion_loss_with_channels(
        model,
        prompt_tokens, prompt_channels,
        completion_tokens, completion_channels,
        device="cuda"
):
    """
    A helper function that concatenates prompt + completion, runs a forward pass,
    and returns the average cross-entropy loss on the completion portion only.
    """
    # Move to the correct device
    prompt_tokens = prompt_tokens.to(device)
    prompt_channels = prompt_channels.to(device)
    completion_tokens = completion_tokens.to(device)
    completion_channels = completion_channels.to(device)

    full_tokens = torch.cat([prompt_tokens, completion_tokens], dim=0).unsqueeze(0)
    full_channels = torch.cat([prompt_channels, completion_channels], dim=0).unsqueeze(0)

    total_len = full_tokens.size(1)
    prompt_len = prompt_tokens.size(0)
    completion_len = completion_tokens.size(0)

    # Build a mask so that we only compute CE on the completion portion
    mask_vals = [0] * prompt_len + [1] * completion_len
    mask = torch.tensor(mask_vals, device=device).unsqueeze(0)

    with torch.no_grad():
        logits, _ = model(idx=full_tokens, channel_idx=full_channels)

    # shift by one
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


def evaluate_multiclass_with_channels(
        model,
        shard_paths,
        device="cuda",
        segment_size=500,
        prompt_stride=256
):
    """
    Evaluates how well the model can distinguish the correct continuation
    from shards vs. “wrong” continuations from other shards.
    """
    # Debug: print which device we are inside
    print(f"evaluate_multiclass_with_channels() called with device={device}")

    # Load all shards into memory, CPU for storage
    shards = []
    for path in shard_paths:
        shard = torch.load(path, weights_only=False)
        tokens = shard['tokens'].cpu()
        channels = shard['channels'].cpu()
        shards.append({
            'tokens': tokens,
            'channels': channels,
            'length': tokens.size(0),
            'path': path
        })
    num_shards = len(shards)
    confusion_matrix = np.zeros((num_shards, num_shards), dtype=int)

    total_evals = 0
    correct_count = 0

    # Evaluate
    for i, shard in enumerate(shards):
        tokens_i = shard['tokens']
        chans_i = shard['channels']
        len_i = shard['length']
        block_index = 0
        pos = 0

        while pos + 2*segment_size <= len_i:
            block_index += 1
            print(f"\n[Shard {i}] Block {block_index} at pos={pos}")
            prompt_offset_candidates = list(range(0, len_i - segment_size + 1, prompt_stride))
            if not prompt_offset_candidates:
                print(f"Not enough tokens in shard {i} for a prompt. Skipping...")
                break

            # find a prompt offset
            while True:
                prompt_offset = random.choice(prompt_offset_candidates)
                prompt_tokens = tokens_i[prompt_offset:prompt_offset+segment_size]
                prompt_chans  = chans_i[prompt_offset:prompt_offset+segment_size]

                all_candidates = list(range(0, len_i - segment_size + 1, prompt_stride))
                correct_offset_candidates = [c for c in all_candidates
                                             if (c + segment_size <= prompt_offset or c >= prompt_offset+segment_size)]
                if not correct_offset_candidates:
                    print(f"No valid correct offsets in shard {i} for prompt offset {prompt_offset}. Retrying...")
                    continue

                correct_offset = random.choice(correct_offset_candidates)
                correct_tokens = tokens_i[correct_offset:correct_offset+segment_size]
                correct_chans  = chans_i[correct_offset:correct_offset+segment_size]
                break

            # Now build the candidate completions
            candidate_info = []
            # correct candidate from same shard
            candidate_info.append({
                'tokens': correct_tokens,
                'channels': correct_chans,
                'label': 'correct',
                'source_shard': i
            })

            # wrong candidates from other shards
            for j, other_shard in enumerate(shards):
                if j == i:
                    continue
                len_j = other_shard['length']
                if len_j < segment_size:
                    print(f"Shard {j} is too short. Skipping candidate from this shard.")
                    continue
                tokens_j = other_shard['tokens']
                chans_j  = other_shard['channels']
                wrong_offset = random.randint(0, len_j - segment_size)
                wrong_tokens = tokens_j[wrong_offset:wrong_offset+segment_size]
                wrong_chans  = chans_j[wrong_offset:wrong_offset+segment_size]
                candidate_info.append({
                    'tokens': wrong_tokens,
                    'channels': wrong_chans,
                    'label': 'wrong',
                    'source_shard': j
                })

            # Evaluate each candidate
            candidate_losses = []
            for cinfo in candidate_info:
                loss_val = compute_completion_loss_with_channels(
                    model,
                    prompt_tokens, prompt_chans,
                    cinfo['tokens'], cinfo['channels'],
                    device=device
                )
                candidate_losses.append(loss_val)

            # pick best candidate
            min_loss_index = np.argmin(candidate_losses)
            chosen = candidate_info[min_loss_index]
            predicted_shard = chosen['source_shard']
            true_shard = i
            confusion_matrix[true_shard, predicted_shard] += 1
            if chosen['label'] == 'correct':
                correct_count += 1

            total_evals += 1

            print(f" -> Losses: {candidate_losses}")
            print(f" -> Selected candidate from shard {predicted_shard} (label={chosen['label']})")

            pos += 2*segment_size

    if total_evals == 0:
        print("No evaluations were performed (not enough tokens?). Returning 0.")
        return 0.0

    accuracy = correct_count / total_evals
    print(f"\n[Evaluation] Final Accuracy = {correct_count}/{total_evals} = {accuracy:.4f}")

    # Show the confusion matrix
    print("\nConfusion Matrix (rows=prompt shard, cols=predicted):")
    header = "      " + " ".join([f"Shd{j}" for j in range(num_shards)])
    print(header)
    for i in range(num_shards):
        row_counts = " ".join([f"{confusion_matrix[i, j]:5d}" for j in range(num_shards)])
        print(f"Shd{i} : {row_counts}")

    return accuracy


# =============================================================================
# 5. MAIN SCRIPT
# =============================================================================

def main():
    # If you have 4 GPUs, define them here:
    gpu_devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]

    # Build (or load) the model on CPU first
    model = GPT(GPTConfig)
    # Load checkpoint
    if small_model:
        checkpoint_path = 'log/model_15000.pt'
    else:
        checkpoint_path = 'log/model_34500.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    orig_sd = checkpoint['model']
    fixed_sd = {}
    for k, v in orig_sd.items():
        new_key = k.replace("_orig_mod.", "")
        fixed_sd[new_key] = v
    model.load_state_dict(fixed_sd, strict=True)
    model.config(checkpoint['config'])
    model.eval()

    print("Model loaded. Parameters now on device:", next(model.parameters()).device)

    epochs = 10
    accs = []
    for epoch in range(epochs):
        # Round robin among GPUs
        device_str = gpu_devices[epoch % len(gpu_devices)]
        device_epoch = torch.device(device_str)

        # Set the default GPU for any new allocations
        torch.cuda.set_device(device_epoch)

        # Move the model parameters to this device
        model = model.to(device_epoch)
        print(f"\n=== Epoch {epoch+1}/{epochs} on device {device_str} ===")
        print("Check model device:", next(model.parameters()).device)

        # (Optional) Clear previous GPU memory cache
        torch.cuda.empty_cache()

        # Evaluate
        acc = evaluate_multiclass_with_channels(
            model=model,
            shard_paths=[
                "output_MEMA/shards/shard_train_0.pt",
                "output_MEMA/shards/shard_train_1.pt",
                "output_MEMA/shards/shard_train_2.pt"
            ],
            device=device_str,
            segment_size=512
        )
        accs.append(acc)

    # After all epochs
    mean_acc = np.mean(accs)
    print(f"\nAccuracies over {epochs} epochs: {accs}")
    print(f"Mean Accuracy: {mean_acc}")

if __name__ == "__main__":
    main()
