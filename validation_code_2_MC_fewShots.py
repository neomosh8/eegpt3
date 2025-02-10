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
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
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
        n_head: int = 12  # number of heads
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

    def encode(self, idx, channel_idx=None):
        """
        Returns the final transformer representations for the given tokens.
        This function is used for computing similarity between sequences.
        """
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
        return x

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


# === Original completion loss function (unused in this variant) ===
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


# === New Attention-Based Similarity Loss Function ===
def compute_similarity_loss_with_attention(
        model,
        prompt_tokens, prompt_channels,
        candidate_tokens, candidate_channels,
        device="cuda"
):
    """
    Computes a similarity loss between the prompt and candidate using an attention-based approach.
    The procedure is:
      1. Encode prompt and candidate to obtain token-level representations.
      2. Normalize the representations.
      3. Compute a similarity matrix between candidate and prompt tokens.
      4. For each candidate token, take the maximum similarity to any prompt token,
         and vice versa; then average these scores.
      5. Define the loss as 1 - (average similarity).
    """
    # Prepare inputs (add batch dimension)
    prompt_tokens = prompt_tokens.unsqueeze(0).to(device)  # shape: [1, L_prompt]
    prompt_channels = prompt_channels.unsqueeze(0).to(device)
    candidate_tokens = candidate_tokens.unsqueeze(0).to(device)  # shape: [1, L_candidate]
    candidate_channels = candidate_channels.unsqueeze(0).to(device)

    with torch.no_grad():
        # Get token-level representations: [1, L, D]
        prompt_rep = model.encode(prompt_tokens, prompt_channels)
        candidate_rep = model.encode(candidate_tokens, candidate_channels)

    # Normalize along the embedding dimension (D)
    prompt_rep = F.normalize(prompt_rep, p=2, dim=-1)  # [1, L_prompt, D]
    candidate_rep = F.normalize(candidate_rep, p=2, dim=-1)  # [1, L_candidate, D]

    # Compute similarity matrix (cosine similarity) between candidate and prompt tokens.
    # Shape: [1, L_candidate, L_prompt]
    sim_matrix = torch.matmul(candidate_rep, prompt_rep.transpose(1, 2))

    # For each candidate token, get maximum similarity with any prompt token.
    candidate_to_prompt_sim = sim_matrix.max(dim=-1)[0]  # shape: [1, L_candidate]
    candidate_sim_avg = candidate_to_prompt_sim.mean(dim=-1)  # shape: [1]

    # For each prompt token, get maximum similarity with any candidate token.
    prompt_to_candidate_sim = sim_matrix.max(dim=1)[0]  # shape: [1, L_prompt]
    prompt_sim_avg = prompt_to_candidate_sim.mean(dim=-1)  # shape: [1]

    # Average both directions to obtain a symmetric similarity score.
    similarity_score = (candidate_sim_avg + prompt_sim_avg) / 2.0  # shape: [1]

    # Define the loss: lower loss for higher similarity.
    loss = 1 - similarity_score
    return loss.item()


# === Modified multi-class few-shot force choice evaluation function ===
def evaluate_multiclass_with_channels(
        model,  # the trained model
        shard_paths,  # list of shard file paths (e.g., ["shard_train_0.pt", "shard_train_1.pt", "shard_train_2.pt"])
        device="cuda",
        segment_size=512,  # candidate completion length remains 512 tokens
        prompt_stride=256
):
    """
    For each shard in shard_paths, we perform an evaluation block where:
      - We sample a few-shot prompt from the shard consisting of n shots (n default=4) of 128 consecutive tokens each.
      - We choose a correct continuation (from the same shard) that does not overlap any of the prompt shots.
      - For every other shard, we sample one candidate continuation.
      - We compute an attention-based similarity loss for each candidate given the prompt.
      - The candidate with the lowest loss (i.e. highest similarity) is considered the model's prediction.
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
            # Few-shot prompt parameters.
            few_shot_n = 4  # number of shots (default 4)
            shot_len = 128  # each shot is 128 tokens long
            total_prompt_len = few_shot_n * shot_len  # total prompt length = 512 tokens

            # Determine valid starting positions for shots.
            possible_shot_starts = list(range(0, len_i - shot_len + 1, 256))
            if len(possible_shot_starts) < few_shot_n:
                print(f"Not enough tokens in shard {i} for a few-shot prompt. Skipping...")
                break
            # Randomly sample few_shot_n distinct shot start indices and sort them.
            chosen_shot_starts = random.sample(possible_shot_starts, few_shot_n)
            chosen_shot_starts.sort()
            # Construct the few-shot prompt by concatenating the shots.
            prompt_tokens = torch.cat([tokens_i[start: start + shot_len] for start in chosen_shot_starts], dim=0)
            prompt_chans = torch.cat([chans_i[start: start + shot_len] for start in chosen_shot_starts], dim=0)
            # For candidate selection, define the prompt intervals.
            prompt_intervals = [(start, start + shot_len) for start in chosen_shot_starts]

            # Helper function: check that a candidate block (of length candidate_len) does not overlap any prompt interval.
            def candidate_non_overlapping(c, candidate_len, intervals):
                candidate_end = c + candidate_len
                for (s, e) in intervals:
                    if c < e and candidate_end > s:
                        return False
                return True

            # For the correct candidate (from the same shard), pick a 512-token block that does not overlap any prompt shot.
            all_candidates = list(range(0, len_i - segment_size + 1, prompt_stride))
            correct_offset_candidates = [c for c in all_candidates if
                                         candidate_non_overlapping(c, segment_size, prompt_intervals)]
            if not correct_offset_candidates:
                print(f"No valid correct candidate offsets in shard {i} for the few-shot prompt. Skipping block...")
                pos += 2 * segment_size
                continue
            correct_offset = random.choice(correct_offset_candidates)
            correct_tokens = tokens_i[correct_offset: correct_offset + segment_size]
            correct_chans = chans_i[correct_offset: correct_offset + segment_size]

            # Build the list of candidate completions:
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
                chans_j = other_shard['channels']
                if len_j < segment_size:
                    print(f"Shard {j} is too short. Skipping candidate from this shard.")
                    continue
                wrong_offset = random.randint(0, len_j - segment_size)
                wrong_tokens = tokens_j[wrong_offset: wrong_offset + segment_size]
                wrong_chans = chans_j[wrong_offset: wrong_offset + segment_size]
                candidate_info.append({
                    'tokens': wrong_tokens,
                    'channels': wrong_chans,
                    'label': 'wrong',
                    'source_shard': j
                })

            # Evaluate each candidate using the new attention-based similarity function.
            candidate_losses = []
            for candidate in candidate_info:
                loss = compute_similarity_loss_with_attention(
                    model,
                    prompt_tokens, prompt_chans,
                    candidate['tokens'], candidate['channels'],
                    device=device
                )
                candidate_losses.append(loss)
            # Find the candidate with the lowest loss (i.e. highest similarity).
            min_loss_index = np.argmin(candidate_losses)
            chosen = candidate_info[min_loss_index]
            predicted_shard = chosen['source_shard']
            true_shard = i  # The prompt always comes from shard i
            confusion_matrix[true_shard, predicted_shard] += 1

            if chosen['label'] == 'correct':
                correct_count += 1
            total_evals += 1

            print(f"[Shard {i}] Block {block_index} similarity losses: {candidate_losses}")
            print(
                f" -> Correct candidate loss: {candidate_losses[0]:.4f} vs. others: {[f'{l:.4f}' for l in candidate_losses[1:]]}")
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
d = 'cpu'
device = torch.device(d)
model = GPT(GPTConfig).to(device)
if small_model:
    checkpoint = torch.load('log/model_15000.pt', map_location=device, weights_only=False)
else:
    checkpoint = torch.load('log/model_34500.pt', map_location=device, weights_only=False)
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
    print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
    acc = evaluate_multiclass_with_channels(
        model=model,
        shard_paths=[
            "output_MEMA/shards/shard_train_0.pt",
            "output_MEMA/shards/shard_train_1.pt",
            "output_MEMA/shards/shard_train_2.pt"
        ],
        device=d,
        segment_size=512
    )
    accs.append(acc)
mean = np.mean(accs)
print(f"\nMean Accuracy over {epochs} epochs: {mean}")
print(f"Accuracies per epoch: {accs}")
