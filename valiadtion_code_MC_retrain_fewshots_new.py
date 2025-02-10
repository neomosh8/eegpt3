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

# Set this flag as desired.
# When False, you get a "large" model configuration (e.g. n_embd=1280) which may not fully match your pretrained checkpoint.
small_model = True
tokenizer = Tokenizer()
tokenizer.load_merges("neo_tokenizer/merges.json")
tokenizer.load_vocab("neo_tokenizer/vocab.json")


# -----------------------------
# Model Components
# -----------------------------
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
    # Use a small or large configuration
    n_layer: int = 12 if small_model else 36  # number of layers
    n_head: int = 12 if small_model else 20  # number of heads
    n_embd: int = 768 if small_model else 1280  # embedding dimension
    num_channels: int = 2
    mlp_dropout: float = 0.05
    attn_dropout: float = 0.05
    resid_dropout: float = 0.05


# -----------------------------
# Learned Similarity Head
# -----------------------------
class BilinearSimilarity(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)

    def forward(self, u, v):
        # u and v are assumed to be [B, D] vectors.
        return torch.sum((u @ self.W) * v, dim=-1)


# -----------------------------
# GPT Model Definition (Without gradient checkpointing or AMP)
# -----------------------------
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
        # Weight tying.
        self.transformer.wte.weight = self.lm_head.weight
        self.similarity_head = BilinearSimilarity(config.n_embd)
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
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits, loss

    def encode(self, idx, channel_idx=None):
        """
        Returns the final transformer representations (after ln_f) for a given sequence.
        Shape: [B, T, D]
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
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and ('cuda' in device)
        optimizer = torch.optim.AdamW(optim_groups,
                                      betas=(0.9, 0.95),
                                      eps=1e-8,
                                      fused=use_fused)
        return optimizer


# -----------------------------
# Helper: If model is wrapped in DataParallel, return underlying module.
# -----------------------------
def get_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


# -----------------------------
# Data & Candidate Sampling Helpers
# -----------------------------
def load_shards(shard_paths):
    shards = []
    for path in shard_paths:
        shard = torch.load(path, map_location="cpu")  # Expected keys: 'tokens', 'channels'
        tokens = shard['tokens'].cpu()
        channels = shard['channels'].cpu()
        shards.append({
            'tokens': tokens,
            'channels': channels,
            'length': tokens.size(0),
            'path': path
        })
    return shards


def candidate_non_overlapping(c, candidate_len, intervals):
    candidate_end = c + candidate_len
    for (s, e) in intervals:
        if c < e and candidate_end > s:
            return False
    return True


# -----------------------------
# Vectorized Candidate Similarity Computation
# -----------------------------
def compute_candidate_similarities_vectorized(model, prompt_tokens, prompt_channels,
                                              candidate_tokens_list, candidate_channels_list,
                                              shot_len, device="cuda"):
    """
    Given a prompt (concatenated shots) and a list of candidate sequences,
    compute similarity scores for all candidates in a vectorized manner.
    Returns a tensor of shape [num_candidates] with similarity scores.
    """
    actual_model = get_model(model)
    few_shot_n = prompt_tokens.size(0) // shot_len
    # Reshape prompt tokens and channels: shape [few_shot_n, shot_len]
    prompt_tokens_shots = prompt_tokens.view(few_shot_n, shot_len)
    prompt_channels_shots = prompt_channels.view(few_shot_n, shot_len)

    shot_embeddings = []
    for i in range(few_shot_n):
        shot_tok = prompt_tokens_shots[i].unsqueeze(0).to(device)  # [1, shot_len]
        shot_cha = prompt_channels_shots[i].unsqueeze(0).to(device)  # [1, shot_len]
        rep = actual_model.encode(shot_tok, shot_cha)  # [1, shot_len, D]
        pooled = rep.mean(dim=1)  # [1, D]
        shot_embeddings.append(pooled)
    shot_embeddings = torch.cat(shot_embeddings, dim=0)  # [few_shot_n, D]
    shot_embeddings = F.normalize(shot_embeddings, p=2, dim=-1)

    # Stack candidate tensors into a batch.
    candidate_tokens_batch = torch.stack(candidate_tokens_list, dim=0).to(device)  # [num_candidates, seg_len]
    candidate_channels_batch = torch.stack(candidate_channels_list, dim=0).to(device)  # [num_candidates, seg_len]

    cand_rep = actual_model.encode(candidate_tokens_batch, candidate_channels_batch)  # [num_candidates, seg_len, D]
    cand_pool = cand_rep.mean(dim=1)  # [num_candidates, D]
    cand_pool = F.normalize(cand_pool, p=2, dim=-1)

    # Compute similarity matrix between each shot and each candidate.
    sims = torch.matmul(shot_embeddings, cand_pool.transpose(0, 1))  # [few_shot_n, num_candidates]
    # Aggregate similarities across shots (using mean).
    candidate_scores = sims.mean(dim=0)  # [num_candidates]
    return candidate_scores


# -----------------------------
# Margin Loss Function
# -----------------------------
def compute_margin_loss(candidate_scores, margin=0.2):
    """
    Computes a margin loss given candidate_scores (tensor of shape [num_candidates]),
    where candidate_scores[0] is the positive (correct) candidate and the rest are negatives.
    The loss is:
        loss = mean( relu( margin + s_neg - s_pos ) )
    """
    s_pos = candidate_scores[0]  # Positive candidate score (scalar)
    s_negs = candidate_scores[1:]  # Negative candidate scores (vector)
    losses = F.relu(margin + s_negs - s_pos)
    return losses.mean()


# -----------------------------
# Training Function (No AMP, No Checkpointing)
# -----------------------------
def train_on_shards(model, shard_paths, optimizer, device="cuda",
                    segment_size=512, shot_len=128, prompt_stride=256, num_epochs=5):
    """
    Fine-tune the pretrained model on prompt shots with positive and negative candidates.
    Uses a margin loss over candidate similarity scores.
    """
    model.train()
    shards = load_shards(shard_paths)
    total_training_steps = 0

    for epoch in range(num_epochs):
        print(f"\n=== Training Epoch {epoch + 1}/{num_epochs} ===")
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for i, shard in enumerate(shards):
            tokens_i = shard['tokens']
            chans_i = shard['channels']
            len_i = shard['length']
            pos = 0
            while pos + 2 * segment_size <= len_i:
                few_shot_n = 4  # number of shots
                possible_shot_starts = list(range(0, len_i - shot_len + 1, 256))
                if len(possible_shot_starts) < few_shot_n:
                    pos += 2 * segment_size
                    continue
                chosen_shot_starts = random.sample(possible_shot_starts, few_shot_n)
                chosen_shot_starts.sort()
                prompt_tokens = torch.cat([tokens_i[start: start + shot_len] for start in chosen_shot_starts], dim=0)
                prompt_chans = torch.cat([chans_i[start: start + shot_len] for start in chosen_shot_starts], dim=0)
                prompt_intervals = [(start, start + shot_len) for start in chosen_shot_starts]

                all_candidates = list(range(0, len_i - segment_size + 1, prompt_stride))
                correct_offset_candidates = [c for c in all_candidates if
                                             candidate_non_overlapping(c, segment_size, prompt_intervals)]
                if not correct_offset_candidates:
                    pos += 2 * segment_size
                    continue
                correct_offset = random.choice(correct_offset_candidates)
                correct_tokens = tokens_i[correct_offset: correct_offset + segment_size]
                correct_chans = chans_i[correct_offset: correct_offset + segment_size]

                # Candidate 0 is the correct one; then sample negatives from other shards.
                candidate_tokens_list = [correct_tokens]
                candidate_channels_list = [correct_chans]
                for j, other_shard in enumerate(shards):
                    if j == i:
                        continue
                    len_j = other_shard['length']
                    tokens_j = other_shard['tokens']
                    chans_j = other_shard['channels']
                    if len_j < segment_size:
                        continue
                    wrong_offset = random.randint(0, len_j - segment_size)
                    wrong_tokens = tokens_j[wrong_offset: wrong_offset + segment_size]
                    wrong_chans = chans_j[wrong_offset: wrong_offset + segment_size]
                    candidate_tokens_list.append(wrong_tokens)
                    candidate_channels_list.append(wrong_chans)

                optimizer.zero_grad()
                sims = compute_candidate_similarities_vectorized(model,
                                                                 prompt_tokens, prompt_chans,
                                                                 candidate_tokens_list, candidate_channels_list,
                                                                 shot_len, device=device)
                loss = compute_margin_loss(sims, margin=0.2)
                loss.backward()
                optimizer.step()

                pred = torch.argmax(sims).item()
                correct = 1 if pred == 0 else 0

                epoch_loss += loss.item()
                epoch_correct += correct
                epoch_total += 1
                total_training_steps += 1

                if total_training_steps % 50 == 0:
                    running_acc = epoch_correct / epoch_total * 100
                    print(
                        f"Epoch {epoch + 1}, Step {total_training_steps}: Loss = {loss.item():.4f}, Running Acc = {running_acc:.2f}%")
                pos += 2 * segment_size

        avg_loss = epoch_loss / epoch_total if epoch_total > 0 else 0.0
        acc = epoch_correct / epoch_total if epoch_total > 0 else 0.0
        print(f"Epoch {epoch + 1} completed: Avg Loss = {avg_loss:.4f}, Accuracy = {acc * 100:.2f}%")
    model.eval()


# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate_multiclass_with_similarity(model, shard_paths, device="cuda",
                                        segment_size=512, prompt_stride=256, shot_len=128):
    """
    Evaluates the fine-tuned model in a multi-class forced-choice setting.
    Prints running accuracy and a confusion matrix.
    """
    shards = load_shards(shard_paths)
    num_shards = len(shards)
    confusion_matrix = np.zeros((num_shards, num_shards), dtype=int)
    total_evals = 0
    correct_count = 0

    for i, shard in enumerate(shards):
        tokens_i = shard['tokens']
        chans_i = shard['channels']
        len_i = shard['length']
        pos = 0
        block_index = 0
        while pos + 2 * segment_size <= len_i:
            block_index += 1
            possible_shot_starts = list(range(0, len_i - shot_len + 1, 256))
            if len(possible_shot_starts) < 4:
                break
            chosen_shot_starts = random.sample(possible_shot_starts, 4)
            chosen_shot_starts.sort()
            prompt_tokens = torch.cat([tokens_i[start: start + shot_len] for start in chosen_shot_starts], dim=0)
            prompt_chans = torch.cat([chans_i[start: start + shot_len] for start in chosen_shot_starts], dim=0)
            prompt_intervals = [(start, start + shot_len) for start in chosen_shot_starts]

            cand_candidates = [c for c in list(range(0, len_i - segment_size + 1, prompt_stride))
                               if candidate_non_overlapping(c, segment_size, prompt_intervals)]
            if not cand_candidates:
                pos += 2 * segment_size
                continue
            correct_offset = random.choice(cand_candidates)
            correct_tokens = tokens_i[correct_offset: correct_offset + segment_size]
            correct_chans = chans_i[correct_offset: correct_offset + segment_size]

            candidate_info = []
            candidate_info.append({
                'tokens': correct_tokens,
                'channels': correct_chans,
                'label': 'correct',
                'source_shard': i
            })
            for j, other_shard in enumerate(shards):
                if j == i:
                    continue
                len_j = other_shard['length']
                tokens_j = other_shard['tokens']
                chans_j = other_shard['channels']
                if len_j < segment_size:
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

            candidate_tokens_list = [cand['tokens'] for cand in candidate_info]
            candidate_channels_list = [cand['channels'] for cand in candidate_info]
            with torch.no_grad():
                sims = compute_candidate_similarities_vectorized(model,
                                                                 prompt_tokens, prompt_chans,
                                                                 candidate_tokens_list, candidate_channels_list,
                                                                 shot_len, device=device)
            pred = torch.argmax(sims).item()
            chosen = candidate_info[pred]
            predicted_shard = chosen['source_shard']
            confusion_matrix[i, predicted_shard] += 1
            if chosen['label'] == 'correct':
                correct_count += 1
            total_evals += 1

            running_accuracy = correct_count / total_evals * 100
            print(f"[Shard {i}] Block {block_index}: Similarity Scores = {[f'{s.item():.4f}' for s in sims]}")
            print(
                f" -> Predicted candidate from shard {predicted_shard} (label: {chosen['label']}). Running Acc = {running_accuracy:.2f}% ({correct_count}/{total_evals})")
            pos += 2 * segment_size

    final_acc = correct_count / total_evals if total_evals > 0 else 0.0
    print(f"\n[Evaluation] Final Accuracy = {correct_count}/{total_evals} = {final_acc * 100:.2f}%")
    print("\nConfusion Matrix (rows: true prompt shard, columns: predicted candidate shard):")
    header = "      " + " ".join([f"Shd{j}" for j in range(num_shards)])
    print(header)
    for i in range(num_shards):
        row = " ".join([f"{confusion_matrix[i, j]:5d}" for j in range(num_shards)])
        print(f"Shd{i} : {row}")
    return final_acc


# -----------------------------
# Main Script: Pretrained Loading, Training, and Evaluation
# -----------------------------
d = 'cuda'
device = torch.device(d)
model = GPT(GPTConfig).to(device)

# Optionally load your pretrained checkpoint to leverage unsupervised training:
pretrained_path = 'log/model_15000.pt'
if os.path.exists(pretrained_path):
    checkpoint = torch.load(pretrained_path, map_location=device)
    fixed_sd = {k.replace("_orig_mod.", ""): v for k, v in checkpoint['model'].items()}
    print("Pretrained model state_dict loaded.")
    model_sd = model.state_dict()
    new_sd = {}
    for key in fixed_sd:
        if key in model_sd:
            if model_sd[key].shape == fixed_sd[key].shape:
                new_sd[key] = fixed_sd[key]
            else:
                print(
                    f"Skipping parameter {key}: checkpoint shape {fixed_sd[key].shape} vs model shape {model_sd[key].shape}")
        else:
            print(f"Parameter {key} not found in model state_dict.")
    model.load_state_dict(new_sd, strict=False)
    print("Pretrained weights loaded (only matching parameters).")
else:
    print("Pretrained checkpoint not found. Training from scratch.")

# Use DataParallel if more than one GPU is available.
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for training.")
    model = nn.DataParallel(model)
model.to(device)

# Get optimizer from the underlying module if using DataParallel.
optimizer = model.module.configure_optimizer(weight_decay=0.1, learning_rate=1e-3, device=d) if isinstance(model,
                                                                                                           nn.DataParallel) else model.configure_optimizer(
    weight_decay=0.1, learning_rate=1e-3, device=d)

# Define shard paths for training and evaluation.
shard_paths_train = [
    "output_EMOTIV/shards/shard_train_0.pt",
    "output_EMOTIV/shards/shard_train_1.pt",
    "output_EMOTIV/shards/shard_train_2.pt"
]

print("\n=== Starting Fine-Tuning ===")
train_on_shards(model, shard_paths_train, optimizer, device=d,
                segment_size=512, shot_len=128, prompt_stride=256, num_epochs=3)

print("\n=== Evaluating the Fine-Tuned Model ===")
eval_shard_paths = [
    "output_MEMA/shards/shard_train_0.pt",
    "output_MEMA/shards/shard_train_1.pt",
    "output_MEMA/shards/shard_train_2.pt"
]
eval_acc = evaluate_multiclass_with_similarity(model, eval_shard_paths, device=d,
                                               segment_size=512, prompt_stride=256, shot_len=128)
print(f"Final Evaluation Accuracy: {eval_acc * 100:.2f}%")
