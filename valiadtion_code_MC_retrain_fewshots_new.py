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

# -----------------------------
# Configuration Flags and Hyperparameters
# -----------------------------
small_model = True        # Set to True for a small model, False for a large model.
freeze_pretrained = False # Set to True to freeze all layers except the similarity head.
margin_value = 0.2        # Margin used in the loss function.
lr_similarity = 5e-4      # Learning rate for the similarity head.
lr_other = 1e-3           # Learning rate for all other parameters.

# -----------------------------
# Tokenizer Loading
# -----------------------------
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
    # Use a small or large configuration based on flag.
    n_layer: int = 12 if small_model else 36  # number of layers
    n_head: int = 12 if small_model else 20     # number of heads
    n_embd: int = 768 if small_model else 1280    # embedding dimension
    num_channels: int = 2
    mlp_dropout: float = 0.05
    attn_dropout: float = 0.05
    resid_dropout: float = 0.05


# -----------------------------
# Original Bilinear Similarity
# -----------------------------
class BilinearSimilarity(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)

    def forward(self, u, v):
        # u and v are [B, D] vectors.
        return torch.sum((u @ self.W) * v, dim=-1)


# -----------------------------
# Scaled (Temperature) Similarity Head
# -----------------------------
class ScaledBilinearSimilarity(nn.Module):
    def __init__(self, embed_dim, init_scale=10.0):
        """
        Computes a scaled cosine similarity.
        :param embed_dim: Dimensionality of the embeddings.
        :param init_scale: Initial scaling factor (temperature). A higher value can magnify small differences.
        """
        super().__init__()
        self.bilinear = BilinearSimilarity(embed_dim)
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float))

    def forward(self, u, v):
        raw_sim = self.bilinear(u, v)  # This computes a cosine-like similarity if u and v are normalized.
        return self.scale * raw_sim


# -----------------------------
# GPT Model Definition (No Gradient Checkpointing)
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
        # Weight tying: share the embedding matrix between input and output.
        self.transformer.wte.weight = self.lm_head.weight
        # Replace the similarity head with our scaled (temperature) version.
        self.similarity_head = ScaledBilinearSimilarity(config.n_embd, init_scale=10.0)
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
        (No gradient checkpointing is applied.)
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
        # Separate out similarity head parameters so we can assign a different learning rate.
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = []
        nodecay_params = []
        similarity_params = []
        for pn, p in param_dict.items():
            if "similarity_head" in pn:
                similarity_params.append(p)
            elif p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay, 'lr': lr_other},
            {'params': nodecay_params, 'weight_decay': 0.0, 'lr': lr_other},
            {'params': similarity_params, 'weight_decay': 0.0, 'lr': lr_similarity},
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and ('cuda' in device)
        optimizer = torch.optim.AdamW(optim_groups,
                                      betas=(0.9, 0.95),
                                      eps=1e-8,
                                      fused=use_fused)
        return optimizer


# -----------------------------
# Data & Candidate Sampling Helpers
# -----------------------------
def load_shards(shard_paths):
    shards = []
    for path in shard_paths:
        # If you don't control the file, consider setting weights_only=True.
        shard = torch.load(path, map_location="cpu")
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
    few_shot_n = prompt_tokens.size(0) // shot_len
    # Reshape prompt tokens and channels: [few_shot_n, shot_len]
    prompt_tokens_shots = prompt_tokens.view(few_shot_n, shot_len)
    prompt_channels_shots = prompt_channels.view(few_shot_n, shot_len)

    shot_embeddings = []
    # If model is wrapped in DataParallel, use the underlying module.
    model_to_use = model.module if hasattr(model, "module") else model

    for i in range(few_shot_n):
        shot_tok = prompt_tokens_shots[i].unsqueeze(0).to(device)
        shot_cha = prompt_channels_shots[i].unsqueeze(0).to(device)
        rep = model_to_use.encode(shot_tok, shot_cha)  # [1, shot_len, D]
        pooled = rep.mean(dim=1)                        # [1, D]
        shot_embeddings.append(pooled)
    shot_embeddings = torch.cat(shot_embeddings, dim=0)  # [few_shot_n, D]
    shot_embeddings = F.normalize(shot_embeddings, p=2, dim=-1)

    # Stack candidate tensors into a batch.
    candidate_tokens_batch = torch.stack(candidate_tokens_list, dim=0).to(device)    # [num_candidates, seg_len]
    candidate_channels_batch = torch.stack(candidate_channels_list, dim=0).to(device)  # [num_candidates, seg_len]

    rep_cand = model_to_use.encode(candidate_tokens_batch, candidate_channels_batch)  # [num_candidates, seg_len, D]
    cand_pool = rep_cand.mean(dim=1)  # [num_candidates, D]
    cand_pool = F.normalize(cand_pool, p=2, dim=-1)

    # Compute similarity matrix between each shot and each candidate.
    sims = torch.matmul(shot_embeddings, cand_pool.transpose(0, 1))  # [few_shot_n, num_candidates]
    # Aggregate similarity across shots (using mean).
    candidate_scores = sims.mean(dim=0)  # [num_candidates]
    return candidate_scores


# -----------------------------
# Margin Loss Function
# -----------------------------
def compute_margin_loss(candidate_scores, margin=0.2):
    """
    Computes a margin loss given candidate_scores (tensor of shape [num_candidates]),
    where candidate_scores[0] is the positive candidate and the rest are negatives.
    Loss = mean( ReLU( margin + s_neg - s_pos ) )
    """
    s_pos = candidate_scores[0]           # Positive candidate score
    s_negs = candidate_scores[1:]         # Negative candidate scores
    losses = F.relu(margin + s_negs - s_pos)
    return losses.mean()


# -----------------------------
# Training Function (No AMP, No Checkpointing)
# -----------------------------
def train_on_shards(model, shard_paths, optimizer, device="cuda",
                    segment_size=512, shot_len=128, prompt_stride=256, num_epochs=3):
    """
    Fine-tune the pretrained model on prompt shots with positive and negative candidates.
    Uses a margin loss over candidate similarity scores.
    Also prints the positive (s_pos) and negative (s_negs) similarity scores every 50 steps.
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
                correct_offset_candidates = [c for c in all_candidates if candidate_non_overlapping(c, segment_size, prompt_intervals)]
                if not correct_offset_candidates:
                    pos += 2 * segment_size
                    continue
                correct_offset = random.choice(correct_offset_candidates)
                correct_tokens = tokens_i[correct_offset: correct_offset + segment_size]
                correct_chans = chans_i[correct_offset: correct_offset + segment_size]

                # Candidate 0 is correct; sample negatives from other shards.
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
                loss = compute_margin_loss(sims, margin=margin_value)
                loss.backward()
                optimizer.step()

                # For inspection: print s_pos and s_negs.
                s_pos = sims[0].item()
                s_negs = sims[1:].tolist()

                pred = torch.argmax(sims).item()
                correct_flag = 1 if pred == 0 else 0

                epoch_loss += loss.item()
                epoch_correct += correct_flag
                epoch_total += 1
                total_training_steps += 1

                if total_training_steps % 50 == 0:
                    running_acc = epoch_correct / epoch_total * 100
                    print(f"Epoch {epoch+1}, Step {total_training_steps}: Loss = {loss.item():.4f}, "
                          f"Running Acc = {running_acc:.2f}%")
                    print(f"   s_pos: {s_pos:.4f}, s_negs: {[f'{v:.4f}' for v in s_negs]}")
                pos += 2 * segment_size

        avg_loss = epoch_loss / epoch_total if epoch_total > 0 else 0.0
        acc = epoch_correct / epoch_total if epoch_total > 0 else 0.0
        print(f"Epoch {epoch+1} completed: Avg Loss = {avg_loss:.4f}, Accuracy = {acc*100:.2f}%")
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
            print(f" -> Predicted candidate from shard {predicted_shard} (label: {chosen['label']}). "
                  f"Running Acc = {running_accuracy:.2f}% ({correct_count}/{total_evals})")
            pos += 2 * segment_size

    final_acc = correct_count / total_evals if total_evals > 0 else 0.0
    print(f"\n[Evaluation] Final Accuracy = {correct_count}/{total_evals} = {final_acc*100:.2f}%")
    print("\nConfusion Matrix (rows: true prompt shard, columns: predicted candidate shard):")
    header = "      " + " ".join([f"Shd{j}" for j in range(num_shards)])
    print(header)
    for i in range(num_shards):
        row = " ".join([f"{confusion_matrix[i, j]:5d}" for j in range(num_shards)])
        print(f"Shd{i} : {row}")
    return final_acc


# -----------------------------
# Main Script: Loading Pretrained, Training, and Evaluation
# -----------------------------
d = 'cuda'
device = torch.device(d)
model = GPT(GPTConfig).to(device)

# Optionally load your pretrained model to leverage unsupervised training.
pretrained_path = 'log/model_15000.pt'
if os.path.exists(pretrained_path):
    checkpoint = torch.load(pretrained_path, map_location=device)
    fixed_sd = {k.replace("_orig_mod.", ""): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(fixed_sd, strict=False)
    print("Pretrained model loaded.")

# Optionally freeze pretrained layers except for the similarity head.
if freeze_pretrained:
    print("Freezing all layers except similarity head.")
    for name, param in model.named_parameters():
        if "similarity_head" not in name:
            param.requires_grad = False

# Use DataParallel if multiple GPUs are available.
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for training.")
    model = nn.DataParallel(model)
model.to(device)

# When using DataParallel, call configure_optimizer on the underlying module.
if isinstance(model, nn.DataParallel):
    optimizer = model.module.configure_optimizer(weight_decay=0.1, learning_rate=lr_other, device=d)
else:
    optimizer = model.configure_optimizer(weight_decay=0.1, learning_rate=lr_other, device=d)

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
print(f"Final Evaluation Accuracy: {eval_acc*100:.2f}%")
