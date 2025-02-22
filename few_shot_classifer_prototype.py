import torch
import random


import Model_class
import glob
import os
import math
import random
import time
import inspect
from dataclasses import dataclass
import contextlib

import torch
import torch.nn as nn
from fontTools.unicodedata import script
from torch.nn import functional as F
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from checkpoint_manager import save_checkpoint
# assumed available; replace or remove if not using S3 logging
from handle_tokenized import upload_folder_to_s3
from plotter import LossPlotter

#########################
# DDP Setup
#########################
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "CUDA is required for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

# Set manual seeds for reproducibility.
torch.manual_seed(9259)
random.seed(88)

if torch.cuda.is_available():
    random.seed(88)
    torch.cuda.manual_seed(9259)

#########################
# Model Components
#########################

class SimpleCrossChannelFusion(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        reduced_dim = 16
        self.proj_reduce = nn.Linear(n_embd, reduced_dim)
        self.proj_expand = nn.Linear(reduced_dim, n_embd)
        self.ln = nn.LayerNorm(n_embd)

    def forward(self, x):
        B, T, C, E = x.size()
        # Average across channels to fuse inter-channel information
        fused = x.mean(dim=2, keepdim=True)  # [B, T, 1, E]
        # Project to a lower-dimensional space
        fused = self.proj_reduce(fused)  # [B, T, 1, reduced_dim]
        # (Optional: you can add a non-linearity here)
        # Project back to the original embedding dimension
        fused = self.proj_expand(fused)  # [B, T, 1, E]
        # Add the reduced fusion back to the original representation and normalize
        x = x + fused.expand_as(x)
        return self.ln(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # NANOGPT uses a special initialization flag.
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.attn_dropout = nn.Dropout(p=getattr(config, 'attn_dropout', 0.05))
        self.resid_dropout = nn.Dropout(p=getattr(config, 'resid_dropout', 0.05))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # Reshape for multi-head attention.
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
        self.attn = CausalSelfAttentionWithRoPE(config)  # Updated to RoPE
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class BlockWithFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttentionWithRoPE(config)  # Updated to RoPE
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.fusion = SimpleCrossChannelFusion(config.n_embd)
        self.ln_3 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        B, T, C, E = x.size()
        x_reshaped = x.view(B * C, T, E)
        x_reshaped = x_reshaped + self.attn(self.ln_1(x_reshaped))
        x = x_reshaped.view(B, T, C, E)
        x = x + self.fusion(self.ln_2(x))
        x_reshaped = x.view(B * C, T, E)
        x_reshaped = x_reshaped + self.mlp(self.ln_3(x_reshaped))
        x = x_reshaped.view(B, T, C, E)
        return x

class CrossChannelFusion(nn.Module):
    def __init__(self, n_embd, num_heads=1):
        super().__init__()
        # use batch_first=True so shapes are [B, seq_len, embd]
        self.attn = nn.MultiheadAttention(embed_dim=n_embd, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        """
        x: [B, time_steps, num_channels, n_embd]
        We flatten (time_steps * num_channels) into a single dimension => "seq_len".
        """
        B, T, C, E = x.size()
        # Flatten time & channels => [B, T*C, E]
        x = x.view(B, T * C, E)

        # MultiheadAttention expects [B, seq_len, embd] if batch_first=True
        fused, _ = self.attn(x, x, x)  # [B, T*C, E]
        # Reshape back to [B, T, C, E] if you still want that 4D layout:
        fused = fused.view(B, T, C, E)

        return fused

class CausalSelfAttentionWithRoPE(nn.Module):
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

    def apply_rotary_emb(self, x, seq_len):
        # Assumes x is [B, n_head, T, head_dim]
        head_dim = x.size(-1)
        freqs = torch.arange(0, head_dim, 2, device=x.device).float() / head_dim
        theta = 1000.0 ** (-freqs)  # Frequency base
        positions = torch.arange(seq_len, device=x.device).float()
        angles = positions[:, None] * theta[None, :]  # [T, head_dim/2]
        sin = angles.sin()
        cos = angles.cos()
        x1, x2 = x[..., 0::2], x[..., 1::2]  # Split even/odd dimensions
        x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rot

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # Apply RoPE to q and k
        q = self.apply_rotary_emb(q, T)
        k = self.apply_rotary_emb(k, T)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = self.attn_dropout(y)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 82
    # vocab_size: int = 10799
    # Small model configuration
    # n_layer: int = 12
    # n_head: int = 12
    # n_embd: int = 768

    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

    # n_layer: int = 12
    # n_head: int = 12
    # n_embd: int = 768
    num_channels: int = 3
    mlp_dropout: float = 0.00
    attn_dropout: float = 0.00
    resid_dropout: float = 0.00
    pad_token: int = 0  # Padding token for inputs


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([BlockWithFusion(config) for _ in range(config.n_layer)]),
            "ln_f": nn.LayerNorm(config.n_embd)
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # Shared intra-channel encoder (replaces per-channel encoder)
        self.intra_channel_encoder = nn.Sequential(
            Block(config),
            Block(config),
            Block(config),
        )

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

    def forward(self, idx, targets=None):
        B, C, T = idx.size()
        tok_emb = self.transformer.wte(idx)  # [B, C, T, n_embd]
        x = tok_emb.transpose(1, 2)  # [B, T, C, n_embd]
        # No positional embeddings added here; RoPE handles it in attention

        channel_outs = []
        for c in range(C):
            x_c = x[:, :, c, :]  # [B, T, n_embd]
            x_c = self.intra_channel_encoder(x_c)  # [B, T, n_embd]
            channel_outs.append(x_c)
        x = torch.stack(channel_outs, dim=2)  # [B, T, C, n_embd]

        for block in self.transformer.h:
            x = block(x)  # [B, T, C, n_embd]

        x = x.transpose(1, 2).reshape(B * C, T, self.config.n_embd)  # [B * C, T, n_embd]
        x = self.transformer.ln_f(x)  # [B * C, T, n_embd]
        logits = self.lm_head(x)  # [B * C, T, vocab_size]
        logits = logits.view(B, C, T, -1)  # [B, C, T, vocab_size]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss


    def configure_optimizer(self, weight_decay, learning_rate, device):
        """
        Configure the optimizer with separate parameter groups for decayed and non-decayed weights.
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = []
        nodecay_params = []

        for pn, p in param_dict.items():
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate},
            {'params': nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and ('cuda' in device)

        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)} with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)} with {num_nodecay_params:,} parameters")
            print(f"Using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(
            optim_groups,
            betas=(0.95, 0.999),
            eps=1e-8,
            fused=use_fused
        )
        return optimizer

    # def configure_optimizer(self, weight_decay, learning_rate, device):
    #     """
    #     Configure the optimizer with separate parameter groups for decayed and non-decayed weights using RMSprop.
    #     """
    #     param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
    #     decay_params = []
    #     nodecay_params = []
    #
    #     for pn, p in param_dict.items():
    #         if p.dim() >= 2:  # Apply weight decay to 2D+ parameters (e.g., weights)
    #             decay_params.append(p)
    #         else:  # No weight decay for 1D parameters (e.g., biases, LayerNorm params)
    #             nodecay_params.append(p)
    #
    #     optim_groups = [
    #         {'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate},
    #         {'params': nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate},
    #     ]
    #
    #     num_decay_params = sum(p.numel() for p in decay_params)
    #     num_nodecay_params = sum(p.numel() for p in nodecay_params)
    #
    #     if master_process:
    #         print(f"num decayed parameter tensors: {len(decay_params)} with {num_decay_params:,} parameters")
    #         print(f"num non-decayed parameter tensors: {len(nodecay_params)} with {num_nodecay_params:,} parameters")
    #
    #     # Initialize RMSprop optimizer
    #     optimizer = torch.optim.RMSprop(
    #         optim_groups,
    #         lr=learning_rate,
    #         alpha=0.99,  # Smoothing constant (default is 0.99, equivalent to Adam's beta2)
    #         eps=1e-8,  # Numerical stability term
    #         weight_decay=weight_decay,  # Weight decay is handled per parameter group
    #         momentum=0.0,  # RMSprop can use momentum, set to 0 if not needed
    #         centered=False  # If True, computes a centered RMSprop (normalizes gradients by variance)
    #     )
    #     return optimizer


import torch
import random

# From the original code
REGIONS = ["frontal", "motor_temporal", "parietal_occipital"]
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# Assuming GPT and GPTConfig classes are defined as in the provided code
# [Insert GPT, GPTConfig, and related classes like BlockWithFusion, CausalSelfAttention, etc., here]

def load_fewshot_data(shard_paths, T=1024, K=3, pad_token=0):
    """
    Load few-shot data from shards, truncate to the same length across classes, and split into support and query sets.

    Args:
        shard_paths (list): List of paths to .pt files, one per class.
        T (int): Sequence length per channel.
        K (int): Number of support samples per class.
        pad_token (int): Token to pad sequences if needed.

    Returns:
        support_data (list): List of (sequence, label) tuples for support set.
        query_data (list): List of (sequence, label) tuples for query set.
    """
    all_sequences = []
    min_num_sequences = float('inf')

    # Step 1: Load and compute number of sequences per shard
    for label, shard_path in enumerate(shard_paths):
        loaded = torch.load(shard_path, map_location="cpu", weights_only=False)
        min_length = min(loaded[region].size(0) for region in REGIONS)
        num_sequences = (min_length - T) // T + 1  # Non-overlapping sequences
        min_num_sequences = min(min_num_sequences, num_sequences)
        if num_sequences < K:
            raise ValueError(f"Shard {shard_path} has too few sequences ({num_sequences}) for K={K}")

        # Extract sequences
        sequences = []
        for i in range(num_sequences):
            start = i * T
            end = start + T
            seq = []
            for region in REGIONS:
                channel_seq = loaded[region][start:end]
                if channel_seq.size(0) < T:
                    padding = torch.full((T - channel_seq.size(0),), pad_token, dtype=channel_seq.dtype)
                    channel_seq = torch.cat((channel_seq, padding), dim=0)
                seq.append(channel_seq.unsqueeze(0))  # [1, T]
            seq = torch.cat(seq, dim=0)  # [C, T]
            sequences.append((seq, label))
        all_sequences.append(sequences)

    # Step 2: Truncate to the minimum number of sequences
    support_data = []
    query_data = []
    for sequences in all_sequences:
        sequences = sequences[:min_num_sequences]  # Truncate to smallest class size
        random.shuffle(sequences)
        support_data.extend(sequences[:K])  # K support samples
        query_data.extend(sequences[K:])  # Remaining as query

    # Verify balance
    query_counts = {}
    for _, label in query_data:
        query_counts[label] = query_counts.get(label, 0) + 1
    print(f"Query set sizes per class: {query_counts}")
    assert len(set(query_counts.values())) == 1, "Query set is not balanced across classes"

    return support_data, query_data


def compute_embeddings(model, sequences, device):
    model.eval()
    with torch.no_grad():
        sequences = sequences.to(device)
        B, C, T = sequences.size()

        tok_emb = model.transformer.wte(sequences)
        x = tok_emb.transpose(1, 2)
        pos = torch.arange(T, device=device).unsqueeze(0)
        pos_emb = model.transformer.wpe(pos)
        x = x + pos_emb.unsqueeze(2)

        channel_outs = []
        for c in range(C):
            x_c = x[:, :, c, :]
            x_c = model.intra_channel_encoder(x_c)
            channel_outs.append(x_c)
        x = torch.stack(channel_outs, dim=2)

        for block in model.transformer.h:
            x = block(x)

        x = x.transpose(1, 2).reshape(B * C, T, model.config.n_embd)
        x = model.transformer.ln_f(x)

        last_tokens = x[:, -1, :]
        last_tokens = last_tokens.view(B, C, model.config.n_embd)
        embeddings = last_tokens.mean(dim=1)

        return embeddings.cpu()


def evaluate_fewshot(model, support_data, query_data, device):
    prototypes = {}
    for label in set(d[1] for d in support_data):
        class_sequences = [d[0] for d in support_data if d[1] == label]
        class_sequences = torch.stack(class_sequences, dim=0)
        embeddings = compute_embeddings(model, class_sequences, device)
        prototypes[label] = embeddings.mean(dim=0)

    query_sequences = torch.stack([d[0] for d in query_data], dim=0)
    query_labels = [d[1] for d in query_data]
    embeddings = compute_embeddings(model, query_sequences, device)

    correct = 0
    for emb, true_label in zip(embeddings, query_labels):
        distances = {label: torch.norm(emb - proto) for label, proto in prototypes.items()}
        pred_label = min(distances, key=distances.get)
        if pred_label == true_label:
            correct += 1
    accuracy = correct / len(query_data)
    print(f"Accuracy: {accuracy:.4f} (Total query samples: {len(query_data)})")


# Main execution
shard_paths = [
    "./local_shards_val/mydata_train_0.pt",
    "./local_shards_val/mydata_train_1.pt",
    "./local_shards_val/mydata_train_2.pt"
]
support_data, query_data = load_fewshot_data(shard_paths, T=1024, K=5)

# Step 1: Random GPT
print("Step 1: Evaluating with random weights")
config = GPTConfig()
gpt_model_random = GPT(config).to(device)
evaluate_fewshot(gpt_model_random, support_data, query_data, device)

# Step 2: Pretrained GPT
print("\nStep 2: Loading pretrained weights and evaluating")
gpt_model_pretrained = GPT(config).to(device)
checkpoint = torch.load("checkpoints/model_03000.pt", map_location=device)  # Adjust path
state_dict = checkpoint['model_state_dict']
new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
gpt_model_pretrained.load_state_dict(new_state_dict)
gpt_model_pretrained.eval()
evaluate_fewshot(gpt_model_pretrained, support_data, query_data, device)