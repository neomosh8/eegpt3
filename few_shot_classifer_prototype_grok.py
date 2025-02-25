import torch
import torch.nn.functional as F
import random
import os
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
from lr_test import CustomLRScheduler
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

#########################
# Model Components
#########################
import torch
import torch.nn as nn


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
        x_reshaped = x.contiguous().view(B * C, T, E)
        x_reshaped = x_reshaped + self.attn(self.ln_1(x_reshaped))
        x = x_reshaped.view(B, T, C, E)
        x = x + self.fusion(self.ln_2(x))
        x_reshaped = x.contiguous().view(B * C, T, E)
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
        seq_len = config.block_size
        head_dim = config.n_embd // config.n_head
        freqs = torch.arange(0, head_dim, 2).float() / head_dim
        theta = 1000.0 ** (-freqs)
        positions = torch.arange(seq_len).float()
        angles = positions[:, None] * theta[None, :]
        self.register_buffer('cos', angles.cos())
        self.register_buffer('sin', angles.sin())

    def apply_rotary_emb(self, x, T):
        # x: [B, n_head, T, head_dim]
        B, n_head, T, head_dim = x.shape
        # Split into pairs for rotation
        x1, x2 = x[..., 0::2], x[..., 1::2]  # Each: [B, n_head, T, head_dim//2]
        # Use precomputed cos and sin, sliced to T if necessary
        cos = self.cos[:T, :]  # [T, head_dim//2]
        sin = self.sin[:T, :]  # [T, head_dim//2]
        # Broadcast to match x1 and x2 shapes
        cos = cos[None, None, :, :]  # [1, 1, T, head_dim//2]
        sin = sin[None, None, :, :]  # [1, 1, T, head_dim//2]
        # Apply rotary transformation
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

    # n_layer: int = 6
    # n_head: int = 6
    # n_embd: int = 384

    n_layer: int = 16  # Moderate depth
    n_head: int = 32  # Fewer heads but still enough for good attention
    n_embd: int = 2048  # Smaller embedding dimension
    num_channels: int = 3
    mlp_dropout: float = 0.05
    attn_dropout: float = 0.02
    resid_dropout: float = 0.05
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

        # Batched operation
        x_reshaped = x.permute(0, 2, 1, 3).contiguous().reshape(B * C, T, self.config.n_embd)  # [B*C, T, n_embd]
        out = self.intra_channel_encoder(x_reshaped)  # [B*C, T, n_embd]
        x = out.view(B, C, T, self.config.n_embd).permute(0, 2, 1, 3).contiguous()  # [B, T, C, n_embd]

        for block in self.transformer.h:
            x = block(x)  # [B, T, C, n_embd]

        x = x.transpose(1, 2).contiguous().reshape(B * C, T, self.config.n_embd)  # [B * C, T, n_embd]
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

    def get_representation(self, idx):
        """
        Extracts a representation for each input example by averaging the last token's
        hidden states across all channels.

        Args:
            idx (torch.Tensor): Input tensor of shape [B, num_channels, T]

        Returns:
            torch.Tensor: Representations of shape [B, n_embd]
        """
        self.eval()  # Ensure the model is in evaluation mode
        B, C, T = idx.size()
        tok_emb = self.transformer.wte(idx)  # [B, C, T, n_embd]
        x = tok_emb.transpose(1, 2)  # [B, T, C, n_embd]

        # Reshape for intra-channel encoder
        x_reshaped = x.permute(0, 2, 1, 3).contiguous().reshape(B * C, T, self.config.n_embd)
        out = self.intra_channel_encoder(x_reshaped)  # [B*C, T, n_embd]
        x = out.view(B, C, T, self.config.n_embd).permute(0, 2, 1, 3).contiguous()  # [B, T, C, n_embd]

        # Process through transformer blocks
        for block in self.transformer.h:
            x = block(x)  # [B, T, C, n_embd]

        # Reshape and apply final layer norm
        x = x.transpose(1, 2).contiguous().reshape(B * C, T, self.config.n_embd)  # [B * C, T, n_embd]
        x = self.transformer.ln_f(x)  # [B * C, T, n_embd]
        x = x.view(B, C, T, self.config.n_embd)

        # Take the last token's hidden state for each channel and average across channels
        representation = x[:, :, -1, :].mean(dim=1)  # [B, n_embd]
        # Normalize the representation to have unit norm (L2 normalization)
        representation = F.normalize(representation, p=2, dim=1)
        return representation
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


def matching_network_evaluation(model, shard_paths, K, Q, T, device):
    """
    Performs few-shot classification using the matching network approach.

    Args:
        model (GPT): The model to use for extracting representations.
        shard_paths (list): List of shard file paths (each corresponds to a class).
        K (int): Number of support examples per class.
        Q (int): Number of query examples per class.
        T (int): Sequence length per channel (e.g., 512).
        device (str): Device to use (e.g., 'cuda' or 'cpu').

    Returns:
        float: Classification accuracy.
    """
    model.eval()
    with torch.no_grad():
        # Load data for each class
        class_data = load_class_data(shard_paths, T)
        N = len(class_data)  # Number of classes

        # Sample support and query examples per class
        support_examples_per_class = []
        query_examples_per_class = []
        for class_idx, examples in enumerate(class_data):
            if len(examples) < K + Q:
                print(f"Warning: Class {class_idx} has only {len(examples)} examples, skipping.")
                continue
            # Shuffle examples
            indices = torch.randperm(len(examples))
            support_indices = indices[:K]
            query_indices = indices[K:K + Q]
            support = [examples[i] for i in support_indices]
            query = [examples[i] for i in query_indices]
            support_examples_per_class.append(support)
            query_examples_per_class.append(query)

        # Compute representations for support examples
        all_support_examples = [ex for class_support in support_examples_per_class for ex in class_support]
        all_support_inputs = torch.stack(all_support_examples, dim=0).to(device)  # [N*K, num_channels, T]
        support_reps = model.get_representation(all_support_inputs).cpu()  # [N*K, n_embd]

        # Compute representations for query examples
        all_query_examples = [ex for class_query in query_examples_per_class for ex in class_query]
        all_query_inputs = torch.stack(all_query_examples, dim=0).to(device)  # [N*Q, num_channels, T]
        query_reps = model.get_representation(all_query_inputs).cpu()  # [N*Q, n_embd]

        # Compute cosine similarities between query and support representations
        similarities = torch.mm(query_reps, support_reps.t())  # [N*Q, N*K]

        # Create class labels for support examples
        support_labels = torch.tensor([class_idx for class_idx in range(N) for _ in range(K)])  # [N*K]

        # For each query, compute the sum of similarities per class
        total_similarities = torch.zeros(N * Q, N)  # [N*Q, N]
        for class_idx in range(N):
            class_mask = (support_labels == class_idx)  # [N*K]
            total_similarities[:, class_idx] = similarities[:, class_mask].sum(dim=1)

        # Predict the class with the highest total similarity
        predictions = total_similarities.argmax(dim=1)  # [N*Q]

        # True labels
        true_labels = torch.tensor([class_idx for class_idx in range(N) for _ in range(Q)])

        # Compute accuracy
        accuracy = (predictions == true_labels).float().mean().item()
        return accuracy
#########################
# DataLoader (All-In-Memory)
#########################
# Ensure these match the channels defined during preprocessing.
REGIONS = ["frontal", "motor_temporal", "parietal_occipital"]

import torch
import os

REGIONS = ["frontal", "motor_temporal", "parietal_occipital"]


def load_class_data(shard_paths, T):
    """
    Loads and prepares data for each class from shard files.

    Args:
        shard_paths (list): List of paths to shard files, each corresponding to a class.
        T (int): Sequence length per channel (e.g., 512).

    Returns:
        list: List of lists, where each sublist contains examples for a class.
              Each example is a tensor of shape [num_channels, T].
    """
    class_data = []
    for shard_path in shard_paths:
        # Load the shard
        loaded = torch.load(shard_path, map_location="cpu", weights_only=False)
        tokens = {region: loaded[region] for region in REGIONS}

        # Find the minimum length across regions
        min_length = min(t.size(0) for t in tokens.values())

        # Number of possible prompt sequences (each of length T)
        num_examples = min_length // T
        examples = []

        # Create non-overlapping prompt sequences
        for i in range(num_examples):
            example = torch.stack(
                [tokens[region][i * T:(i + 1) * T] for region in REGIONS],
                dim=0
            )  # [num_channels, T]
            examples.append(example)
        class_data.append(examples)
    return class_data


import torch.nn.functional as F


def few_shot_evaluation(model, shard_paths, K, Q, T, device):
    """
    Performs few-shot classification using the prototype method.

    Args:
        model (GPT): The model to use for extracting representations.
        shard_paths (list): List of shard file paths (each corresponds to a class).
        K (int): Number of support examples per class.
        Q (int): Number of query examples per class.
        T (int): Sequence length per channel (e.g., 512).
        device (str): Device to use (e.g., 'cuda' or 'cpu').

    Returns:
        float: Classification accuracy.
    """
    model.eval()
    with torch.no_grad():
        # Load data for each class
        class_data = load_class_data(shard_paths, T)
        N = len(class_data)  # Number of classes

        # Sample support and query examples per class
        support_examples_per_class = []
        query_examples_per_class = []
        for class_idx, examples in enumerate(class_data):
            if len(examples) < K + Q:
                print(f"Warning: Class {class_idx} has only {len(examples)} examples, skipping.")
                continue
            # Shuffle examples
            indices = torch.randperm(len(examples))
            support_indices = indices[:K]
            query_indices = indices[K:K + Q]
            support = [examples[i] for i in support_indices]
            query = [examples[i] for i in query_indices]
            support_examples_per_class.append(support)
            query_examples_per_class.append(query)

        # Compute representations for support examples
        all_support_examples = [ex for class_support in support_examples_per_class for ex in class_support]
        all_support_inputs = torch.stack(all_support_examples, dim=0).to(device)  # [N*K, num_channels, T]
        support_reps = model.get_representation(all_support_inputs).cpu()  # [N*K, n_embd]

        # Compute representations for query examples
        all_query_examples = [ex for class_query in query_examples_per_class for ex in class_query]
        all_query_inputs = torch.stack(all_query_examples, dim=0).to(device)  # [N*Q, num_channels, T]
        query_reps = model.get_representation(all_query_inputs).cpu()  # [N*Q, n_embd]

        # Compute prototypes
        prototypes = []
        for i in range(N):
            start = i * K
            end = (i + 1) * K
            class_reps = support_reps[start:end]  # [K, n_embd]
            prototype = class_reps.mean(dim=0)  # [n_embd]
            prototypes.append(prototype)
        prototypes = torch.stack(prototypes, dim=0)  # [N, n_embd]

        # Compute distances and predict
        distances = torch.cdist(query_reps, prototypes)  # [N*Q, N]
        predictions = distances.argmin(dim=1)  # [N*Q]

        # True labels
        true_labels = torch.tensor([class_idx for class_idx in range(N) for _ in range(Q)])

        # Compute accuracy
        accuracy = (predictions == true_labels).float().mean().item()
        return accuracy
# --- Hyperparameters & Configurations ---
## Set up device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define model configuration (must match pre-trained model)
config = GPTConfig()

# Create randomly initialized model
model = GPT(config).to(device)

# Define shard paths (each corresponds to a class)
shard_paths = [
    "./local_shards_val/mydata_train_2.pt",
    "./local_shards_val/mydata_train_0.pt",
    "./local_shards_val/mydata_train_1.pt",
]

# Set few-shot parameters
K = 5  # Number of support examples per class
Q = 10 # Number of query examples per class
T = 1024  # Sequence length per channel

# Evaluate with randomly initialized model
random_accuracy = matching_network_evaluation(model, shard_paths, K, Q, T, device)
print(f"Accuracy with randomly initialized model: {random_accuracy:.4f}")

# Load pre-trained weights
try:
    checkpoint = torch.load(
        "checkpoints/model_last_checkpoint.pt",
        map_location=device,
        weights_only=False
    )
    state_dict = checkpoint['model_state_dict']
    # Fix state dict keys if needed (remove _orig_mod prefix common in DDP training)
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state_dict.items()})
    model.eval()
except Exception as e:
    print(f"Error loading pre-trained weights: {e}")
    pretrained_accuracy = 0.0
else:
    # Evaluate with pre-trained model
    pretrained_accuracy = matching_network_evaluation(model, shard_paths, K, Q, T, device)
    print(f"Accuracy with pre-trained model: {pretrained_accuracy:.4f}")