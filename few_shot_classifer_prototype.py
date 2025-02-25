import inspect
from dataclasses import dataclass

import torch
import os
import glob
import random
import numpy as np
from torch.distributed import init_process_group
from torch.nn import functional as F

# Define regions globally for consistency
REGIONS = ["frontal", "motor_temporal", "parietal_occipital"]

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
if torch.cuda.is_available():
    torch.cuda.manual_seed(9259)


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

def compute_prototypes(model, support_data, device, batch_size=8):
    """
    Compute class prototypes from support set with improved batching and representation.

    Args:
        model: The model to extract embeddings
        support_data: List of (sequence, label) tuples
        device: Device to run computation on
        batch_size: Number of samples to process at once

    Returns:
        Dictionary mapping class labels to prototype vectors
    """
    model.eval()
    label_to_embeddings = {}

    # Process in batches to save memory
    for i in range(0, len(support_data), batch_size):
        batch = support_data[i:i + batch_size]
        sequences = torch.stack([seq for seq, _ in batch], dim=0).to(device)
        batch_labels = [label for _, label in batch]

        with torch.no_grad():
            # Get embeddings using the consistent method
            embeddings = extract_embeddings(model, sequences)

            # Store embeddings by class
            for emb, label in zip(embeddings, batch_labels):
                if label not in label_to_embeddings:
                    label_to_embeddings[label] = []
                label_to_embeddings[label].append(emb.cpu())  # Store on CPU to save GPU memory

    # Compute prototypes for each class
    prototypes = {}
    for label, embs in label_to_embeddings.items():
        embs = torch.stack(embs, dim=0).to(device)
        # Apply L2 normalization before averaging for better prototype quality
        normalized_embs = F.normalize(embs, p=2, dim=1)
        prototype = normalized_embs.mean(dim=0)
        # Re-normalize the prototype
        prototype = F.normalize(prototype, p=2, dim=0)
        prototypes[label] = prototype

    return prototypes


def extract_embeddings(model, sequences):
    """
    Extract embeddings from sequences using the same processing pipeline as training.

    Args:
        model: The model to extract embeddings
        sequences: Tensor of shape [B, C, T] where B is batch size, C is channels, T is sequence length

    Returns:
        Tensor of embeddings with shape [B, E] where E is embedding dimension
    """
    B, C, T = sequences.size()
    config = model.config

    # Follow the same processing as in forward()
    tok_emb = model.transformer.wte(sequences)  # [B, C, T, n_embd]
    x = tok_emb.transpose(1, 2)  # [B, T, C, n_embd]

    # Use the same batched operation as in forward()
    x_reshaped = x.permute(0, 2, 1, 3).contiguous().reshape(B * C, T, config.n_embd)
    out = model.intra_channel_encoder(x_reshaped)
    x = out.view(B, C, T, config.n_embd).permute(0, 2, 1, 3).contiguous()

    # Process through transformer blocks
    for block in model.transformer.h:
        x = block(x)

    # Use a more comprehensive representation than just the last token
    # Get the last 4 tokens and average them for a better representation
    last_n_tokens = x[:, -4:, :, :].mean(dim=1)  # [B, C, n_embd]

    # Take mean across channels
    embedding = last_n_tokens.mean(dim=1)  # [B, n_embd]

    # Apply layer norm for consistency with forward pass
    embedding = model.transformer.ln_f(embedding)

    return embedding


def compute_cosine_similarities(query_embeddings, prototypes):
    """
    Compute cosine similarities between query embeddings and prototypes.

    Args:
        query_embeddings: Tensor of shape [B, E]
        prototypes: Dictionary mapping class labels to prototype vectors

    Returns:
        Dictionary mapping each query index to a dictionary of {label: similarity}
    """
    similarities = {}
    for i, query_emb in enumerate(query_embeddings):
        query_emb_normalized = F.normalize(query_emb, p=2, dim=0)
        similarities[i] = {}
        for label, prototype in prototypes.items():
            # Compute cosine similarity (dot product of normalized vectors)
            similarity = torch.dot(query_emb_normalized, prototype).item()
            similarities[i][label] = similarity
    return similarities


def evaluate_fewshot(model, support_data, query_data, device, batch_size=8, return_predictions=False):
    """
    Evaluate few-shot classification performance.

    Args:
        model: The model to use for evaluation
        support_data: List of (sequence, label) tuples for support set
        query_data: List of (sequence, label) tuples for query set
        device: Device to run computation on
        batch_size: Number of samples to process at once
        return_predictions: Whether to return predicted labels

    Returns:
        Dictionary of evaluation metrics including accuracy, and optionally predictions
    """
    model.eval()

    # Compute prototypes from support set
    prototypes = compute_prototypes(model, support_data, device, batch_size)

    # Process query data in batches
    all_similarities = {}
    all_true_labels = []

    for i in range(0, len(query_data), batch_size):
        batch = query_data[i:i + batch_size]
        sequences = torch.stack([seq for seq, _ in batch], dim=0).to(device)
        batch_labels = [label for _, label in batch]
        all_true_labels.extend(batch_labels)

        with torch.no_grad():
            # Extract embeddings
            embeddings = extract_embeddings(model, sequences)

            # Compute similarities
            batch_similarities = compute_cosine_similarities(embeddings, prototypes)

            # Add to all similarities with adjusted indices
            for j, sims in batch_similarities.items():
                all_similarities[i + j] = sims

    # Predict labels and compute metrics
    predictions = []
    correct = 0

    for i, true_label in enumerate(all_true_labels):
        sims = all_similarities[i]
        pred_label = max(sims, key=sims.get)
        predictions.append(pred_label)
        if pred_label == true_label:
            correct += 1

    total = len(all_true_labels)
    accuracy = correct / total if total > 0 else 0

    # Compute per-class metrics
    class_metrics = {}
    for label in set(all_true_labels):
        class_correct = sum(1 for p, t in zip(predictions, all_true_labels)
                            if p == t and t == label)
        class_total = sum(1 for t in all_true_labels if t == label)
        class_accuracy = class_correct / class_total if class_total > 0 else 0
        class_metrics[label] = {
            'accuracy': class_accuracy,
            'support': class_total
        }

    # Print results
    print(f"Overall Accuracy: {accuracy:.4f} (Total query samples: {total})")
    print("Per-class Accuracy:")
    for label, metrics in class_metrics.items():
        print(f"  Class {label}: {metrics['accuracy']:.4f} (Support: {metrics['support']})")

    # Return results
    results = {
        'accuracy': accuracy,
        'class_metrics': class_metrics,
    }

    if return_predictions:
        results['predictions'] = predictions
        results['true_labels'] = all_true_labels

    return results


def load_fewshot_data(shard_paths, T=1024, K=3, pad_token=0, num_channels=len(REGIONS),
                      balance_classes=True, max_samples_per_class=None, seed=42):
    """
    Load few-shot data with improved handling of class imbalance.

    Args:
        shard_paths: List of paths to shard files, one per class
        T: Sequence length
        K: Number of support samples per class
        pad_token: Token used for padding
        num_channels: Number of channels
        balance_classes: Whether to balance classes in query set
        max_samples_per_class: Maximum samples to use per class (None = use all)
        seed: Random seed for reproducibility

    Returns:
        support_data: List of (sequence, label) tuples for support set
        query_data: List of (sequence, label) tuples for query set
    """
    random.seed(seed)
    np.random.seed(seed)

    if not shard_paths:
        raise ValueError("No shard paths provided.")

    all_sequences = []
    min_num_sequences = float('inf')

    # Load data from shards
    for label, shard_path in enumerate(shard_paths):
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Shard file not found: {shard_path}")

        try:
            loaded = torch.load(shard_path, map_location="cpu", weights_only=False)

            # Handle missing regions
            for region in REGIONS:
                if region not in loaded:
                    available_regions = list(loaded.keys())
                    if available_regions:
                        print(f"Warning: Shard {shard_path} missing {region}, using {available_regions[0]}.")
                        loaded[region] = loaded[available_regions[0]]
                    else:
                        raise ValueError(f"Shard {shard_path} has no channels.")

            # Ensure all channels have the same length
            max_length = max(loaded[region].size(0) for region in REGIONS)
            for region in REGIONS:
                current_length = loaded[region].size(0)
                if current_length < max_length:
                    padding = torch.full((max_length - current_length,), pad_token,
                                         dtype=loaded[region].dtype)
                    loaded[region] = torch.cat((loaded[region], padding), dim=0)
                elif current_length > max_length:
                    loaded[region] = loaded[region][:max_length]

            # Extract non-overlapping sequences
            min_length = min(loaded[region].size(0) for region in REGIONS)
            num_sequences = (min_length - T) // T + 1
            min_num_sequences = min(min_num_sequences, num_sequences)

            if num_sequences < K:
                raise ValueError(f"Shard {shard_path} has too few sequences ({num_sequences}) for K={K}")

            sequences = []
            for i in range(num_sequences):
                start = i * T
                end = start + T
                seq = []
                for region in REGIONS:
                    channel_seq = loaded[region][start:end]
                    if channel_seq.size(0) < T:
                        padding = torch.full((T - channel_seq.size(0),), pad_token,
                                             dtype=channel_seq.dtype)
                        channel_seq = torch.cat((channel_seq, padding), dim=0)
                    seq.append(channel_seq.unsqueeze(0))
                seq = torch.cat(seq, dim=0)  # [C, T]
                sequences.append((seq, label))

            all_sequences.append(sequences)

        except Exception as e:
            print(f"Error loading shard {shard_path}: {e}")
            raise

    # Split into support and query sets
    support_data = []
    query_data = []

    for sequences in all_sequences:
        # Apply max_samples_per_class if specified
        if max_samples_per_class and len(sequences) > max_samples_per_class:
            sequences = random.sample(sequences, max_samples_per_class)

        random.shuffle(sequences)
        support_data.extend(sequences[:K])
        query_data.extend(sequences[K:])

    # Balance query set if requested
    if balance_classes:
        query_by_class = {}
        for seq, label in query_data:
            if label not in query_by_class:
                query_by_class[label] = []
            query_by_class[label].append((seq, label))

        # Find minimum size for balancing
        min_class_size = min(len(samples) for samples in query_by_class.values())

        # Balance classes
        balanced_query_data = []
        for label, samples in query_by_class.items():
            balanced_query_data.extend(random.sample(samples, min_class_size))

        query_data = balanced_query_data

    # Shuffle query data
    random.shuffle(query_data)

    # Verify balance
    query_counts = {}
    for _, label in query_data:
        query_counts[label] = query_counts.get(label, 0) + 1

    print(f"Support set: {len(support_data)} samples")
    print(f"Query set: {len(query_data)} samples")
    print(f"Class distribution in query set: {query_counts}")

    return support_data, query_data


def extract_embeddings(model, sequences):
    """
    Extract embeddings using continuous segments of tokens for better context representation.

    Args:
        model: The model to extract embeddings
        sequences: Tensor of shape [B, C, T] where B is batch size, C is channels, T is sequence length

    Returns:
        Tensor of embeddings with shape [B, E] where E is embedding dimension
    """
    B, C, T = sequences.size()
    config = model.config

    # Process through the embedding layer and encoder
    tok_emb = model.transformer.wte(sequences)  # [B, C, T, n_embd]
    x = tok_emb.transpose(1, 2)  # [B, T, C, n_embd]

    # Use the same batched operation as in forward()
    x_reshaped = x.permute(0, 2, 1, 3).contiguous().reshape(B * C, T, config.n_embd)
    out = model.intra_channel_encoder(x_reshaped)
    x = out.view(B, C, T, config.n_embd).permute(0, 2, 1, 3).contiguous()

    # Process through transformer blocks
    for block in model.transformer.h:
        x = block(x)

    # Apply the final layer norm to maintain consistency with forward pass
    x_reshaped = x.transpose(1, 2).contiguous().reshape(B * C, T, config.n_embd)
    x_norm = model.transformer.ln_f(x_reshaped)
    x = x_norm.view(B, C, T, config.n_embd)

    # Use continuous segments of tokens for context
    # Divide the sequence into 4 segments and extract embedding from each
    segment_length = T // 4
    segment_embeddings = []

    for i in range(4):
        start_idx = i * segment_length
        end_idx = start_idx + segment_length

        # Get embedding for this segment (average across tokens in segment)
        segment_emb = x[:, start_idx:end_idx, :, :].mean(dim=1)  # [B, C, E]
        segment_emb = segment_emb.mean(dim=1)  # [B, E] - average across channels
        segment_embeddings.append(segment_emb)

    # Concatenate segment embeddings to capture the full temporal context
    full_embedding = torch.cat(segment_embeddings, dim=1)  # [B, 4*E]

    # Project back to original embedding dimension to keep consistent size
    if not hasattr(model, 'context_projection'):
        model.context_projection = torch.nn.Linear(
            4 * config.n_embd, config.n_embd,
            device=full_embedding.device
        )
        # Initialize weights to average the segments
        torch.nn.init.constant_(model.context_projection.weight, 0.25)
        torch.nn.init.zeros_(model.context_projection.bias)

    final_embedding = model.context_projection(full_embedding)

    return final_embedding


def evaluate_fewshot_with_augmentation(model, support_data, query_data, device, batch_size=8):
    """
    Evaluate few-shot classification with temporal augmentation for more robust results.

    Args:
        model: The model to use for evaluation
        support_data: List of (sequence, label) tuples for support set
        query_data: List of (sequence, label) tuples for query set
        device: Device to run computation on
        batch_size: Number of samples to process at once

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    # 1. Create multiple augmented views of the support set through minor time shifts
    augmented_support_data = []
    for seq, label in support_data:
        B, C, T = 1, seq.size(0), seq.size(1)

        # Original sample
        augmented_support_data.append((seq, label))

        # Add time-shifted versions (shifts of 5% of sequence length)
        shift_amount = max(1, int(0.05 * T))
        for shift in [shift_amount, 2 * shift_amount]:
            # Shift right (take from beginning)
            shifted_seq = torch.roll(seq.clone(), shifts=shift, dims=1)
            augmented_support_data.append((shifted_seq, label))

            # Shift left (take from end)
            shifted_seq = torch.roll(seq.clone(), shifts=-shift, dims=1)
            augmented_support_data.append((shifted_seq, label))

    # 2. Compute prototypes from augmented support set
    prototypes = compute_prototypes(model, augmented_support_data, device, batch_size)

    # 3. Process query data with multiple views through time offsets
    all_true_labels = []
    all_pred_labels = []

    for i in range(0, len(query_data), batch_size):
        batch = query_data[i:i + batch_size]
        batch_sequences = [seq for seq, _ in batch]
        batch_labels = [label for _, label in batch]
        all_true_labels.extend(batch_labels)

        # For each sequence, create multiple temporal views
        batch_predictions = []

        for j, orig_seq in enumerate(batch_sequences):
            views = [orig_seq]  # Start with original sequence

            # Add time-shifted versions for robustness
            T = orig_seq.size(1)
            shift_amount = max(1, int(0.05 * T))

            # Add right-shifted version
            shifted_seq = torch.roll(orig_seq.clone(), shifts=shift_amount, dims=1)
            views.append(shifted_seq)

            # Add left-shifted version
            shifted_seq = torch.roll(orig_seq.clone(), shifts=-shift_amount, dims=1)
            views.append(shifted_seq)

            # Process all views to get embeddings
            view_embeddings = []
            for view in views:
                view_tensor = view.unsqueeze(0).to(device)  # [1, C, T]
                with torch.no_grad():
                    embedding = extract_embeddings(model, view_tensor)
                    view_embeddings.append(embedding)

            # Combine view predictions with voting
            view_predictions = []
            for embedding in view_embeddings:
                # Compute similarity to each prototype
                similarities = {}
                for label, prototype in prototypes.items():
                    # Normalize for cosine similarity
                    embedding_norm = F.normalize(embedding, p=2, dim=1)
                    similarity = torch.matmul(embedding_norm, prototype.unsqueeze(1)).item()
                    similarities[label] = similarity

                # Get prediction from this view
                pred = max(similarities, key=similarities.get)
                view_predictions.append(pred)

            # Take majority vote across views
            from collections import Counter
            votes = Counter(view_predictions)
            final_pred = votes.most_common(1)[0][0]
            batch_predictions.append(final_pred)

        all_pred_labels.extend(batch_predictions)

    # 4. Compute metrics
    correct = sum(1 for p, t in zip(all_pred_labels, all_true_labels) if p == t)
    total = len(all_true_labels)
    accuracy = correct / total if total > 0 else 0

    # Compute per-class metrics
    class_metrics = {}
    for label in set(all_true_labels):
        class_correct = sum(1 for p, t in zip(all_pred_labels, all_true_labels)
                            if p == t and t == label)
        class_total = sum(1 for t in all_true_labels if t == label)
        class_accuracy = class_correct / class_total if class_total > 0 else 0
        class_metrics[label] = {
            'accuracy': class_accuracy,
            'support': class_total
        }

    # Print results
    print(f"Overall Accuracy: {accuracy:.4f} (Total query samples: {total})")
    print("Per-class Accuracy:")
    for label, metrics in class_metrics.items():
        print(f"  Class {label}: {metrics['accuracy']:.4f} (Support: {metrics['support']})")

    # Return results
    results = {
        'accuracy': accuracy,
        'class_metrics': class_metrics,
        'predictions': all_pred_labels,
        'true_labels': all_true_labels
    }

    return results


# Usage example
if __name__ == "__main__":
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model configuration

    config = GPTConfig()
    model = GPT(config).to(device)

    # Example shard paths (replace with actual paths)
    shard_paths = [
        "./local_shards_val/mydata_train_2.pt",
        "./local_shards_val/mydata_train_0.pt",
    ]

    # Load data
    support_data, query_data = load_fewshot_data(
        shard_paths,
        T=config.block_size,
        K=10,
        pad_token=config.pad_token,
        num_channels=config.num_channels,
        balance_classes=True
    )

    # Evaluate with random weights
    print("Evaluating with random weights and continuous context extraction")
    random_results = evaluate_fewshot_with_augmentation(
        model, support_data, query_data, device, batch_size=16
    )

    # Optionally, load pretrained weights and evaluate
    try:
        checkpoint = torch.load(
            "checkpoints/model_last_checkpoint.pt",
            map_location=device,
            weights_only=False
        )
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state_dict.items()})
        model.eval()

        print("\nEvaluating with pretrained weights and continuous context extraction")
        pretrained_results = evaluate_fewshot_with_augmentation(
            model, support_data, query_data, device, batch_size=16
        )

    except FileNotFoundError:
        print("\nPretrained weights not found; skipping pretrained evaluation.")
