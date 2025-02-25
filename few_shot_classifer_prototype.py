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


#########################
# DataLoader (All-In-Memory)
#########################
# Ensure these match the channels defined during preprocessing.
REGIONS = ["frontal", "motor_temporal", "parietal_occipital"]

class DataLoaderLiteAllInMemory:
    def __init__(self, B: int, T: int, process_rank: int, num_processes: int,
                 local_data_dir: str = "./local_shards", shard_prefix: str = "mydata",
                 split: str = "train", shuffle_shards: bool = False, pad_token: int = 0):
        self.B = B
        self.per_channel_length = T  # Sequence length per channel
        self.num_channels = len(REGIONS)
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.pad_token = pad_token  # Token to use for padding

        # Locate shard files
        pattern = os.path.join(local_data_dir, f"{shard_prefix}_{split}_*.pt")
        self.shard_files = sorted(glob.glob(pattern))
        if not self.shard_files:
            raise ValueError(f"No {split} shards found in {local_data_dir} with prefix {shard_prefix}_{split}_")
        if shuffle_shards:
            random.shuffle(self.shard_files)

        # Load and concatenate tokens
        self.tokens = {region: [] for region in REGIONS}
        for shard_path in self.shard_files:
            loaded = torch.load(shard_path, map_location="cpu", weights_only=False)
            for region in REGIONS:
                if region not in loaded:
                    available_regions = list(loaded.keys())
                    if available_regions:
                        loaded[region] = loaded[available_regions[0]]
                        print(f"Warning: Shard {shard_path} missing {region}, using {available_regions[0]}.")
                    else:
                        raise ValueError(f"Shard {shard_path} has no channels for {region}.")
                self.tokens[region].append(loaded[region])
        for region in REGIONS:
            self.tokens[region] = torch.cat(self.tokens[region], dim=0)

        # Check minimum length
        min_length = min(t.size(0) for t in self.tokens.values())
        required_length = self.B * self.per_channel_length * self.num_processes
        if min_length < required_length:
            print(f"Warning: Shortest channel has {min_length} tokens, less than required {required_length}. Padding will be used.")

        self.start_ptr = self.B * self.per_channel_length * self.process_rank
        self.ptr = self.start_ptr

    def _get_slice(self, token_tensor: torch.Tensor, start: int, length: int, pad_value: int) -> torch.Tensor:
        total_length = token_tensor.size(0)
        start = start % total_length  # Wrap around if exceeding length
        end = start + length
        if end <= total_length:
            return token_tensor[start:end]
        else:
            first_part = token_tensor[start:]
            remaining = length - first_part.size(0)
            second_part = token_tensor[:remaining]
            return torch.cat((first_part, second_part), dim=0)
    def next_batch(self):
        inputs_list = []
        targets_list = []

        for region in REGIONS:
            token_tensor = self.tokens[region]
            channel_inputs = []
            channel_targets = []
            for b in range(self.B):
                start = self.ptr + b * self.per_channel_length
                # Input: T tokens, pad with pad_token
                seq = self._get_slice(token_tensor, start, self.per_channel_length, pad_value=self.pad_token)  # [T]
                channel_inputs.append(seq.unsqueeze(0))  # [1, T]
                # Target: Next T tokens, pad with -100
                target = self._get_slice(token_tensor, start + 1, self.per_channel_length, pad_value=-100)  # [T]
                channel_targets.append(target.unsqueeze(0))  # [1, T]
            channel_inputs = torch.cat(channel_inputs, dim=0)  # [B, T]
            channel_targets = torch.cat(channel_targets, dim=0)  # [B, T]
            inputs_list.append(channel_inputs.unsqueeze(1))  # [B, 1, T]
            targets_list.append(channel_targets.unsqueeze(1))  # [B, 1, T]

        inputs = torch.cat(inputs_list, dim=1)  # [B, num_channels, T]
        targets = torch.cat(targets_list, dim=1)  # [B, num_channels, T]

        self.ptr += self.B * self.per_channel_length * self.num_processes
        return inputs, targets

    def reset(self):
        self.ptr = self.start_ptr

    @property
    def total_len(self):
        return self.tokens[REGIONS[0]].size(0)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch()

# --- Hyperparameters & Configurations ---
# We use 512 tokens for prompt and 512 tokens for completion.
PROMPT_LEN = 512
COMP_LEN = 512
# For the in-context (few-shot) classification, we prepend the candidate’s support pair.
# Total sequence length = support (1024) + query prompt (512) + query completion (512) = 2048.
SEQ_LEN = PROMPT_LEN + COMP_LEN + PROMPT_LEN + COMP_LEN

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Utility Functions ---
def sample_support_and_query(token_tensor, prompt_len=PROMPT_LEN, comp_len=COMP_LEN):
    """
    Given a 1D tensor of tokens (from a shard corresponding to one class),
    randomly sample two nonoverlapping contiguous segments:
      - A support example: first prompt_len tokens as prompt and next comp_len tokens as support completion.
      - A query example: same split from the next contiguous block.
    """
    total_needed = 2 * (prompt_len + comp_len)
    total_tokens = token_tensor.size(0)
    if total_tokens < total_needed:
        raise ValueError("Not enough tokens to sample support and query examples.")
    # Randomly choose a starting index such that the support and query segments fit
    start = random.randint(0, total_tokens - total_needed)
    support_seq = token_tensor[start : start + prompt_len + comp_len]
    query_seq = token_tensor[start + prompt_len + comp_len : start + total_needed]
    support_prompt = support_seq[:prompt_len]
    support_completion = support_seq[prompt_len:]
    query_prompt = query_seq[:prompt_len]
    query_completion = query_seq[prompt_len:]
    return support_prompt, support_completion, query_prompt, query_completion

def compute_loss_for_class(model, support_prompt, support_completion, query_prompt, query_completion):
    """
    For a given candidate class (with its support pair), construct the input sequence:
      [support_prompt, support_completion, query_prompt, query_completion]
    Set targets to -100 for all tokens except for the query completion,
    and then compute the LM loss (averaged over the query tokens).
    """
    # Concatenate into one long sequence
    # Order: support prompt (512) + support completion (512) + query prompt (512) + query completion (512)
    input_seq = torch.cat([support_prompt, support_completion, query_prompt, query_completion], dim=0)
    # Create targets: ignore support and query prompt tokens (set to -100), then use query_completion tokens.
    num_ignore = support_prompt.size(0) + support_completion.size(0) + query_prompt.size(0)
    ignore_tokens = torch.full((num_ignore,), -100, dtype=torch.long)
    targets = torch.cat([ignore_tokens, query_completion], dim=0)
    # Reshape as required by the model: [B, num_channels, T]. Here B = 1 and num_channels = 1.
    input_seq = input_seq.unsqueeze(0).unsqueeze(0).to(device)   # shape: [1, 1, SEQ_LEN]
    targets = targets.unsqueeze(0).unsqueeze(0).to(device)         # shape: [1, 1, SEQ_LEN]
    with torch.no_grad():
        _, loss = model(input_seq, targets=targets)
    return loss.item()

def evaluate_few_shot(model, class_tokens, num_trials=10):
    """
    For each trial, for every class (true label), sample a query example and then
    compute the loss (i.e. negative log-likelihood) of generating its query completion
    when conditioned on each candidate class’s support pair.
    The predicted class is the one with the minimum loss.
    Returns the overall accuracy.
    """
    correct = 0
    total = 0
    # Loop for several trials for stability
    for _ in range(num_trials):
        for true_class, token_tensor in class_tokens.items():
            # Sample support and query for the true class (for the query example)
            sp_true, sc_true, query_prompt, query_completion = sample_support_and_query(token_tensor)
            losses = []
            # For every candidate class, sample its own support pair (as the prototype)
            for candidate_class, candidate_tokens in class_tokens.items():
                sp_candidate, sc_candidate, _, _ = sample_support_and_query(candidate_tokens)
                loss = compute_loss_for_class(model, sp_candidate, sc_candidate, query_prompt, query_completion)
                losses.append(loss)
            # The candidate with the smallest loss is predicted.
            predicted_class = min(range(len(losses)), key=lambda i: losses[i])
            if predicted_class == true_class:
                correct += 1
            total += 1
    return correct / total

# --- Load Class Data ---
# Here each shard file corresponds to one class. Adjust shard_paths as needed.
shard_paths = [
    "./local_shards_val/mydata_train_2.pt",
    "./local_shards_val/mydata_train_0.pt",
    # Add more paths if available...
]

# For simplicity, we assume each shard contains a dict with one or more channels;
# we use the "frontal" channel if available, else the first key.
class_tokens = {}
for i, shard_path in enumerate(shard_paths):
    if not os.path.exists(shard_path):
        raise FileNotFoundError(f"Shard file {shard_path} not found.")
    data = torch.load(shard_path, map_location="cpu")
    if "frontal" in data:
        tokens = data["frontal"]
    else:
        tokens = next(iter(data.values()))
    class_tokens[i] = tokens

# --- Initialize the Model ---
# We update block_size to 2048 (to accommodate support+query) and set num_channels to 1.


config = GPTConfig()
model = GPT(config).to(device)
model.eval()  # Set to evaluation mode

# --- Evaluate with the Raw (Randomly Initialized) Model ---
print("Evaluating raw model (random initialization)...")
acc_raw = evaluate_few_shot(model, class_tokens, num_trials=10)
print(f"Raw model few-shot accuracy: {acc_raw * 100:.2f}%")

# --- Load Pretrained Weights & Evaluate Again ---
checkpoint_path = "checkpoints/model_last_checkpoint.pt"
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found.")

try:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )
    state_dict = checkpoint['model_state_dict']
    # Fix keys if needed (e.g. remove DDP prefix)
    fixed_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(fixed_state_dict)
    model.eval()
except Exception as e:
    print("Error loading pretrained weights:", e)

print("Evaluating pretrained model...")
acc_pretrained = evaluate_few_shot(model, class_tokens, num_trials=10)
print(f"Pretrained model few-shot accuracy: {acc_pretrained * 100:.2f}%")
