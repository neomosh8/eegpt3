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
if torch.cuda.is_available():
    torch.cuda.manual_seed(9259)

#########################
# Model Components
#########################
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
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approximate='tanh')
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

class CrossChannelFusion(nn.Module):
    def __init__(self, n_embd, num_heads=1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=n_embd, num_heads=num_heads)

    def forward(self, x):
        # x: [B, time_steps, num_channels, n_embd]
        B, T, C, E = x.size()
        x = x.view(B * T, C, E)         # [B*T, num_channels, n_embd]
        x = x.transpose(0, 1)           # [num_channels, B*T, n_embd]
        fused, _ = self.attn(x, x, x)     # [num_channels, B*T, n_embd]
        fused = fused.mean(dim=0)        # [B*T, n_embd]
        return fused.view(B, T, E)       # [B, time_steps, n_embd]

@dataclass
class GPTConfig:
    block_size: int = 1032
    vocab_size: int = 10799
    # Small model configuration
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    num_channels: int = 3
    mlp_dropout: float = 0.05
    attn_dropout: float = 0.05
    resid_dropout: float = 0.05

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "wpe": nn.Embedding(config.block_size, config.n_embd),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": nn.LayerNorm(config.n_embd)
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying between token embedding and output projection.
        self.transformer.wte.weight = self.lm_head.weight

        # Per-channel encoder: 2 blocks per channel.
        self.channel_encoder = nn.ModuleList([
            nn.Sequential(Block(config), Block(config))
            for _ in range(config.num_channels)
        ])
        self.cross_channel_fusion = CrossChannelFusion(config.n_embd, num_heads=1)

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
        B, T = idx.size()
        # Ensure that T is divisible by the number of channels.
        assert T % self.config.num_channels == 0, "T must be divisible by num_channels"
        time_steps = T // self.config.num_channels

        tok_emb = self.transformer.wte(idx)  # [B, T, n_embd]
        # Reshape tokens so each contiguous block corresponds to one channel.
        x = tok_emb.view(B, time_steps, self.config.num_channels, self.config.n_embd)
        channel_outs = []
        for c in range(self.config.num_channels):
            x_c = x[:, :, c, :]  # [B, time_steps, n_embd]
            x_c = self.channel_encoder[c](x_c)
            channel_outs.append(x_c)
        x = torch.stack(channel_outs, dim=2)  # [B, time_steps, num_channels, n_embd]
        x_fused = self.cross_channel_fusion(x)  # [B, time_steps, n_embd]
        # Replicate fused output to recover original sequence length.
        x_fused_rep = x_fused.unsqueeze(2).repeat(1, 1, self.config.num_channels, 1)
        x_flat = x_fused_rep.view(B, T, self.config.n_embd)

        pos = torch.arange(0, T, device=x_flat.device).unsqueeze(0)
        pos_emb = self.transformer.wpe(pos)
        x_flat = x_flat + pos_emb

        for block in self.transformer.h:
            x_flat = block(x_flat)
        x_flat = self.transformer.ln_f(x_flat)
        logits = self.lm_head(x_flat)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
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
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused
        )
        return optimizer

#########################
# DataLoader (All-In-Memory)
#########################
# Ensure these match the channels defined during preprocessing.
REGIONS = ["frontal", "motor_temporal", "parietal_occipital"]


class DataLoaderLiteAllInMemory:
    """
    A DataLoader that:
      - Loads all shard files (each containing a dict with keys in REGIONS) from a local directory.
      - For each shard, verifies that all channels (regions) are present and have the same length.
      - Partitions shards among DDP processes (using modulo of the shard index).

    When next_batch() is called, it extracts (non-overlapping) contiguous blocks from a shard.
    In each shard every channel is tokenized separately. For each sample the loader:
      1. From each channel, extracts T+1 tokens starting at a pointer.
      2. Forms x_region = tokens[:-1] and y_region = tokens[1:].
      3. Concatenates the blocks from all regions (in the fixed order of REGIONS) to create x and y.

    Thus, if T=1000 tokens per channel and there are 3 channels, each sample will be a vector
    of length 3000 for x and similarly for y.

    The loader uses round-robin sampling over its assigned shards. If a shard does not have enough
    tokens for the next block, its pointer is reset (i.e. we “wrap‐around”).
    """

    def __init__(
            self,
            B: int,
            T: int,
            process_rank: int,
            num_processes: int,
            local_data_dir: str = "./local_shards",
            shard_prefix: str = "mydata",
            split: str = "train",
            shuffle_shards: bool = False,
    ):
        """
        Args:
          B: Batch size (number of samples per batch)
          T: Number of tokens per channel to be used as input (each sample uses T tokens per channel,
             so the final sample will be of length [num_channels * T])
          process_rank: Current DDP process rank.
          num_processes: Total number of DDP processes.
          local_data_dir: Directory where .pt shard files are stored.
          shard_prefix: Shard file prefix.
          split: e.g. "train" or "val".
          shuffle_shards: Whether to shuffle the list of shard files.
        """
        self.B = B
        self.T = T
        self.num_channels = len(REGIONS)

        # Locate shard files matching the expected naming convention.
        pattern = os.path.join(local_data_dir, f"{shard_prefix}_{split}_*.pt")
        all_shard_files = sorted(glob.glob(pattern))
        if not all_shard_files:
            raise ValueError(f"No {split} shards found in {local_data_dir} with prefix {shard_prefix}_{split}_*")

        if shuffle_shards:
            random.shuffle(all_shard_files)

        # Load all shards into memory.
        loaded_shards = []
        for shard_path in all_shard_files:
            shard = torch.load(shard_path, map_location="cpu")
            # Ensure all expected regions are present.
            for region in REGIONS:
                if region not in shard:
                    raise ValueError(f"Shard {shard_path} is missing expected region '{region}'.")
            # Verify that all channels have the same length.
            lengths = [shard[region].size(0) for region in REGIONS]
            if not all(l == lengths[0] for l in lengths):
                raise ValueError(f"Shard {shard_path} has mismatched channel lengths: {lengths}.")
            loaded_shards.append(shard)

        # Partition shards among DDP processes.
        self.shards = [s for i, s in enumerate(loaded_shards) if i % num_processes == process_rank]
        if not self.shards:
            raise ValueError(f"No shards assigned to process {process_rank} (num_processes={num_processes}).")
        # Store the length (number of tokens) for each shard (all channels in a shard have the same length).
        self.shard_lengths = [s[REGIONS[0]].size(0) for s in self.shards]

        # For each shard, we maintain a pointer (in terms of token index per channel) for sampling.
        self.shard_ptrs = [0 for _ in self.shards]

        # Global index (over self.shards) for round-robin sampling.
        self.shard_index = 0

        # Precompute a “channel vector” of length (num_channels * T). This tells you, for each token
        # in the concatenated sample, from which channel it came (here we simply use the region index).
        channel_ids = []
        for i in range(self.num_channels):
            channel_ids.extend([i] * T)
        self.channel_vector = torch.tensor(channel_ids, dtype=torch.long)
    @property
    def total_len(self):
        """
        Returns the total number of tokens per channel across all assigned shards.
        """
        return sum(self.shard_lengths)
    def next_batch(self):
        """
        For each sample in the batch, selects a shard (round-robin) and extracts a contiguous block
        from each channel. For each region, it takes T+1 tokens (so that we can form an input/target pair),
        then creates:
           x_region = block[:-1]
           y_region = block[1:]
        Finally, it concatenates the per-region x and y blocks (in the order given by REGIONS).

        Returns:
          x: Tensor of shape [B, num_channels * T]
          c: Tensor of shape [B, num_channels * T] containing region indices for each token
          y: Tensor of shape [B, num_channels * T]
        """
        # We need T+1 tokens per channel to form the shift (input/target pair).
        sample_tokens_per_channel = self.T + 1
        xs = []
        ys = []

        for i in range(self.B):
            # Select a shard in round-robin fashion.
            shard_idx = (self.shard_index + i) % len(self.shards)
            shard = self.shards[shard_idx]
            ptr = self.shard_ptrs[shard_idx]
            shard_length = self.shard_lengths[shard_idx]

            # If there are not enough tokens remaining in this shard, wrap around.
            if ptr + sample_tokens_per_channel > shard_length:
                ptr = 0

            # For each region, extract the block and form input and target.
            sample_x_parts = []
            sample_y_parts = []
            for region in REGIONS:
                block = shard[region][ptr: ptr + sample_tokens_per_channel]
                sample_x_parts.append(block[:-1])  # first T tokens
                sample_y_parts.append(block[1:])  # last T tokens

            # Concatenate the blocks from all regions (preserving region order).
            sample_x = torch.cat(sample_x_parts, dim=0)  # shape: [num_channels * T]
            sample_y = torch.cat(sample_y_parts, dim=0)
            xs.append(sample_x)
            ys.append(sample_y)

            # Update this shard’s pointer (advance by T tokens, i.e. non-overlapping blocks).
            self.shard_ptrs[shard_idx] = ptr + self.T

        # Update the global shard round-robin pointer.
        self.shard_index = (self.shard_index + self.B) % len(self.shards)

        # Stack the samples to form a batch.
        batch_x = torch.stack(xs, dim=0)  # [B, num_channels * T]
        batch_y = torch.stack(ys, dim=0)

        # Replicate the channel vector for each sample in the batch.
        batch_c = self.channel_vector.unsqueeze(0).expand(self.B, -1)  # [B, num_channels * T]

        return batch_x, batch_c, batch_y

    def reset(self):
        """Reset the per-shard pointers and the round-robin index (useful for restarting an epoch)."""
        self.shard_ptrs = [0 for _ in self.shards]
        self.shard_index = 0


#########################
# Training Setup & Loop (No Epochs)
#########################
# Training hyperparameters
B = 16              # micro-batch size (sequences per mini-batch)
T = 1032           # sequence length (tokens per sequence)
desired_B_eff = 32  # effective batch size (number of sequences per optimizer step)
grad_accum_steps = desired_B_eff // B  # number of micro-steps to accumulate gradients
if master_process:
    print(f"Using grad_accum_steps: {grad_accum_steps}")

# Create dataloaders for training and validation.
train_loader = DataLoaderLiteAllInMemory(
    B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size,
    local_data_dir="./local_shards", shard_prefix="mydata", split='train', shuffle_shards=True
)
val_loader = DataLoaderLiteAllInMemory(
    B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size,
    local_data_dir="./local_shards", shard_prefix="mydata", split='val', shuffle_shards=True
)

# Calculate max_steps based on passes through all data.
# For example, if you want to run 5 full passes over the training data:
num_passes = 5
tokens_per_optim = B * T * grad_accum_steps * ddp_world_size  # tokens processed per optimizer step
steps_per_pass = (train_loader.total_len - 1) // tokens_per_optim
max_steps = num_passes * steps_per_pass
if master_process:
    print(f"Total tokens in training set: {train_loader.total_len}")
    print(f"Steps per pass: {steps_per_pass}")
    print(f"Running for {max_steps} optimization steps ({num_passes} passes over the data)")

# Instantiate the model.
model = GPT(GPTConfig())
model.to(device)
# Optionally compile the model for potential speedups.
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

# Set up the optimizer.
base_lr = 6e-4
optimizer = raw_model.configure_optimizer(weight_decay=0.1, learning_rate=base_lr, device=device)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=base_lr,
    total_steps=max_steps,              # total number of training steps
    pct_start=0.15,   # fraction of steps for warmup
    anneal_strategy='cos',                # cosine annealing for decay
    cycle_momentum=False                  # typically False for AdamW
)

# Log file for training (will be appended at every optimizer step)
log_file = "training.log"
if master_process:
    with open(log_file, "w") as f:
        f.write("step train_loss\n")

#########################
# Training Loop (No Epochs)
#########################


def train_step(model, optimizer, scheduler, train_loader, grad_accum_steps, device, device_type, ddp):
    """
    Performs a single training step with proper gradient accumulation in DDP mode.
    """
    optimizer.zero_grad()
    loss_accum = torch.zeros(1, device=device)

    # Disable DDP sync initially
    if ddp:
        model.require_backward_grad_sync = False

    # # Accumulate gradients locally on each GPU
    # for micro_step in range(grad_accum_steps):
    #     x, y = train_loader.next_batch()
    #     x, y = x.to(device), y.to(device)
    #
    #     with torch.autocast(device_type=device_type, dtype=torch.float16):
    #         logits, loss = model(x, y)
    #
    #     # Scale loss by accumulation steps
    #     loss = loss / grad_accum_steps
    #     loss_accum += loss.detach()
    #     loss.backward()

    # Now sync gradients across all GPUs
    if ddp:
        # Re-enable gradient sync
        model.require_backward_grad_sync = True

        # Average the accumulated gradients across GPUs
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

        # Average the loss
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # Clip gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

    # Step optimizer and scheduler
    optimizer.step()
    scheduler.step()

    return loss_accum.item(), grad_norm




def train_step_TESLA(model, optimizer, scheduler, train_loader, grad_accum_steps, device, device_type, ddp, scaler):
    """
    Performs a single training step with gradient accumulation in DDP mode using AMP.

    Args:
        model: The model (optionally wrapped in DDP).
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler.
        train_loader: Data loader providing training batches.
        grad_accum_steps: Number of gradient accumulation steps.
        device: Device (e.g. 'cuda:0').
        device_type: 'cuda' or 'cpu'.
        ddp: Boolean flag indicating if Distributed Data Parallel is used.
        scaler: torch.cuda.amp.GradScaler instance.

    Returns:
        Tuple of (accumulated loss, gradient norm).
    """
    optimizer.zero_grad()
    loss_accum = torch.zeros(1, device=device)

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        # Use no_sync() on all micro-steps except the last one to reduce inter-GPU communication.
        context = model.no_sync() if ddp and micro_step < grad_accum_steps - 1 else contextlib.nullcontext()
        with context:
            with torch.autocast(device_type=device_type, dtype=torch.float16):
                logits, loss = model(x, y)
            # Scale the loss by the accumulation steps to average gradients
                # Add the full loss to accumulator BEFORE scaling for backward
            loss_accum += loss.detach()

            # Scale the loss for gradient accumulation
            scaled_loss = loss / grad_accum_steps
            # Backward pass with AMP scaling
            scaler.scale(scaled_loss).backward()

    # Average the accumulated loss
    loss_accum = loss_accum / grad_accum_steps

    # Unscale gradients before clipping
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

    # Take an optimizer step using the scaler and update it.
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    return loss_accum.item(), grad_norm


val_steps_needed = (val_loader.total_len + B * T * ddp_world_size - 1) // (
            B * T * ddp_world_size)  # Ceiling division
if master_process:
    print("Starting training...")
    loss_plotter = LossPlotter(plot_interval=50, window=100)
    print(f"validation steps: {val_steps_needed}")


scaler = torch.amp.GradScaler(device='cuda')

for step in range(max_steps):
    t0 = time.time()
    model.train()

    loss, grad_norm = train_step_TESLA(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        grad_accum_steps=grad_accum_steps,
        device=device,
        device_type=device_type,
        ddp=ddp,
        scaler=scaler
    )
    if master_process:
        loss_plotter.update_train(loss)
    if device_type == "cuda":
        torch.cuda.synchronize()

    t1 = time.time()
    dt = t1 - t0  # time difference in seconds
    tokens_processed = B * T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    current_lrs = [pg['lr'] for pg in optimizer.param_groups]
    formatted_lrs = ", ".join(f"{lr:.4e}" for lr in current_lrs)
    if master_process:
        print(f"Step {step:5d} | Loss: {loss:.6f} | LR: {formatted_lrs} | "
              f"Grad Norm: {grad_norm:.4f} | dt: {dt*1000:.2f}ms | tokens/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} {loss:.6f}\n")

    # (Optional) Every so often, run a quick validation pass.
    if ((step % 500 == 0) and step > 0):
        model.eval()
        val_loader.reset()
        if master_process:
            print(f"--- Validation Step (Training Step: {step}) ---") # Indicate start of validation
        with torch.no_grad():
            val_loss_accum = torch.zeros(1, device=device)
            val_loss_steps = val_steps_needed
            # val_loss_steps = 200

            for val_step_num in range(val_loss_steps): # Add step counter for validation
                x_val, y_val = val_loader.next_batch()
                x_val, y_val = x_val.to(device), y_val.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.float16):
                    logits, loss = model(x_val, y_val)
                # No longer divide loss by val_loss_steps here for step-wise logging
                val_loss_accum += loss.detach() # Still accumulate for average
                if ddp: # Reduce loss across processes at each step for accurate step-wise loss
                    step_loss_reduced = loss.clone()
                    dist.all_reduce(step_loss_reduced, op=dist.ReduceOp.AVG)
                else:
                    step_loss_reduced = loss

                if master_process:
                    print(f"  Val Step: {val_step_num+1}/{val_loss_steps} | Step Val Loss: {step_loss_reduced.item():.6f}") # Log step-wise loss

            avg_val_loss = val_loss_accum / val_loss_steps # Calculate average after loop
        if ddp:
            dist.all_reduce(avg_val_loss, op=dist.ReduceOp.AVG) # Reduce average loss

            current_val_loss = avg_val_loss
            if master_process:
                print(f"Step {step} | Average val_loss over {val_loss_steps} steps: {current_val_loss.item():.4f} ") # Log average loss
                loss_plotter.update_val(current_val_loss.item())
    if master_process:
        loss_plotter.maybe_plot(step)

# Clean up DDP resources.
if ddp:
    destroy_process_group()

# (Optional) Upload log files to S3.
# if master_process:
#     upload_folder_to_s3(local_folder_path="./", bucket_name="dataframes--use1-az6--x-s3", s3_prefix="training/log")
