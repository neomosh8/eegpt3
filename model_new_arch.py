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
if torch.cuda.is_available():
    torch.cuda.manual_seed(9259)


#########################
# Model Components
#########################
import torch
import torch.nn as nn

class SimpleCrossChannelFusion(nn.Module):
    def __init__(self, n_embd, num_heads=1):
        """
        Args:
            n_embd: Embedding dimension (hidden size).
            num_heads: Number of attention heads for cross-channel interaction.
        """
        super().__init__()
        # Simple attention to fuse across channels, using batch_first for shape consistency
        self.attn = nn.MultiheadAttention(embed_dim=n_embd, num_heads=num_heads, batch_first=True)
        # Layer norm to stabilize outputs
        self.ln = nn.LayerNorm(n_embd)

    def forward(self, x):
        """
        Args:
            x: [B, time_steps, num_channels, n_embd] — Input with batch, time, channel, and embedding dims.
        Returns:
            fused: [B, time_steps, num_channels, n_embd] — Output with cross-channel info integrated.
        """
        B, T, C, E = x.size()  # Batch, Time, Channels, Embedding

        # Reshape to treat channels as the sequence dimension for attention
        # [B, T, C, E] -> [B * T, C, E], where C becomes the "sequence length"
        x_reshaped = x.view(B * T, C, E)

        # Apply causal attention across channels (no masking needed since we’re not attending over time here)
        # Query, key, value are all the same input; output shape remains [B * T, C, E]
        fused, _ = self.attn(x_reshaped, x_reshaped, x_reshaped)

        # Residual connection to preserve original channel info
        fused = fused + x_reshaped

        # Normalize and reshape back to [B, T, C, E]
        fused = self.ln(fused).view(B, T, C, E)

        return fused


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


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 10799
    # Small model configuration
    # n_layer: int = 12
    # # n_head: int = 12
    # # n_embd: int = 768

    # n_layer: int = 6
    # n_head: int = 6
    # n_embd: int = 384

    n_layer: int = 12
    n_head: int = 16
    n_embd: int = 1024
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
        self.transformer.wte.weight = self.lm_head.weight

        # Per-channel encoder (optional, kept for consistency)
        self.channel_encoder = nn.ModuleList([
            nn.Sequential(Block(config), Block(config))
            for _ in range(config.num_channels)
        ])

        # Replace with the simpler fusion module
        self.cross_channel_fusion = SimpleCrossChannelFusion(config.n_embd, num_heads=1)

        self.apply(self._init_weights)

    def forward(self, idx, targets=None):
        B, C, T = idx.size()
        assert C == self.config.num_channels, f"Expected {self.config.num_channels} channels, but got {C}"

        # Token embedding and reshape
        tok_emb = self.transformer.wte(idx)  # [B, C, T, n_embd]
        x = tok_emb.transpose(1, 2)  # [B, T, C, n_embd]

        # Positional embeddings
        pos = torch.arange(T, device=x.device).unsqueeze(0)  # [1, T]
        pos_emb = self.transformer.wpe(pos)  # [1, T, n_embd]
        x = x + pos_emb.unsqueeze(2)  # [B, T, C, n_embd]

        # Per-channel encoding (optional)
        channel_outs = []
        for c in range(self.config.num_channels):
            x_c = x[:, :, c, :]  # [B, T, n_embd]
            x_c = self.channel_encoder[c](x_c)
            channel_outs.append(x_c)
        x = torch.stack(channel_outs, dim=2)  # [B, T, C, n_embd]

        # Simple cross-channel fusion
        x = self.cross_channel_fusion(x)  # [B, T, C, n_embd]

        # Transformer blocks
        x = x.transpose(1, 2)  # [B, C, T, n_embd]
        x = x.reshape(B * C, T, self.config.n_embd)  # [B * C, T, n_embd]
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)  # [B * C, T, n_embd]

        # Next-token prediction
        x_last = x[:, -1, :]  # [B * C, n_embd]
        logits = self.lm_head(x_last)  # [B * C, vocab_size]
        logits = logits.view(B, C, -1)  # [B, C, vocab_size]

        # Loss computation
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    # _init_weights and configure_optimizer remain unchanged

#########################
# DataLoader (All-In-Memory)
#########################
# Ensure these match the channels defined during preprocessing.
REGIONS = ["frontal", "motor_temporal", "parietal_occipital"]


class DataLoaderLiteAllInMemory:
    """
    Loads all .pt shard files from a local directory.
    Each shard contains separate token tensors for each channel.
    This dataloader concatenates tokens per channel across shards,
    then extracts contiguous blocks per channel (of length per_channel_length)
    and interleaves them so that the final tensor has shape [B, num_channels, per_channel_length],
    where per_channel_length = T_total / num_channels.

    Multi–process logic:
      - Each process starts at an offset: B * per_channel_length * process_rank.
      - After producing a batch, the pointer advances by B * per_channel_length * num_processes.
    """

    def __init__(self, B: int, T: int, process_rank: int, num_processes: int,
                 local_data_dir: str = "./local_shards", shard_prefix: str = "mydata",
                 split: str = "train", shuffle_shards: bool = False):
        self.B = B
        self.T_total = T
        self.num_channels = len(REGIONS)
        # if T % self.num_channels != 0:
        #     raise ValueError("T_total must be divisible by the number of channels")
        self.per_channel_length = T

        self.process_rank = process_rank
        self.num_processes = num_processes

        # Locate shard files.
        pattern = os.path.join(local_data_dir, f"{shard_prefix}_{split}_*.pt")
        self.shard_files = sorted(glob.glob(pattern))
        if not self.shard_files:
            raise ValueError(
                f"No {split} shards found in {local_data_dir} with prefix {shard_prefix}_{split}_"
            )
        if shuffle_shards:
            random.shuffle(self.shard_files)

        # Load and concatenate tokens separately for each channel.
        self.tokens = {region: [] for region in REGIONS}
        for shard_path in self.shard_files:
            loaded = torch.load(shard_path, map_location="cpu",weights_only=False)
            for region in REGIONS:
                # Instead of raising an error, handle the missing region gracefully.
                if region not in loaded:
                    available_regions = list(loaded.keys())
                    if available_regions:
                        # Choose the first available region as a substitute.
                        alternative_region = available_regions[0]
                        print(
                            f"Warning: Shard {shard_path} is missing channel {region}. Using channel {alternative_region} as a replacement.")
                        # Copy the data from the available region.
                        loaded[region] = loaded[alternative_region]
                    else:
                        # If no region is available at all, then raise an error.
                        raise ValueError(
                            f"Shard {shard_path} does not contain any channels to copy from for missing channel {region}")

                self.tokens[region].append(loaded[region])
        # Concatenate tokens for each channel along dimension 0.
        for region in REGIONS:
            self.tokens[region] = torch.cat(self.tokens[region], dim=0)

        # Initialize the pointer using the multi-process start offset.
        self.start_ptr = self.B * self.per_channel_length * self.process_rank
        self.ptr = self.start_ptr

    def _get_slice(self, token_tensor: torch.Tensor, start: int, length: int) -> torch.Tensor:
        """
        Returns a slice of `length` tokens from token_tensor starting at `start`.
        If the slice extends past the end, it wraps around.
        """
        total_length = token_tensor.size(0)
        if start + length <= total_length:
            return token_tensor[start: start + length]
        else:
            # Wrap-around: take remainder from the beginning.
            first_part = token_tensor[start:]
            remaining = length - (total_length - start)
            second_part = token_tensor[:remaining]
            return torch.cat((first_part, second_part), dim=0)

    def next_batch(self):
        """
        Produces a batch for the current process.

        Returns:
            A tuple (inputs, targets) where:
              - inputs is a tensor of shape [B, num_channels, per_channel_length]
              - targets is a tensor of shape [B, num_channels], where each target token
                is the token that immediately follows the corresponding input sequence
                in that channel.
        """
        inputs_list = []   # To collect per-channel inputs
        targets_list = []  # To collect per-channel targets

        # For each channel, for each batch sample, extract a sequence and its target.
        for region in REGIONS:
            token_tensor = self.tokens[region]
            channel_inputs = []
            channel_targets = []
            for b in range(self.B):
                start = self.ptr + b * self.per_channel_length
                # Extract input sequence of length per_channel_length.
                seq = self._get_slice(token_tensor, start, self.per_channel_length)
                channel_inputs.append(seq.unsqueeze(0))  # shape: [1, per_channel_length]
                # Extract the target token (the token immediately after the sequence).
                target = self._get_slice(token_tensor, start + self.per_channel_length, 1)
                channel_targets.append(target)
            # Stack along the batch dimension.
            channel_inputs = torch.cat(channel_inputs, dim=0)   # shape: [B, per_channel_length]
            channel_targets = torch.cat(channel_targets, dim=0)   # shape: [B]
            # Add a channel dimension.
            inputs_list.append(channel_inputs.unsqueeze(1))   # shape: [B, 1, per_channel_length]
            targets_list.append(channel_targets.unsqueeze(1))   # shape: [B, 1]

        # Concatenate the channels: resulting shape [B, num_channels, per_channel_length]
        inputs = torch.cat(inputs_list, dim=1)
        # Concatenate targets: resulting shape [B, num_channels]
        targets = torch.cat(targets_list, dim=1)

        # Advance the pointer by B * per_channel_length * num_processes.
        self.ptr += self.B * self.per_channel_length * self.num_processes

        return inputs, targets

    def reset(self):
        """
        Resets the dataloader pointer to its initial start position.
        """
        self.ptr = self.start_ptr

    @property
    def total_len(self):
        """
        Returns the total number of tokens for one channel.
        Assumes all channels have the same length.
        """
        # Using the first region as representative.
        return self.tokens[REGIONS[0]].size(0)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch()

#########################
# Training Setup & Loop (No Epochs)
#########################
# Training hyperparameters
B = 2  # micro-batch size (sequences per mini-batch)
T = 1024  # sequence length (tokens per sequence)
desired_B_eff = 32*8  # effective batch size (number of sequences per optimizer step)
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
print(f"Total tokens in training set for {ddp_local_rank}: {train_loader.total_len}")

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
    total_steps=max_steps,  # total number of training steps
    pct_start=0.10,  # fraction of steps for warmup
    anneal_strategy='cos',  # cosine annealing for decay
    cycle_momentum=False  # typically False for AdamW
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
            with torch.autocast(device_type=device_type, dtype=torch.float32):
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
val_steps_needed = 100
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
              f"Grad Norm: {grad_norm:.4f} | dt: {dt * 1000:.2f}ms | tokens/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} {loss:.6f}\n")

    # (Optional) Every so often, run a quick validation pass.
    if ((step % 500 == 0)):
        model.eval()
        val_loader.reset()
        if master_process:
            print(f"--- Validation Step (Training Step: {step}) ---")  # Indicate start of validation
        with torch.no_grad():
            val_loss_accum = torch.zeros(1, device=device)
            val_loss_steps = val_steps_needed
            # val_loss_steps = 200

            for val_step_num in range(val_loss_steps):  # Add step counter for validation
                x_val, y_val = val_loader.next_batch()
                x_val, y_val = x_val.to(device), y_val.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.float32):
                    logits, loss = model(x_val, y_val)
                # No longer divide loss by val_loss_steps here for step-wise logging
                val_loss_accum += loss.detach()  # Still accumulate for average
                if ddp:  # Reduce loss across processes at each step for accurate step-wise loss
                    step_loss_reduced = loss.clone()
                    dist.all_reduce(step_loss_reduced, op=dist.ReduceOp.AVG)
                else:
                    step_loss_reduced = loss

                if master_process:
                    print(
                        f"  Val Step: {val_step_num + 1}/{val_loss_steps} | Step Val Loss: {step_loss_reduced.item():.6f}")  # Log step-wise loss

            avg_val_loss = val_loss_accum / val_loss_steps  # Calculate average after loop
        if ddp:
            dist.all_reduce(avg_val_loss, op=dist.ReduceOp.AVG)  # Reduce average loss

            current_val_loss = avg_val_loss
            if master_process:
                print(
                    f"Step {step} | Average val_loss over {val_loss_steps} steps: {current_val_loss.item():.4f} ")  # Log average loss
                loss_plotter.update_val(current_val_loss.item())
    if master_process:
        loss_plotter.maybe_plot(step)
    if step % 1000 == 0 and master_process and step>0:
        save_checkpoint(
            model=raw_model,
            optimizer=optimizer,
            config=raw_model.config,
            step=step,
            val_loss=current_val_loss.item(),
            log_dir="./checkpoints"
        )


# Clean up DDP resources.
if ddp:
    destroy_process_group()

# (Optional) Upload log files to S3.
# if master_process:
#     upload_folder_to_s3(local_folder_path="./", bucket_name="dataframes--use1-az6--x-s3", s3_prefix="training/log")