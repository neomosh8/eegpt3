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
class MultiScaleCrossChannelFusion(nn.Module):
    def __init__(self, n_embd, num_heads=1, scales=[1, 2, 4]):
        """
        Args:
            n_embd: hidden size.
            num_heads: number of attention heads.
            scales: list of downsampling factors (1 means no downsampling).
                    For each scale > 1, we pool over 'scale' time steps, apply attention,
                    then upsample back.
        """
        super().__init__()
        self.scales = scales
        self.attn_blocks = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=n_embd, num_heads=num_heads, batch_first=True)
            for _ in scales
        ])
        # Combine multi-scale outputs back to n_embd dimensions.
        self.out_linear = nn.Linear(len(scales) * n_embd, n_embd)

    def forward(self, x):
        """
        x: [B, T, C, n_embd] -- B=batch size, T=time steps, C=channels.
        Since channel order is unimportant, we pool over channels with a permutation-invariant operation.
        Then we perform multi-scale self-attention along time.
        """
        B, T, C, E = x.size()
        # Permutation-invariant pooling over channels (mean pooling)
        x_pooled = x.mean(dim=2)  # [B, T, E]

        scale_outputs = []
        for scale, attn in zip(self.scales, self.attn_blocks):
            if scale > 1:
                new_T = T // scale
                # Downsample: average pool over non-overlapping windows of size 'scale'
                x_down = x_pooled[:, :new_T * scale, :].view(B, new_T, scale, E).mean(dim=2)  # [B, new_T, E]
            else:
                x_down = x_pooled  # [B, T, E]

            # Apply self-attention on the (possibly downsampled) sequence.
            attn_out, _ = attn(x_down, x_down, x_down)  # [B, new_T, E]

            if scale > 1:
                # Upsample back to T by repeating each token 'scale' times.
                attn_out = attn_out.unsqueeze(2).repeat(1, 1, scale, 1).view(B, -1, E)
                attn_out = attn_out[:, :T, :]  # ensure shape [B, T, E]
            scale_outputs.append(attn_out)  # each: [B, T, E]

        # Concatenate multi-scale outputs along the feature dimension and project back to n_embd.
        fused = torch.cat(scale_outputs, dim=-1)  # [B, T, len(scales)*E]
        fused = self.out_linear(fused)  # [B, T, E]
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

        # Per-channel encoder: 3 blocks per channel.
        self.channel_encoder = nn.ModuleList([
            nn.Sequential(Block(config), Block(config), Block(config))
            for _ in range(config.num_channels)
        ])

        # Use the new multi-scale cross-channel fusion.
        self.cross_channel_fusion = MultiScaleCrossChannelFusion(config.n_embd, num_heads=1, scales=[1, 2, 4])

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
        """
        Args:
            idx: [B, T] token IDs, where T = time_steps * num_channels.
        """
        B, T = idx.size()
        # Ensure that T is divisible by the number of channels.
        assert T % self.config.num_channels == 0, "T must be divisible by num_channels"
        time_steps = T // self.config.num_channels

        # Token embeddings: [B, T, n_embd]
        tok_emb = self.transformer.wte(idx)

        # Reshape to [B, time_steps, num_channels, n_embd]
        x = tok_emb.view(B, time_steps, self.config.num_channels, self.config.n_embd)

        # Add position embeddings so that every channel at a given time-step gets the same time index.
        pos = torch.arange(time_steps, device=x.device).unsqueeze(0)  # [1, time_steps]
        pos_emb = self.transformer.wpe(pos)  # [1, time_steps, n_embd]
        x = x + pos_emb.unsqueeze(2)  # [B, time_steps, num_channels, n_embd]

        # Apply per-channel encoder.
        channel_outs = []
        for c in range(self.config.num_channels):
            x_c = x[:, :, c, :]  # [B, time_steps, n_embd]
            x_c = self.channel_encoder[c](x_c)  # [B, time_steps, n_embd]
            channel_outs.append(x_c)
        # Re-stack to get shape [B, time_steps, num_channels, n_embd]
        x = torch.stack(channel_outs, dim=2)

        # Fuse across channels using multi-scale temporal self-attention.
        # Output shape: [B, time_steps, n_embd]
        x = self.cross_channel_fusion(x)

        # Continue with the final GPT transformer blocks.
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Note: if your targets were originally shaped [B, T],
            # you might need to adjust them to [B, time_steps] for this design.
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
    Loads all .pt shard files from a local directory.
    Each shard contains separate token tensors for each channel.
    This dataloader concatenates tokens per channel across shards,
    then extracts contiguous blocks per channel (of length per_channel_length)
    and interleaves them so that the final tensor has shape [B, T_total],
    where T_total (e.g., 1032) is divisible by the number of channels.
    """
    def __init__(self, B: int, T: int, process_rank: int, num_processes: int,
                 local_data_dir: str = "./local_shards", shard_prefix: str = "mydata",
                 split: str = "train", shuffle_shards: bool = False):
        self.B = B
        self.T_total = T
        self.num_channels = len(REGIONS)
        if T % self.num_channels != 0:
            raise ValueError("T_total must be divisible by the number of channels")
        self.per_channel_length = T // self.num_channels

        self.process_rank = process_rank
        self.num_processes = num_processes

        # Locate shard files.
        pattern = os.path.join(local_data_dir, f"{shard_prefix}_{split}_*.pt")
        self.shard_files = sorted(glob.glob(pattern))
        if not self.shard_files:
            raise ValueError(f"No {split} shards found in {local_data_dir} with prefix {shard_prefix}_{split}_")
        if shuffle_shards:
            random.shuffle(self.shard_files)

        # Load and concatenate tokens separately for each channel.
        self.tokens = {region: [] for region in REGIONS}
        for shard_path in self.shard_files:
            loaded = torch.load(shard_path, map_location="cpu")
            for region in REGIONS:
                if region not in loaded:
                    raise ValueError(f"Shard {shard_path} is missing channel {region}")
                self.tokens[region].append(loaded[region])
        # Concatenate tokens for each channel along the 0-dimension.
        for region in REGIONS:
            self.tokens[region] = torch.cat(self.tokens[region], dim=0)

        # Assume all channels have the same total length.
        self.total_len = len(self.tokens[REGIONS[0]])
        # Starting position is offset by process_rank.
        self.current_position = self.B * self.per_channel_length * self.process_rank

    @property
    def total_length(self):
        """Return the total number of tokens per channel."""
        return self.total_len

    def next_batch(self):
        """
        For each channel, extract a contiguous block of tokens and then interleave them.
        Returns:
            x: tensor of shape [B, T_total] with tokens arranged as:
               [ch0_time0, ch1_time0, ch2_time0, ch0_time1, ch1_time1, ch2_time1, ...]
            y: similarly structured target tensor.
        """
        B = self.B
        L = self.per_channel_length  # tokens per channel
        needed = B * L + 1  # extra token for shifting.
        x_dict, y_dict = {}, {}

        # Extract tokens for each channel separately.
        for region in REGIONS:
            tokens_region = self.tokens[region]
            if self.current_position + needed <= self.total_len:
                buf_tokens = tokens_region[self.current_position: self.current_position + needed]
            else:
                leftover = self.total_len - self.current_position
                wrap_amount = needed - leftover
                part1_toks = tokens_region[self.current_position:]
                part2_toks = tokens_region[:wrap_amount]
                buf_tokens = torch.cat([part1_toks, part2_toks], dim=0)
            if len(buf_tokens) != needed:
                raise RuntimeError(f"Unexpected length for channel {region}. Expected {needed}, got {len(buf_tokens)}")
            # x: all tokens except the last; y: shifted by one.
            x_dict[region] = buf_tokens[:-1].view(B, L)
            y_dict[region] = buf_tokens[1:].view(B, L)

        self.current_position = (self.current_position + needed) % self.total_len

        # Stack the tokens from all channels along a new dimension.
        # Shape becomes [B, L, num_channels]
        x_stacked = torch.stack([x_dict[r] for r in REGIONS], dim=2)
        y_stacked = torch.stack([y_dict[r] for r in REGIONS], dim=2)

        # Swap the time and channel dimensions so each channel's tokens are contiguous.
        # New shape becomes [B, num_channels, L]
        x_swapped = x_stacked.transpose(1, 2)
        y_swapped = y_stacked.transpose(1, 2)

        # Flatten the swapped tensors to get the final shape [B, T_total]
        # x_combined = x_swapped.reshape(B, self.num_channels * L)
        # y_combined = y_swapped.reshape(B, self.num_channels * L)
        x_combined = x_stacked.reshape(B, self.num_channels * L)
        y_combined = y_stacked.reshape(B, self.num_channels * L)

        return x_combined, y_combined
    def reset(self):
        self.current_position = self.B * self.per_channel_length * self.process_rank


#########################
# Training Setup & Loop (No Epochs)
#########################
# Training hyperparameters
B = 8  # micro-batch size (sequences per mini-batch)
T = 1032  # sequence length (tokens per sequence)
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
    if step % 3000 == 0 and master_process and step>0:
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