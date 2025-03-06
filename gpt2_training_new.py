import glob
import os
import math
import random
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


small_model = True
resume = False

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from matplotlib import pyplot as plt
def moving_average(values, window_size=10):
    """
    Compute the simple moving average of a list of values.
    """
    if len(values) < window_size:
        return values  # not enough data, just return as-is

    # We'll output an array of the same length,
    # where each index i is the average of the last `window_size` points
    # (or fewer at the start).
    averaged = []
    for i in range(len(values)):
        start = max(0, i - window_size + 1)
        chunk = values[start : i + 1]
        averaged.append(sum(chunk) / len(chunk))
    return averaged

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(9259)
if torch.cuda.is_available():
    torch.cuda.manual_seed(9259)


# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # optional attention dropout
        self.attn_dropout = nn.Dropout(p=getattr(config, 'attn_dropout', 0.05))
        self.resid_dropout = nn.Dropout(p=getattr(config, 'resid_dropout', 0.05))

        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention
        y = self.attn_dropout(y)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        y = self.resid_dropout(y)

        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.dropout = nn.Dropout(p=getattr(config, 'mlp_dropout', 0.05))
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.dropout(x)  # dropout after activation

        x = self.c_proj(x)
        x = self.dropout(x)  # dropout again after projection

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
    vocab_size: int = 65  # Update this based on your VQCAE tokenizer vocab size
    if small_model:
        n_layer: int = 12  # number of layers
        n_head: int = 12  # number of heads
        n_embd: int = 768  # embedding dimension
    else:
        # n_layer: int = 36
        # n_head: int = 20
        # n_embd: int = 1280
        # model xL
        n_layer: int = 48  # reduced from 64 (multiple of 8)
        n_head: int = 24  # reduced from 32 (multiple of 8)
        n_embd: int = 1536  # reduced from 2048 (multiple of 128)
    mlp_dropout: float = 0.05
    attn_dropout: float = 0.05
    resid_dropout: float = 0.05


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Normal token + positional embeddings
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

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)

        x = tok_emb + pos_emb

        # pass through transformer
        for block in self.transformer.h:
            x = block(x)

        # final layernorm + linear head
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        return logits, loss

    def configure_optimizer(self, weight_decay, learning_rate, device):
        """
        Configure the optimizer, separating parameters into weight decay and no weight decay groups
        """
        # Gather all trainable params with their names
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        decay_params = []
        nodecay_params = []

        for pn, p in param_dict.items():
            # If tensor has 2+ dims, we apply weight decay
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)

        # Set up param groups
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate},
            {'params': nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate},
        ]

        # Count how many parameters in each group for logging
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        # Check fused AdamW availability
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and ('cuda' in device)

        # Only print this info on master process
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)} with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)} with {num_nodecay_params:,} parameters")
            print(f"Using fused AdamW: {use_fused}")

        # Create the optimizer
        optimizer = torch.optim.AdamW(
            optim_groups,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused
        )

        return optimizer


def get_train_val_files(data_dir, split_ratio=0.9, shuffle=True, seed=42):
    """
    Get consistent training and validation file splits across all processes
    """
    # Find all token files in the data directory
    pattern = os.path.join(data_dir, "*_tokens.pt")
    token_files = sorted(glob.glob(pattern))

    if not token_files:
        raise ValueError(f"No token files found in {data_dir} with pattern '*_tokens.pt'")

    if master_process:
        print(f"Found {len(token_files)} token files in {data_dir}")

    # Use same seed for all processes to ensure consistent splits
    rng = random.Random(seed)
    if shuffle:
        # Make a copy before shuffling
        files_to_split = token_files.copy()
        rng.shuffle(files_to_split)
    else:
        files_to_split = token_files

    # Split into train and validation sets
    split_idx = int(len(files_to_split) * split_ratio)
    train_files = files_to_split[:split_idx]
    val_files = files_to_split[split_idx:]

    if master_process:
        print(f"Split: {len(train_files)} training files, {len(val_files)} validation files")

    return train_files, val_files


class EEGTokenDataLoader:
    """
    A data loader that loads EEG token files from a directory and iterates
    through the tokens for training/validation with proper epoch boundaries.
    """

    def __init__(
            self,
            B: int,  # Batch size
            T: int,  # Sequence length (context window)
            process_rank: int,  # For DDP
            num_processes: int,  # For DDP
            files: list,  # List of files to use
            shuffle_files: bool = True,  # Whether to shuffle files between epochs
            seed: int = 42  # Random seed for shuffling
    ):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.files = files
        self.shuffle_files = shuffle_files
        self.seed = seed
        self.current_epoch = 0

        if not self.files:
            raise ValueError(f"No token files provided to the data loader")

        if master_process:
            print(f"Data loader initialized with {len(self.files)} files")

        # Initialize with first epoch
        self._load_epoch_data()

    def _load_epoch_data(self):
        """Load data for the current epoch"""
        # Use deterministic shuffling based on epoch and seed
        epoch_files = self.files.copy()
        if self.shuffle_files:
            # Different shuffle for each epoch, but consistent across processes
            epoch_seed = self.seed + self.current_epoch * 10000
            rng = random.Random(epoch_seed)
            rng.shuffle(epoch_files)

        if master_process:
            print(f"Loading data for epoch {self.current_epoch}")

        # Load files for this epoch
        all_tokens = []
        for file_path in epoch_files:
            try:
                token_tensor = torch.load(file_path, map_location='cpu')
                all_tokens.append(token_tensor)
            except Exception as e:
                if master_process:
                    print(f"Error loading {file_path}: {e}")
                continue

        if not all_tokens:
            raise ValueError(f"No valid token tensors loaded for epoch {self.current_epoch}")

        # Concatenate all tokens into a single tensor
        self.tokens = torch.cat(all_tokens, dim=0)
        self.total_len = len(self.tokens)

        # Calculate steps per epoch
        tokens_per_step = self.B * self.T
        self.total_steps = self.total_len // tokens_per_step

        # Each process handles a subset of steps
        self.steps_per_epoch = self.total_steps // self.num_processes
        if self.steps_per_epoch == 0:
            self.steps_per_epoch = 1  # At least one step per process

        # Reset for the new epoch
        self.current_step = 0

        if master_process:
            print(f"Epoch {self.current_epoch}: Loaded {self.total_len} tokens, "
                  f"{self.steps_per_epoch} steps per process")

    def next_batch(self):
        """
        Fetch next batch of tokens as (x, y) for training.
        Returns None if epoch is completed.
        """
        # Check if we've reached the end of the epoch
        if self.current_step >= self.steps_per_epoch:
            return None

        B, T = self.B, self.T
        needed = B * T + 1  # need one extra token for targets

        # Calculate position based on process rank and current step
        # Each process starts at a different position and skips other processes' positions
        pos = (self.process_rank * self.B * self.T) + (self.current_step * self.B * self.T * self.num_processes)
        pos = pos % self.total_len  # Wrap around if needed

        # Check if we need to wrap around the end of the token sequence
        if pos + needed <= self.total_len:
            # No wrap needed
            buf_tokens = self.tokens[pos:pos + needed]
        else:
            # Wrap around
            leftover = self.total_len - pos
            wrap_amount = needed - leftover

            # Get tokens from end and beginning
            part1_tokens = self.tokens[pos:]
            part2_tokens = self.tokens[:wrap_amount]

            # Concatenate
            buf_tokens = torch.cat([part1_tokens, part2_tokens], dim=0)

        # Ensure we have exactly the needed number of tokens
        if len(buf_tokens) != needed:
            # This might happen at the end of datasets - we'll pad or truncate
            if len(buf_tokens) < needed:
                # Pad with zeros (or repeat the last token)
                padding = torch.zeros(needed - len(buf_tokens), dtype=buf_tokens.dtype, device=buf_tokens.device)
                buf_tokens = torch.cat([buf_tokens, padding], dim=0)
            else:
                # Truncate
                buf_tokens = buf_tokens[:needed]

        # Create inputs (x) and targets (y)
        x = buf_tokens[:-1].view(B, T)
        y = buf_tokens[1:].view(B, T)

        # Increment step
        self.current_step += 1

        return x, y

    def start_new_epoch(self):
        """Start a new epoch"""
        self.current_epoch += 1
        self._load_epoch_data()

    def epoch_finished(self):
        """Check if current epoch is finished"""
        return self.current_step >= self.steps_per_epoch

    def reset_for_validation(self):
        """Reset position to beginning for validation run"""
        self.current_step = 0


# ---- Main Training Code ----

# Set hyperparameters based on model size
if small_model:
    epoch_num = 10
    total_batch_size = 524288
    B = 8
    T = 1024
else:
    # for XL
    epoch_num = 20
    total_batch_size = 1638400
    B = 8
    T = 1024

assert total_batch_size % (
            B * T * ddp_world_size) == 0, "make sure Total batch size is divisible by B*T* ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

torch.set_float32_matmul_precision('high')

# Get consistent train/val split
train_files, val_files = get_train_val_files(
    data_dir="training_data_shards",
    split_ratio=0.9,
    shuffle=True,
    seed=42
)

# Initialize the data loaders with the fixed splits
train_loader = EEGTokenDataLoader(
    B=B,
    T=T,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    files=train_files,
    shuffle_files=True,
    seed=42
)

val_loader = EEGTokenDataLoader(
    B=B,
    T=T,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    files=val_files,
    shuffle_files=False,
    seed=42
)

# Calculate max steps based on epochs and steps per epoch
steps_per_epoch = train_loader.steps_per_epoch
max_steps = steps_per_epoch * epoch_num

if master_process:
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps for {epoch_num} epochs: {max_steps}")

# Initialize model
model = GPT(GPTConfig())
model.to(device)
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model  # always contains the "raw" unwrapped model

# Learning rate settings
max_lr = 4e-3
min_lr = 1e-4
warmup_steps = int(0.02 * max_steps)

# Setup optimizer
optimizer = raw_model.configure_optimizer(weight_decay=0.1, learning_rate=max_lr, device=device)

# Setup scheduler - warmup + cosine annealing
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

warmup_scheduler = LinearLR(
    optimizer,
    start_factor=0.1,
    end_factor=1.0,
    total_iters=warmup_steps
)

cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=max_steps - warmup_steps,
    eta_min=min_lr
)

scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_steps]
)

# Training state variables
start_step = 0
current_epoch = 0
plateau_count = 0
best_val_loss = float('inf')
no_improvement_count = 0
patience = 3
plateau_flag = False

# Setup tracking for losses
train_losses = []
val_losses = []
train_steps = []
val_steps = []

# Setup logging and checkpointing
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")

# Resume from checkpoint if needed
if resume:
    # Find the most recent checkpoint file if any exist
    def get_latest_checkpoint(log_dir):
        ckpts = sorted(glob.glob(os.path.join(log_dir, "model_*.pt")))
        if len(ckpts) == 0:
            return None
        return ckpts[-1]  # return the most recent


    latest_ckpt_path = get_latest_checkpoint(log_dir)

    if latest_ckpt_path is not None and os.path.isfile(latest_ckpt_path):
        if master_process:
            print(f"Resuming from checkpoint: {latest_ckpt_path}")

        checkpoint = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_step = checkpoint['step'] + 1  # resume from the next step

        # Extract epoch if available
        if 'epoch' in checkpoint:
            current_epoch = checkpoint['epoch']
        else:
            current_epoch = start_step // steps_per_epoch

        # Restore scheduler state if available
        if 'scheduler_states' in checkpoint:
            warmup_state, cosine_state = checkpoint['scheduler_states']
            warmup_scheduler.load_state_dict(warmup_state)
            cosine_scheduler.load_state_dict(cosine_state)

            # Update sequential scheduler's index based on current step
            if start_step >= warmup_steps:
                scheduler._current_scheduler_idx = 1  # Using cosine scheduler
        elif start_step >= warmup_steps:
            scheduler._current_scheduler_idx = 1  # Using cosine scheduler

        # Skip to correct position in data loader
        train_loader.current_epoch = current_epoch
        train_loader.current_step = start_step % steps_per_epoch
    else:
        # Start a fresh log file
        with open(log_file, "w") as f:
            pass
else:
    # Start a fresh log file
    with open(log_file, "w") as f:
        pass

if master_process:
    print(f"Starting training from epoch {current_epoch}, step {start_step}")

# Main training loop
global_step = start_step
while current_epoch < epoch_num:
    if master_process:
        print(f"Epoch {current_epoch}/{epoch_num}")

    # Training phase for current epoch
    model.train()
    epoch_finished = False

    while not epoch_finished:
        t0 = time.time()
        last_step = (global_step == max_steps - 1)

        # Run validation periodically
        if global_step % 100 == 0 or last_step:
            model.eval()
            val_loader.reset_for_validation()

            with torch.no_grad():
                val_loss_accum = 0.0
                val_steps_done = 0

                # Process all validation data
                while not val_loader.epoch_finished():
                    x_val, y_val = val_loader.next_batch()
                    if x_val is None:  # End of validation data
                        break

                    x_val, y_val = x_val.to(device), y_val.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.float16):
                        logits, loss = model(x_val, y_val)

                    val_loss_accum += loss.detach()
                    val_steps_done += 1

                # Average validation loss
                if val_steps_done > 0:
                    val_loss_accum = val_loss_accum / val_steps_done

                    if ddp:
                        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

                # Handle validation results
                if master_process:
                    val_loss_val = val_loss_accum.item()
                    print(f"Validation loss: {val_loss_val:.4f}")
                    with open(log_file, "a") as f:
                        f.write(f"{global_step} val {val_loss_val:.4f}\n")

                    val_losses.append(val_loss_val)
                    val_steps.append(global_step)

                    # Check for plateaus
                    current_val_loss = val_loss_accum.item()
                    threshold = 1e-3
                    if current_val_loss < best_val_loss - threshold:
                        best_val_loss = current_val_loss
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                        if no_improvement_count >= patience:
                            plateau_count += 1
                            no_improvement_count = 0
                            plateau_flag = True
                            print(f"[!] Plateau detected => plateau_count={plateau_count}")

                            # Apply learning rate reduction
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = param_group['lr'] * 0.1

                            # Update scheduler states to match the new learning rates
                            if global_step < warmup_steps:
                                # If still in warmup phase, adjust end learning rate
                                warmup_scheduler.end_factor = warmup_scheduler.end_factor * 0.1
                            else:
                                # If in cosine phase, adjust base_lrs
                                cosine_scheduler.base_lrs = [param_group['lr'] for param_group in
                                                             optimizer.param_groups]

                            print(f"Learning rate reduced to {optimizer.param_groups[0]['lr']:.6e}")

            if ddp:
                plateau_tensor = torch.tensor([1 if plateau_flag else 0], device=device,
                                              dtype=torch.int64) if ddp_rank == 0 else torch.zeros(1, device=device,
                                                                                                   dtype=torch.int64)
                dist.broadcast(plateau_tensor, src=0)
                plateau_flag = (plateau_tensor.item() == 1)

            # Save checkpoint
            if global_step > 0 and (global_step % 500 == 0 or last_step):
                checkpoint_path = os.path.join(log_dir, f"model_{global_step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': global_step,
                    'epoch': current_epoch,
                    'val_loss': val_loss_accum.item(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_states': [warmup_scheduler.state_dict(), cosine_scheduler.state_dict()]
                }
                torch.save(checkpoint, checkpoint_path)

            model.train()

        # Get training batch
        batch = train_loader.next_batch()
        if batch is None:
            # End of epoch
            epoch_finished = True
            continue

        x, y = batch

        # Training step with gradient accumulation
        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = x.to(device), y.to(device)

            if device_type == 'cuda':
                with torch.autocast(device_type=device_type, dtype=torch.float16):
                    logits, loss = model(idx=x, targets=y)
            else:
                logits, loss = model(idx=x, targets=y)

            loss = loss / grad_accum_steps
            loss_accum += loss.detach()

            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

            loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        # Log gradient norms
        if master_process:
            with torch.no_grad():
                wte_grad = raw_model.transformer.wte.weight.grad
                wte_grad_norm = wte_grad.norm(2).item() if wte_grad is not None else 0.0
                c_attn_grad = raw_model.transformer.h[0].attn.c_attn.weight.grad
                c_attn_grad_norm = c_attn_grad.norm(2).item() if c_attn_grad is not None else 0.0

                print(f"[Grad Norms] wte={wte_grad_norm:.4f}, c_attn={c_attn_grad_norm:.4f}")

        # Clip gradients
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2)

        # Update weights
        optimizer.step()

        # Update learning rate
        scheduler.step()
        plateau_flag = False  # Reset flag if it was set

        # Timing and logging
        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0

        tokens_processed = B * T * grad_accum_steps * ddp_world_size
        token_per_second = tokens_processed / dt

        current_lrs = [pg['lr'] for pg in optimizer.param_groups]
        formatted_lrs = ", ".join(f"{lr:.4e}" for lr in current_lrs)

        if master_process:
            train_loss_val = loss_accum.item()
            print(
                f"Epoch {current_epoch} | Step {global_step}/{max_steps} | "
                f"Loss:{train_loss_val:.6f} | lr: {formatted_lrs} | "
                f"norm {norm:.4f} | dt: {1000 * dt:.2f}ms | tok/sec: {token_per_second:.1f}"
            )

            with open(log_file, "a") as f:
                f.write(f"{global_step} train loss: {train_loss_val:.6f} lr: {formatted_lrs} | norm {norm:.4f}\n")

            # Update tracking lists
            train_losses.append(train_loss_val)
            train_steps.append(global_step)

            # Plot progress periodically
            if global_step % 50 == 0:
                plt.figure(figsize=(10, 6))
                plt.plot(train_steps, train_losses, label='Train Loss', color='#63B8FF', alpha=0.6)
                plt.plot(val_steps, val_losses, label='Val Loss', color='#1E56A0')

                ma_train_losses = moving_average(train_losses, window_size=50)
                plt.plot(train_steps, ma_train_losses, label='Train Loss (MA)',
                         color='black', linestyle='--')
                plt.xlabel('Steps')
                plt.ylabel('Loss')
                plt.title(f'Training and Validation Loss (Epoch {current_epoch})')
                plt.legend()
                plt.grid(True)

                train_val_png_path = os.path.join(log_dir, "train_val_loss_plot.png")
                plt.savefig(train_val_png_path)
                plt.close()

        # Increment global step
        global_step += 1

    # End of epoch - prepare for next epoch
    current_epoch += 1
    train_loader.start_new_epoch()

# End of training
if ddp:
    destroy_process_group()