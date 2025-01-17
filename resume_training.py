import glob
import os
import math
import random
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.ao.quantization.backend_config.onednn import rnn_op_dtype_configs
from torch.nn import functional as F
import numpy as np
from torch.special import logit
import boto3

from handle_tokenized import upload_folder_to_s3
from tokenizer2 import BPE_RLE_Tokenizer as Tokenizer

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from tokenizer2 import apply_alignment_to_channels

# -----------------------------------------------------------------------------
# 1. Setup DDP
# -----------------------------------------------------------------------------

ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "For now we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing, etc.
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

device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(9259)
if torch.cuda.is_available():
    torch.cuda.manual_seed(9259)

# -----------------------------------------------------------------------------
# 2. Model Definitions
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
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

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 4140
    n_layer: int = 18
    n_head: int = 12
    n_embd: int = 768
    num_channels: int = 2


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            wce = nn.Embedding(config.num_channels, config.n_embd),
            h   = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f= nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # scale down the init for certain layers
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, channel_idx=None, targets=None):
        """
        idx: (B, T) tokens
        channel_idx: (B, T) channel IDs (0..num_channels-1)
        targets: (B, T) next-token predictions
        """
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        )

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)

        if channel_idx is not None:
            cha_emb = self.transformer.wce(channel_idx)
            x = tok_emb + pos_emb + cha_emb
        else:
            x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        return logits, loss

    def configure_optimizer(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params =  [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params,   'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device

        if master_process:
            print(f"num decayed parameter tensors {len(decay_params)} , with {num_decay_params:,} parameters ")
            print(f"num non-decayed parameter tensors {len(nodecay_params)} , with {num_nodecay_params:,} parameters ")
            print(f"using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused
        )
        return optimizer

# -----------------------------------------------------------------------------
# 3. DataLoaderLite
# -----------------------------------------------------------------------------

class DataLoaderLite:
    """
    A simplified DataLoader that loads .pt shard files from a local directory.
    Each shard is either 'train' or 'val'.
    """

    def __init__(self,
                 B: int,
                 T: int,
                 process_rank: int,
                 num_processes: int,
                 local_data_dir: str = "./local_shards",
                 shard_prefix: str = "mydata",
                 split: str = "train",
                 shuffle_shards: bool = False):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        pattern = os.path.join(local_data_dir, f"{shard_prefix}_{split}_*.pt")
        self.shard_files = sorted(glob.glob(pattern))
        if not self.shard_files:
            raise ValueError(f"No {split} shards found in {local_data_dir} with prefix={shard_prefix}_{split}_")

        if shuffle_shards:
            random.shuffle(self.shard_files)

        self.current_shard_idx = 0
        self.tokens = None
        self.channels = None
        self.current_position = 0

        self._load_shard(self.shard_files[self.current_shard_idx])

    def _load_shard(self, shard_path: str):
        loaded = torch.load(shard_path, weights_only=False)
        self.tokens = loaded['tokens']
        self.channels = loaded['channels']
        if len(self.tokens) != len(self.channels):
            raise ValueError("tokens and channels length mismatch in shard!")
        self.current_position = self.B * self.T * self.process_rank

    def _advance_shard(self):
        self.current_shard_idx = (self.current_shard_idx + 1) % len(self.shard_files)
        self._load_shard(self.shard_files[self.current_shard_idx])

    def next_batch(self):
        B, T = self.B, self.T
        attempt_count = 0
        max_attempts = len(self.shard_files)

        while True:
            start = self.current_position
            end = start + (B * T + 1)

            buf_tokens = self.tokens[start:end]
            buf_channels = self.channels[start:end]

            if len(buf_tokens) >= (B * T + 1):
                x = buf_tokens[:-1].view(B, T)
                y = buf_tokens[1:].view(B, T)
                c = buf_channels[:-1].view(B, T)

                self.current_position += B * T * self.num_processes

                if (self.current_position + (B * T * self.num_processes + 1)) > len(self.tokens):
                    self._advance_shard()

                return x, c, y

            self._advance_shard()
            attempt_count += 1
            if attempt_count > max_attempts:
                raise RuntimeError(
                    f"Unable to get a full batch of size {B}x{T} from any shard."
                )

    def reset(self):
        self.current_shard_idx = 0
        self._load_shard(self.shard_files[self.current_shard_idx])

# -----------------------------------------------------------------------------
# 4. Training Configuration
# -----------------------------------------------------------------------------

epoch_num = 13
total_batch_size = 524288
B = 64
T = 1024
assert total_batch_size % (B * T * ddp_world_size) == 0, "Total batch size must be divisible by B*T*ddp_world_size"

grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

torch.set_float32_matmul_precision('high')

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
val_loader   = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')

model = GPT(GPTConfig())
model.to(device)
model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

max_lr = 3e-4
min_lr = 1e-6
warmup_steps = (300000000 // total_batch_size)
# For demonstration, the dataset size is big, so max_steps is computed from total tokens / total_batch_size, times epochs
max_steps = math.ceil(771479260 / total_batch_size) * epoch_num
if master_process:
    print("Max Steps: ", max_steps)

def get_lr(it, max_lr=max_lr, min_lr=min_lr, warmup_steps=warmup_steps, max_steps=max_steps*1.8):
    if it < warmup_steps:
        lr = max_lr * (it + 1) / warmup_steps
    elif it > max_steps:
        lr = min_lr
    else:
        decay_steps = it - warmup_steps
        total_decay_steps = max_steps - warmup_steps
        decay_rate = math.log(min_lr / max_lr) / total_decay_steps
        lr = max_lr * math.exp(decay_rate * decay_steps)
        lr = max(lr, min_lr)
    return lr

optimizer = raw_model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device)

# -----------------------------------------------------------------------------
# 5. Checkpoint Resuming
# -----------------------------------------------------------------------------

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")

# Find the most recent checkpoint file if any exist
def get_latest_checkpoint(log_dir):
    ckpts = sorted(glob.glob(os.path.join(log_dir, "model_*.pt")))
    if len(ckpts) == 0:
        return None
    return ckpts[-1]  # return the most recent

latest_ckpt_path = get_latest_checkpoint(log_dir)
start_step = 0

# If there is a checkpoint, load it
if latest_ckpt_path is not None and os.path.isfile(latest_ckpt_path):
    if master_process:
        print(f"Resuming from checkpoint: {latest_ckpt_path}")
    checkpoint = torch.load(latest_ckpt_path, map_location=device)
    raw_model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_step = checkpoint['step'] + 1  # resume from the next step
else:
    # start a fresh log file
    with open(log_file, "w") as f:
        pass

# -----------------------------------------------------------------------------
# 6. Loss Tracking Lists
# -----------------------------------------------------------------------------

train_losses = []
val_losses   = []
train_steps  = []
val_steps    = []

# -----------------------------------------------------------------------------
# 7. Training Loop
# -----------------------------------------------------------------------------

for step in range(start_step, max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # Evaluate on validation set periodically
    if True:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 100
            for _ in range(val_loss_steps):
                x_val, c_val, y_val = val_loader.next_batch()
                x_val, c_val, y_val = x_val.to(device), c_val.to(device), y_val.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x_val, c_val, y_val)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            val_loss_val = val_loss_accum.item()
            print(f"[Step {step}] validation loss: {val_loss_val:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_val:.4f}\n")
            val_losses.append(val_loss_val)
            val_steps.append(step)

        # Optionally write model checkpoints
        if step > 0 and (step % 1000 == 0 or last_step):
            checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
            checkpoint = {
                'model': raw_model.state_dict(),
                'config': raw_model.config,
                'step': step,
                'val_loss': val_loss_accum.item(),
                'optimizer_state': optimizer.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)

    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0

    # Gradient accumulation
    for micro_step in range(grad_accum_steps):
        x, c, y = train_loader.next_batch()
        x, c, y = x.to(device), c.to(device), y.to(device)
        if device_type == 'cuda':
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(idx=x, channel_idx=c, targets=y)
        else:
            logits, loss = model(idx=x, channel_idx=c, targets=y)

        loss = loss / grad_accum_steps
        loss_accum += loss.detach()

        if ddp:
            # Sync gradients only on the last micro_step
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # Clip gradients
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

    # Update learning rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()

    if device_type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    token_per_second = tokens_processed / dt

    if master_process:
        train_loss_val = loss_accum.item()
        print(f"Step {step}: "
              f"Loss: {train_loss_val:.6f} | "
              f"lr: {lr:.4e} | "
              f"grad_norm: {norm:.4f} | "
              f"dt: {1000*dt:.2f}ms | "
              f"tok/sec: {token_per_second:.1f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {train_loss_val:.6f}\n")
        train_losses.append(train_loss_val)
        train_steps.append(step)

        # Plot losses every 100 steps (optional)
        if step % 100 == 0:
            plt.figure(figsize=(10, 6))
            plt.plot(train_steps, train_losses, label='Train Loss')
            plt.plot(val_steps,   val_losses,   label='Val Loss')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            png_path = os.path.join(log_dir, "loss_plot.png")
            plt.savefig(png_path)
            plt.close()

# -----------------------------------------------------------------------------
# 8. Upload logs to S3 and clean up
# -----------------------------------------------------------------------------
if master_process:
    upload_folder_to_s3(
        local_folder_path="./log",
        bucket_name="dataframes--use1-az6--x-s3",
        s3_prefix="training_small_cont/log"
    )

if ddp:
    destroy_process_group()
