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
from matplotlib import pyplot as plt
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from handle_tokenized import upload_folder_to_s3  # assumed available

# DDP setup
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
torch.manual_seed(9259)
if torch.cuda.is_available():
    torch.cuda.manual_seed(9259)

###############################################################################
# Model components
###############################################################################
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
    block_size: int = 1024
    vocab_size: int = 10799
    # Small model flag (set to False for larger model)
    if True:
        n_layer: int = 12
        n_head: int = 12
        n_embd: int = 768
    else:
        n_layer: int = 48
        n_head: int = 24
        n_embd: int = 1536
    num_channels: int = 2
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

        # Per-channel encoder (2 blocks per channel)
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
        # T must be divisible by num_channels.
        assert T % self.config.num_channels == 0, "T must be divisible by num_channels"
        time_steps = T // self.config.num_channels

        tok_emb = self.transformer.wte(idx)  # [B, T, n_embd]
        # Reshape tokens so that each contiguous block corresponds to one channel.
        x = tok_emb.view(B, time_steps, self.config.num_channels, self.config.n_embd)
        channel_outs = []
        for c in range(self.config.num_channels):
            x_c = x[:, :, c, :]  # [B, time_steps, n_embd]
            x_c = self.channel_encoder[c](x_c)
            channel_outs.append(x_c)
        x = torch.stack(channel_outs, dim=2)  # [B, time_steps, num_channels, n_embd]
        x_fused = self.cross_channel_fusion(x)  # [B, time_steps, n_embd]
        # Replicate the fused output to recover the original sequence length.
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
        Configure the optimizer with separate groups for decayed and non-decayed parameters.
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = []
        nodecay_params = []

        # Group parameters: if the parameter has 2 or more dimensions, apply weight decay.
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


###############################################################################
# DataLoader (channels are no longer used)
###############################################################################
class DataLoaderLiteAllInMemory:
    """
    Loads all .pt shard files from a local directory into memory,
    concatenates them, and provides batches.
    (Now only the 'tokens' field is used.)
    """
    def __init__(self, B: int, T: int, process_rank: int, num_processes: int,
                 local_data_dir: str = "./local_shards", shard_prefix: str = "mydata",
                 split: str = "train", shuffle_shards: bool = False):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        # Use an instance of GPTConfig to get the number of channels
        if self.T % GPTConfig().num_channels != 0:
            raise ValueError("T must be divisible by num_channels")
        pattern = os.path.join(local_data_dir, f"{shard_prefix}_{split}_*.pt")
        self.shard_files = sorted(glob.glob(pattern))
        if not self.shard_files:
            raise ValueError(f"No {split} shards found in {local_data_dir} with prefix={shard_prefix}_{split}_")
        if shuffle_shards:
            random.shuffle(self.shard_files)

        all_tokens = []
        for shard_path in self.shard_files:
            loaded = torch.load(shard_path, weights_only=False)
            shard_tokens = loaded['tokens']
            all_tokens.append(shard_tokens)
        self.tokens = torch.cat(all_tokens, dim=0)
        self.current_position = self.B * self.T * self.process_rank
        self.total_len = len(self.tokens)

    def next_batch(self):
        B, T = self.B, self.T
        needed = B * T + 1
        if self.current_position + needed <= self.total_len:
            buf_tokens = self.tokens[self.current_position: self.current_position + needed]
            self.current_position += needed
        else:
            leftover = self.total_len - self.current_position
            wrap_amount = needed - leftover
            part1_toks = self.tokens[self.current_position:]
            part2_toks = self.tokens[: wrap_amount]
            buf_tokens = torch.cat([part1_toks, part2_toks], dim=0)
            self.current_position = wrap_amount
        if len(buf_tokens) != needed:
            raise RuntimeError(f"Unexpected length. Expected {needed}, got {len(buf_tokens)}")
        x = buf_tokens[:-1].view(B, T)
        y = buf_tokens[1:].view(B, T)
        return x, y

    def reset(self):
        self.current_position = self.B * self.T * self.process_rank


###############################################################################
# Training Loop (Epoch-Based)
###############################################################################

# For the small model flag
if True:  # small_model flag enabled
    epoch_num = 5
    B = 4  # micro-batch size (number of sequences per mini-batch)
    T = 1032  # sequence length (tokens per sequence)
else:
    epoch_num = 20
    B = 8
    T = 1024

# Define your desired effective batch size (in number of sequences)
desired_B_eff = 500000  # e.g. you want 256 sequences per optimizer update

# Compute the number of gradient accumulation steps.
# Each micro-step processes B sequences, so we need:
grad_accum_steps = desired_B_eff // B
if master_process:
    print(f"Using grad_accum_steps: {grad_accum_steps}")

# Calculate the number of tokens processed per micro-step and per optimizer step.
tokens_per_micro = B * T  # tokens per micro-batch
tokens_per_optim = tokens_per_micro * grad_accum_steps  # tokens per optimizer update

# Compute steps per epoch based on the total token count available in the training set.
# (Subtracting 1 because targets are shifted by one token.)
steps_per_epoch = (train_loader.total_len - 1) // tokens_per_optim
if master_process:
    print(f"Epochs: {epoch_num}, Steps per epoch: {steps_per_epoch}")

# (The scheduler was already created with epochs=epoch_num and steps_per_epoch=steps_per_epoch)

for epoch in range(epoch_num):
    print(f"\n--- Epoch {epoch + 1}/{epoch_num} ---")
    train_loader.reset()
    model.train()
    epoch_loss = 0.0
    epoch_start_time = time.time()

    for step in range(steps_per_epoch):
        step_start_time = time.time()
        optimizer.zero_grad()
        loss_accum = 0.0

        # Perform grad accumulation over several micro-steps
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            # For Distributed Data Parallel: only sync gradients on the final micro-step.
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

            with torch.autocast(device_type=device_type, dtype=torch.float16):
                logits, loss = model(idx=x, targets=y)

            # Scale loss to account for accumulation (we want an average)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        # Optionally clip gradients before stepping.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Optimizer and scheduler step.
        optimizer.step()
        scheduler.step()

        # Wait for GPU to finish work (if using CUDA)
        if device_type == "cuda":
            torch.cuda.synchronize()

        step_time = time.time() - step_start_time
        tokens_processed = tokens_per_optim * ddp_world_size
        tokens_per_sec = tokens_processed / step_time

        epoch_loss += loss_accum.item()
        if master_process and ((step + 1) % 50 == 0 or step == steps_per_epoch - 1):
            print(f"Epoch {epoch + 1:3d} | Step {step + 1:5d}/{steps_per_epoch} | "
                  f"Loss: {loss_accum.item():.6f} | Grad Norm: {grad_norm:.4f} | "
                  f"Step Time: {step_time * 1000:.2f} ms | Tokens/sec: {tokens_per_sec:.2f}")

    avg_epoch_loss = epoch_loss / steps_per_epoch
    if master_process:
        print(f"Epoch {epoch + 1} Average Training Loss: {avg_epoch_loss:.6f}")

    # Validation loop
    val_loader.reset()
    model.eval()
    val_loss_total = 0.0
    # Calculate validation steps (each batch is of size B x T)
    val_steps = (val_loader.total_len - 1) // (B * T)
    with torch.no_grad():
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            x_val, y_val = x_val.to(device), y_val.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.float16):
                _, v_loss = model(idx=x_val, targets=y_val)
            val_loss_total += v_loss.item()
    avg_val_loss = val_loss_total / val_steps
    if master_process:
        print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.6f}")
        os.makedirs("log", exist_ok=True)
        checkpoint_path = os.path.join("log", f"model_epoch_{epoch + 1:03d}.pt")
        checkpoint = {
            'model': raw_model.state_dict(),
            'epoch': epoch + 1,
            'val_loss': avg_val_loss,
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        upload_folder_to_s3(local_folder_path="./log",
                            bucket_name="dataframes--use1-az6--x-s3",
                            s3_prefix="training/log")

if ddp:
    destroy_process_group()
