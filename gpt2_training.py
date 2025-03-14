import glob
import os
import math
import random
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matplotlib import pyplot as plt
from torch.ao.quantization.backend_config.onednn import rnn_op_dtype_configs
from torch.nn import functional as F
import numpy as np
from torch.special import logit
import boto3
small_model = False
resume = False
from handle_tokenized import upload_folder_to_s3
from tokenizer2 import BPE_RLE_Tokenizer as Tokenizer

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from tokenizer2 import apply_alignment_to_channels

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
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
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = self.attn_dropout(y)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        y = self.resid_dropout(y)

        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        # self.dropout = nn.Dropout(p=getattr(config, 'mlp_dropout', 0.05))
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        # x = self.dropout(x)     # dropout after activation

        x = self.c_proj(x)
        # x = self.dropout(x)     # optional dropout again

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
    vocab_size: int = 6460
    if small_model:
        n_layer: int = 12  # number of layers
        n_head: int = 12  # number of heads
        n_embd: int = 768  # embedding dimension
    else:
        # n_layer: int = 36
        # n_head: int = 20
        # n_embd: int = 1280
        #model xL
        n_layer: int = 48  # reduced from 64 (multiple of 8)
        n_head: int = 24  # reduced from 32 (multiple of 8)
        n_embd: int = 1536  # reduced from 2048 (multiple of 128)
    num_channels: int = 2
    mlp_dropout: float = 0.05
    attn_dropout: float = 0.05
    resid_dropout: float = 0.05


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.channel_dim = 32
        self.wce = nn.Embedding(config.num_channels, self.channel_dim)

        # We'll project up to n_embd so we can add it directly
        self.channel_proj = nn.Linear(self.channel_dim, config.n_embd)

        # Optionally include a learnable scale
        self.channel_scale = nn.Parameter(torch.tensor(1.0))

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

    # NOTE: We now take an extra argument: channel_idx
    def forward(self, idx, channel_idx=None, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)

        # smaller channel embedding
        if channel_idx is not None:
            cha_emb_small = self.wce(channel_idx)  # (B, T, channel_dim)
            cha_emb_large = self.channel_proj(cha_emb_small)  # (B, T, n_embd)
            cha_emb_scaled = self.channel_scale * cha_emb_large  # apply the learnable scale
            x = tok_emb + pos_emb + cha_emb_scaled
        else:
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
        Configure the optimizer, separating channel embedding parameters into their own group
        for potentially different hyperparameters (e.g., lower learning rate, no weight decay, etc.).
        """

        # Gather all trainable params with their names
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        decay_params = []
        nodecay_params = []
        channel_params = []

        # Decide on a separate (potentially smaller) LR for channel-related parameters
        channel_lr = learning_rate * 0.1  # e.g., 10x smaller, tune as needed

        for pn, p in param_dict.items():
            # If the parameter name indicates it's part of the channel embedding/projection/scale
            if 'wce' in pn or 'channel_proj' in pn or 'channel_scale' in pn:
                channel_params.append(p)
            # If tensor has 2+ dims, we apply weight decay
            elif p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)

        # Set up param groups
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate},
            {'params': nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate},
            {'params': channel_params, 'weight_decay': 0.0, 'lr': channel_lr},
        ]

        # Count how many parameters in each group for logging
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        num_channel_params = sum(p.numel() for p in channel_params)

        # Check fused AdamW availability
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and ('cuda' in device)

        # Only print this info on master process (assuming you have 'master_process' defined globally)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)} with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)} with {num_nodecay_params:,} parameters")
            print(f"num channel parameter tensors: {len(channel_params)} with {num_channel_params:,} parameters")
            print(f"Using fused AdamW: {use_fused}")

        # Create the optimizer
        optimizer = torch.optim.AdamW(
            optim_groups,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused
        )

        return optimizer


class DataLoaderLite:
    """
    A version of your DataLoaderLite that:
      - loads .pt shard files from a local directory
      - each shard is either 'train' or 'val'
      - you specify which split to load
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
        """
        Args:
            B: Batch size
            T: Sequence length
            process_rank: (DDP) process rank
            num_processes: total DDP processes
            local_data_dir: directory containing the preprocessed .pt shards
            shard_prefix: prefix used in naming the shards (e.g. "mydata")
            split: "train" or "val"
            shuffle_shards: whether to shuffle the shard ordering
        """
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        # Collect shards for the requested split
        pattern = os.path.join(local_data_dir, f"{shard_prefix}_{split}_*.pt")
        self.shard_files = sorted(glob.glob(pattern))
        if not self.shard_files:
            raise ValueError(f"No {split} shards found in {local_data_dir} with prefix={shard_prefix}_{split}_")

        if shuffle_shards:
            import random
            random.shuffle(self.shard_files)

        self.current_shard_idx = 0
        self.tokens = None
        self.channels = None
        self.current_position = 0

        self._load_shard(self.shard_files[self.current_shard_idx])

    def _load_shard(self, shard_path: str):
        """
        Load a single shard (tokens, channels).
        Reset current_position.
        """
        loaded = torch.load(shard_path, weights_only=False)
        # ^ You can explicitly set weights_only=False to avoid future PyTorch warnings.

        self.tokens = loaded['tokens']
        self.channels = loaded['channels']

        if len(self.tokens) != len(self.channels):
            raise ValueError("tokens and channels length mismatch in shard!")

        self.current_position = self.B * self.T * self.process_rank

    def _advance_shard(self):
        """
        Move to the next shard (cyclically).
        """
        self.current_shard_idx = (self.current_shard_idx + 1) % len(self.shard_files)
        self._load_shard(self.shard_files[self.current_shard_idx])

    def next_batch(self):
        """
        Fetch the next batch: (x, c, y).
        If the current shard is exhausted, move to the next shard.
        """
        B, T = self.B, self.T

        attempt_count = 0
        max_attempts = len(self.shard_files)  # how many times we'll try loading

        while True:
            start = self.current_position
            end = start + (B * T + 1)

            buf_tokens = self.tokens[start:end]
            buf_channels = self.channels[start:end]

            if len(buf_tokens) >= (B * T + 1):
                # We have enough tokens. Make the batch
                x = buf_tokens[:-1].view(B, T)
                y = buf_tokens[1:].view(B, T)
                c = buf_channels[:-1].view(B, T)

                # Advance position
                self.current_position += B * T * self.num_processes

                # If the next batch fetch would exceed the current shard,
                # we move to the next shard for subsequent calls
                if (self.current_position + (B * T * self.num_processes + 1)) > len(self.tokens):
                    self._advance_shard()

                return x, c, y

            # If not enough tokens, move on to the next shard
            self._advance_shard()
            attempt_count += 1

            if attempt_count > max_attempts:
                # We’ve tried all shards and none has enough tokens
                raise RuntimeError(
                    f"Unable to get a full batch of size {B}x{T} from any shard. "
                    f"All shards may be too small."
                )

    def reset(self):
        """
        Reset the current shard index and position, useful for e.g. validation loops.
        """
        self.current_shard_idx = 0
        self._load_shard(self.shard_files[self.current_shard_idx])
class DataLoaderLiteAllInMemory:
    """
    A DataLoader that loads *all* .pt shard files from a local directory up front,
    concatenates them, and iterates through them in memory.
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
        shuffle_shards: bool = False
    ):
        """
        Args:
            B: Batch size
            T: Sequence length
            process_rank: (DDP) process rank
            num_processes: total DDP processes
            local_data_dir: directory containing .pt shards
            shard_prefix: prefix used in naming the shards
            split: "train" or "val"
            shuffle_shards: whether to shuffle the order of shards before concatenation
        """
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        # Collect shards for the requested split
        pattern = os.path.join(local_data_dir, f"{shard_prefix}_{split}_*.pt")
        self.shard_files = sorted(glob.glob(pattern))
        if not self.shard_files:
            raise ValueError(f"No {split} shards found in {local_data_dir} with prefix={shard_prefix}_{split}_")

        if shuffle_shards:
            random.shuffle(self.shard_files)

        # Load them all up front
        all_tokens = []
        all_channels = []
        for shard_path in self.shard_files:
            loaded = torch.load(shard_path, weights_only=False)
            shard_tokens = loaded['tokens']        # shape [N]
            shard_channels = loaded['channels']    # shape [N]
            if len(shard_tokens) != len(shard_channels):
                raise ValueError("tokens and channels length mismatch in shard!")
            all_tokens.append(shard_tokens)
            all_channels.append(shard_channels)

        # Concatenate
        self.tokens = torch.cat(all_tokens, dim=0)      # shape [sum_of_all_lengths]
        self.channels = torch.cat(all_channels, dim=0)  # same shape as tokens

        # Current read position
        self.current_position = self.B * self.T * self.process_rank

        # For convenience, store the total length
        self.total_len = len(self.tokens)

    def next_batch(self):
        B, T = self.B, self.T
        needed = B * T + 1

        if self.current_position + needed <= self.total_len:
            # no wrap
            buf_tokens = self.tokens[self.current_position: self.current_position + needed]
            buf_channels = self.channels[self.current_position: self.current_position + needed]
            self.current_position += needed
        else:
            # wrap
            leftover = self.total_len - self.current_position
            wrap_amount = needed - leftover
            part1_toks = self.tokens[self.current_position:]
            part1_chans = self.channels[self.current_position:]
            part2_toks = self.tokens[: wrap_amount]
            part2_chans = self.channels[: wrap_amount]

            buf_tokens = torch.cat([part1_toks, part2_toks], dim=0)
            buf_channels = torch.cat([part1_chans, part2_chans], dim=0)

            # Now we wrap around, so new position is wrap_amount
            self.current_position = wrap_amount

        # Should now have exactly needed = B*T+1
        if len(buf_tokens) != needed:
            raise RuntimeError(f"Unexpected length. Expected {needed}, got {len(buf_tokens)}")

        # Final reshape
        x = buf_tokens[:-1].view(B, T)
        y = buf_tokens[1:].view(B, T)
        c = buf_channels[:-1].view(B, T)

        return x, c, y

    def reset(self):
        """
        Reset the iteration index (useful for validation).
        """
        self.current_position = self.B * self.T * self.process_rank



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

if small_model:
    epoch_num = 50
    total_batch_size = 524288
    B = 64
    T = 1024
else:
    # epoch_num = 20
    # total_batch_size = 524288
    # B = 16
    # T = 1024
    #for XL
    epoch_num = 20
    total_batch_size = 1638400
    B = 8
    T = 1024


assert total_batch_size % (B*T* ddp_world_size) == 0 , "make sure Total batch size is divisible by B*T* ddp_world_size"
grad_accum_steps = total_batch_size //(B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

torch.set_float32_matmul_precision('high')


# train_loader = DataLoaderLite(B=B, T=T , process_rank=ddp_rank, num_processes=ddp_world_size,split='train')
# val_loader = DataLoaderLite(B=B//4, T=T , process_rank=ddp_rank, num_processes=ddp_world_size,split='val')

train_loader = DataLoaderLiteAllInMemory(B=B, T=T,
                                         process_rank=ddp_rank,
                                         num_processes=ddp_world_size,
                                         local_data_dir="./local_shards",
                                         shard_prefix="mydata",
                                         split='train',
                                         shuffle_shards=True)  # or False

val_loader   = DataLoaderLiteAllInMemory(B=B, T=T,
                                         process_rank=ddp_rank,
                                         num_processes=ddp_world_size,
                                         local_data_dir="./local_shards",
                                         shard_prefix="mydata",
                                         split='val',
                                         shuffle_shards=False)


# imageNet_data_by_subject = build_forced_choice_data(
#     shards_dir="validation_datasets_imageNet/shards",
#     file_pattern="shard_train_",  # or "shard_val_" if that's how your files are named
#     map_location='cpu'
# )

model = GPT(GPTConfig())
model.to(device)
model = torch.compile(model)
if ddp:
    model = DDP(model,device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

max_lr = 1e-3
min_lr = 1e-9
max_steps = math.ceil(1e9//total_batch_size) * epoch_num
warmup_steps =int(0.02*max_steps)

if master_process:
    print("Max Steps: ",max_steps)

plateau_count = 0
best_val_loss = float('inf')
no_improvement_count = 0
patience = 3

# def get_lr(step, max_lr=max_lr, min_lr=min_lr, warmup_steps=warmup_steps, total_steps=max_steps):
#     if step < warmup_steps:
#         lr = max_lr * (step + 1) / warmup_steps
#     else:
#         ratio = (step - warmup_steps) / float(total_steps - warmup_steps)
#         ratio = min(1.0, max(0.0, ratio))
#         lr = max_lr * (min_lr / max_lr) ** ratio
#         lr = max(lr, min_lr)
#     # multiply by 0.1^plateau_count
#     factor = 0.1 ** plateau_count
#     lr_final = lr * factor
#     return max(lr_final, 1e-10)

optimizer = raw_model.configure_optimizer(weight_decay=0.1,learning_rate=6e-3,device=device)
max_lr_main = 6e-3
max_lr_channel = max_lr_main * 0.1

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[max_lr_main, max_lr_main, max_lr_channel],
    total_steps=max_steps,              # total number of training steps
    pct_start=warmup_steps / max_steps,   # fraction of steps for warmup
    anneal_strategy='cos',                # cosine annealing for decay
    cycle_momentum=False                  # typically False for AdamW
)
start_step = 0

####RESUME
if resume:
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
        checkpoint = torch.load(latest_ckpt_path, map_location=device,weights_only=False)
        raw_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_step = checkpoint['step'] + 1  # resume from the next step
    else:
        # start a fresh log file
        with open(log_file, "w") as f:
            pass
##########

plateau_flag = False
# keep track of losses to plot later  ### ADDED LINES ###
train_losses = []
val_losses   = []
train_steps  = []
val_steps    = []
mc_val_losses=[]
mc_val_steps =[]

if not resume:
    # create the log directory we will write checkpoints to and log to
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f: # open for writing to clear the file
        pass

print("step:",start_step,max_steps)
for step in range(start_step,max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)
    # once in a while evaluate our validation loss
    if step % 100 == 0 or last_step:
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
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                val_loss_val = val_loss_accum.item()
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                val_losses.append(val_loss_val)
                val_steps.append(step)
            current_val_loss = val_loss_accum
            print(f"Step {step} | val_loss {current_val_loss:.4f} | plateau_flag={plateau_flag}")

            threshold = 1e-3
            if current_val_loss < best_val_loss - threshold:
                best_val_loss = current_val_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    plateau_count += 1
                    no_improvement_count = 0
                    print(f"[!] Plateau detected => plateau_count={plateau_count}, LR will drop by factor of 0.1.")
        if ddp:
            plateau_tensor = torch.tensor([1 if plateau_flag else 0], device=device,
                                          dtype=torch.int64) if ddp_rank == 0 else torch.zeros(1, device=device,
                                                                                               dtype=torch.int64)
            dist.broadcast(plateau_tensor, src=0)
            plateau_flag = (plateau_tensor.item() == 1)

        if step > 0 and (step % 500 == 0 or last_step):
            # optionally write model checkpoints
            checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
            checkpoint = {
                'model': raw_model.state_dict(),
                'config': raw_model.config,
                'step': step,
                'val_loss': val_loss_accum.item(),
                'optimizer_state':optimizer.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)
            if master_process:
                upload_folder_to_s3(
                    local_folder_path="./log",
                    bucket_name="dataframes--use1-az6--x-s3",
                    s3_prefix="training_XL/log"
                )



    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for mico_step in range(grad_accum_steps):
        x, c, y = train_loader.next_batch()
        x, c, y = x.to(device), c.to(device), y.to(device)
        if device == 'cuda':
            with torch.autocast(device_type=device,dtype=torch.bfloat16):
                logits, loss = model(idx=x, channel_idx=c, targets=y)
        else:
            logits, loss = model(idx=x, channel_idx=c, targets=y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (mico_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    # --- after finishing grad accumulation, before clipping ---
    if master_process:  # so we only print once in DDP
        with torch.no_grad():
            # Example: get gradient norm for the word embedding (wte) weights
            wte_grad = raw_model.transformer.wte.weight.grad
            wte_grad_norm = wte_grad.norm(2).item() if wte_grad is not None else 0.0
            # Example: get gradient norm for the channel embedding (wce) weights
            wce_grad = raw_model.wce.weight.grad  # note: no ".transformer" anymore
            wce_grad_norm = wce_grad.norm(2).item() if wce_grad is not None else 0.0
            # Example: get gradient norm for the first block's attention projection
            c_attn_grad = raw_model.transformer.h[0].attn.c_attn.weight.grad
            c_attn_grad_norm = c_attn_grad.norm(2).item() if c_attn_grad is not None else 0.0

            print(f"[Grad Norms] wte={wte_grad_norm:.4f}, c_attn={c_attn_grad_norm:.4f}, wce={wce_grad_norm:.4f}")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),2)
    # lr = get_lr(step)
    # for param_group in optimizer.param_groups:
    #     param_group['lr']=lr
    optimizer.step()
    scheduler.step()  # updates the learning rate according to OneCycleLR

    torch.cuda.synchronize()
    t1=time.time()
    dt = t1-t0
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    token_per_second = tokens_processed/dt
    current_lrs = [pg['lr'] for pg in optimizer.param_groups]
    formatted_lrs = ", ".join(f"{lr:.4e}" for lr in current_lrs)

    if master_process:
        print(f"Step {step }: Loss:{loss_accum.item():.6f} | lr: {formatted_lrs} | norm {norm:.4f} | dt: {1000*dt:.2f}ms | tok/sec: {token_per_second:.1f}")
        with open(log_file, "a") as f:
            train_loss_val = loss_accum.item()
            f.write(f"{step} train loss: {train_loss_val:.6f} lr: {formatted_lrs} | norm {norm:.4f}\n")
        # update train_losses and steps  ### ADDED LINES ###
        train_losses.append(train_loss_val)
        train_steps.append(step)
        # Plot every several steps  ### ADDED LINES ###
        if step % 50 == 0:
            # ---- 1) Figure for Train Loss & Val Loss ----
            plt.figure(figsize=(10, 6))
            plt.plot(train_steps, train_losses, label='Train Loss', color='#63B8FF', alpha=0.6)
            plt.plot(val_steps, val_losses, label='Val Loss', color='#1E56A0')
            # NEW LINES: compute and plot the MA
            ma_train_losses = moving_average(train_losses, window_size=50)
            plt.plot(train_steps, ma_train_losses, label='Train Loss (MA)',
                     color='black', linestyle='--')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)

            # Save figure for train/val
            train_val_png_path = os.path.join(log_dir, "train_val_loss_plot.png")
            plt.savefig(train_val_png_path)
            plt.close()

            # ---- 2) Figure for MC Loss with Random Baseline ----
            plt.figure(figsize=(10, 6))
            plt.plot(mc_val_steps, mc_val_losses, label='MC-Val Loss', color='green')

            # Horizontal random baseline
            random_baseline = 100/80
            plt.axhline(y=random_baseline, color='red', linestyle='--', label='Random Baseline')

            plt.xlabel('Steps')
            plt.ylabel('MC Loss')
            plt.title('MC Validation Loss')
            plt.legend()
            plt.grid(True)

            # Save figure for MC loss
            mc_png_path = os.path.join(log_dir, "mc_loss_plot.png")
            plt.savefig(mc_png_path)
            plt.close()
if master_process:
    upload_folder_to_s3(
        local_folder_path="./log",
        bucket_name="dataframes--use1-az6--x-s3",
        s3_prefix="training/log"
    )
if ddp:
    destroy_process_group()

