import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.ao.quantization.backend_config.onednn import rnn_op_dtype_configs
from torch.nn import functional as F
import numpy as np
from torch.special import logit

from tokenizer import BPE_RLE_Tokenizer as Tokenizer

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

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

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

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
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
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
    block_size: int = 1024  # max sequence length
    vocab_size: int = 4085  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 48  # Number of transformer layers
    n_head: int = 25  # Number of attention heads
    n_embd: int = 1600
    num_channels: int = 2  # channel number


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            wce = nn.Embedding(config.num_channels, config.n_embd),  # <-- new
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # init params
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
        """
        idx: (B, T) tokens
        channel_idx: (B, T) channel IDs (e.g., 0 or 1)
        targets: (B, T) next-token predictions
        """
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, "
            f"block size is only {self.config.block_size}"
        )

        # forward the token, position, and (optionally) channel embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T,)
        pos_emb = self.transformer.wpe(pos)   # (T, n_embd)
        tok_emb = self.transformer.wte(idx)   # (B, T, n_embd)

        if channel_idx is not None:
            # Make sure channel_idx is the same shape as idx
            # channel_idx must be in [0..num_channels-1]
            cha_emb = self.transformer.wce(channel_idx)  # (B, T, n_embd)
            x = tok_emb + pos_emb + cha_emb
        else:
            # fallback if no channel_idx is provided
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
    def configure_optimizer(self,weight_decay,learning_rate,device):
        param_dict = {pn:p for pn,p in self.named_parameters()}
        param_dict = {pn:p for pn,p in param_dict.items() if p.requires_grad}
        decay_params =  [p for n,p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params':decay_params,'weight_decay': weight_decay},
            {'params':nodecay_params,'weight_decay':0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors {len(decay_params)} , with {num_decay_params:,} parameters ")
        print(f"num non-decayed parameter tensors {len(nodecay_params)} , with {num_nodecay_params:,} parameters ")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups,lr=learning_rate,betas=(0.9,0.95),eps=1e-8,fused=use_fused)
        return optimizer




class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        # 1) load token file
        with open('quantized_coeffs.txt', 'r') as f:
            text = f.read()
        # tokenize
        tokenizer = Tokenizer()
        tokenizer.load_merges("neo_tokenizer/merges.json")
        tokenizer.load_vocab("neo_tokenizer/vocab.json")
        encoded = tokenizer.encode(text.strip().split(), as_ids=True)
        self.tokens = torch.tensor(encoded)

        # 2) load channel file
        with open('final_channels.txt', 'r') as f:
            chan_text = f.read().strip().split()
        # Convert e.g. '1' -> 0, '2' -> 1
        self.channels = torch.tensor([int(x)-1 for x in chan_text])

        print(f"Loaded {len(self.tokens)} tokens")
        print(f"Loaded {len(self.channels)} channel-ids")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")
        assert len(self.tokens) == len(self.channels), "Token count != channel count!"

        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        start = self.current_position
        end   = self.current_position + B * T + 1

        buf_tokens = self.tokens[start:end]
        buf_channels = self.channels[start:end]

        x = buf_tokens[:-1].view(B, T)  # (B, T)
        y = buf_tokens[1:].view(B, T)   # (B, T)

        c = buf_channels[:-1].view(B, T)  # (B, T) for the channel IDs
        # we won't shift c by 1, because the channel "belongs" to the same token that we feed in

        self.current_position += B * T * self.num_processes
        if self.current_position + (B*T * self.num_processes +1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank

        return x, c, y






total_batch_size = 133632
B = 2
T = 522
assert total_batch_size % (B*T* ddp_world_size) == 0 , "make sure Total batch size is divisible by B*T* ddp_world_size"
grad_accum_steps = total_batch_size //(B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

torch.set_float32_matmul_precision('high')


train_loader = DataLoaderLite(B=B, T=T , process_rank=ddp_rank, num_processes=ddp_world_size)

model = GPT(GPTConfig())
model.to(device)
model = torch.compile(model)
if ddp:
    model = DDP(model,device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

max_lr = 3e-4
min_lr = max_lr*0.1
warmup_steps = 10
max_steps = 147

def get_lr(it):
    if it<warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr

    decay_ratio = ( it - warmup_steps ) / (max_steps - warmup_steps)
    assert 0<=decay_ratio<=1
    coeff = 0.5  * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


optimizer = raw_model.configure_optimizer(weight_decay=0.1,learning_rate=6e-4,device=device)
for step in range(max_steps):
    t0 = time.time()
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
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr']=lr
    optimizer.step()
    torch.cuda.synchronize()
    t1=time.time()
    dt = t1-t0
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    token_per_second = tokens_processed/dt
    if master_process:
        print(f"Step {step }: Loss:{loss_accum.item():.6f} | lr: {lr:.4e} | norm {norm:.4f} | dt: {1000*dt:.2f}ms | tok/sec: {token_per_second:.1f}")

if ddp:
    destroy_process_group()

import sys; sys.exit(0)
