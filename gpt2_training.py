import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import List

# ---------------------------------------------------------------------
# 1) BPE+RLE Tokenizer and Simple Vocab
#    We'll assume you already have these stored on disk:
#       - "bpe_merges.json"  (merges learned)
#       - "vocab.json"       (mapping token->id)
#    and you'll read them here.
# ---------------------------------------------------------------------

# We'll assume you have a file "bpe_rle_tokenizer.py" with your BPE_RLE_Tokenizer
# class, which can do tokenizer.load_merges(...), tokenizer.encode(...), etc.
# For demonstration, we just import it:
from tokenizer import BPE_RLE_Tokenizer  # <--- adapt if needed
# ---------------------------------------------------------------------
# 2) GPT Config & Model (with Channel Embeddings)
#    A minimal GPT-2 style adaptation that includes channel embeddings
# ---------------------------------------------------------------------
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50000
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_channels: int = 3  # up to 3 channels (1, 2, or 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        # qkv projection
        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)  # each (B, T, C)
        # reshape for multi-head
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # scaled dot-product attention
        # PyTorch >=2.0 has F.scaled_dot_product_attention with `is_causal=True`
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, n_head, T, head_size)
        # reassemble
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # final projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.gelu   = nn.GELU()

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

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),  # position embedding
            wce = nn.Embedding(config.n_channels, config.n_embd),  # channel embedding
            h   = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f= nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # optional weight tying
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, channel_idx=None, targets=None):
        """
        idx:         (B, T) LongTensor of token indices
        channel_idx: (B, T) LongTensor of channel IDs (1,2,3) but we typically store them as 0..n_channels-1
        targets:     (B, T) LongTensor of next-token labels
        """
        B, T = idx.size()
        assert T <= self.config.block_size, "Sequence length too big"
        # Token + positional embedding
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T,)
        pos_emb = self.transformer.wpe(pos).unsqueeze(0)  # (1, T, n_embd)
        # Channel embedding
        if channel_idx is not None:
            # Convert channel IDs to 0-based (if they are 1,2,3).
            # (depends on how you store them)
            chan_zero_based = channel_idx - 1
            chan_emb = self.transformer.wce(chan_zero_based)
        else:
            chan_emb = 0

        x = tok_emb + pos_emb + chan_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1  # if we want to ignore some padded indices
            )
        return logits, loss

# ---------------------------------------------------------------------
# 3) Dataset & DataLoader: read your data_file + channel_file, tokenize them,
#    build a big array of token IDs + channel IDs, then yield (x,y,channel_idx)
# ---------------------------------------------------------------------

class ChannelDataset(Dataset):
    """
    1. Loads lines from data_file.txt (raw tokens).
    2. Loads lines from channel_file.txt (channel strings).
    3. For each line:
       - tokenize using the loaded BPE tokenizer
       - build a parallel channel ID array
    4. Store everything in one big list of IDs.
    5. We'll produce sliding windows of size block_size for training.
    """
    def __init__(self,
                 data_file: str,
                 channel_file: str,
                 tokenizer: BPE_RLE_Tokenizer,
                 vocab: SimpleVocab,
                 block_size: int = 1024):
        super().__init__()
        self.block_size = block_size

        # Step 1: read lines
        with open(data_file, "r", encoding="utf-8") as f:
            data_lines = [line.strip() for line in f if line.strip()]

        with open(channel_file, "r", encoding="utf-8") as f:
            channel_lines = [line.strip() for line in f if line.strip()]

        if len(data_lines) != len(channel_lines):
            raise ValueError("Mismatch: data_file lines != channel_file lines")

        # Step 2: encode all lines
        all_token_ids = []
        all_channel_ids = []

        for text_token, chan_text in zip(data_lines, channel_lines):
            # tokenize BPE + RLE
            bpe_tokens = tokenizer.encode([text_token])  # we assume each line is a single token, or you might split
            # or if each line might have multiple tokens:
            # bpe_tokens = tokenizer.encode(text_token.split())

            # convert to vocab IDs
            token_ids = vocab.encode_ids(bpe_tokens)

            # Decide channel ID:
            #   if ends with an odd digit -> channel=1
            #   if ends with an even digit-> channel=2
            #   else -> channel=3
            # We do this once per line, but if you have multiple tokens, do it for each token in that line
            # For demonstration, let's assume 1 line = 1 token
            if chan_text[-1].isdigit():
                last_digit = int(chan_text[-1])
                if last_digit % 2 == 1:
                    ch_id = 1
                else:
                    ch_id = 2
            else:
                ch_id = 3

            # If multiple tokens in that line, they'd share the same channel ID
            channel_ids = [ch_id] * len(token_ids)

            all_token_ids.extend(token_ids)
            all_channel_ids.extend(channel_ids)

        self.data = torch.tensor(all_token_ids, dtype=torch.long)
        self.channel_data = torch.tensor(all_channel_ids, dtype=torch.long)

        # We'll verify shapes match
        if self.data.size(0) != self.channel_data.size(0):
            raise ValueError("Token ID array and Channel ID array differ in length")

        self.n = self.data.size(0)

    def __len__(self):
        return self.n - self.block_size  # number of possible chunks

    def __getitem__(self, idx):
        # x = [idx : idx+block_size]
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx+1 : idx + self.block_size + 1]
        c = self.channel_data[idx : idx + self.block_size]

        # if the last chunk index is at self.n-1, y would be out of range, so
        # carefully handle that. We'll assume inrange for now.

        return x, y, c

# ---------------------------------------------------------------------
# 4) Main Training Script
# ---------------------------------------------------------------------

def train_gpt2_with_channels():
    # hyperparams
    block_size = 1024
    batch_size = 2
    max_steps = 1000
    lr = 3e-4

    # files
    data_file    = "data_file.txt"
    channel_file = "channel_file.txt"
    merges_file  = "bpe_merges.json"
    vocab_file   = "vocab.json"

    # 1) Load tokenizer merges
    tokenizer = BPE_RLE_Tokenizer()
    tokenizer.load_merges(merges_file)

    # 2) Load vocab
    vocab = SimpleVocab.load_json(vocab_file)
    vocab_size = len(vocab.id_to_token)
    print(f"Loaded vocab size: {vocab_size}")

    # 3) Create dataset
    dataset = ChannelDataset(
        data_file=data_file,
        channel_file=channel_file,
        tokenizer=tokenizer,
        vocab=vocab,
        block_size=block_size
    )
    print("Dataset length:", len(dataset))

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True)

    # 4) Create GPT model config
    config = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=4,     # smaller for example
        n_head=4,
        n_embd=256,    # smaller for example
        n_channels=3
    )
    model = GPT(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 5) Training loop
    model.train()
    step_count = 0
    for epoch in range(3):  # say we do 3 epochs
        for x, y, c in loader:
            x, y, c = x.to(device), y.to(device), c.to(device)
            optimizer.zero_grad()
            logits, loss = model(idx=x, channel_idx=c, targets=y)
            loss.backward()
            optimizer.step()

            step_count += 1
            if step_count % 50 == 0:
                print(f"step {step_count}, loss={loss.item():.4f}")

            if step_count >= max_steps:
                break
        if step_count >= max_steps:
            break

    # 6) Done
    print("Training finished.")
    torch.save(model.state_dict(), "gpt_channels.pt")

# ---------------------------------------------------------------------

if __name__ == "__main__":
    train_gpt2_with_channels()
