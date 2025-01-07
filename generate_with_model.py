#!/usr/bin/env python3
import pickle
from dataclasses import dataclass

import torch
import math
import os
from torch.nn import functional as F

from torch import nn

import tokenizer2
from sandbox3 import tokenizer
from tokenizer2 import BPE_RLE_Tokenizer as Tokenizer
from utils import wavelet_reconstruct_window, dequantize_number

tokenizer = Tokenizer()
tokenizer.load_merges("neo_tokenizer/merges.json")
tokenizer.load_vocab("neo_tokenizer/vocab.json")

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
    vocab_size: int = 4140  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 36          # number of transformer blocks
    n_head: int = 20           # number of attention heads
    n_embd: int = 1280         # embedding (hidden) dimension
    # n_layer: int = 12 # number of layers
    # n_head: int = 12 # number of heads
    # n_embd: int = 768 # embedding dimension
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
device='cpu'
from tokenizer2 import BPE_RLE_Tokenizer as Tokenizer
# model = GPT(GPTConfig)

# checkpoint = torch.load('log/model_02000.pt', map_location=torch.device('cpu'),weights_only=False)
# # retrieve the state_dict
# orig_sd = checkpoint['model']
#
# # remove "_orig_mod." from each key
# fixed_sd = {}
# for k, v in orig_sd.items():
#     new_key = k.replace("_orig_mod.", "")
#     fixed_sd[new_key] = v
#
# # now load with the fixed state dict
# model.load_state_dict(fixed_sd, strict=True)
# model.config(checkpoint['config'])
# model.eval()
#
# num_return_sequences = 1
# max_length = 1024
# prompt_text = "D35 C18 B6 A1 A2 B5 B6 B11 B9 B7 A3 A3 C13 B7 A3 A2 B7 E56 D47 D43 D45 D38 D36 D37 D41 D40 D38 D36 D26 D34 D36 C21 D31 D46 E58 F68 F74 F74 F73 F69 E63 E62 F73 E59 E65 E58 E60 E65 F69 F67 F67 F68 E65 E65 E62 E58 E59 D54 D42 D34 D28 D29 D39 D44 D41 D43 E57 D34 D35 D35 D35 D39 D38 D36 D37 D43 D39 D46 D49 D30 D36 C22 D42 D44 D39 D46 D39 D43 D40 D38 D43 D48 D43 D36 D44 D41 D34 D45 D32 D48 D35 D33 D39 D42 D47 D36 D35 D42 D40 D33 D39 D43 D49 D48 D39 D44 D44 D35 D40 D40 D38 D45 D29 D34 D31 D40 D37 D34 D43 D50 D40 D28 F68 D36 D33 D40 D39 D41 D37 D38 D43 D35 D41 D37 D43 D36 D39 D40 D41 D35 D41 D41 D36 D38 D39 D44 D33 D43 D34 D45 D39 D28 D49 D34 D41 D38 D45 D33 D40 D40 D39 D39 D39 D39 D39 D39 D41 D36 D39 D44 D35 D37 D45 D36 D37 D42 D39 D36 D39 D41 D38 D41 D34 D41 D45 D31 D42 D40 D41 D34 D43 D39 D37 D42 D36 D40 D42 D35 D40 D43 D35 D40 D39 D39 D42 D34 D43 D38 D36 D41 D39 D41 D34 D41 D43 D33 D45 D35 D39 D42 D33 D43 D40 D37 D38 D42 D36 D39 D41 D35 D43 D35 D44 D35 D39 D41 D36 D43 D36 D40 D38 D41 D38 D37 D41 D40 D41 D35 D43 D37 D42 E65 D40 A3 A3 B5 B5 B6 B8 B8 B10 B9 B9 B10 C14 C15 C18 C16 C13 C13 C14 C15 C15 C16 C15 C18 C18 C20 C23 D26 D29 D31 D32 D34 D36 D41 D42 D45 D45 D48 D49 D51 D53 E56 E58 E59 E57 E59 E61 E64 E66 F67 F69 F68 F70 F70 F71 F71 F71 F70 F71 F71 F71 F73 G75 G76 G76 E63 D25 D39 D38 D39 D40 D38 D40 D39 D40 D40 D40 D42 D39 D40 D38 D38 D40 D39 D40 D41 D40 D38 D39 D41 D39 D40 D38 D40 D40 D39 D41 D38 D41 D40 D38 D39 D38 D40 D39 D39 D40 D37 D38 D40 D39 D40 D40 D39 D40 D39 D38 D39 D39 D40 D39 D40 D40 D38 D41 D38 D37 D39 D39 D39 D40 G75 D30 D32 D41 D41 D38 D38 D39 D40 D38 D39 D38 D40 D38 D39 D39 D39 D38 D39 D39 D38 D39 D38 D41 D38 D40 D38 D40 D39 D38 D40 D38 D39 D39 D39 D39 D39 D39 D40 D39 D39 D39 D39 D39 D39 D39 D39 D40 D38 D39 D40 D38 D40 D38 D40 D39 D39 D40 D39 D39 D39 D38 D40 D39 D37 D40 D40 D38 D39 D40 D38 D40 D38 D39 D40 D39 D38 D39 D39 D38 D39 D39 D39 D39 D39 D39 D39 D39 D39 D38 D40 D38 D40 D37 D41 D38 D39 D39 D40 D38 D40 D39 D38 D39 D39 D40 D39 D38 D40 D39 D39 D39 D40 D39 D39 D39 D39 D39 D39 D39 D39 D39 D39 D39 D39 D39 D39 D40 D38 C15 C14 B9 B10"
# raw_tokens = prompt_text.strip().split()[0:500]
# # raw_tokens.insert(0, "|trial|")
# print(len(raw_tokens))
#
# # Encode the prompt
# tokens, pos = tokenizer.encode_with_alignment(raw_tokens,as_ids=True)
# print(len(tokens))
#
# tokens = torch.tensor(tokens, dtype=torch.long)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1).to(device)
# xgen = tokens
#
# # Create channel tensor for the prompt
# chan_text = """1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
# 1 1 1 1""".strip().split()[0:500]
# # chan_text.insert(0, "1")
# print(len(chan_text))
# final_channels = tokenizer2.apply_alignment_to_channels(chan_text, pos)
# cgen = torch.tensor([int(x) - 1 for x in final_channels], dtype=torch.long).to(device)
# cgen = cgen.unsqueeze(0).repeat(num_return_sequences, 1).to(device)
# sample_rng = torch.Generator(device=device)
# sample_rng.manual_seed(42)
# k=0
# while xgen.size(1) < max_length:
#     print(k)
#     with torch.no_grad():
#             # forward pass with channels
#         logits, _ = model(idx=xgen, channel_idx=cgen)  # (B, T, vocab_size)
#
#         # take the logits at the last position
#         logits = logits[:, -1, :]  # (B, vocab_size)
#         probs = F.softmax(logits, dim=-1)
#
#         # top-k sampling of k=50
#         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
#         ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # shape (B, 1)
#         xcol = torch.gather(topk_indices, -1, ix)  # shape (B, 1)
#
#         # For each newly generated token, we also create the corresponding channel ID
#         # e.g., all zeros => channel 0
#         ccol = torch.zeros_like(xcol)
#
#         # append both the new token and its channel
#         xgen = torch.cat((xgen, xcol), dim=1)  # shape (B, T+1)
#         cgen = torch.cat((cgen, ccol), dim=1)  # shape (B, T+1)
#         k += 1

import numpy as np
import matplotlib.pyplot as plt

# Assume these helper functions and variables are already defined in your code:
#   dequantize_number(qid) -> float
#   wavelet_reconstruct_window(dequantized_coeffs, coeffs_lengths, num_samples, wavelet) -> np.array
#   wvlet: wavelet name or object
#   coeffs_lengths: an iterable describing lengths of wavelet subbands
#   decomposed_channels.shape: the shape you need to reshape your wavelet coefficients
#   etc.

# with open('goh','wb') as f:
#     pickle.dump([xgen,cgen],f)
#     f.close()

with open('goh','rb') as f:
    a = pickle.load(f)
    xgen, cgen = a[0],a[1]
    f.close()

chunk_size = (256+5)
wvlet = 'db2'
level = 2
# For first epoch, we fix these as requested:
INIT_MEAN = 40.0
INIT_STD  = 0.5

for i in range(1):
    # 1) Extract tokens (IDs) and channels from model's output
    tokens_i = xgen[i, :1024].tolist()
    channels_i = cgen[i, :1024].tolist()

    # 2) Decode with alignment (expanding R=3 and "A_B") while duplicating channels
    final_decoded_tokens, final_expanded_channels = tokenizer.decode_with_alignment(
        encoded=tokens_i,
        channels=channels_i,
        from_ids=True  # True if tokens_i are integer IDs
    )

    print(f"\n--- Sample {i} ---")
    print(f"Total decoded tokens = {len(final_decoded_tokens)}")

    # We'll store the wavelet-based reconstructions for each epoch, so we can plot them later
    reconstructed_epochs = []

    # We also track the left-/right-channel words separately (optional, if you need them)
    left_channel_epochs = []
    right_channel_epochs = []

    # For each epoch, we might define a 'prev_mean' and 'prev_std' from the previous epoch.
    prev_mean = INIT_MEAN
    prev_std  = INIT_STD
    # We'll store *all* left or right channel tokens across all chunks
    left_channel_coeffs = []
    right_channel_coeffs = []

    # Initialize trackers
    left_collected = []  # to accumulate tokens for the left channel
    right_collected = []  # to accumulate tokens for the right channel


    for epoch_idx, start_idx in enumerate(range(0, len(final_decoded_tokens), chunk_size)):
        # 1) Get the next chunk of 266 tokens from the decoded stream
        chunk_tokens = final_decoded_tokens[start_idx: start_idx + chunk_size]
        chunk_chs = final_expanded_channels[start_idx: start_idx + chunk_size]

        # 2) Separate those 266 tokens by channel
        left_list = [t for t, ch in zip(chunk_tokens, chunk_chs) if ch == 0]
        right_list = [t for t, ch in zip(chunk_tokens, chunk_chs) if ch == 1]

        # (Optional) Keep a record of exactly which tokens landed in each channel this epoch
        left_channel_epochs.append(left_list)
        right_channel_epochs.append(right_list)

        # 3) Accumulate them. (Because sometimes a chunk might be all left or all right.)
        left_collected.extend(left_list)
        right_collected.extend(right_list)

        # 4) Once we have at least 266 tokens in both channels, we can reconstruct
        #    one "epoch" from the wavelet domain back to time domain.
        if len(left_collected) >= chunk_size and len(right_collected) >= chunk_size:
            # Take exactly 266 from each channel
            left_chunk = left_collected[:chunk_size]
            right_chunk = right_collected[:chunk_size]

            # Remove them from the front
            left_collected = left_collected[chunk_size:]
            right_collected = right_collected[chunk_size:]

            # -------------------------------------------------------------
            # a) Dequantize each channel’s tokens into floats
            # -------------------------------------------------------------
            left_coeffs = [dequantize_number(x) for x in left_chunk]
            right_coeffs = [dequantize_number(x) for x in right_chunk]

            # -------------------------------------------------------------
            # b) Build shape (2, 266) so wavelet_reconstruct_window
            #    sees 2 channels at once
            # -------------------------------------------------------------
            combined_coeffs = np.array([left_coeffs, right_coeffs])  # shape: (2, 266)

            # -------------------------------------------------------------
            # c) Call your wavelet reconstruction function
            #    It expects:
            #       decomposed_channels: shape (num_channels, flattened_coeffs)
            #       coeffs_lengths: shape (num_channels, something)
            #       num_samples: how many time-domain samples to reconstruct
            # -------------------------------------------------------------
            reconstructed_window = wavelet_reconstruct_window(
                combined_coeffs,  # (2, 266)
                chunk_size,  # e.g. shape (2, num_levels+1)
                2*128,  # how many samples per channel
                wavelet=wvlet
            )
            # result shape: (2, num_samples)
            #   reconstructed_window[0, :] => left channel
            #   reconstructed_window[1, :] => right channel

            # This is still in z-score domain (per your docstring).
            # We'll unify them for the next step. You could handle them separately if you prefer.
            # For now, let's say we just do a "mono" approach, or store them both.

            # -------------------------------------------------------------
            # d) Apply your prev_mean / prev_std re-scaling for both channels
            #    For the first time (epoch_idx == 0), we use (40, 0.5).
            #    For subsequent times, we derive from the last epoch’s scaled signal
            #    if that’s your pipeline design. Adjust as needed.
            # -------------------------------------------------------------
            if epoch_idx > 0:
                # You might want to compute new mean/std from the
                # previous epoch’s final reconstruction. For example:
                prev_epoch_signal = reconstructed_epochs[-1]  # shape (2, num_samples) from last iteration
                prev_mean = np.mean(prev_epoch_signal)
                prev_std = np.std(prev_epoch_signal)
                if prev_std < 1e-9:
                    prev_std = 1.0

            # Rescale from z-space => real amplitude
            reconstructed_signal_scaled = reconstructed_window * prev_std + prev_mean
            # shape is still (2, num_samples)

            # -------------------------------------------------------------
            # e) Store it for subsequent steps (plotting, saving, etc.)
            # -------------------------------------------------------------
            reconstructed_epochs.append(reconstructed_signal_scaled)

    twoch_signal = np.concatenate(reconstructed_epochs, axis=1)
    # Index 0 => left channel, index 1 => right channel

    # 2) Plot both channels, overlaid on the same figure
    plt.figure(figsize=(10, 5))
    plt.plot(twoch_signal[0, :], label='Left Channel')
    plt.plot(twoch_signal[1, :], label='Right Channel', alpha=0.7)
    plt.title("Reconstructed 2-Channel Signal Across Epochs")
    plt.xlabel("Sample Index (across concatenated epochs)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

    # -------------------------------------------------------------------------
    # (Alternative) If you prefer two separate subplots:
    # -------------------------------------------------------------------------
    # fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    #
    # axs[0].plot(twoch_signal[0, :], color='blue')
    # axs[0].set_title('Left Channel')
    # axs[0].set_ylabel('Amplitude')
    #
    # axs[1].plot(twoch_signal[1, :], color='orange')
    # axs[1].set_title('Right Channel')
    # axs[1].set_xlabel('Sample Index (across concatenated epochs)')
    # axs[1].set_ylabel('Amplitude')
    #
    # plt.tight_layout()
    # plt.show()