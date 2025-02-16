#!/usr/bin/env python3
import inspect
import random
import time
import pickle
from dataclasses import dataclass
import numpy as np
import torch
import math
import os
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch import nn

from tokenizer2 import BPE_RLE_Tokenizer as Tokenizer
small_model = False
tokenizer = Tokenizer()
tokenizer.load_merges("neo_tokenizer/merges.json")
tokenizer.load_vocab("neo_tokenizer/vocab.json")

import os, glob, random
import numpy as np
import torch
import torch.nn.functional as F

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
            targets: [B, T] with different target tokens for each channel.
        """
        B, T = idx.size()
        # Ensure T is divisible by the number of channels.
        assert T % self.config.num_channels == 0, "T must be divisible by num_channels"
        time_steps = T // self.config.num_channels

        # 1. Token Embeddings and Reshape:
        tok_emb = self.transformer.wte(idx)  # [B, T, n_embd]
        # Reshape to [B, time_steps, num_channels, n_embd]
        x = tok_emb.view(B, time_steps, self.config.num_channels, self.config.n_embd)

        # 2. Add Positional Embeddings (same time index for every channel):
        pos = torch.arange(time_steps, device=x.device).unsqueeze(0)  # [1, time_steps]
        pos_emb = self.transformer.wpe(pos)  # [1, time_steps, n_embd]
        x = x + pos_emb.unsqueeze(2)  # [B, time_steps, num_channels, n_embd]

        # (Optional) Add a learnable channel embedding if you want to inform the model of channel identity.
        # Even if channel ordering is unimportant, if targets differ per channel you might benefit from it.
        # Uncomment the following if desired:
        # channel_ids = torch.arange(self.config.num_channels, device=x.device).unsqueeze(0).unsqueeze(0)  # [1,1,num_channels]
        # channel_emb = self.channel_embedding(channel_ids)  # [1,1,num_channels, n_embd]
        # x = x + channel_emb  # [B, time_steps, num_channels, n_embd]

        # 3. Per-Channel Encoding:
        channel_outs = []
        for c in range(self.config.num_channels):
            x_c = x[:, :, c, :]  # [B, time_steps, n_embd]
            x_c = self.channel_encoder[c](x_c)  # Process each channel separately.
            channel_outs.append(x_c)
        # Stack back: [B, time_steps, num_channels, n_embd]
        x = torch.stack(channel_outs, dim=2)

        # 4. Cross-Channel Fusion (Global Fusion Over Channels at Each Time Step):
        # This module is expected to take an input of shape [B, time_steps, num_channels, n_embd]
        # and output a fused representation of shape [B, time_steps, n_embd].
        fused = self.cross_channel_fusion(x)  # [B, time_steps, n_embd]

        # 5. Combine the Fused Representation with the Original Per-Channel Features:
        # Broadcast the fused representation over the channel dimension and add it.
        x = x + fused.unsqueeze(2)  # [B, time_steps, num_channels, n_embd]

        # 6. Flatten the (time, channel) dimensions to get a sequence of length T:
        x = x.view(B, T, self.config.n_embd)  # [B, T, n_embd]

        # 7. Process with the Final GPT Transformer Blocks:
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]

        loss = None
        if targets is not None:
            # Now targets have shape [B, T] (with T = time_steps * num_channels)
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



# Helper: compute completion loss for one prompt–candidate pair.
import torch
import torch.nn.functional as F


def compute_completion_loss_with_channels(model, prompt_tokens, candidate_tokens, device="cuda"):
    """
    Given a prompt and a candidate (both already interleaved as 1D tensors),
    compute the summed negative log-likelihood loss on the candidate tokens.

    Args:
        model: the trained model (expects input shape [B, seq_length])
        prompt_tokens: LongTensor of shape [prompt_length] (1D) on CPU
        candidate_tokens: LongTensor of shape [candidate_length] (1D) on CPU
        device: device string ("cuda" or "cpu")

    Returns:
        loss: scalar loss (float)
    """
    # Move tokens to the target device.
    prompt_tokens = prompt_tokens.to(device)
    candidate_tokens = candidate_tokens.to(device)

    # Create batch dimension.
    prompt_tokens = prompt_tokens.unsqueeze(0)  # [1, prompt_length]
    candidate_tokens = candidate_tokens.unsqueeze(0)  # [1, candidate_length]

    # Concatenate prompt and candidate tokens.
    input_seq = torch.cat([prompt_tokens, candidate_tokens], dim=1)  # [1, total_length]

    # Forward pass through the model.
    logits, _ = model(input_seq)  # logits: [1, total_length, vocab_size]

    # Compute log probabilities only for the candidate portion.
    prompt_length = prompt_tokens.size(1)
    candidate_logits = logits[:, prompt_length:, :]  # [1, candidate_length, vocab_size]
    log_probs = F.log_softmax(candidate_logits, dim=-1)

    # Gather the log-probabilities corresponding to the ground-truth candidate tokens.
    token_log_probs = log_probs.gather(dim=-1, index=candidate_tokens.unsqueeze(-1)).squeeze(
        -1)  # [1, candidate_length]

    # Sum over candidate tokens to get the total log-likelihood loss.
    loss = - token_log_probs.sum()
    return loss.item()


REGIONS = ["frontal", "motor_temporal", "parietal_occipital"]

def evaluate_multiclass_with_channels(
        model,  # the trained model
        shard_paths,  # list of shard file paths (e.g., ["shard_train_0.pt", "shard_train_1.pt", ...])
        device="cuda",
        segment_size=512,  # total candidate (completion) segment size (must be divisible by num_channels)
        prompt_stride=258  # stride used when sampling prompt/candidate offsets
):
    """
    For each shard in shard_paths, we perform an evaluation block as follows:
      - For the chosen (prompt) shard, sample a prompt by taking several random chunks
        (here, 4 chunks of 128 tokens each) from each channel.
      - Interleave the per–channel prompt blocks (so that the final prompt has shape
        [num_channels * (4*128)]).
      - Choose a correct candidate continuation from the same shard—a contiguous block
        from each channel (so that the total candidate length equals segment_size).
      - For every other shard, sample one candidate continuation similarly.
      - For each candidate, compute the summed negative log–likelihood loss on the candidate tokens,
        given the prompt.
      - The candidate with the lowest loss is considered the model's prediction.
      - Record the ground truth (the prompt’s shard index) and the predicted candidate’s shard index
        in order to build a confusion matrix.
    Returns the overall accuracy and prints the confusion matrix.
    """
    # Load all shards into memory.
    shards = []
    for path in shard_paths:
        loaded = torch.load(path, map_location="cpu")
        tokens_by_region = {}
        for region in REGIONS:
            if region not in loaded:
                raise ValueError(f"Shard {path} is missing channel {region}")
            tokens_by_region[region] = loaded[region].cpu()
        # Assume all channels have the same length.
        shard_length = tokens_by_region[REGIONS[0]].size(0)
        shards.append({
            'tokens': tokens_by_region,  # dict: region -> 1D tensor of tokens
            'length': shard_length,
            'path': path
        })
    num_shards = len(shards)
    # Initialize confusion matrix: rows = true (prompt) shard, columns = predicted candidate shard.
    confusion_matrix = np.zeros((num_shards, num_shards), dtype=int)

    total_evals = 0
    correct_count = 0

    # Parameters for prompt sampling.
    chunk_size = 258  # tokens per chunk per channel
    num_chunks = 2  # number of chunks sampled per channel for the prompt
    # For candidate: candidate_segment_size is given (total candidate length)
    # Per-channel candidate length:
    num_channels = len(REGIONS)
    if segment_size % num_channels != 0:
        raise ValueError("segment_size must be divisible by the number of channels")
    per_channel_candidate_length = segment_size // num_channels

    # Evaluate using each shard as the prompt source.
    for i, shard in enumerate(shards):
        tokens_by_region = shard['tokens']  # dict: region -> tensor
        len_i = shard['length']
        # Sample prompt chunks for each channel.
        valid_offsets = list(range(0, len_i - chunk_size + 1, prompt_stride))
        if not valid_offsets:
            print(f"Not enough tokens in shard {i} for a prompt. Skipping...")
            continue

        prompt_chunks_by_region = {}
        prompt_offsets_by_region = {}
        for region in REGIONS:
            # Sample num_chunks offsets (with replacement) from valid offsets.
            offsets = random.choices(valid_offsets, k=num_chunks)
            prompt_offsets_by_region[region] = offsets
            # For each offset, extract a chunk of size chunk_size.
            chunks = [tokens_by_region[region][off: off + chunk_size] for off in offsets]
            # Concatenate chunks for this region.
            prompt_chunks_by_region[region] = torch.cat(chunks, dim=0)  # shape: [num_chunks*chunk_size]

        # Interleave the per-channel prompt blocks.
        # Stack per-channel prompts into tensor of shape [num_channels, num_chunks*chunk_size]
        prompt_stack = torch.stack([prompt_chunks_by_region[region] for region in REGIONS], dim=0)
        # Transpose to [num_chunks*chunk_size, num_channels] then flatten to 1D.
        prompt_interleaved = prompt_stack.transpose(0, 1).reshape(-1)

        # --- Candidate sampling for the correct candidate (from the same shard) ---
        # For each region, define a helper to check overlap with prompt chunks.
        def region_overlaps(candidate_offset, offsets):
            for off in offsets:
                # Candidate block [candidate_offset, candidate_offset+per_channel_candidate_length]
                # overlaps with prompt chunk [off, off+chunk_size] if they are not completely separate.
                if not (candidate_offset + per_channel_candidate_length <= off or candidate_offset >= off + chunk_size):
                    return True
            return False

        candidate_offsets_by_region = {}
        valid_candidate_exists = True
        for region in REGIONS:
            valid_candidate_offsets = [
                c for c in range(0, len_i - per_channel_candidate_length + 1, prompt_stride)
                if not region_overlaps(c, prompt_offsets_by_region[region])
            ]
            if not valid_candidate_offsets:
                print(f"No valid candidate offsets in shard {i} for region {region}. Skipping this block...")
                valid_candidate_exists = False
                break
            candidate_offsets_by_region[region] = random.choice(valid_candidate_offsets)
        if not valid_candidate_exists:
            continue

        candidate_chunks_by_region = {}
        for region in REGIONS:
            off = candidate_offsets_by_region[region]
            candidate_chunks_by_region[region] = tokens_by_region[region][off: off + per_channel_candidate_length]
        # Interleave candidate tokens.
        candidate_stack = torch.stack([candidate_chunks_by_region[region] for region in REGIONS], dim=0)
        candidate_interleaved = candidate_stack.transpose(0, 1).reshape(-1)  # shape: [segment_size]

        # Build candidate_info list. First candidate is the correct one.
        candidate_info = []
        candidate_info.append({
            'tokens': candidate_interleaved,
            'source_shard': i,
            'label': 'correct'
        })
        # --- For every other shard, sample one candidate continuation. ---
        for j, other_shard in enumerate(shards):
            if j == i:
                continue
            len_j = other_shard['length']
            tokens_by_region_j = other_shard['tokens']
            candidate_chunks_j = {}
            for region in REGIONS:
                off = random.randint(0, len_j - per_channel_candidate_length)
                candidate_chunks_j[region] = tokens_by_region_j[region][off: off + per_channel_candidate_length]
            candidate_stack_j = torch.stack([candidate_chunks_j[region] for region in REGIONS], dim=0)
            candidate_interleaved_j = candidate_stack_j.transpose(0, 1).reshape(-1)
            candidate_info.append({
                'tokens': candidate_interleaved_j,
                'source_shard': j,
                'label': 'wrong'
            })

        # Evaluate each candidate using the helper.
        candidate_losses = []
        for candidate in candidate_info:
            loss = compute_completion_loss_with_channels(model, prompt_interleaved, candidate['tokens'], device=device)
            candidate_losses.append(loss)
        # The candidate with the lowest loss is chosen.
        min_loss_index = np.argmin(candidate_losses)
        chosen = candidate_info[min_loss_index]
        predicted_shard = chosen['source_shard']
        confusion_matrix[i, predicted_shard] += 1
        if chosen['label'] == 'correct':
            correct_count += 1
        total_evals += 1

        print(f"[Shard {i}] Candidate losses: {candidate_losses}")
        print(
            f" -> Correct candidate loss: {candidate_losses[0]:.4f} vs. others: {[f'{l:.4f}' for l in candidate_losses[1:]]}")
        print(f" -> Model selected candidate from shard {predicted_shard} (label: {chosen['label']})")

    if total_evals == 0:
        print("No evaluations were performed (possibly not enough tokens in the shards).")
        return 0.0

    accuracy = correct_count / total_evals
    print(f"\n[Multi-class Evaluation] Final Accuracy = {correct_count}/{total_evals} = {accuracy:.4f}")
    print("\nConfusion Matrix (rows: true prompt shard, columns: predicted candidate shard):")
    header = "      " + " ".join([f"Shd{j}" for j in range(num_shards)])
    print(header)
    for i in range(num_shards):
        row = " ".join([f"{confusion_matrix[i, j]:5d}" for j in range(num_shards)])
        print(f"Shd{i}: {row}")
    return accuracy


import os
import torch
import numpy as np

# Set device.
d = 'cuda'
device = torch.device(d)

# Load the checkpoint.
checkpoint = torch.load('checkpoints/model_06000.pt', map_location=device)

# The checkpoint's 'config' is already a GPTConfig instance.
config = checkpoint['config']

# Instantiate the model with the loaded config.
model = GPT(config).to(device)

# Load the state dict.
orig_sd = checkpoint['model_state_dict']  # note: new key name from save_checkpoint
fixed_sd = {}
for k, v in orig_sd.items():
    # Fix key names if necessary.
    new_key = k.replace("_orig_mod.", "")
    fixed_sd[new_key] = v
model.load_state_dict(fixed_sd, strict=True)

# Set the model's configuration attribute to the loaded config.
model.config = config

model.eval()

# Example: Evaluate over 10 epochs using three shards.
accs = []
epochs = 10
for epoch in range(epochs):
    print(f"\n=== Epoch {epoch+1}/{epochs} ===")
    acc = evaluate_multiclass_with_channels(
        model=model,
        shard_paths=[
            "local_shards_val/mydata_train_0.pt",
            "local_shards_val/mydata_train_9.pt",
            "local_shards_val/mydata_train_18.pt"
        ],
        device=d,
        segment_size=1032//2  # candidate continuation remains 512 tokens total
    )
    accs.append(acc)

mean_acc = np.mean(accs)
print(f"\nMean Accuracy over {epochs} epochs: {mean_acc}")
print(f"Accuracies per epoch: {accs}")
