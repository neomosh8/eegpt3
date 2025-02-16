import os
import glob
import random
import torch
import torch.nn.functional as F
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

from checkpoint_manager import save_checkpoint, load_checkpoint
# assumed available; replace or remove if not using S3 logging
from handle_tokenized import upload_folder_to_s3
from plotter import LossPlotter

# Assume REGIONS is defined as in your training code:
REGIONS = ["frontal", "motor_temporal", "parietal_occipital"]
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


#########################

class ForcedChoiceClassifier:
    def __init__(self, model, device, data_dir, sequence_length=1032):
        """
        Args:
            model: A trained GPT model.
            device: Device (e.g., 'cuda:0' or 'cpu') on which to run the model.
            data_dir: Directory containing the .pt files.
            sequence_length: Total sequence length (must be divisible by the number of channels).
                             For few-shot, we'll interpret this as the sum of 3 prompt segments (each T/4)
                             and 1 candidate segment (T/4).
        """
        self.model = model
        self.device = device
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        # For few-shot, define segment length as one quarter of the sequence.
        self.segment_length = sequence_length // 4

        # Find all .pt files in the provided directory.
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        if not self.file_paths:
            raise ValueError(f"No .pt files found in directory {data_dir}")

        # Preload interleaved token sequences from each file.
        self.file_tokens = {}
        for path in self.file_paths:
            self.file_tokens[path] = self.load_interleaved_tokens(path)
        print(f"Loaded {len(self.file_tokens)} files for forced choice evaluation.")

    def load_interleaved_tokens(self, file_path, T_total=None):
        """
        Loads tokens from a single shard file, extracts a contiguous block
        from each channel, and interleaves them to form a single sample.
        (Here we use the same logic as in training.)

        If T_total is not provided, we use the entire sequence in the file.

        Returns:
            A tensor of shape [1, T_total] containing interleaved tokens.
        """
        data = torch.load(file_path, map_location="cpu")
        tokens = {}
        for region in REGIONS:
            if region not in data:
                raise ValueError(f"File {file_path} is missing channel {region}")
            tokens[region] = data[region]
        lengths = [t.numel() for t in tokens.values()]
        if len(set(lengths)) != 1:
            raise ValueError(f"Channel lengths differ in {file_path}")
        total_length_per_channel = lengths[0]
        num_channels = len(REGIONS)
        # If no T_total is given, use all tokens interleaved.
        if T_total is None:
            T_total = total_length_per_channel * num_channels

        if T_total % num_channels != 0:
            raise ValueError("T_total must be divisible by the number of channels")
        L = T_total // num_channels  # tokens per channel to extract

        max_start = total_length_per_channel - L
        if max_start < 0:
            raise ValueError(f"Not enough tokens in each channel in {file_path} (need at least {L})")
        start = random.randint(0, max_start)

        blocks = []
        for region in REGIONS:
            block = tokens[region][start: start + L]
            blocks.append(block)
        stacked = torch.stack(blocks, dim=0)
        interleaved = stacked.t().reshape(-1)
        return interleaved.unsqueeze(0)

    def compute_completion_logprob(self, prompt, candidate):
        """
        Computes the total log probability of the candidate tokens conditioned on the prompt.
        """
        # Ensure both prompt and candidate are 1D.
        prompt = prompt.view(-1)
        candidate = candidate.view(-1)
        full_seq = torch.cat([prompt, candidate], dim=0).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(full_seq)
        pl = prompt.size(0)
        cl = candidate.size(0)
        candidate_logits = logits[0, pl - 1: pl - 1 + cl, :]
        log_probs = F.log_softmax(candidate_logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=candidate.to(self.device).unsqueeze(-1)).squeeze(-1)
        total_log_prob = token_log_probs.sum()
        return total_log_prob.item()

    def evaluate_few_shot(self, num_samples=100):
        """
        Runs few-shot forced-choice evaluation. For each sample:
          - From a randomly chosen file, sample 3 random non-overlapping segments (each of length T/4)
            and concatenate them (in order) to form the prompt.
          - Also sample a candidate (completion) segment of length T/4 from a non-overlapping region.
          - For the "wrong" candidate, sample a segment of length T/4 from a different file.
          - Compare log probabilities and count as correct if the prompt yields a higher log likelihood
            for the correct candidate.
        """
        correct_count = 0
        total_evaluated = 0
        seg_len = self.segment_length  # T/4

        for i in range(num_samples):
            # ----- Sample a correct example from one file -----
            correct_file = random.choice(self.file_paths)
            tokens_correct = self.file_tokens[correct_file]  # shape [1, T_total_file]
            file_length = tokens_correct.size(1)
            if file_length < seg_len:
                continue

            # Sample 3 random segments for the prompt.
            # We sample start indices (0 to file_length - seg_len) without replacement.
            if file_length - seg_len < 3:
                continue  # not enough space to sample 3 segments
            prompt_indices = random.sample(range(0, file_length - seg_len + 1), 3)
            prompt_indices.sort()  # so the prompt is in order
            prompt_segments = [tokens_correct[0, start:start + seg_len] for start in prompt_indices]
            prompt = torch.cat(prompt_segments, dim=0)  # shape: [3 * seg_len]

            # Now sample a candidate segment (the "correct" completion) that does not overlap with any prompt segments.
            attempts = 0
            candidate_start = None
            while attempts < 100:
                cand_start = random.randint(0, file_length - seg_len)
                overlap = False
                for start in prompt_indices:
                    # Check if the candidate interval [cand_start, cand_start+seg_len) overlaps with [start, start+seg_len).
                    if not (cand_start + seg_len <= start or cand_start >= start + seg_len):
                        overlap = True
                        break
                if not overlap:
                    candidate_start = cand_start
                    break
                attempts += 1
            if candidate_start is None:
                continue  # skip this sample if no candidate found
            correct_completion = tokens_correct[0, candidate_start:candidate_start + seg_len]

            # ----- Sample a wrong candidate from a different file -----
            wrong_file = random.choice([fp for fp in self.file_paths if fp != correct_file])
            tokens_wrong = self.file_tokens[wrong_file]
            if tokens_wrong.size(1) < seg_len:
                continue
            wrong_start = random.randint(0, tokens_wrong.size(1) - seg_len)
            wrong_completion = tokens_wrong[0, wrong_start:wrong_start + seg_len]

            # ----- Compute log probabilities given the prompt.
            logprob_correct = self.compute_completion_logprob(prompt, correct_completion)
            logprob_wrong = self.compute_completion_logprob(prompt, wrong_completion)

            predicted_correct = logprob_correct > logprob_wrong
            if predicted_correct:
                correct_count += 1
            total_evaluated += 1

            print(f"Few-shot sample {i + 1}:")
            print(f"  Correct candidate logprob: {logprob_correct:.4f}")
            print(f"  Wrong candidate   logprob: {logprob_wrong:.4f}")
            print(f"  Model choice: {'Correct' if predicted_correct else 'Wrong'}\n")

        if total_evaluated == 0:
            print("No few-shot samples were evaluated. Check that your files have enough tokens.")
            return 0.0

        accuracy = (correct_count / total_evaluated) * 100
        print(f"Few-shot Forced Choice Accuracy over {total_evaluated} samples: {accuracy:.2f}%")
        return accuracy


# -------------------------------------------
# Example usage:
# (This snippet assumes you have already instantiated and/or trained your GPT model.)
# -------------------------------------------

if __name__ == "__main__":
    # Set the device.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Instantiate the model and move it to the correct device.
    config = GPTConfig()  # Ensure this configuration matches the one used during training.
    model = GPT(config)
    model.to(device)
    base_lr = 6e-4

    # Create an optimizer that matches the one used during training.
    # Using model.configure_optimizer(...) so that parameter groups match.
    optimizer = model.configure_optimizer(weight_decay=0.1, learning_rate=base_lr, device=device)

    # Specify the path to the checkpoint file you want to load.
    checkpoint_path = "./checkpoints/model_06000.pt"  # Update the filename as needed.

    # Load the checkpoint.
    checkpoint = load_checkpoint(checkpoint_path, model=model, optimizer=optimizer, device=device)

    # Fix the keys in the state_dict by removing the '_orig_mod.' prefix.
    orig_sd = checkpoint['model_state_dict']
    fixed_sd = {}
    for k, v in orig_sd.items():
        new_key = k.replace("_orig_mod.", "")
        fixed_sd[new_key] = v
    model.load_state_dict(fixed_sd, strict=True)

    print(f"Loaded checkpoint from {checkpoint_path} at step {checkpoint['step']} with val loss {checkpoint['val_loss']}")

    # If using DDP, unwrap the model (for now, we assume a non-DDP setting).
    model_for_eval = model  # or model.module if using DDP

    # Set the model to evaluation mode.
    model_for_eval.eval()

    # Instantiate the forced-choice classifier.
    # Here we assume the token shards for evaluation are in "./local_shards_val" (adjust as needed).
    fc_classifier = ForcedChoiceClassifier(
        model=model_for_eval,
        device=device,
        data_dir="./local_shards_val",
        sequence_length=1032
    )

    # Run forced-choice evaluation over a desired number of samples.
    fc_classifier.evaluate(num_samples=10)