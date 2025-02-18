import os
import random
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# Import your checkpoint manager functions.
from checkpoint_manager import load_checkpoint, save_checkpoint

#############################################
# Global Constants & Regions (Channels)
#############################################

REGIONS = ["frontal", "motor_temporal", "parietal_occipital"]


#############################################
# Dataset: Load Shards for Classification
#############################################

class ShardClassificationDataset(Dataset):
    """
    Loads tokens from shard files.

    Each shard file corresponds to one class and contains a dictionary
    mapping region names to a 1D tensor of token IDs.

    For each shard, contiguous segments of length T are extracted (using a given stride)
    and the shard’s index is used as the class label.

    The entire dataset is built here; later, we split it into training and validation sets.
    """

    def __init__(self, shard_paths, T, stride=None):
        """
        Args:
            shard_paths (list): List of paths to shard files (one per class).
            T (int): Sequence length (number of tokens per sample).
            stride (int): Step size for sampling segments (default: T, i.e. non-overlapping).
        """
        self.T = T
        self.stride = stride if stride is not None else T
        self.samples = []  # Each sample is a tuple: (shard_idx, start_index, label)
        self.shard_data = []  # Loaded shard data (one per class)

        for i, path in enumerate(shard_paths):
            data = torch.load(path, map_location="cpu")
            # Verify that each expected region exists.
            for region in REGIONS:
                if region not in data:
                    raise ValueError(f"Shard {path} is missing region '{region}'")
            self.shard_data.append(data)
            # Assume all regions in a shard have the same total length.
            total_length = self.shard_data[-1][REGIONS[0]].size(0)
            indices = list(range(0, total_length - T, self.stride))
            for start_idx in indices:
                # Use the shard index as the class label.
                self.samples.append((i, start_idx, i))
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        shard_idx, start_idx, label = self.samples[idx]
        channels = []
        # For each region, extract a segment of length T.
        for region in REGIONS:
            tokens = self.shard_data[shard_idx][region]
            segment = tokens[start_idx: start_idx + self.T]
            channels.append(segment.unsqueeze(0))  # Shape: [1, T]
        # Stack channels: final shape [num_channels, T]
        x = torch.cat(channels, dim=0)
        return x, label


#############################################
# Model Components
#############################################



#########################
# DDP Setup
#########################
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "CUDA is required for DDP"
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
    def __init__(self, n_embd, num_heads=6, scales=[1, 2]):
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
    block_size: int = 1024
    vocab_size: int = 10799
    # Small model configuration
    # n_layer: int = 12
    # # n_head: int = 12
    # # n_embd: int = 768

    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

    # n_layer: int = 12
    # n_head: int = 16
    # n_embd: int = 1024
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
            nn.Sequential(Block(config), Block(config))
            for _ in range(config.num_channels)
        ])

        # Use the new multi-scale cross-channel fusion.
        self.cross_channel_fusion = MultiScaleCrossChannelFusion(config.n_embd, num_heads=2, scales=[1, 2])

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
            idx: [B, num_channels, time_steps] token IDs.
            targets: [B, num_channels] with the target token for each channel
                     (the token immediately following the input sequence).
        Returns:
            logits: [B, num_channels, vocab_size] — prediction for the next token per channel.
            loss: cross-entropy loss if targets is provided, else None.
        """
        # Unpack dimensions.
        B, C, T = idx.size()
        assert C == self.config.num_channels, f"Expected {self.config.num_channels} channels, but got {C}"
        time_steps = T  # number of time steps per channel

        # 1. Token Embedding and Reshape:
        # The embedding layer now expects [B, num_channels, time_steps] and returns [B, num_channels, time_steps, n_embd]
        tok_emb = self.transformer.wte(idx)  # [B, num_channels, time_steps, n_embd]
        tok_emb = tok_emb.transpose(1, 2)  # -> [B, time_steps, num_channels, n_embd]
        x = tok_emb  # [B, time_steps, num_channels, n_embd]

        # 2. Add Positional Embeddings (along the time dimension):
        pos = torch.arange(time_steps, device=x.device).unsqueeze(0)  # [1, time_steps]
        pos_emb = self.transformer.wpe(pos)  # [1, time_steps, n_embd]
        x = x + pos_emb.unsqueeze(2)  # broadcast -> [B, time_steps, num_channels, n_embd]

        # (Optional) Add channel embeddings if desired.
        # For example:
        # channel_ids = torch.arange(self.config.num_channels, device=x.device).unsqueeze(0).unsqueeze(0)
        # channel_emb = self.channel_embedding(channel_ids)  # [1, 1, num_channels, n_embd]
        # x = x + channel_emb

        # 3. Per-Channel Encoding:
        channel_outs = []
        for c in range(self.config.num_channels):
            x_c = x[:, :, c, :]  # Extract channel c: [B, time_steps, n_embd]
            x_c = self.channel_encoder[c](x_c)  # Process with that channel's encoder (3 blocks)
            channel_outs.append(x_c)
        # Stack channels back: [B, time_steps, num_channels, n_embd]
        x = torch.stack(channel_outs, dim=2)

        # 4. Cross-Channel Fusion:
        # This module expects input of shape [B, time_steps, num_channels, n_embd]
        # and outputs a fused representation per time step: [B, time_steps, n_embd]
        fused = self.cross_channel_fusion(x)
        # Broadcast the fused representation across channels and add it.
        x = x + fused.unsqueeze(2)  # [B, time_steps, num_channels, n_embd]

        # 5. Final Transformer Blocks with Causal Masking (process each channel separately):
        # Rearrange to process each channel's time sequence independently.
        x = x.transpose(1, 2)  # -> [B, num_channels, time_steps, n_embd]
        B, C, T, E = x.size()
        x = x.reshape(B * C, T, E)  # -> [B * num_channels, time_steps, n_embd]
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)  # -> [B * num_channels, time_steps, n_embd]

        # 6. Use only the last time step's representation for next-token prediction.
        x_last = x[:, -1, :]  # [B * num_channels, n_embd]
        logits = self.lm_head(x_last)  # [B * num_channels, vocab_size]
        logits = logits.view(B, C, -1)  # Reshape to [B, num_channels, vocab_size]

        # 7. Compute Loss:
        loss = None
        if targets is not None:
            # targets should be of shape [B, num_channels]
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



        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)} with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)} with {num_nodecay_params:,} parameters")
            print(f"Using fused AdamW: false")

        optimizer = torch.optim.AdamW(
            optim_groups,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=False
        )
        return optimizer



class GPTForClassification(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.gpt = GPT(config)
        self.classifier = nn.Linear(config.n_embd, num_classes)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, idx, labels=None):
        """
        idx: [B, num_channels, T]
        labels: [B] (class indices)
        """
        B, C, T = idx.size()
        tok_emb = self.gpt.wte(idx)  # [B, C, T, n_embd]
        tok_emb = tok_emb.transpose(1, 2)  # [B, T, C, n_embd]
        x = tok_emb
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        pos_emb = self.gpt.wpe(pos)
        x = x + pos_emb.unsqueeze(2)
        channel_outs = []
        for c in range(self.gpt.config.num_channels):
            x_c = x[:, :, c, :]
            x_c = self.gpt.channel_encoder[c](x_c)
            channel_outs.append(x_c)
        x = torch.stack(channel_outs, dim=2)  # [B, T, C, n_embd]
        fused = self.gpt.cross_channel_fusion(x)  # [B, T, n_embd]
        x = x + fused.unsqueeze(2)
        x = x.transpose(1, 2)  # [B, C, T, n_embd]
        B, C, T, E = x.size()
        x = x.reshape(B * C, T, E)
        for block in self.gpt.h:
            x = block(x)
        x = self.gpt.ln_f(x)
        x_last = x[:, -1, :]  # [B * C, n_embd]
        x_last = x_last.view(B, C, -1)  # [B, C, n_embd]
        pooled = x_last.mean(dim=1)  # [B, n_embd]
        logits = self.classifier(pooled)  # [B, num_classes]
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return logits, loss


#############################################
# Training & Evaluation Functions
#############################################

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, labels in dataloader:
        x = x.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, labels in dataloader:
            x = x.to(device)
            labels = labels.to(device)
            logits, _ = model(x, labels)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


#############################################
# Main: Fine-Tune & Test
#############################################

def main():
    import os
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split

    # Hyperparameters.
    num_classes = 3
    T = 1024  # Sequence length.
    batch_size = 16
    learning_rate = 1e-4
    num_epochs = 5
    val_pct = 0.2  # 20% holdout for validation.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Shard file paths (one per class).
    shard_paths = [
        "./local_shards_val/mydata_train_0.pt",
        "./local_shards_val/mydata_train_1.pt",
        "./local_shards_val/mydata_train_2.pt"
    ]

    # Load the entire dataset from shards.
    full_dataset = ShardClassificationDataset(shard_paths, T=T, stride=T)
    total_samples = len(full_dataset)
    train_size = int((1 - val_pct) * total_samples)
    val_size = total_samples - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Total samples: {total_samples}, Train: {train_size}, Val: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model configuration.
    config = GPTConfig()

    # --------- Step 1: Load raw GPT model from checkpoint -----------
    raw_gpt = GPT(config)
    raw_gpt.to(device)
    optimizer_raw = optim.AdamW(raw_gpt.parameters(), lr=learning_rate)
    checkpoint_path = "./checkpoints/model_01000.pt"  # Update filename as needed.
    if os.path.exists(checkpoint_path):
        # Load checkpoint into raw_gpt
        checkpoint = load_checkpoint(checkpoint_path, model=raw_gpt, optimizer=optimizer_raw, device=device)
        raw_gpt.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print(
            f"Raw GPT model state loaded from {checkpoint_path} at step {checkpoint['step']} with val loss {checkpoint['val_loss']}")
        try:
            optimizer_raw.load_state_dict(checkpoint['optimizer_state_dict'])
        except ValueError as e:
            print("Warning: Optimizer state dict mismatch, skipping optimizer state load.")
    else:
        print("Checkpoint not found; training from scratch.")

    # --------- Step 2: Wrap the pretrained GPT into the classification model -----------
    classification_model = GPTForClassification(config, num_classes=num_classes)
    classification_model.to(device)
    # Replace the internal GPT module with our loaded raw_gpt.
    classification_model.gpt = raw_gpt

    # Create optimizer for classification model.
    optimizer = optim.AdamW(classification_model.parameters(), lr=learning_rate)

    # --------- Fine-tuning loop -----------
    for epoch in range(num_epochs):
        train_loss = train_epoch(classification_model, train_loader, optimizer, device)
        val_acc = evaluate(classification_model, val_loader, device)
        print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss = {train_loss:.4f}, Val Accuracy = {val_acc:.4f}")
        # Save checkpoint after each epoch.
        save_checkpoint(model=classification_model, optimizer=optimizer, config=config, step=epoch,
                        val_loss=1 - val_acc, log_dir="./checkpoints")

    # Save the final fine-tuned classification model.
    torch.save(classification_model.state_dict(), "gpt_classification_finetuned.pth")
    print("Saved fine-tuned classification model.")

    # --------- Testing: Run the model on one validation batch -----------
    classification_model.eval()
    x, labels = next(iter(val_loader))
    x = x.to(device)
    logits, _ = classification_model(x)
    preds = logits.argmax(dim=1)
    print("Test Batch - True Labels:", labels)
    print("Test Batch - Predicted:", preds.cpu().numpy())


if __name__ == "__main__":
    main()
