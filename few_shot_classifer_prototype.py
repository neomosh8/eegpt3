import torch
import torch.nn.functional as F
import random
import os
import glob
import math
import time
import inspect
from dataclasses import dataclass
import contextlib

import torch.nn as nn
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from checkpoint_manager import save_checkpoint
from handle_tokenized import upload_folder_to_s3
from lr_test import CustomLRScheduler
from plotter import LossPlotter

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


#########################
# Model Components
#########################
class SimpleCrossChannelFusion(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        reduced_dim = 16
        self.proj_reduce = nn.Linear(n_embd, reduced_dim)
        self.proj_expand = nn.Linear(reduced_dim, n_embd)
        self.ln = nn.LayerNorm(n_embd)

    def forward(self, x):
        B, T, C, E = x.size()
        fused = x.mean(dim=2, keepdim=True)  # [B, T, 1, E]
        fused = self.proj_reduce(fused)
        fused = self.proj_expand(fused)
        x = x + fused.expand_as(x)
        return self.ln(x)


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
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


# Using RoPE in attention for positional encoding
class CausalSelfAttentionWithRoPE(nn.Module):
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
        seq_len = config.block_size
        head_dim = config.n_embd // config.n_head
        freqs = torch.arange(0, head_dim, 2).float() / head_dim
        theta = 1000.0 ** (-freqs)
        positions = torch.arange(seq_len).float()
        angles = positions[:, None] * theta[None, :]
        self.register_buffer('cos', angles.cos())
        self.register_buffer('sin', angles.sin())

    def apply_rotary_emb(self, x, T):
        B, n_head, T, head_dim = x.shape
        x1, x2 = x[..., 0::2], x[..., 1::2]
        cos = self.cos[:T, :][None, None, :, :]
        sin = self.sin[:T, :][None, None, :, :]
        x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rot

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        q = self.apply_rotary_emb(q, T)
        k = self.apply_rotary_emb(k, T)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = self.attn_dropout(y)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttentionWithRoPE(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class BlockWithFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttentionWithRoPE(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.fusion = SimpleCrossChannelFusion(config.n_embd)
        self.ln_3 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        B, T, C, E = x.size()
        x_reshaped = x.contiguous().view(B * C, T, E)
        x_reshaped = x_reshaped + self.attn(self.ln_1(x_reshaped))
        x = x_reshaped.view(B, T, C, E)
        x = x + self.fusion(self.ln_2(x))
        x_reshaped = x.contiguous().view(B * C, T, E)
        x_reshaped = x_reshaped + self.mlp(self.ln_3(x_reshaped))
        x = x_reshaped.view(B, T, C, E)
        return x


class CrossChannelFusion(nn.Module):
    def __init__(self, n_embd, num_heads=1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=n_embd, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        B, T, C, E = x.size()
        x = x.view(B, T * C, E)
        fused, _ = self.attn(x, x, x)
        fused = fused.view(B, T, C, E)
        return fused


@dataclass
class GPTConfig:
    block_size: int = 2048
    vocab_size: int = 82
    n_layer: int = 16  # Moderate depth
    n_head: int = 32  # Number of attention heads
    n_embd: int = 2048  # Embedding dimension
    num_channels: int = 3
    mlp_dropout: float = 0.05
    attn_dropout: float = 0.02
    resid_dropout: float = 0.05
    pad_token: int = 0


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([BlockWithFusion(config) for _ in range(config.n_layer)]),
            "ln_f": nn.LayerNorm(config.n_embd)
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # Shared intra-channel encoder (replaces per-channel encoder)
        self.intra_channel_encoder = nn.Sequential(
            Block(config),
            Block(config),
            Block(config),
        )

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
        # Standard forward pass for language modeling
        B, C, T = idx.size()
        tok_emb = self.transformer.wte(idx)  # [B, C, T, n_embd]
        x = tok_emb.transpose(1, 2)  # [B, T, C, n_embd]
        x_reshaped = x.permute(0, 2, 1, 3).contiguous().reshape(B * C, T, self.config.n_embd)
        out = self.intra_channel_encoder(x_reshaped)
        x = out.view(B, C, T, self.config.n_embd).permute(0, 2, 1, 3).contiguous()
        for block in self.transformer.h:
            x = block(x)
        x = x.transpose(1, 2).contiguous().reshape(B * C, T, self.config.n_embd)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        logits = logits.view(B, C, T, -1)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

    def configure_optimizer(self, weight_decay, learning_rate, device):
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
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and ('cuda' in device)
        if master_process:
            print(f"num decayed parameters: {sum(p.numel() for p in decay_params):,}")
            print(f"num non-decayed parameters: {sum(p.numel() for p in nodecay_params):,}")
            print(f"Using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups,
            betas=(0.95, 0.999),
            eps=1e-8,
            fused=use_fused
        )
        return optimizer

    # --- New method to extract features for downstream evaluation ---
    def extract_features(self, idx):
        """
        Given input tokens (shape [B, C, T]), returns a feature vector for each sequence.
        Here we average over the sequence dimension after the final layer norm.
        """
        B, C, T = idx.size()
        tok_emb = self.transformer.wte(idx)  # [B, C, T, n_embd]
        x = tok_emb.transpose(1, 2)  # [B, T, C, n_embd]
        x_reshaped = x.permute(0, 2, 1, 3).contiguous().reshape(B * C, T, self.config.n_embd)
        out = self.intra_channel_encoder(x_reshaped)
        x = out.view(B, C, T, self.config.n_embd).permute(0, 2, 1, 3).contiguous()
        for block in self.transformer.h:
            x = block(x)
        x = x.transpose(1, 2).contiguous().reshape(B * C, T, self.config.n_embd)
        x = self.transformer.ln_f(x)
        # Average over sequence tokens
        features = x.mean(dim=1)  # [B * C, n_embd]
        # Reshape back to [B, n_embd] (assuming C=1 for downstream tasks)
        features = features.view(B, -1)
        return features


#########################
# DataLoader (All-In-Memory)
#########################
REGIONS = ["frontal", "motor_temporal", "parietal_occipital"]


class DataLoaderLiteAllInMemory:
    def __init__(self, B: int, T: int, process_rank: int, num_processes: int,
                 local_data_dir: str = "./local_shards", shard_prefix: str = "mydata",
                 split: str = "train", shuffle_shards: bool = False, pad_token: int = 0):
        self.B = B
        self.per_channel_length = T
        self.num_channels = len(REGIONS)
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.pad_token = pad_token
        pattern = os.path.join(local_data_dir, f"{shard_prefix}_{split}_*.pt")
        self.shard_files = sorted(glob.glob(pattern))
        if not self.shard_files:
            raise ValueError(f"No {split} shards found in {local_data_dir} with prefix {shard_prefix}_{split}_")
        if shuffle_shards:
            random.shuffle(self.shard_files)
        self.tokens = {region: [] for region in REGIONS}
        for shard_path in self.shard_files:
            loaded = torch.load(shard_path, map_location="cpu", weights_only=False)
            for region in REGIONS:
                if region not in loaded:
                    available_regions = list(loaded.keys())
                    if available_regions:
                        loaded[region] = loaded[available_regions[0]]
                        print(f"Warning: Shard {shard_path} missing {region}, using {available_regions[0]}.")
                    else:
                        raise ValueError(f"Shard {shard_path} has no channels for {region}.")
                self.tokens[region].append(loaded[region])
        for region in REGIONS:
            self.tokens[region] = torch.cat(self.tokens[region], dim=0)
        min_length = min(t.size(0) for t in self.tokens.values())
        required_length = self.B * self.per_channel_length * self.num_processes
        if min_length < required_length:
            print(
                f"Warning: Shortest channel has {min_length} tokens, less than required {required_length}. Padding will be used.")
        self.start_ptr = self.B * self.per_channel_length * self.process_rank
        self.ptr = self.start_ptr

    def _get_slice(self, token_tensor: torch.Tensor, start: int, length: int, pad_value: int) -> torch.Tensor:
        total_length = token_tensor.size(0)
        start = start % total_length
        end = start + length
        if end <= total_length:
            return token_tensor[start:end]
        else:
            first_part = token_tensor[start:]
            remaining = length - first_part.size(0)
            second_part = token_tensor[:remaining]
            return torch.cat((first_part, second_part), dim=0)

    def next_batch(self):
        inputs_list = []
        targets_list = []
        for region in REGIONS:
            token_tensor = self.tokens[region]
            channel_inputs = []
            channel_targets = []
            for b in range(self.B):
                start = self.ptr + b * self.per_channel_length
                seq = self._get_slice(token_tensor, start, self.per_channel_length, pad_value=self.pad_token)
                channel_inputs.append(seq.unsqueeze(0))
                target = self._get_slice(token_tensor, start + 1, self.per_channel_length, pad_value=-100)
                channel_targets.append(target.unsqueeze(0))
            channel_inputs = torch.cat(channel_inputs, dim=0)
            channel_targets = torch.cat(channel_targets, dim=0)
            inputs_list.append(channel_inputs.unsqueeze(1))
            targets_list.append(channel_targets.unsqueeze(1))
        inputs = torch.cat(inputs_list, dim=1)
        targets = torch.cat(targets_list, dim=1)
        self.ptr += self.B * self.per_channel_length * self.num_processes
        return inputs, targets

    def reset(self):
        self.ptr = self.start_ptr

    @property
    def total_len(self):
        return self.tokens[REGIONS[0]].size(0)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch()


#########################
# Utility Functions for Transfer Evaluation
#########################
def sample_sequence(token_tensor, seq_len, pad_token=0):
    """
    Randomly sample a contiguous sequence of length seq_len from token_tensor.
    """
    total_length = token_tensor.size(0)
    start = random.randint(0, total_length - seq_len)
    return token_tensor[start: start + seq_len]


def evaluate_linear_probe(model, class_tokens, seq_len=512, num_train=100, num_test=20, num_epochs=50):
    """
    For each class (assumed to be represented by a token tensor in class_tokens),
    sample sequences to create training and test sets, extract features using the model,
    and train a simple linear classifier.
    """
    model.eval()
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []

    for label, token_tensor in class_tokens.items():
        for _ in range(num_train):
            seq = sample_sequence(token_tensor, seq_len, pad_token=0)
            # Shape: [1, 1, seq_len]
            seq = seq.unsqueeze(0).unsqueeze(0).to(device)
            feat = model.extract_features(seq)  # [1, n_embd]
            train_features.append(feat.squeeze(0))
            train_labels.append(label)
        for _ in range(num_test):
            seq = sample_sequence(token_tensor, seq_len, pad_token=0)
            seq = seq.unsqueeze(0).unsqueeze(0).to(device)
            feat = model.extract_features(seq)
            test_features.append(feat.squeeze(0))
            test_labels.append(label)

    train_features = torch.stack(train_features)
    train_labels = torch.tensor(train_labels, device=device)
    test_features = torch.stack(test_features)
    test_labels = torch.tensor(test_labels, device=device)

    n_embd = train_features.size(1)
    num_classes = len(class_tokens)
    classifier = nn.Linear(n_embd, num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print("Starting linear probe training...")
    for epoch in range(num_epochs):
        classifier.train()
        optimizer.zero_grad()
        logits = classifier(train_features)
        loss = loss_fn(logits, train_labels)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            classifier.eval()
            with torch.no_grad():
                logits_test = classifier(test_features)
                preds = torch.argmax(logits_test, dim=1)
                acc = (preds == test_labels).float().mean().item()
                print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item():.4f}, Test Accuracy: {acc * 100:.2f}%")
    classifier.eval()
    with torch.no_grad():
        logits_test = classifier(test_features)
        preds = torch.argmax(logits_test, dim=1)
        acc = (preds == test_labels).float().mean().item()
        print(f"Final Linear Probe Test Accuracy: {acc * 100:.2f}%")
    return acc


def plot_tsne_features(model, class_tokens, seq_len=512, num_samples_per_class=50):
    """
    Extract features from a few samples per class and use t-SNE to visualize them.
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import numpy as np

    model.eval()
    features = []
    labels = []
    for label, token_tensor in class_tokens.items():
        for _ in range(num_samples_per_class):
            seq = sample_sequence(token_tensor, seq_len, pad_token=0)
            seq = seq.unsqueeze(0).unsqueeze(0).to(device)
            feat = model.extract_features(seq)
            features.append(feat.squeeze(0).cpu().numpy())
            labels.append(label)
    features = np.stack(features)
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title("t-SNE of Model Representations")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig('goh.png')


#########################
# Load Class Data (for downstream evaluation)
#########################
# Here each shard file corresponds to one class.
shard_paths = [
    "./local_shards_val/mydata_train_2.pt",
    "./local_shards_val/mydata_train_0.pt",
]

class_tokens = {}
for i, shard_path in enumerate(shard_paths):
    if not os.path.exists(shard_path):
        raise FileNotFoundError(f"Shard file {shard_path} not found.")
    data = torch.load(shard_path, map_location="cpu")
    if "frontal" in data:
        tokens = data["frontal"]
    else:
        tokens = next(iter(data.values()))
    class_tokens[i] = tokens

#########################
# Initialize the Model
#########################
config = GPTConfig()
model = GPT(config).to(device)

# --- Optionally load a pretrained checkpoint ---
checkpoint_path = "checkpoints/model_last_checkpoint.pt"
if os.path.exists(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        fixed_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(fixed_state_dict)
        model.eval()
        print("Loaded pretrained weights.")
    except Exception as e:
        print("Error loading pretrained weights:", e)
else:
    print(f"Checkpoint {checkpoint_path} not found. Using randomly initialized model.")

#########################
# Evaluate Transferable Knowledge
#########################
print("\n=== Linear Probe Evaluation ===")
_ = evaluate_linear_probe(model, class_tokens, seq_len=512, num_train=100, num_test=20, num_epochs=50)

print("\n=== t-SNE Representation Analysis ===")
plot_tsne_features(model, class_tokens, seq_len=512, num_samples_per_class=50)
