import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import random
import os
import glob

# REGIONS for channel mapping (example, adjust as per your data)
REGIONS = ["frontal", "motor_temporal", "parietal_occipital"]  # Define your regions


# Model Components
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
        fused = self.proj_reduce(fused)  # [B, T, 1, reduced_dim]
        fused = self.proj_expand(fused)  # [B, T, 1, E]
        x = x + fused.expand_as(x)
        return self.ln(x)


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

    def apply_rotary_emb(self, x, seq_len):
        head_dim = x.size(-1)
        freqs = torch.arange(0, head_dim, 2, device=x.device).float() / head_dim
        theta = 1000.0 ** (-freqs)
        positions = torch.arange(seq_len, device=x.device).float()
        angles = positions[:, None] * theta[None, :]
        sin = angles.sin()
        cos = angles.cos()
        x1, x2 = x[..., 0::2], x[..., 1::2]
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
        x_reshaped = x.view(B * C, T, E)
        x_reshaped = x_reshaped + self.attn(self.ln_1(x_reshaped))
        x = x_reshaped.view(B, T, C, E)
        x = x + self.fusion(self.ln_2(x))
        x_reshaped = x.view(B * C, T, E)
        x_reshaped = x_reshaped + self.mlp(self.ln_3(x_reshaped))
        x = x_reshaped.view(B, T, C, E)
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 82
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    num_channels: int = len(REGIONS)
    mlp_dropout: float = 0.00
    attn_dropout: float = 0.00
    resid_dropout: float = 0.00
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
        B, C, T = idx.size()
        tok_emb = self.transformer.wte(idx)
        x = tok_emb.transpose(1, 2)
        channel_outs = []
        for c in range(C):
            x_c = x[:, :, c, :]
            x_c = self.intra_channel_encoder(x_c)
            channel_outs.append(x_c)
        x = torch.stack(channel_outs, dim=2)
        for block in self.transformer.h:
            x = block(x)
        x = x.transpose(1, 2).reshape(B * C, T, self.config.n_embd)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        logits = logits.view(B, C, T, -1)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

    def get_embeddings(data, batch_size):
        embeddings = []
        labels = []
        with torch.no_grad():  # Prevent gradient tracking
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                sequences = torch.stack([seq for seq, _ in batch], dim=0).to(device)
                batch_emb = model.get_embedding(sequences).detach()
                embeddings.append(batch_emb)
                labels.extend([label for _, label in batch])
        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.tensor(labels, device=device)
        return embeddings, labels


# Few-Shot Classification Functions
def load_fewshot_data(shard_paths, T=1024, K=3, pad_token=0, num_channels=len(REGIONS)):
    """
    Load few-shot data from shard files, splitting into support and query sets.

    Args:
        shard_paths (list): List of paths to .pt shard files, one per class.
        T (int): Sequence length per channel.
        K (int): Number of support samples per class.
        pad_token (int): Token to pad sequences.
        num_channels (int): Number of channels.

    Returns:
        support_data (list): List of (sequence, label) tuples for support set.
        query_data (list): List of (sequence, label) tuples for query set.
    """
    if not shard_paths:
        raise ValueError("No shard paths provided.")

    all_sequences = []
    min_num_sequences = float('inf')

    for label, shard_path in enumerate(shard_paths):
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Shard file not found: {shard_path}")

        loaded = torch.load(shard_path, map_location="cpu", weights_only=False)
        for region in REGIONS:
            if region not in loaded:
                available_regions = list(loaded.keys())
                if available_regions:
                    loaded[region] = loaded[available_regions[0]]
                    print(f"Warning: Shard {shard_path} missing {region}, using {available_regions[0]}.")
                else:
                    raise ValueError(f"Shard {shard_path} has no channels for {region}.")

        lengths = [loaded[region].size(0) for region in REGIONS]
        max_length = max(lengths)
        for region in REGIONS:
            current_length = loaded[region].size(0)
            if current_length < max_length:
                padding = torch.full((max_length - current_length,), pad_token, dtype=loaded[region].dtype)
                loaded[region] = torch.cat((loaded[region], padding), dim=0)
            elif current_length > max_length:
                loaded[region] = loaded[region][:max_length]

        min_length = min(loaded[region].size(0) for region in REGIONS)
        num_sequences = (min_length - T) // T + 1
        min_num_sequences = min(min_num_sequences, num_sequences)

        if num_sequences < K:
            raise ValueError(f"Shard {shard_path} has too few sequences ({num_sequences}) for K={K}")

        sequences = []
        for i in range(num_sequences):
            start = i * T
            end = start + T
            seq = []
            for region in REGIONS:
                channel_seq = loaded[region][start:end]
                if channel_seq.size(0) < T:
                    padding = torch.full((T - channel_seq.size(0),), pad_token, dtype=channel_seq.dtype)
                    channel_seq = torch.cat((channel_seq, padding), dim=0)
                seq.append(channel_seq.unsqueeze(0))
            seq = torch.cat(seq, dim=0)
            sequences.append((seq, label))
        all_sequences.append(sequences)

    support_data = []
    query_data = []
    for sequences in all_sequences:
        sequences = sequences[:min_num_sequences]
        random.shuffle(sequences)
        support_data.extend(sequences[:K])
        query_data.extend(sequences[K:])

    query_counts = {}
    for _, label in query_data:
        query_counts[label] = query_counts.get(label, 0) + 1
    if len(set(query_counts.values())) > 1:
        print(f"Warning: Query set is unbalanced: {query_counts}")

    return support_data, query_data


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def evaluate_fewshot(model, support_data, query_data, device, num_classes, batch_size=16, epochs=100, lr=0.01):
    """
    Evaluate few-shot classification by training a linear classifier on support embeddings with batching.

    Args:
        model: Pretrained GPT model.
        support_data: List of (sequence, label) tuples for support.
        query_data: List of (sequence, label) tuples for query.
        device: Device to run on (e.g., 'cuda' or 'cpu').
        num_classes: Number of classes in the task.
        batch_size: Batch size for processing (default=16).
        epochs: Number of training epochs for the classifier (default=100).
        lr: Learning rate for the classifier (default=0.01).
    """
    model.eval()

    # Helper function to compute embeddings in batches
    def get_embeddings(data, batch_size):
        embeddings = []
        labels = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            sequences = torch.stack([seq for seq, _ in batch], dim=0).to(device)
            batch_emb = model.get_embeddings(sequences)
            embeddings.append(batch_emb)
            labels.extend([label for _, label in batch])
        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.tensor(labels, device=device)
        return embeddings, labels

    # Compute support and query embeddings in batches
    support_emb, support_labels = get_embeddings(support_data, batch_size)
    query_emb, query_labels = get_embeddings(query_data, batch_size)

    # Train classifier with batching
    classifier = nn.Linear(model.config.n_embd, num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Create a DataLoader for support embeddings
    support_dataset = TensorDataset(support_emb, support_labels)
    support_loader = DataLoader(support_dataset, batch_size=batch_size, shuffle=True)

    # Train the classifier using mini-batches
    for epoch in range(epochs):
        for batch_emb, batch_labels in support_loader:
            optimizer.zero_grad()
            logits = classifier(batch_emb)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

    # Evaluate on query set in batches
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, query_emb.size(0), batch_size):
            batch_emb = query_emb[i:i + batch_size]
            batch_labels = query_labels[i:i + batch_size]
            logits = classifier(batch_emb)
            pred = logits.argmax(dim=1)
            correct += (pred == batch_labels).sum().item()
            total += batch_labels.size(0)

    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.4f} (Total query samples: {total})")

# Main Execution
if __name__ == "__main__":
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model configuration
    config = GPTConfig()
    model = GPT(config).to(device)

    # Shard paths (example, replace with your actual paths)
    shard_paths = [
        "./local_shards_val/mydata_train_2.pt",
        "./local_shards_val/mydata_train_0.pt",
        "./local_shards_val/mydata_train_1.pt",
    ]
    # Alternatively, use glob to load all shards:
    # shard_paths = glob.glob("./local_shards_val/mydata_train_*.pt")

    # Holdout setup
    holdout_percentage = 0.2
    num_holdout = int(len(shard_paths) * holdout_percentage)
    base_shards = shard_paths[:-num_holdout] if num_holdout > 0 else shard_paths
    holdout_shards = shard_paths[-num_holdout:] if num_holdout > 0 else []

    # Load data
    support_data_base, query_data_base = load_fewshot_data(
        base_shards, T=config.block_size, K=5, pad_token=config.pad_token, num_channels=config.num_channels
    )
    if holdout_shards:
        support_data_holdout, query_data_holdout = load_fewshot_data(
            holdout_shards, T=config.block_size, K=5, pad_token=config.pad_token, num_channels=config.num_channels
        )
    else:
        support_data_holdout, query_data_holdout = [], []

    # Evaluate with random weights
    print("\nEvaluating with random weights on base classes")
    num_classes_base = len(base_shards)
    evaluate_fewshot(model, support_data_base, query_data_base, device, num_classes=num_classes_base, batch_size=1)
    if holdout_shards:
        print("Evaluating with random weights on holdout classes")
        num_classes_holdout = len(holdout_shards)
        evaluate_fewshot(model, support_data_base, query_data_base, device, num_classes=num_classes_base, batch_size=1)

    # Load pretrained weights and evaluate
    checkpoint_path = "checkpoints/model_03000.pt"  # Adjust path as needed
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state_dict.items()})
        model.eval()

        print("\nEvaluating with pretrained weights on base classes")
        evaluate_fewshot(model, support_data_base, query_data_base, device, num_classes=num_classes_base, batch_size=1)

        if holdout_shards:
            print("Evaluating with pretrained weights on holdout classes")
            evaluate_fewshot(model, support_data_base, query_data_base, device, num_classes=num_classes_base, batch_size=1)
    except FileNotFoundError:
        print(f"\nPretrained weights not found at {checkpoint_path}; skipping pretrained evaluation.")
