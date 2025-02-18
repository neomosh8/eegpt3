from collections import Counter
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn.functional as F  # Add this import

# Model definitions (condensed from earlier)
REGIONS = ["frontal", "motor_temporal", "parietal_occipital"]


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 10799
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    num_channels: int = 3
    mlp_dropout: float = 0.05
    attn_dropout: float = 0.05
    resid_dropout: float = 0.05


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.attn_dropout = nn.Dropout(p=config.attn_dropout)
        self.resid_dropout = nn.Dropout(p=config.resid_dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
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


class SimpleCrossChannelFusion(nn.Module):
    def __init__(self, n_embd, num_heads=1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=n_embd, num_heads=num_heads, batch_first=True)
        self.ln = nn.LayerNorm(n_embd)

    def forward(self, x):
        B, T, C, E = x.size()
        x_reshaped = x.view(B * T, C, E)
        fused, _ = self.attn(x_reshaped, x_reshaped, x_reshaped)
        fused = fused + x_reshaped
        fused = self.ln(fused).view(B, T, C, E)
        return fused


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
        self.channel_encoder = nn.ModuleList([
            nn.Sequential(Block(config), Block(config))
            for _ in range(config.num_channels)
        ])
        self.cross_channel_fusion = SimpleCrossChannelFusion(config.n_embd, num_heads=1)
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
        assert C == self.config.num_channels
        tok_emb = self.transformer.wte(idx).transpose(1, 2)
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb.unsqueeze(2)

        channel_outs = [self.channel_encoder[c](x[:, :, c, :]) for c in range(C)]
        x = torch.stack(channel_outs, dim=2)
        x = self.cross_channel_fusion(x)

        x = x.transpose(1, 2).reshape(B * C, T, self.config.n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        x_last = x[:, -1, :]
        logits = self.lm_head(x_last).view(B, C, -1)

        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, x_last, loss  # Modified to return x_last


class GPTWithClassifier(nn.Module):
    def __init__(self, gpt_model, num_classes):
        super().__init__()
        self.gpt = gpt_model
        for param in self.gpt.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(gpt_model.config.n_embd, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )

    def forward(self, idx):
        B, C, T = idx.size()
        with torch.no_grad():
            _, x_last, _ = self.gpt(idx)  # [B * C, n_embd]
        x_last = x_last.view(B, C, -1).mean(dim=1)  # [B, n_embd]
        logits = self.classifier(x_last)  # [B, num_classes]
        return logits


# Dataloader for specific shard paths
class ShardDataset(Dataset):
    def __init__(self, shard_paths, sequence_length, split_ratio=0.8, is_train=True):
        self.shard_paths = shard_paths
        self.sequence_length = sequence_length
        self.num_channels = len(REGIONS)

        self.tokens = {region: [] for region in REGIONS}
        self.labels = []

        # Load all data
        for shard_path in shard_paths:
            label = int(os.path.basename(shard_path).split('_')[-1].replace('.pt', ''))
            loaded = torch.load(shard_path, map_location="cpu", weights_only=False)
            token_count = min(loaded[REGIONS[0]].size(0), 39490)
            for region in REGIONS:
                self.tokens[region].append(loaded[region][:token_count])
            self.labels.extend([label] * token_count)

        for region in REGIONS:
            self.tokens[region] = torch.cat(self.tokens[region], dim=0)
        self.labels = torch.tensor(self.labels)

        self.total_length = self.tokens[REGIONS[0]].size(0)
        assert all(self.tokens[r].size(0) == self.total_length for r in REGIONS)
        assert len(self.labels) == self.total_length

        # Randomly shuffle indices and split
        indices = torch.randperm(self.total_length)
        split_idx = int(self.total_length * split_ratio)
        if is_train:
            indices = indices[:split_idx]
        else:
            indices = indices[split_idx:]

        # Apply split
        for region in REGIONS:
            self.tokens[region] = self.tokens[region][indices]
        self.labels = self.labels[indices]
        self.total_length = len(self.labels)

    def __len__(self):
        return self.total_length // self.sequence_length

    def __getitem__(self, idx):
        start = idx * self.sequence_length
        end = start + self.sequence_length
        channel_inputs = []
        for region in REGIONS:
            seq = self.tokens[region][start:end]
            if seq.size(0) < self.sequence_length:
                seq = F.pad(seq, (0, self.sequence_length - seq.size(0)))
            channel_inputs.append(seq.unsqueeze(0))
        inputs = torch.cat(channel_inputs, dim=0)
        label = self.labels[start]
        return inputs, label

# Updated training function with validation
def train_classifier(model, train_loader, val_loader, num_epochs, device):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=4e-4)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        avg_loss = total_loss / len(train_loader)
        train_accuracy = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_accuracy = val_correct / val_total if val_total > 0 else 0

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, "
              f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

    torch.save(model.state_dict(), "gpt_with_classifier.pt")
    print("Model saved to gpt_with_classifier.pt")

# Evaluation function
def evaluate_performance(model, dataloader, device, desc="Evaluation"):
    model.eval()
    correct = 0
    total = 0
    batch_count = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            batch_count += 1
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            running_accuracy = correct / total
            print(f"{desc} - Batch {batch_count}, Samples: {total}, Running Accuracy: {running_accuracy:.4f}")
    final_accuracy = correct / total
    print(f"{desc} - Final Accuracy (Full Dataset, {total} samples, {batch_count} batches): {final_accuracy:.4f}")
    return final_accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "./checkpoints/model_00300.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    shard_paths = ["./local_shards_val/mydata_train_0.pt", "./local_shards_val/mydata_train_1.pt",
                   "./local_shards_val/mydata_train_2.pt"]
    train_dataset = ShardDataset(shard_paths, sequence_length=config.block_size, split_ratio=0.6, is_train=True)
    val_dataset = ShardDataset(shard_paths, sequence_length=config.block_size, split_ratio=.6, is_train=False)
    print("Train label distribution:", Counter(train_dataset.labels.tolist()))
    print("Val label distribution:", Counter(val_dataset.labels.tolist()))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Step 1: Random GPT
    print("Step 1: Evaluating with randomly initialized GPT model")
    gpt_model_random = GPT(config)
    gpt_model_random.eval()
    model_random = GPTWithClassifier(gpt_model_random, num_classes=3)
    model_random = model_random.to(device)
    evaluate_performance(model_random, val_loader, device, desc="Random GPT (Validation)")

    # Step 2: Pretrained GPT
    print("\nStep 2: Loading pretrained weights and evaluating")
    gpt_model_pretrained = GPT(config)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    gpt_model_pretrained.load_state_dict(new_state_dict)
    gpt_model_pretrained.eval()
    model_pretrained = GPTWithClassifier(gpt_model_pretrained, num_classes=3)
    model_pretrained = model_pretrained.to(device)
    evaluate_performance(model_pretrained, val_loader, device, desc="Pretrained GPT (Validation)")

    # Step 3: Train
    print("\nStep 3: Starting classifier training")
    print("Training random GPT model")
    train_classifier(model_random, train_loader, val_loader, num_epochs=10, device=device)  # Reduced for testing
    print("\nTraining pretrained GPT model")
    train_classifier(model_pretrained, train_loader, val_loader, num_epochs=10, device=device)

if __name__ == "__main__":
    main()