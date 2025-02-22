import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import random

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
    num_channels: int = 3
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
        tok_emb = self.transformer.wte(idx)  # [B, C, T, n_embd]
        x = tok_emb.transpose(1, 2)  # [B, T, C, n_embd]
        channel_outs = []
        for c in range(C):
            x_c = x[:, :, c, :]
            x_c = self.intra_channel_encoder(x_c)
            channel_outs.append(x_c)
        x = torch.stack(channel_outs, dim=2)  # [B, T, C, n_embd]
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

    def get_embedding(self, idx):
        B, C, T = idx.size()
        tok_emb = self.transformer.wte(idx)  # [B, C, T, n_embd]
        x = tok_emb.transpose(1, 2)  # [B, T, C, n_embd]
        channel_outs = []
        for c in range(C):
            x_c = x[:, :, c, :]
            x_c = self.intra_channel_encoder(x_c)
            channel_outs.append(x_c)
        x = torch.stack(channel_outs, dim=2)  # [B, T, C, n_embd]
        for block in self.transformer.h:
            x = block(x)
        last_tokens = x[:, -1, :, :]  # [B, C, n_embd]
        embedding = last_tokens.mean(dim=1)  # [B, n_embd]
        return embedding

# Few-Shot Classification Functions
def compute_prototypes(model, support_data, device):
    model.eval()
    with torch.no_grad():
        sequences = torch.stack([seq for seq, _ in support_data], dim=0).to(device)
        embeddings = model.get_embedding(sequences)
        labels = [label for _, label in support_data]
        label_to_embeddings = {}
        for emb, label in zip(embeddings, labels):
            if label not in label_to_embeddings:
                label_to_embeddings[label] = []
            label_to_embeddings[label].append(emb)
        prototypes = {}
        for label, embs in label_to_embeddings.items():
            embs = torch.stack(embs, dim=0)
            prototype = embs.mean(dim=0)
            prototypes[label] = prototype
        return prototypes

def evaluate_fewshot(model, support_data, query_data, device, batch_size=4):
    prototypes = compute_prototypes(model, support_data, device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(query_data), batch_size):
            batch = query_data[i:i + batch_size]
            sequences = torch.stack([seq for seq, _ in batch], dim=0).to(device)
            labels = [label for _, label in batch]
            embeddings = model.get_embedding(sequences)
            for emb, true_label in zip(embeddings, labels):
                distances = {label: torch.norm(emb - proto) for label, proto in prototypes.items()}
                pred_label = min(distances, key=distances.get)
                if pred_label == true_label:
                    correct += 1
                total += 1
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.4f} (Total query samples: {total})")

# Data Loading (Simple Example)
def load_fewshot_data(shard_paths, T=1024, K=3, num_channels=3, vocab_size=82):
    support_data = []
    query_data = []
    for label, shard_path in enumerate(shard_paths):
        # Simulate loading data; replace with actual torch.load logic
        num_samples = K + 10  # K support + 10 query
        sequences = torch.randint(0, vocab_size, (num_samples, num_channels, T))
        support_data.extend([(sequences[i], label) for i in range(K)])
        query_data.extend([(sequences[i + K], label) for i in range(10)])
    return support_data, query_data

# Main Execution
if __name__ == "__main__":
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model configuration
    config = GPTConfig()
    model = GPT(config).to(device)

    # Example shard paths (replace with actual paths)
    shard_paths = [
        "./local_shards_val/mydata_train_0.pt",
        "./local_shards_val/mydata_train_1.pt",
        "./local_shards_val/mydata_train_2.pt"
    ]

    # Load data
    support_data, query_data = load_fewshot_data(shard_paths, T=config.block_size, K=10,
                                                 num_channels=config.num_channels, vocab_size=config.vocab_size)

    # Evaluate with random weights
    print("Evaluating with random weights")
    evaluate_fewshot(model, support_data, query_data, device, batch_size=16)

    # Optionally, load pretrained weights and evaluate
    try:
        checkpoint = torch.load("checkpoints/model_03000.pt", map_location=device)
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state_dict.items()})
        model.eval()
        print("\nEvaluating with pretrained weights")
        evaluate_fewshot(model, support_data, query_data, device, batch_size=4)
    except FileNotFoundError:
        print("\nPretrained weights not found; skipping pretrained evaluation.")