import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# Configuration class
@dataclass
class TestConfig:
    block_size: int = 12  # 3 channels * 4 time steps
    vocab_size: int = 100
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 8
    num_channels: int = 3
    mlp_dropout: float = 0.0  # Set to 0 for deterministic testing
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0


# Model components
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.resid_dropout)
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
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

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
        self.attn = nn.MultiheadAttention(embed_dim=n_embd, num_heads=num_heads)

    def forward(self, x):
        B, T, C, E = x.size()
        x = x.view(B * T, C, E)
        x = x.transpose(0, 1)
        fused, _ = self.attn(x, x, x)
        fused = fused.mean(dim=0)
        return fused.view(B, T, E)


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Main transformer components
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd)
        })

        # Output head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # Channel processing
        self.channel_encoder = nn.ModuleList([
            nn.Sequential(Block(config), Block(config))
            for _ in range(config.num_channels)
        ])
        self.cross_channel_fusion = CrossChannelFusion(config.n_embd, num_heads=1)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T % self.config.num_channels == 0
        time_steps = T // self.config.num_channels

        # Get embeddings
        tok_emb = self.transformer.wte(idx)
        debug_tensor("Initial token embeddings", tok_emb)

        # Process channels
        x = tok_emb.view(B, time_steps, self.config.num_channels, self.config.n_embd)
        debug_tensor("Reshaped for channels", x)

        # Process each channel
        channel_outs = []
        for c in range(self.config.num_channels):
            x_c = x[:, :, c, :]
            debug_tensor(f"Channel {c} input", x_c)
            x_c = self.channel_encoder[c](x_c)
            debug_tensor(f"Channel {c} output", x_c)
            channel_outs.append(x_c)

        x = torch.stack(channel_outs, dim=2)
        x_fused = self.cross_channel_fusion(x)
        debug_tensor("After fusion", x_fused)

        # Replicate fused output
        x_fused_rep = x_fused.unsqueeze(2).repeat(1, 1, self.config.num_channels, 1)
        x_flat = x_fused_rep.view(B, T, self.config.n_embd)

        # Add positional embeddings
        pos = torch.arange(0, T, device=x_flat.device).unsqueeze(0)
        pos_emb = self.transformer.wpe(pos)
        x_flat = x_flat + pos_emb

        # Process through transformer blocks
        for block in self.transformer.h:
            x_flat = block(x_flat)
        x_flat = self.transformer.ln_f(x_flat)

        # Generate logits
        logits = self.lm_head(x_flat)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


def debug_tensor(name, tensor, shape_only=False):
    """Helper function to print tensor information"""
    print(f"\n{'-' * 50}")
    print(f"Debug {name}:")
    print(f"Shape: {tensor.shape}")
    if not shape_only:
        print(f"Content:\n{tensor}")


def create_sample_data():
    """Create sample data with distinct patterns per channel"""
    batch1 = torch.tensor([
        # Channel 1 tokens
        11, 12, 13, 14,
        # Channel 2 tokens
        21, 22, 23, 24,
        # Channel 3 tokens
        31, 32, 33, 34
    ])

    batch2 = torch.tensor([
        # Channel 1 tokens
        15, 16, 17, 18,
        # Channel 2 tokens
        25, 26, 27, 28,
        # Channel 3 tokens
        35, 36, 37, 38
    ])

    return torch.stack([batch1, batch2])


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create config and model
    config = TestConfig()
    model = GPT(config)

    # Create and process sample data
    input_data = create_sample_data()
    print("\nInput Data Structure:")
    debug_tensor("input_data", input_data)

    # Forward pass
    logits, _ = model(input_data)
    print("\nFinal Output Structure:")
    debug_tensor("logits", logits, shape_only=True)


if __name__ == "__main__":
    main()