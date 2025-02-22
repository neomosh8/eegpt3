import torch
import torch.nn as nn
import random
from torch.cuda.amp import autocast

# Constants (assumed from context; adjust as needed)
REGIONS = ["frontal", "motor_temporal", "parietal_occipital"]
VOCAB_SIZE = 82  # Example vocab size; adjust based on your tokenizer
N_EMBD = 384
N_LAYER = 6
N_HEAD = 6
T = 1024  # Sequence length
PAD_TOKEN = 0  # Padding token index


# Configuration class (unchanged model design)
class GPTConfig:
    def __init__(self):
        self.vocab_size = VOCAB_SIZE
        self.n_embd = N_EMBD
        self.n_layer = N_LAYER
        self.n_head = N_HEAD
        self.block_size = T  # Max sequence length


# Placeholder GPT class (model design unchanged)
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd)
        })
        self.intra_channel_encoder = nn.Sequential(
            Block(config),
            Block(config),
            Block(config)
        )
        self.lm_head = self.transformer.wte  # Shared embedding weights

    def forward(self, x):
        # Simplified forward pass for embedding computation
        B, C, T = x.size()
        tok_emb = self.transformer.wte(x)  # [B, C, T, n_embd]
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
        last_tokens = x[:, -1, :].view(B, C, self.config.n_embd)
        embeddings = last_tokens.mean(dim=1)  # [B, n_embd]
        return embeddings


# Placeholder Block class (model design unchanged)
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = nn.MultiheadAttention(config.n_embd, config.n_head, batch_first=True)  # Simplified
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd)
        )

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.ln_1(x + attn_output)
        mlp_output = self.mlp(x)
        x = self.ln_2(x + mlp_output)
        return x


# Optimized data loading function
def load_fewshot_data(shard_paths, T=1024, K=3, pad_token=0):
    """
    Load few-shot data from shard files, storing sequences as int8 to save memory.

    Args:
        shard_paths (list): Paths to shard files (.pt format).
        T (int): Sequence length.
        K (int): Number of support samples per class.
        pad_token (int): Token index for padding.

    Returns:
        support_data (list): List of (sequence, label) tuples for support set.
        query_data (list): List of (sequence, label) tuples for query set.
    """
    all_sequences = []
    min_num_sequences = float('inf')
    for label, shard_path in enumerate(shard_paths):
        loaded = torch.load(shard_path, map_location="cpu", weights_only=False)
        min_length = min(loaded[region].size(0) for region in REGIONS)
        num_sequences = (min_length - T) // T + 1
        min_num_sequences = min(min_num_sequences, num_sequences)
        sequences = []
        for i in range(num_sequences):
            start = i * T
            end = start + T
            seq = []
            for region in REGIONS:
                channel_seq = loaded[region][start:end].to(torch.int8)
                if channel_seq.size(0) < T:
                    padding = torch.full((T - channel_seq.size(0),), pad_token, dtype=torch.int8)
                    channel_seq = torch.cat((channel_seq, padding), dim=0)
                seq.append(channel_seq.unsqueeze(0))
            seq = torch.cat(seq, dim=0)  # [C, T], dtype=int8
            sequences.append((seq, label))
        all_sequences.append(sequences)
    support_data = []
    query_data = []
    for sequences in all_sequences:
        sequences = sequences[:min_num_sequences]
        random.shuffle(sequences)
        support_data.extend(sequences[:K])
        query_data.extend(sequences[K:])
    return support_data, query_data


# Optimized embedding computation function
def compute_embeddings(model, sequences, device):
    """
    Compute embeddings for a batch of sequences using mixed precision.

    Args:
        model (nn.Module): The GPT model.
        sequences (Tensor): Input sequences [B, C, T], possibly int8.
        device (torch.device): Device to run computation on.

    Returns:
        embeddings (Tensor): Computed embeddings [B, n_embd] on CPU.
    """
    model.eval()
    with torch.no_grad():
        with autocast():
            sequences = sequences.to(device).to(torch.int64)  # Cast to int64 for nn.Embedding
            embeddings = model(sequences)  # Forward pass
            return embeddings.float().cpu()  # Convert to FP32 and move to CPU


# Optimized few-shot evaluation function
def evaluate_fewshot(model, support_data, query_data, device, batch_size=4):
    """
    Evaluate the model on a few-shot task, processing query data in batches.

    Args:
        model (nn.Module): The GPT model.
        support_data (list): List of (sequence, label) tuples for support set.
        query_data (list): List of (sequence, label) tuples for query set.
        device (torch.device): Device to run computation on.
        batch_size (int): Batch size for processing query data.
    """
    # Compute prototypes from support data
    prototypes = {}
    for label in set(d[1] for d in support_data):
        class_sequences = [d[0] for d in support_data if d[1] == label]
        class_sequences = torch.stack(class_sequences, dim=0).to(device)  # [K, C, T], int8
        embeddings = compute_embeddings(model, class_sequences, device)  # [K, n_embd]
        prototypes[label] = embeddings.mean(dim=0).cpu()  # [n_embd]

    # Evaluate query data in batches
    correct = 0
    total = 0
    for i in range(0, len(query_data), batch_size):
        batch = query_data[i:i + batch_size]
        query_sequences = torch.stack([d[0] for d in batch], dim=0).to(device)  # [batch_size, C, T], int8
        query_labels = [d[1] for d in batch]
        embeddings = compute_embeddings(model, query_sequences, device)  # [batch_size, n_embd]
        for emb, true_label in zip(embeddings, query_labels):
            distances = {label: torch.norm(emb - proto) for label, proto in prototypes.items()}
            pred_label = min(distances, key=distances.get)
            if pred_label == true_label:
                correct += 1
            total += 1
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.4f} (Total query samples: {total})")


# Main execution
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    shard_paths = [
        "./local_shards_val/mydata_train_0.pt",
        "./local_shards_val/mydata_train_1.pt",
        "./local_shards_val/mydata_train_2.pt"
    ]
    support_data, query_data = load_fewshot_data(shard_paths, T=T, K=2, pad_token=PAD_TOKEN)

    # Step 1: Evaluate with random weights
    print("Step 1: Evaluating with random weights")
    config = GPTConfig()
    gpt_model_random = GPT(config).to(device)
    evaluate_fewshot(gpt_model_random, support_data, query_data, device, batch_size=4)

    # Step 2: Evaluate with pretrained weights
    print("\nStep 2: Loading pretrained weights and evaluating")
    gpt_model_pretrained = GPT(config).to(device)
    checkpoint = torch.load("checkpoints/model_03000.pt", map_location=device)
    state_dict = checkpoint['model_state_dict']
    del checkpoint  # Free memory after extracting state_dict
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    gpt_model_pretrained.load_state_dict(new_state_dict)
    gpt_model_pretrained.eval()
    evaluate_fewshot(gpt_model_pretrained, support_data, query_data, device, batch_size=4)

    # Optional: Clear GPU memory cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()