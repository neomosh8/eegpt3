import torch
import random


import Model_class

# From the original code
REGIONS = ["frontal", "motor_temporal", "parietal_occipital"]
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")



def load_fewshot_data(shard_paths, T=1024, K=5, pad_token=0):
    support_data = []
    query_data = []
    for label, shard_path in enumerate(shard_paths):
        loaded = torch.load(shard_path, map_location="cpu", weights_only=False)
        min_length = min(loaded[region].size(0) for region in REGIONS)
        num_sequences = (min_length - T) // T + 1
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
            sequences.append(seq)

        random.shuffle(sequences)
        support_data.extend((seq, label) for seq in sequences[:K])
        query_data.extend((seq, label) for seq in sequences[K:])

    return support_data, query_data


def compute_embeddings(model, sequences, device):
    model.eval()
    with torch.no_grad():
        sequences = sequences.to(device)
        B, C, T = sequences.size()

        tok_emb = model.transformer.wte(sequences)
        x = tok_emb.transpose(1, 2)
        pos = torch.arange(T, device=device).unsqueeze(0)
        pos_emb = model.transformer.wpe(pos)
        x = x + pos_emb.unsqueeze(2)

        channel_outs = []
        for c in range(C):
            x_c = x[:, :, c, :]
            x_c = model.intra_channel_encoder(x_c)
            channel_outs.append(x_c)
        x = torch.stack(channel_outs, dim=2)

        for block in model.transformer.h:
            x = block(x)

        x = x.transpose(1, 2).reshape(B * C, T, model.config.n_embd)
        x = model.transformer.ln_f(x)

        last_tokens = x[:, -1, :]
        last_tokens = last_tokens.view(B, C, model.config.n_embd)
        embeddings = last_tokens.mean(dim=1)

        return embeddings.cpu()


def evaluate_fewshot(model, support_data, query_data, device):
    prototypes = {}
    for label in set(d[1] for d in support_data):
        class_sequences = [d[0] for d in support_data if d[1] == label]
        class_sequences = torch.stack(class_sequences, dim=0)
        embeddings = compute_embeddings(model, class_sequences, device)
        prototypes[label] = embeddings.mean(dim=0)

    query_sequences = torch.stack([d[0] for d in query_data], dim=0)
    query_labels = [d[1] for d in query_data]
    embeddings = compute_embeddings(model, query_sequences, device)

    correct = 0
    for emb, true_label in zip(embeddings, query_labels):
        distances = {label: torch.norm(emb - proto) for label, proto in prototypes.items()}
        pred_label = min(distances, key=distances.get)
        if pred_label == true_label:
            correct += 1
    accuracy = correct / len(query_data)
    print(f"Accuracy: {accuracy:.4f}")


# Main execution
shard_paths = [
    "./local_shards_val/mydata_train_0.pt",
    "./local_shards_val/mydata_train_1.pt",
    "./local_shards_val/mydata_train_2.pt"
]
support_data, query_data = load_fewshot_data(shard_paths, T=1024, K=5)

# Step 1: Random GPT
print("Step 1: Evaluating with random weights")
config = Model_class.GPTConfig()
gpt_model_random = Model_class.GPT(config).to(device)
evaluate_fewshot(gpt_model_random, support_data, query_data, device)

# Step 2: Pretrained GPT
print("\nStep 2: Loading pretrained weights and evaluating")
gpt_model_pretrained = Model_class.GPT(config).to(device)
checkpoint = torch.load("/checkpoints/model_03400.pt", map_location=device)  # Adjust path
state_dict = checkpoint['model_state_dict']
new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
gpt_model_pretrained.load_state_dict(new_state_dict)
gpt_model_pretrained.eval()
evaluate_fewshot(gpt_model_pretrained, support_data, query_data, device)