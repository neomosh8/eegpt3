import torch
import os
import glob
import random
import numpy as np
from torch.nn import functional as F

# Define regions globally for consistency
REGIONS = ["frontal", "motor_temporal", "parietal_occipital"]


def compute_prototypes(model, support_data, device, batch_size=8):
    """
    Compute class prototypes from support set with improved batching and representation.

    Args:
        model: The model to extract embeddings
        support_data: List of (sequence, label) tuples
        device: Device to run computation on
        batch_size: Number of samples to process at once

    Returns:
        Dictionary mapping class labels to prototype vectors
    """
    model.eval()
    label_to_embeddings = {}

    # Process in batches to save memory
    for i in range(0, len(support_data), batch_size):
        batch = support_data[i:i + batch_size]
        sequences = torch.stack([seq for seq, _ in batch], dim=0).to(device)
        batch_labels = [label for _, label in batch]

        with torch.no_grad():
            # Get embeddings using the consistent method
            embeddings = extract_embeddings(model, sequences)

            # Store embeddings by class
            for emb, label in zip(embeddings, batch_labels):
                if label not in label_to_embeddings:
                    label_to_embeddings[label] = []
                label_to_embeddings[label].append(emb.cpu())  # Store on CPU to save GPU memory

    # Compute prototypes for each class
    prototypes = {}
    for label, embs in label_to_embeddings.items():
        embs = torch.stack(embs, dim=0).to(device)
        # Apply L2 normalization before averaging for better prototype quality
        normalized_embs = F.normalize(embs, p=2, dim=1)
        prototype = normalized_embs.mean(dim=0)
        # Re-normalize the prototype
        prototype = F.normalize(prototype, p=2, dim=0)
        prototypes[label] = prototype

    return prototypes


def extract_embeddings(model, sequences):
    """
    Extract embeddings from sequences using the same processing pipeline as training.

    Args:
        model: The model to extract embeddings
        sequences: Tensor of shape [B, C, T] where B is batch size, C is channels, T is sequence length

    Returns:
        Tensor of embeddings with shape [B, E] where E is embedding dimension
    """
    B, C, T = sequences.size()
    config = model.config

    # Follow the same processing as in forward()
    tok_emb = model.transformer.wte(sequences)  # [B, C, T, n_embd]
    x = tok_emb.transpose(1, 2)  # [B, T, C, n_embd]

    # Use the same batched operation as in forward()
    x_reshaped = x.permute(0, 2, 1, 3).contiguous().reshape(B * C, T, config.n_embd)
    out = model.intra_channel_encoder(x_reshaped)
    x = out.view(B, C, T, config.n_embd).permute(0, 2, 1, 3).contiguous()

    # Process through transformer blocks
    for block in model.transformer.h:
        x = block(x)

    # Use a more comprehensive representation than just the last token
    # Get the last 4 tokens and average them for a better representation
    last_n_tokens = x[:, -4:, :, :].mean(dim=1)  # [B, C, n_embd]

    # Take mean across channels
    embedding = last_n_tokens.mean(dim=1)  # [B, n_embd]

    # Apply layer norm for consistency with forward pass
    embedding = model.transformer.ln_f(embedding)

    return embedding


def compute_cosine_similarities(query_embeddings, prototypes):
    """
    Compute cosine similarities between query embeddings and prototypes.

    Args:
        query_embeddings: Tensor of shape [B, E]
        prototypes: Dictionary mapping class labels to prototype vectors

    Returns:
        Dictionary mapping each query index to a dictionary of {label: similarity}
    """
    similarities = {}
    for i, query_emb in enumerate(query_embeddings):
        query_emb_normalized = F.normalize(query_emb, p=2, dim=0)
        similarities[i] = {}
        for label, prototype in prototypes.items():
            # Compute cosine similarity (dot product of normalized vectors)
            similarity = torch.dot(query_emb_normalized, prototype).item()
            similarities[i][label] = similarity
    return similarities


def evaluate_fewshot(model, support_data, query_data, device, batch_size=8, return_predictions=False):
    """
    Evaluate few-shot classification performance.

    Args:
        model: The model to use for evaluation
        support_data: List of (sequence, label) tuples for support set
        query_data: List of (sequence, label) tuples for query set
        device: Device to run computation on
        batch_size: Number of samples to process at once
        return_predictions: Whether to return predicted labels

    Returns:
        Dictionary of evaluation metrics including accuracy, and optionally predictions
    """
    model.eval()

    # Compute prototypes from support set
    prototypes = compute_prototypes(model, support_data, device, batch_size)

    # Process query data in batches
    all_similarities = {}
    all_true_labels = []

    for i in range(0, len(query_data), batch_size):
        batch = query_data[i:i + batch_size]
        sequences = torch.stack([seq for seq, _ in batch], dim=0).to(device)
        batch_labels = [label for _, label in batch]
        all_true_labels.extend(batch_labels)

        with torch.no_grad():
            # Extract embeddings
            embeddings = extract_embeddings(model, sequences)

            # Compute similarities
            batch_similarities = compute_cosine_similarities(embeddings, prototypes)

            # Add to all similarities with adjusted indices
            for j, sims in batch_similarities.items():
                all_similarities[i + j] = sims

    # Predict labels and compute metrics
    predictions = []
    correct = 0

    for i, true_label in enumerate(all_true_labels):
        sims = all_similarities[i]
        pred_label = max(sims, key=sims.get)
        predictions.append(pred_label)
        if pred_label == true_label:
            correct += 1

    total = len(all_true_labels)
    accuracy = correct / total if total > 0 else 0

    # Compute per-class metrics
    class_metrics = {}
    for label in set(all_true_labels):
        class_correct = sum(1 for p, t in zip(predictions, all_true_labels)
                            if p == t and t == label)
        class_total = sum(1 for t in all_true_labels if t == label)
        class_accuracy = class_correct / class_total if class_total > 0 else 0
        class_metrics[label] = {
            'accuracy': class_accuracy,
            'support': class_total
        }

    # Print results
    print(f"Overall Accuracy: {accuracy:.4f} (Total query samples: {total})")
    print("Per-class Accuracy:")
    for label, metrics in class_metrics.items():
        print(f"  Class {label}: {metrics['accuracy']:.4f} (Support: {metrics['support']})")

    # Return results
    results = {
        'accuracy': accuracy,
        'class_metrics': class_metrics,
    }

    if return_predictions:
        results['predictions'] = predictions
        results['true_labels'] = all_true_labels

    return results


def load_fewshot_data(shard_paths, T=1024, K=3, pad_token=0, num_channels=len(REGIONS),
                      balance_classes=True, max_samples_per_class=None, seed=42):
    """
    Load few-shot data with improved handling of class imbalance.

    Args:
        shard_paths: List of paths to shard files, one per class
        T: Sequence length
        K: Number of support samples per class
        pad_token: Token used for padding
        num_channels: Number of channels
        balance_classes: Whether to balance classes in query set
        max_samples_per_class: Maximum samples to use per class (None = use all)
        seed: Random seed for reproducibility

    Returns:
        support_data: List of (sequence, label) tuples for support set
        query_data: List of (sequence, label) tuples for query set
    """
    random.seed(seed)
    np.random.seed(seed)

    if not shard_paths:
        raise ValueError("No shard paths provided.")

    all_sequences = []
    min_num_sequences = float('inf')

    # Load data from shards
    for label, shard_path in enumerate(shard_paths):
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Shard file not found: {shard_path}")

        try:
            loaded = torch.load(shard_path, map_location="cpu", weights_only=False)

            # Handle missing regions
            for region in REGIONS:
                if region not in loaded:
                    available_regions = list(loaded.keys())
                    if available_regions:
                        print(f"Warning: Shard {shard_path} missing {region}, using {available_regions[0]}.")
                        loaded[region] = loaded[available_regions[0]]
                    else:
                        raise ValueError(f"Shard {shard_path} has no channels.")

            # Ensure all channels have the same length
            max_length = max(loaded[region].size(0) for region in REGIONS)
            for region in REGIONS:
                current_length = loaded[region].size(0)
                if current_length < max_length:
                    padding = torch.full((max_length - current_length,), pad_token,
                                         dtype=loaded[region].dtype)
                    loaded[region] = torch.cat((loaded[region], padding), dim=0)
                elif current_length > max_length:
                    loaded[region] = loaded[region][:max_length]

            # Extract non-overlapping sequences
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
                        padding = torch.full((T - channel_seq.size(0),), pad_token,
                                             dtype=channel_seq.dtype)
                        channel_seq = torch.cat((channel_seq, padding), dim=0)
                    seq.append(channel_seq.unsqueeze(0))
                seq = torch.cat(seq, dim=0)  # [C, T]
                sequences.append((seq, label))

            all_sequences.append(sequences)

        except Exception as e:
            print(f"Error loading shard {shard_path}: {e}")
            raise

    # Split into support and query sets
    support_data = []
    query_data = []

    for sequences in all_sequences:
        # Apply max_samples_per_class if specified
        if max_samples_per_class and len(sequences) > max_samples_per_class:
            sequences = random.sample(sequences, max_samples_per_class)

        random.shuffle(sequences)
        support_data.extend(sequences[:K])
        query_data.extend(sequences[K:])

    # Balance query set if requested
    if balance_classes:
        query_by_class = {}
        for seq, label in query_data:
            if label not in query_by_class:
                query_by_class[label] = []
            query_by_class[label].append((seq, label))

        # Find minimum size for balancing
        min_class_size = min(len(samples) for samples in query_by_class.values())

        # Balance classes
        balanced_query_data = []
        for label, samples in query_by_class.items():
            balanced_query_data.extend(random.sample(samples, min_class_size))

        query_data = balanced_query_data

    # Shuffle query data
    random.shuffle(query_data)

    # Verify balance
    query_counts = {}
    for _, label in query_data:
        query_counts[label] = query_counts.get(label, 0) + 1

    print(f"Support set: {len(support_data)} samples")
    print(f"Query set: {len(query_data)} samples")
    print(f"Class distribution in query set: {query_counts}")

    return support_data, query_data


# Update the model's get_embedding method for consistency
def update_model_with_consistent_embedding(model):
    """
    Replace the model's get_embedding method with a consistent version.

    Args:
        model: The model to update

    Returns:
        Updated model
    """

    def new_get_embedding(self, idx):
        return extract_embeddings(self, idx)

    # Replace the method
    import types
    model.get_embedding = types.MethodType(new_get_embedding, model)

    return model


# Example usage in main
if __name__ == "__main__":
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model configuration (assuming GPTConfig and GPT are defined elsewhere)
    from your_model_file import GPTConfig, GPT

    config = GPTConfig()
    model = GPT(config).to(device)

    # Update model with consistent embedding method
    model = update_model_with_consistent_embedding(model)

    # Example shard paths (replace with actual paths)
    shard_paths = [
        "./local_shards_val/mydata_train_2.pt",
        "./local_shards_val/mydata_train_0.pt",
    ]

    # Load data with improved handling
    support_data, query_data = load_fewshot_data(
        shard_paths,
        T=config.block_size,
        K=10,
        pad_token=config.pad_token,
        num_channels=config.num_channels,
        balance_classes=True,
        max_samples_per_class=100  # Limit samples per class for faster training/evaluation
    )

    # Evaluate with random weights
    print("Evaluating with random weights")
    random_results = evaluate_fewshot(model, support_data, query_data, device, batch_size=16)

    # Optionally, load pretrained weights and evaluate
    try:
        checkpoint = torch.load(
            "checkpoints/model_last_checkpoint.pt",
            map_location=device,
            weights_only=False
        )
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state_dict.items()})
        model.eval()

        print("\nEvaluating with pretrained weights")
        pretrained_results = evaluate_fewshot(model, support_data, query_data, device, batch_size=16)

        # Compare results
        print(f"\nImprovement: {pretrained_results['accuracy'] - random_results['accuracy']:.4f}")

    except FileNotFoundError:
        print("\nPretrained weights not found; skipping pretrained evaluation.")