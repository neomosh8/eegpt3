import os
import glob
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import re


# Define model classes (should match those in your training script)
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


class GPTConfig:
    def __init__(self, block_size=2048, vocab_size=257, n_layer=24, n_head=16, n_embd=768,
                 mlp_dropout=0.05, attn_dropout=0.05, resid_dropout=0.05):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.mlp_dropout = mlp_dropout
        self.attn_dropout = attn_dropout
        self.resid_dropout = resid_dropout


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
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        return logits, loss


# Feature extractor using pre-trained model
class EEGFeatureExtractor(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.pretrained_model = pretrained_model

        # Freeze the pre-trained model
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Extract features from the pre-trained model
        with torch.no_grad():
            # Forward through token embedding
            tok_emb = self.pretrained_model.transformer.wte(x)

            # Get positional embeddings
            pos = torch.arange(0, x.size(1), dtype=torch.long, device=x.device)
            pos_emb = self.pretrained_model.transformer.wpe(pos)

            # Combine token and positional embeddings
            hidden = tok_emb + pos_emb

            # Pass through transformer blocks
            for block in self.pretrained_model.transformer.h:
                hidden = block(hidden)

            # Final layer norm
            hidden = self.pretrained_model.transformer.ln_f(hidden)

        # Mean pooling across sequence dimension
        pooled = hidden.mean(dim=1)

        return pooled


# Classification model
class BCIClassifier(nn.Module):
    def __init__(self, pretrained_model, num_classes, freeze_backbone=True):
        super().__init__()
        self.feature_extractor = EEGFeatureExtractor(pretrained_model)

        # Define classification head
        embedding_dim = pretrained_model.config.n_embd
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)


# Dataset class for tokenized BCI data
class BCIDataset(Dataset):
    def __init__(self, token_files, labels):
        self.token_files = token_files
        self.labels = labels
        self.max_length = 2048  # Adjust based on your model's input size

    def __len__(self):
        return len(self.token_files)

    def __getitem__(self, idx):
        # Load tokenized data
        tokens = torch.load(self.token_files[idx])

        # Truncate or pad tokens to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            pad_length = self.max_length - len(tokens)
            tokens = torch.cat([tokens, torch.zeros(pad_length, dtype=tokens.dtype)])

        return tokens, self.labels[idx]


# Training function
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    all_preds = []
    all_labels = []

    for tokens, labels in tqdm(dataloader, desc="Training"):
        tokens = tokens.to(device)
        labels = labels.to(device)

        # Forward pass
        logits = model(tokens)
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        epoch_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return epoch_loss / len(dataloader), accuracy, f1


# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for tokens, labels in tqdm(dataloader, desc="Evaluating"):
            tokens = tokens.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(tokens)
            loss = criterion(logits, labels)

            # Track metrics
            epoch_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    return epoch_loss / len(dataloader), accuracy, f1, cm, all_preds, all_labels


# Load the pre-trained model
def load_pretrained_model(checkpoint_path):
    """
    Load a pre-trained model with handling for compiled model checkpoints
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Create model with the saved configuration
    config = checkpoint.get('config', None)

    # If config is not in the checkpoint, create a default one
    if config is None:
        print("No config found in checkpoint, using default GPT configuration")
        config = GPTConfig()

    model = GPT(config)

    # Check if this is a compiled model (keys start with "_orig_mod.")
    state_dict = checkpoint['model']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print("Detected compiled model checkpoint, removing '_orig_mod.' prefix from keys")
        # Remove the "_orig_mod." prefix from all keys
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                new_key = k[len('_orig_mod.'):]
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict

    # Load the modified state dict
    model.load_state_dict(state_dict)

    print(
        f"Successfully loaded model with config: n_layer={config.n_layer}, n_head={config.n_head}, n_embd={config.n_embd}")

    return model
# Prepare tokenized BCI data for classification
def prepare_bci_data(tokenized_dir):
    # Load label mapping
    with open(os.path.join(tokenized_dir, "label_mapping.json"), "r") as f:
        mapping_data = json.load(f)

    label_mapping = mapping_data["label_mapping"]
    print(f"Loaded label mapping with {len(label_mapping)} classes: {label_mapping}")

    # Find all tokenized files for each class
    token_files = []
    labels = []

    for class_name, label in label_mapping.items():
        class_token_files = glob.glob(os.path.join(tokenized_dir, f"*_{class_name}_tokens.pt"))
        print(f"Found {len(class_token_files)} token files for class '{class_name}'")
        token_files.extend(class_token_files)
        labels.extend([label] * len(class_token_files))

    return token_files, labels, label_mapping


# Main evaluation function
def evaluate_bci_classification(pretrained_model_path, tokenized_dir, output_dir,
                                epochs=15, batch_size=16, learning_rate=1e-4):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load pre-trained model
    print("Loading pre-trained model...")
    pretrained_model = load_pretrained_model(pretrained_model_path)
    pretrained_model.to(device)
    pretrained_model.eval()

    # Prepare BCI data
    print("Preparing BCI data...")
    token_files, labels, label_mapping = prepare_bci_data(tokenized_dir)

    if not token_files:
        print("No tokenized files found!")
        return

    # Create dataset
    dataset = BCIDataset(token_files, labels)

    # Split into train, validation, test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Dataset splits: Train={train_size}, Val={val_size}, Test={test_size}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Run two evaluations: pretrained vs. random initialization
    results = {}

    # 1. Evaluate with pretrained model
    num_classes = len(label_mapping)
    print(f"\n--- Evaluating with pre-trained model ({num_classes} classes) ---")

    model = BCIClassifier(pretrained_model, num_classes, freeze_backbone=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # Training loop
    best_val_acc = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, val_f1, _, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        # Update learning rate
        scheduler.step(val_loss)

        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Print progress
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, "best_pretrained_model.pt"))
            print("Saved new best model!")

    # Load best model for testing
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_pretrained_model.pt")))

    # Test on test set
    test_loss, test_acc, test_f1, conf_mat, _, _ = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\nPretrained Model Test Results:")
    print(f"Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, F1 Score: {test_f1:.4f}")

    # Save pretrained results
    results['pretrained'] = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'confusion_matrix': conf_mat.tolist(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }

    # 2. Evaluate with randomly initialized model
    print("\n--- Evaluating with randomly initialized model ---")

    # Create a new model with same architecture but random weights
    config = pretrained_model.config
    random_model = GPT(config)

    # Initialize the model with random weights
    for p in random_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)

    random_model.to(device)

    # Create classifier with random model
    random_classifier = BCIClassifier(random_model, num_classes, freeze_backbone=False).to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(random_classifier.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # Training loop
    best_val_acc = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            random_classifier, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, val_f1, _, _, _ = evaluate(
            random_classifier, val_loader, criterion, device
        )

        # Update learning rate
        scheduler.step(val_loss)

        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Print progress
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(random_classifier.state_dict(), os.path.join(output_dir, "best_random_model.pt"))
            print("Saved new best model!")

    # Load best model for testing
    random_classifier.load_state_dict(torch.load(os.path.join(output_dir, "best_random_model.pt")))

    # Test on test set
    test_loss, test_acc, test_f1, conf_mat, _, _ = evaluate(
        random_classifier, test_loader, criterion, device
    )

    print(f"\nRandom Model Test Results:")
    print(f"Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, F1 Score: {test_f1:.4f}")

    # Save random model results
    results['random'] = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'confusion_matrix': conf_mat.tolist(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }

    # Create comparison plots
    create_comparison_plots(results, label_mapping, output_dir)

    # Save overall results
    with open(os.path.join(output_dir, "classification_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    return results


def create_comparison_plots(results, label_mapping, output_dir):
    """Create plots comparing pretrained and random models"""
    # Extract class names
    class_names = list(label_mapping.keys())

    # 1. Accuracy comparison
    plt.figure(figsize=(10, 6))
    plt.bar(['Pretrained Model', 'Random Initialization'],
            [results['pretrained']['test_accuracy'], results['random']['test_accuracy']],
            color=['#3498db', '#e74c3c'])
    plt.ylabel('Test Accuracy')
    plt.title('Transfer Learning Effectiveness: Pretrained vs Random Initialization')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')

    # 2. Confusion matrices
    plt.figure(figsize=(18, 8))

    plt.subplot(1, 2, 1)
    sns.heatmap(results['pretrained']['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Pretrained Model: Confusion Matrix')

    plt.subplot(1, 2, 2)
    sns.heatmap(results['random']['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Random Model: Confusion Matrix')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')

    # 3. Training curves
    plt.figure(figsize=(18, 10))

    # Loss curves
    plt.subplot(2, 2, 1)
    plt.plot(results['pretrained']['train_losses'], label='Train Loss', color='#3498db')
    plt.plot(results['pretrained']['val_losses'], label='Val Loss', color='#2980b9', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Pretrained Model: Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(results['random']['train_losses'], label='Train Loss', color='#e74c3c')
    plt.plot(results['random']['val_losses'], label='Val Loss', color='#c0392b', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Random Model: Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Accuracy curves
    plt.subplot(2, 2, 3)
    plt.plot(results['pretrained']['train_accs'], label='Train Acc', color='#3498db')
    plt.plot(results['pretrained']['val_accs'], label='Val Acc', color='#2980b9', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Pretrained Model: Accuracy Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(results['random']['train_accs'], label='Train Acc', color='#e74c3c')
    plt.plot(results['random']['val_accs'], label='Val Acc', color='#c0392b', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Random Model: Accuracy Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=300, bbox_inches='tight')


# Run low-data regime experiment
def run_low_data_experiment(pretrained_model_path, tokenized_dir, output_dir,
                            ratios=[0.1, 0.25, 0.5, 0.75, 1.0],  # Adjusted ratios for small datasets
                            epochs=10, batch_size=4, learning_rate=1e-4):  # Smaller batch size
    """Test performance with different amounts of training data"""
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output directory
    low_data_dir = os.path.join(output_dir, "low_data_regime")
    os.makedirs(low_data_dir, exist_ok=True)

    # Load pre-trained model
    pretrained_model = load_pretrained_model(pretrained_model_path)
    pretrained_model.to(device)

    # Also create a randomly initialized model
    random_model = GPT(pretrained_model.config)
    for p in random_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)
    random_model.to(device)

    # Prepare BCI data
    token_files, labels, label_mapping = prepare_bci_data(tokenized_dir)
    num_classes = len(label_mapping)

    # Create full dataset
    full_dataset = BCIDataset(token_files, labels)

    # For very small datasets, adjust the test/train split
    full_size = len(full_dataset)
    print(f"Total dataset size: {full_size} samples")

    # Use a smaller test set for very small datasets
    test_size = max(1, min(int(0.2 * full_size), 2))  # At least 1, at most 2 samples for test
    train_val_size = full_size - test_size

    # Ensure we have at least 1 sample per class in the test set if possible
    if test_size >= num_classes:
        # Try to create a stratified test set
        class_indices = {}
        for i, label in enumerate(labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(i)

        # Take one sample from each class for testing
        test_indices = []
        for class_label, indices in class_indices.items():
            if indices:  # If we have samples for this class
                test_indices.append(indices[0])  # Take the first sample

        # If we need more test samples, add randomly
        if len(test_indices) < test_size:
            remaining = [i for i in range(full_size) if i not in test_indices]
            additional = random.sample(remaining, min(test_size - len(test_indices), len(remaining)))
            test_indices.extend(additional)

        train_val_indices = [i for i in range(full_size) if i not in test_indices]

        # Create stratified test dataset
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
        train_val_dataset = torch.utils.data.Subset(full_dataset, train_val_indices)
    else:
        # If dataset is too small for stratification, use random split
        train_val_dataset, test_dataset = random_split(
            full_dataset, [train_val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

    print(f"Split dataset: Train+Val={len(train_val_dataset)}, Test={len(test_dataset)}")
    test_loader = DataLoader(test_dataset, batch_size=min(batch_size, test_size), shuffle=False)

    # Results dictionary
    results = {
        'pretrained': {},
        'random': {},
        'dataset_info': {
            'total_size': full_size,
            'test_size': test_size,
            'train_val_size': train_val_size
        }
    }

    # Adjusted ratios based on dataset size
    adjusted_ratios = []
    for ratio in ratios:
        # Calculate how many samples this would be
        sample_count = max(1, int(ratio * train_val_size))
        # Back-calculate the actual ratio
        actual_ratio = sample_count / train_val_size
        adjusted_ratios.append(actual_ratio)
        print(f"Adjusted ratio {ratio:.2f} to {actual_ratio:.2f} ({sample_count} samples)")

    # Run experiments for each data ratio
    for ratio in adjusted_ratios:
        # Ensure at least 1 sample
        train_size = max(1, int(ratio * train_val_size))
        print(f"\n--- Running low-data experiment with {ratio * 100:.1f}% of training data ({train_size} samples) ---")

        # Sample without replacement
        indices = torch.randperm(train_val_size)[:train_size].tolist()

        # Create training subset
        train_dataset = torch.utils.data.Subset(train_val_dataset, indices)
        train_loader = DataLoader(train_dataset, batch_size=min(batch_size, train_size), shuffle=True)

        print(f"Training with {len(train_dataset)} samples")

        # 1. Train with pretrained model
        print("Training with pretrained model...")
        pretrained_classifier = BCIClassifier(pretrained_model, num_classes, freeze_backbone=True).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(pretrained_classifier.parameters(), lr=learning_rate)

        # Train
        for epoch in range(epochs):
            train_loss, train_acc, _ = train_epoch(
                pretrained_classifier, train_loader, criterion, optimizer, device
            )
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

        # Evaluate
        _, pretrained_acc, pretrained_f1, _, _, _ = evaluate(
            pretrained_classifier, test_loader, criterion, device
        )

        # 2. Train with random model
        print("Training with random model...")
        random_classifier = BCIClassifier(random_model, num_classes, freeze_backbone=False).to(device)

        optimizer = optim.AdamW(random_classifier.parameters(), lr=learning_rate)

        # Train
        for epoch in range(epochs):
            train_loss, train_acc, _ = train_epoch(
                random_classifier, train_loader, criterion, optimizer, device
            )
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

        # Evaluate
        _, random_acc, random_f1, _, _, _ = evaluate(
            random_classifier, test_loader, criterion, device
        )

        # Store results
        results['pretrained'][f"{ratio:.2f}"] = {
            'accuracy': pretrained_acc,
            'f1_score': pretrained_f1,
            'num_samples': train_size
        }

        results['random'][f"{ratio:.2f}"] = {
            'accuracy': random_acc,
            'f1_score': random_f1,
            'num_samples': train_size
        }

        print(f"Results with {ratio * 100:.1f}% data ({train_size} samples):")
        print(f"  Pretrained: Acc={pretrained_acc:.4f}, F1={pretrained_f1:.4f}")
        print(f"  Random: Acc={random_acc:.4f}, F1={random_f1:.4f}")

    # Create low-data regime plot
    plt.figure(figsize=(12, 7))

    # Extract the results
    ratio_values = [float(r) for r in results['pretrained'].keys()]
    sample_counts = [results['pretrained'][f"{r:.2f}"]['num_samples'] for r in ratio_values]
    pretrained_accs = [results['pretrained'][f"{r:.2f}"]['accuracy'] for r in ratio_values]
    random_accs = [results['random'][f"{r:.2f}"]['accuracy'] for r in ratio_values]

    # Plot by number of samples instead of ratio
    plt.plot(sample_counts, pretrained_accs, 'o-', label='Pretrained Model', linewidth=2, markersize=8, color='#3498db')
    plt.plot(sample_counts, random_accs, 'o-', label='Random Model', linewidth=2, markersize=8, color='#e74c3c')

    plt.xlabel('Number of Training Samples')
    plt.ylabel('Test Accuracy')
    plt.title('Low-Data Regime: Transfer Learning Effectiveness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.xticks(sample_counts, [str(s) for s in sample_counts])

    plt.tight_layout()
    plt.savefig(os.path.join(low_data_dir, 'low_data_regime.png'), dpi=300, bbox_inches='tight')

    # Save results
    with open(os.path.join(low_data_dir, 'low_data_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    return results
if __name__ == "__main__":
    # Set paths
    pretrained_model_path = "log/model_03047.pt"  # Update with your model path
    tokenized_dir = "tokenized_bci_data"
    output_dir = "bci_evaluation_results"

    # Run main evaluation
    print("Running main evaluation experiment...")
    evaluate_bci_classification(
        pretrained_model_path=pretrained_model_path,
        tokenized_dir=tokenized_dir,
        output_dir=output_dir,
        epochs=15,
        batch_size=16,
        learning_rate=1e-4
    )

    # Run low-data regime experiment
    print("\nRunning low-data regime experiment...")
    run_low_data_experiment(
        pretrained_model_path=pretrained_model_path,
        tokenized_dir=tokenized_dir,
        output_dir=output_dir,
        ratios=[0.05, 0.1, 0.25, 0.5, 1.0],
        epochs=10
    )

    print("\nAll experiments completed!")