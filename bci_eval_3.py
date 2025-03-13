import os
import json
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd


# Model architecture components
class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, window_size, dropout=0.1):
        super().__init__()
        self.attn = HierarchicalMemoryEfficientAttention(
            d_model, n_heads, window_size, dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attn(self.norm1(x))
        x = x + self.dropout(attn_output)
        ff_output = self.ff(self.norm2(x))
        x = x + self.dropout(ff_output)
        return x


class HierarchicalMemoryEfficientAttention(nn.Module):
    def __init__(self, d_model, n_heads, window_size, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size

        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Project inputs to queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Calculate number of windows in the sequence
        num_windows = (seq_len + self.window_size - 1) // self.window_size

        # Initialize output tensor
        context = torch.zeros_like(q)

        # Process each window separately for memory efficiency
        for i in range(num_windows):
            # Define current window's range
            q_start = i * self.window_size
            q_end = min((i + 1) * self.window_size, seq_len)

            if q_start >= seq_len:
                break

            # Get query for current window
            q_window = q[:, q_start:q_end]  # [B, window_size, H, D]

            # This window can attend to itself (bidirectional) and all previous windows (causal)
            context_window = torch.zeros_like(q_window)

            # 1. WITHIN-WINDOW ATTENTION (bidirectional)
            # Process current window attending to itself (full bidirectional)
            scores = torch.matmul(
                q_window.transpose(1, 2),  # [B, H, window_size, D]
                k[:, q_start:q_end].transpose(1, 2).transpose(-1, -2)  # [B, H, D, window_size]
            ) / math.sqrt(self.head_dim)

            # Apply attention
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Apply attention to values
            context_window += torch.matmul(
                attn_weights,
                v[:, q_start:q_end].transpose(1, 2)  # [B, H, window_size, D]
            ).transpose(1, 2)  # [B, window_size, H, D]

            # 2. CROSS-WINDOW ATTENTION (causal only)
            # For each previous window (causal attention)
            for j in range(i):
                k_start = j * self.window_size
                k_end = min((j + 1) * self.window_size, seq_len)

                # Compute attention scores with previous window
                scores = torch.matmul(
                    q_window.transpose(1, 2),  # [B, H, window_size, D]
                    k[:, k_start:k_end].transpose(1, 2).transpose(-1, -2)  # [B, H, D, window_size]
                ) / math.sqrt(self.head_dim)

                # Apply attention
                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = self.dropout(attn_weights)

                # Apply attention to values and accumulate
                context_window += torch.matmul(
                    attn_weights,
                    v[:, k_start:k_end].transpose(1, 2)  # [B, H, window_size, D]
                ).transpose(1, 2)  # [B, window_size, H, D]

            # Place result in output tensor
            context[:, q_start:q_end] = context_window

        # Reshape back to [batch, seq_len, d_model]
        context = context.reshape(batch_size, seq_len, self.d_model)

        return self.out_proj(context)


class HierarchicalEEGTransformer(nn.Module):
    def __init__(self,
                 codebook_size,
                 window_size=2304,
                 d_model=768,
                 n_heads=12,
                 n_layers=6,
                 max_windows=50,
                 pad_token_id=129):
        super().__init__()
        self.d_model = d_model

        self.window_size = window_size
        self.pad_token_id = pad_token_id

        # Token embedding
        self.token_embedding = nn.Embedding(codebook_size, d_model)

        # Position encodings for both window-level and token-level
        self.window_pos_embed = nn.Parameter(torch.zeros(1, max_windows, d_model))
        self.token_pos_embed = nn.Parameter(torch.zeros(1, window_size, d_model))

        # Initialize positional embeddings
        nn.init.normal_(self.window_pos_embed, std=0.02)
        nn.init.normal_(self.token_pos_embed, std=0.02)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, window_size=window_size) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, codebook_size)

    def _extract_windows_with_pad_tokens(self, x):
        """
        Extract windows from sequences that have PAD tokens between windows.
        Each window consists of window_size tokens followed by a PAD token.

        Returns:
            - windows: Tensor of shape [batch_size, num_windows, window_size]
            - pad_mask: Boolean mask indicating positions of pad tokens
        """
        batch_size, seq_length = x.shape
        device = x.device

        # Create a mask where True indicates pad tokens
        pad_mask = (x == self.pad_token_id)

        # Now we'll separate the windows based on the pad tokens
        windows = []

        for b in range(batch_size):
            # Find all pad token positions
            pad_positions = torch.where(pad_mask[b])[0]

            if len(pad_positions) == 0:
                # No pad tokens found, try to handle as single window
                if seq_length >= self.window_size:
                    windows.append(x[b, :self.window_size])
                else:
                    # Not enough tokens for a window, pad to window_size
                    padded_window = torch.cat([
                        x[b, :],
                        torch.full((self.window_size - seq_length,), self.pad_token_id,
                                   dtype=x.dtype, device=device)
                    ])
                    windows.append(padded_window)
            else:
                # Extract windows based on pad positions
                start_idx = 0
                b_windows = []

                for pad_pos in pad_positions:
                    # Window should be tokens from start_idx to pad_pos
                    if pad_pos - start_idx > 0:  # Ensure window has tokens
                        window = x[b, start_idx:pad_pos]

                        # Make sure window has correct size
                        if len(window) == self.window_size:
                            b_windows.append(window)
                        elif len(window) < self.window_size:
                            # Pad short window
                            padded_window = torch.cat([
                                window,
                                torch.full((self.window_size - len(window),), self.pad_token_id,
                                           dtype=x.dtype, device=device)
                            ])
                            b_windows.append(padded_window)
                        else:
                            # Truncate long window
                            b_windows.append(window[:self.window_size])

                    # Move to tokens after this pad
                    start_idx = pad_pos + 1

                # Check if there are tokens after the last pad (before EOS)
                if start_idx < seq_length:
                    window = x[b, start_idx:seq_length]
                    if len(window) > 0:
                        if len(window) == self.window_size:
                            b_windows.append(window)
                        elif len(window) < self.window_size:
                            # Pad short window
                            padded_window = torch.cat([
                                window,
                                torch.full((self.window_size - len(window),), self.pad_token_id,
                                           dtype=x.dtype, device=device)
                            ])
                            b_windows.append(padded_window)
                        else:
                            # Truncate long window
                            b_windows.append(window[:self.window_size])

                # Stack all windows for this batch item
                if b_windows:
                    windows.append(torch.stack(b_windows))

        # Pad to make all batches have same number of windows
        max_windows = max([w.size(0) for w in windows]) if windows else 0
        padded_windows = []

        for w in windows:
            num_windows = w.size(0)
            if num_windows < max_windows:
                # Pad with zeros
                padding = torch.zeros(
                    max_windows - num_windows, self.window_size,
                    dtype=x.dtype, device=device
                )
                padded_windows.append(torch.cat([w, padding], dim=0))
            else:
                padded_windows.append(w)

        if not padded_windows:
            # Handle case where no valid windows were found
            return torch.zeros(batch_size, 1, self.window_size, device=device), pad_mask

        # Stack across batch dimension
        return torch.stack(padded_windows), pad_mask

    def forward(self, x):
        """
        Args:
            x: Tensor of token indices [batch_size, seq_length]
        """
        batch_size, seq_length = x.shape
        device = x.device

        # Handle the special case of sequences with pad tokens between windows
        windows, pad_mask = self._extract_windows_with_pad_tokens(x)

        # Get the shape after extracting windows
        batch_size, num_windows, window_size = windows.shape

        # Reshape for embedding lookup
        flat_windows = windows.reshape(-1, window_size)

        # Get token embeddings
        embedded = self.token_embedding(flat_windows)  # [B*N, W, D]

        # Reshape to separate batch and window dimensions
        embedded = embedded.reshape(batch_size, num_windows, window_size, -1)

        # Add positional encodings
        # 1. Window-level positions
        embedded = embedded + self.window_pos_embed[:, :num_windows, :].unsqueeze(2)

        # 2. Token-level positions
        embedded = embedded + self.token_pos_embed[:, :window_size, :].unsqueeze(1)

        # Reshape back to sequence for transformer processing
        embedded = embedded.reshape(batch_size, num_windows * window_size, -1)

        # Create batch-specific hierarchical attention mask
        mask = self._create_hierarchical_mask(batch_size, num_windows, window_size, device)

        # Apply transformer layers
        x = embedded
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)

        # Output projection
        logits = self.output_head(x)

        return logits

    def _create_hierarchical_mask(self, batch_size, num_windows, window_size, device):
        """
        Create mask that allows:
        1. Full attention within each window (bidirectional)
        2. Causal attention between windows
        """
        seq_length = num_windows * window_size
        # Create batch-specific masks [B, seq_len, seq_len]
        mask = torch.ones(batch_size, seq_length, seq_length, device=device) * float('-inf')

        # Allow full attention within each window
        for i in range(num_windows):
            start_idx = i * window_size
            end_idx = (i + 1) * window_size
            mask[:, start_idx:end_idx, start_idx:end_idx] = 0

        # Allow causal attention between windows
        for i in range(num_windows):
            for j in range(i):
                i_start = i * window_size
                i_end = (i + 1) * window_size
                j_start = j * window_size
                j_end = (j + 1) * window_size
                mask[:, i_start:i_end, j_start:j_end] = 0

        return mask


# Evaluation DataLoader
class EEGTransformerEvaluationDataLoader:
    def __init__(self, data_dir, pad_token_id=129, eos_token_id=128, window_size=2304):
        """
        DataLoader for evaluating the EEG Transformer model.

        Args:
            data_dir (str): Directory containing tokenized EEG data files
            pad_token_id (int): ID of the padding token
            eos_token_id (int): ID of the end-of-sequence token
            window_size (int): Size of each EEG window (default: 2304 for 72x32)
        """
        self.data_dir = data_dir
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.window_size = window_size

        # Load all token files and parse class information
        self.token_files = sorted(glob.glob(os.path.join(data_dir, "*_tokens.pt")))
        if not self.token_files:
            raise ValueError(f"No token files found in {data_dir}")

        # Extract class names from filenames
        self.class_names = []
        self.file_class_mapping = {}

        for file_path in self.token_files:
            filename = os.path.basename(file_path)
            # Extract class name from filename (assuming format: dataset_class_tokens.pt)
            parts = filename.split('_')
            if len(parts) >= 3 and parts[-1] == "tokens.pt":
                class_name = parts[-2]  # Second to last part before _tokens.pt
                self.class_names.append(class_name)
                self.file_class_mapping[file_path] = class_name

        # Get unique class names
        self.class_names = sorted(list(set(self.class_names)))

        # Check for label mapping file
        mapping_file = os.path.join(data_dir, "label_mapping.json")
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                mapping_data = json.load(f)
                self.label_mapping = mapping_data.get("label_mapping", {})
        else:
            # Create default mapping (alphabetical)
            self.label_mapping = {name: i for i, name in enumerate(self.class_names)}

        # Create reverse mapping for convenience
        self.index_to_class = {v: k for k, v in self.label_mapping.items()}

        # Organize files by class
        self.files_by_class = {}
        for file_path, class_name in self.file_class_mapping.items():
            if class_name not in self.files_by_class:
                self.files_by_class[class_name] = []
            self.files_by_class[class_name].append(file_path)

        # Cache for loaded windows
        self.window_cache = {}

        print(f"Found {len(self.token_files)} token files across {len(self.class_names)} classes")
        for class_name in self.class_names:
            if class_name in self.files_by_class:
                print(
                    f"  Class '{class_name}' (index {self.label_mapping[class_name]}): {len(self.files_by_class[class_name])} files")

    def load_windows_from_file(self, file_path):
        """
        Load tokenized EEG windows from a file.

        Args:
            file_path (str): Path to tokenized EEG file

        Returns:
            list: List of window tensors
        """
        # Check cache first
        if file_path in self.window_cache:
            return self.window_cache[file_path]

        # Load tokens from file
        tokens = torch.load(file_path)

        # Extract windows based on pad tokens
        windows = []
        start_idx = 0

        # Find pad token positions
        pad_mask = (tokens == self.pad_token_id)
        pad_positions = torch.where(pad_mask)[0]

        if len(pad_positions) == 0:
            # No pad tokens found, try to handle as single window
            if len(tokens) >= self.window_size:
                windows.append(tokens[:self.window_size])
        else:
            # Extract windows based on pad positions
            for pad_pos in pad_positions:
                # Window should be tokens from start_idx to pad_pos
                if pad_pos - start_idx > 0:  # Ensure window has tokens
                    window = tokens[start_idx:pad_pos]

                    # Check if the window is exactly the right size
                    if len(window) == self.window_size:
                        windows.append(window)
                    elif len(window) < self.window_size:
                        # Skip windows that are too small
                        pass
                    else:
                        # Truncate windows that are too large
                        windows.append(window[:self.window_size])

                # Move to tokens after this pad
                start_idx = pad_pos + 1

            # Check if there are tokens after the last pad (before EOS)
            if start_idx < len(tokens):
                window = tokens[start_idx:]
                # Check for EOS token at the end
                eos_pos = torch.where(window == self.eos_token_id)[0]
                if len(eos_pos) > 0:
                    window = window[:eos_pos[0]]  # Truncate at EOS

                if len(window) == self.window_size:
                    windows.append(window)
                elif len(window) > self.window_size:
                    windows.append(window[:self.window_size])

        # Cache the result
        self.window_cache[file_path] = windows

        return windows

    def get_few_shot_batch(self, n_way, n_shot, n_query):
        """
        Create a few-shot batch for evaluation.

        Args:
            n_way (int): Number of classes to discriminate between
            n_shot (int): Number of examples per class for support set
            n_query (int): Number of examples per class for query set

        Returns:
            dict: Dictionary containing support and query sets with labels
        """
        # Make sure we have enough classes
        if n_way > len(self.class_names):
            raise ValueError(f"Requested {n_way}-way task but only have {len(self.class_names)} classes")

        # Select n_way classes randomly
        selected_classes = random.sample(self.class_names, n_way)

        support_windows = []
        support_labels = []
        query_windows = []
        query_labels = []

        for class_idx, class_name in enumerate(selected_classes):
            class_files = self.files_by_class[class_name]

            # Need at least n_shot + n_query windows for this class
            all_windows = []

            # Load windows until we have enough
            random.shuffle(class_files)
            for file_path in class_files:
                if len(all_windows) >= n_shot + n_query:
                    break

                file_windows = self.load_windows_from_file(file_path)
                all_windows.extend(file_windows)

            # If we still don't have enough windows, repeat some
            if len(all_windows) < n_shot + n_query:
                if len(all_windows) == 0:
                    raise ValueError(f"No windows found for class {class_name}")
                all_windows = all_windows * ((n_shot + n_query) // len(all_windows) + 1)

            # Shuffle the windows
            random.shuffle(all_windows)

            # Select support and query sets
            support_windows.extend(all_windows[:n_shot])
            support_labels.extend([class_idx] * n_shot)  # Use sequential indices for few-shot

            query_windows.extend(all_windows[n_shot:n_shot + n_query])
            query_labels.extend([class_idx] * n_query)

        # Convert to tensors and add pad tokens for the model's expected format
        support_windows_with_pad = []
        for window in support_windows:
            # Add a single pad token after the window (as expected by the model)
            window_with_pad = torch.cat([
                window,
                torch.tensor([self.pad_token_id])
            ])
            support_windows_with_pad.append(window_with_pad.unsqueeze(0))  # Add batch dimension

        query_windows_with_pad = []
        for window in query_windows:
            window_with_pad = torch.cat([
                window,
                torch.tensor([self.pad_token_id])
            ])
            query_windows_with_pad.append(window_with_pad.unsqueeze(0))

        return {
            'support_windows': support_windows_with_pad,
            'support_labels': torch.tensor(support_labels),
            'query_windows': query_windows_with_pad,
            'query_labels': torch.tensor(query_labels),
            'class_names': [selected_classes[i] for i in range(n_way)],
            'n_way': n_way,
            'n_shot': n_shot
        }


import os
import json
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
class EEGSimpleEvaluator:
    def __init__(self,
                 checkpoint_dir,
                 data_dir,
                 device="cuda",
                 codebook_size=130,
                 window_size=2304,
                 d_model=768,
                 n_heads=12,
                 n_layers=6,
                 max_windows=50,
                 pad_token_id=129,
                 eos_token_id=128):
        """
        Simple evaluator for the Hierarchical EEG Transformer model.

        Args:
            checkpoint_dir (str): Directory containing model checkpoints
            data_dir (str): Directory containing tokenized EEG data
            device (str): Device to run evaluation on ('cuda' or 'cpu')
            codebook_size (int): Size of the VQVAE codebook
            window_size (int): Size of each EEG window (default: 2304 for 72x32)
            d_model (int): Hidden dimension of the model
            n_heads (int): Number of attention heads
            n_layers (int): Number of transformer layers
            max_windows (int): Maximum number of windows in sequence
            pad_token_id (int): ID of the padding token
            eos_token_id (int): ID of the end-of-sequence token
        """
        self.checkpoint_dir = checkpoint_dir
        self.data_dir = data_dir
        self.device = torch.device(device)
        self.codebook_size = codebook_size
        self.window_size = window_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        # Initialize model
        self.model = HierarchicalEEGTransformer(
            codebook_size=codebook_size,
            window_size=window_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_windows=max_windows,
            pad_token_id=pad_token_id
        ).to(self.device)

        # Initialize dataloader
        self.dataloader = EEGTransformerEvaluationDataLoader(
            data_dir=data_dir,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            window_size=window_size
        )

        # Latest checkpoint path
        self.latest_checkpoint = self._find_latest_checkpoint()
        if self.latest_checkpoint:
            print(f"Found latest checkpoint: {self.latest_checkpoint}")

    def _find_latest_checkpoint(self):
        """Find the latest checkpoint in the checkpoint directory"""
        checkpoints = sorted(glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_epoch_*.pt")))
        return checkpoints[-1] if checkpoints else None

    def load_checkpoint(self, checkpoint_path=None):
        """
        Load a model checkpoint.

        Args:
            checkpoint_path (str, optional): Path to specific checkpoint. If None, loads latest.

        Returns:
            bool: True if checkpoint loaded successfully, False otherwise
        """
        if checkpoint_path is None:
            checkpoint_path = self.latest_checkpoint

        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print(f"Checkpoint {checkpoint_path} not found")
            return False

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from {checkpoint_path}")
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False

    def _get_window_representation(self, window_tensor):
        """
        Extract representation from a single window using the transformer.

        Args:
            window_tensor (torch.Tensor): Window tensor [1, seq_len] with pad token

        Returns:
            torch.Tensor: Window representation vector
        """
        self.model.eval()
        with torch.no_grad():
            # Forward pass through the model
            output = self.model(window_tensor.to(self.device))

            # Get the representation from the last non-pad token
            # We use the output right before the pad token
            batch_size, seq_len, _ = output.shape

            # For each batch, find the position right before the pad token
            pad_positions = (window_tensor == self.pad_token_id).nonzero()
            if len(pad_positions) > 0:
                # If pad token found, use position right before it
                rep_pos = pad_positions[0, 1] - 1
                # Make sure it's valid
                rep_pos = max(0, min(rep_pos, seq_len - 1))
            else:
                # If no pad token found, use the last position
                rep_pos = seq_len - 1

            # Extract representation
            representation = output[0, rep_pos]

            return representation

    def evaluate_few_shot(self, n_way=2, n_shot=5, n_query=5, n_trials=10, verbose=True):
        """
        Evaluate few-shot learning performance.

        Args:
            n_way (int): Number of classes to discriminate between
            n_shot (int): Number of examples per class for support set
            n_query (int): Number of examples per class for query set
            n_trials (int): Number of random trials to run
            verbose (bool): Whether to print detailed results

        Returns:
            float: Average accuracy across trials
        """
        # Make sure model is loaded
        if not hasattr(self, 'model') or self.model is None:
            if not self.load_checkpoint():
                raise ValueError("No model loaded and no checkpoint found")

        self.model.eval()

        accuracies = []
        for trial in tqdm(range(n_trials), desc=f"Evaluating {n_way}-way {n_shot}-shot"):
            # Get a few-shot batch
            batch = self.dataloader.get_few_shot_batch(n_way, n_shot, n_query)

            # Extract representations for support set
            support_reps = []
            for window in batch['support_windows']:
                rep = self._get_window_representation(window)
                support_reps.append(rep)
            support_reps = torch.stack(support_reps)

            # Extract representations for query set
            query_reps = []
            for window in batch['query_windows']:
                rep = self._get_window_representation(window)
                query_reps.append(rep)
            query_reps = torch.stack(query_reps)

            # Compute prototypes for each class (mean of support examples)
            prototypes = torch.zeros(n_way, support_reps.shape[1], device=self.device)
            for c in range(n_way):
                # Get all support representations for this class
                class_mask = (batch['support_labels'] == c)
                class_reps = support_reps[class_mask]
                # Compute prototype as mean
                if len(class_reps) > 0:
                    prototypes[c] = class_reps.mean(dim=0)

            # Compute distances between query examples and prototypes
            # Using negative L2 distance (squared Euclidean)
            dists = torch.cdist(query_reps, prototypes, p=2)

            # Predict classes based on distances
            _, predictions = torch.min(dists, dim=1)

            # Compute accuracy
            correct = (predictions == batch['query_labels'].to(self.device)).sum().item()
            accuracy = correct / len(batch['query_labels'])
            accuracies.append(accuracy)

            if verbose:
                print(f"Trial {trial + 1}/{n_trials} - Accuracy: {accuracy:.4f}")

                # Print class-wise accuracies
                for c in range(n_way):
                    class_mask = (batch['query_labels'] == c)
                    if class_mask.sum() > 0:
                        class_correct = (predictions[class_mask] == c).sum().item()
                        class_acc = class_correct / class_mask.sum().item()
                        print(
                            f"  Class '{batch['class_names'][c]}': {class_acc:.4f} ({class_correct}/{class_mask.sum().item()})")

        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        if verbose:
            print(f"\n{n_way}-way {n_shot}-shot evaluation results:")
            print(f"  Mean accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")

        return mean_accuracy

    def evaluate_across_shots(self, n_way=2, shots=[1, 5, 10], n_query=5, n_trials=10):
        """
        Evaluate few-shot learning across different numbers of shots.

        Args:
            n_way (int): Number of classes
            shots (list): List of shot values to evaluate
            n_query (int): Number of query examples per class
            n_trials (int): Number of trials per configuration

        Returns:
            dict: Results with shot values as keys and accuracies as values
        """
        results = {}

        for n_shot in shots:
            print(f"\nEvaluating {n_way}-way {n_shot}-shot learning...")
            accuracy = self.evaluate_few_shot(
                n_way=n_way,
                n_shot=n_shot,
                n_query=n_query,
                n_trials=n_trials
            )
            results[n_shot] = accuracy

        # Plot results
        plt.figure(figsize=(10, 6))
        shots_list = list(results.keys())
        accuracies = [results[shot] for shot in shots_list]

        plt.plot(shots_list, accuracies, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Number of shots', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.title(f'{n_way}-way few-shot learning performance', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.xticks(shots_list)

        # Set y-axis limits with some padding
        plt.ylim(max(0, min(accuracies) - 0.05), min(1.0, max(accuracies) + 0.05))

        # Add values on plot
        for i, acc in enumerate(accuracies):
            plt.annotate(f'{acc:.4f}',
                         (shots_list[i], accuracies[i]),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center')

        plt.tight_layout()
        return results

    def evaluate_generalization(self, source_classes, target_classes, n_shot=5, n_query=5, n_trials=10):
        """
        Evaluate cross-class generalization by training on some classes and testing on others.

        Args:
            source_classes (list): Classes to use in support set
            target_classes (list): Classes to use in query set
            n_shot (int): Number of examples per source class
            n_query (int): Number of examples per target class
            n_trials (int): Number of random trials to run

        Returns:
            float: Average generalization accuracy
        """
        # Validate class names
        for c in source_classes + target_classes:
            if c not in self.dataloader.class_names:
                raise ValueError(f"Class '{c}' not found in dataset")

        self.model.eval()

        accuracies = []
        for trial in tqdm(range(n_trials), desc=f"Evaluating generalization"):
            # Get support examples from source classes
            support_windows = []
            support_labels = []

            for i, class_name in enumerate(source_classes):
                class_files = self.dataloader.files_by_class[class_name]
                all_windows = []

                # Load windows
                random.shuffle(class_files)
                for file_path in class_files:
                    if len(all_windows) >= n_shot:
                        break
                    file_windows = self.dataloader.load_windows_from_file(file_path)
                    all_windows.extend(file_windows)

                # Ensure we have enough, repeat if necessary
                if len(all_windows) < n_shot:
                    all_windows = all_windows * ((n_shot) // len(all_windows) + 1)

                # Add to support set
                support_windows.extend(all_windows[:n_shot])
                support_labels.extend([i] * n_shot)

            # Get query examples from target classes
            query_windows = []
            query_labels = []
            query_class_to_idx = {c: i for i, c in enumerate(target_classes)}

            for class_name in target_classes:
                class_idx = query_class_to_idx[class_name]
                class_files = self.dataloader.files_by_class[class_name]
                all_windows = []

                # Load windows
                random.shuffle(class_files)
                for file_path in class_files:
                    if len(all_windows) >= n_query:
                        break
                    file_windows = self.dataloader.load_windows_from_file(file_path)
                    all_windows.extend(file_windows)

                # Ensure we have enough, repeat if necessary
                if len(all_windows) < n_query:
                    all_windows = all_windows * ((n_query) // len(all_windows) + 1)

                # Add to query set
                query_windows.extend(all_windows[:n_query])
                query_labels.extend([class_idx] * n_query)

            # Format windows for model
            support_windows_with_pad = []
            for window in support_windows:
                window_with_pad = torch.cat([
                    window,
                    torch.tensor([self.pad_token_id])
                ])
                support_windows_with_pad.append(window_with_pad.unsqueeze(0))

            query_windows_with_pad = []
            for window in query_windows:
                window_with_pad = torch.cat([
                    window,
                    torch.tensor([self.pad_token_id])
                ])
                query_windows_with_pad.append(window_with_pad.unsqueeze(0))

            # Extract representations
            support_reps = []
            for window in support_windows_with_pad:
                rep = self._get_window_representation(window)
                support_reps.append(rep)
            support_reps = torch.stack(support_reps)

            query_reps = []
            for window in query_windows_with_pad:
                rep = self._get_window_representation(window)
                query_reps.append(rep)
            query_reps = torch.stack(query_reps)

            # Compute prototypes for source classes
            prototypes = torch.zeros(len(source_classes), support_reps.shape[1], device=self.device)
            for c in range(len(source_classes)):
                # Get all support representations for this class
                class_mask = (torch.tensor(support_labels) == c)
                class_reps = support_reps[class_mask]
                # Compute prototype as mean
                if len(class_reps) > 0:
                    prototypes[c] = class_reps.mean(dim=0)

            # For each target class, find the closest source class prototype
            target_to_source = {}
            for tc in range(len(target_classes)):
                # Get all query examples for this target class
                tc_mask = (torch.tensor(query_labels) == tc)
                tc_reps = query_reps[tc_mask]

                if len(tc_reps) > 0:
                    # Compute mean representation for this target class
                    tc_mean = tc_reps.mean(dim=0, keepdim=True)

                    # Find closest source prototype
                    dists = torch.cdist(tc_mean, prototypes, p=2)[0]
                    closest_source = dists.argmin().item()

                    target_to_source[tc] = closest_source

            # Compute accuracy based on mapping
            correct = 0
            total = len(query_labels)

            for i, query_label in enumerate(query_labels):
                query_rep = query_reps[i]

                # Find closest source prototype
                dists = torch.cdist(query_rep.unsqueeze(0), prototypes, p=2)[0]
                pred_source = dists.argmin().item()

                # Check if mapping is correct
                if pred_source == target_to_source.get(query_label, -1):
                    correct += 1

            accuracy = correct / total
            accuracies.append(accuracy)

        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        print(f"\nGeneralization from {source_classes} to {target_classes}:")
        print(f"  Mean accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")

        return mean_accuracy


# Utility functions for creating and visualizing embeddings
def create_tsne_visualization(evaluator, classes=None, samples_per_class=20, perplexity=30):
    """
    Create t-SNE visualization of window embeddings.

    Args:
        evaluator (EEGSimpleEvaluator): Evaluator instance
        classes (list, optional): List of classes to visualize. If None, use all classes.
        samples_per_class (int): Number of samples per class to use
        perplexity (int): t-SNE perplexity parameter

    Returns:
        plt.Figure: Matplotlib figure with t-SNE plot
    """
    from sklearn.manifold import TSNE

    if classes is None:
        classes = evaluator.dataloader.class_names

    # Collect windows and labels
    windows = []
    labels = []
    class_indices = []

    for i, class_name in enumerate(classes):
        class_files = evaluator.dataloader.files_by_class.get(class_name, [])
        if not class_files:
            continue

        # Get windows from files
        class_windows = []
        for file_path in class_files:
            file_windows = evaluator.dataloader.load_windows_from_file(file_path)
            class_windows.extend(file_windows)
            if len(class_windows) >= samples_per_class:
                break

        # Select random samples if we have more than needed
        if len(class_windows) > samples_per_class:
            class_windows = random.sample(class_windows, samples_per_class)

        # Add to collection
        windows.extend(class_windows)
        labels.extend([class_name] * len(class_windows))
        class_indices.extend([i] * len(class_windows))

    # Extract embeddings
    embeddings = []
    for window in tqdm(windows, desc="Extracting embeddings"):
        # Add pad token and batch dimension
        window_with_pad = torch.cat([
            window,
            torch.tensor([evaluator.pad_token_id])
        ]).unsqueeze(0)

        # Get representation
        with torch.no_grad():
            rep = evaluator._get_window_representation(window_with_pad)
            embeddings.append(rep.cpu().numpy())

    # Convert to numpy array
    embeddings = np.array(embeddings)

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each class with a different color
    for i, class_name in enumerate(classes):
        mask = [j == i for j in class_indices]
        if sum(mask) > 0:  # Only plot if we have samples
            x = reduced_embeddings[mask, 0]
            y = reduced_embeddings[mask, 1]
            ax.scatter(x, y, label=class_name, alpha=0.7)

    ax.legend()
    ax.set_title("t-SNE Visualization of EEG Window Embeddings")

    return fig


def run_evaluation_suite(evaluator, output_dir="evaluation_results"):
    """
    Run a comprehensive evaluation suite and save results.

    Args:
        evaluator (EEGSimpleEvaluator): Evaluator instance
        output_dir (str): Directory to save results

    Returns:
        dict: Dictionary with all evaluation results
    """
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # 1. Evaluate standard few-shot learning
    print("\nEvaluating standard few-shot learning...")
    standard_results = {}

    # a. 2-way classification
    for n_shot in [1, 5, 10]:
        accuracy = evaluator.evaluate_few_shot(n_way=2, n_shot=n_shot, n_query=5, n_trials=10)
        standard_results[f"2way_{n_shot}shot"] = accuracy

    # b. 5-way classification
    for n_shot in [1, 5, 10]:
        accuracy = evaluator.evaluate_few_shot(n_way=5, n_shot=n_shot, n_query=5, n_trials=10)
        standard_results[f"5way_{n_shot}shot"] = accuracy

    results["standard_few_shot"] = standard_results

    # 2. Create t-SNE visualization
    print("\nCreating t-SNE visualization...")
    try:
        fig = create_tsne_visualization(evaluator)
        fig.savefig(os.path.join(output_dir, "tsne_visualization.png"), dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f"Error creating t-SNE visualization: {e}")

    # 3. Save all results
    print("\nSaving results...")
    results_df = pd.DataFrame({
        'Configuration': list(standard_results.keys()),
        'Accuracy': list(standard_results.values())
    })
    results_df.to_csv(os.path.join(output_dir, "evaluation_results.csv"), index=False)

    # Create summary plot
    plt.figure(figsize=(12, 6))

    # Group by n_way
    way2_results = {k: v for k, v in standard_results.items() if k.startswith("2way_")}
    way5_results = {k: v for k, v in standard_results.items() if k.startswith("5way_")}

    # Extract shots
    shots = [int(k.split("_")[1].replace("shot", "")) for k in way2_results.keys()]

    # Plot
    plt.plot(shots, list(way2_results.values()), 'o-', label="2-way", linewidth=2, markersize=8)
    plt.plot(shots, list(way5_results.values()), 'o-', label="5-way", linewidth=2, markersize=8)

    plt.xlabel("Number of shots", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title("Few-shot Learning Performance", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(shots)

    # Add values on plot
    for i, (shot, acc) in enumerate(zip(shots, list(way2_results.values()))):
        plt.annotate(f'{acc:.4f}',
                     (shot, acc),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center')

    for i, (shot, acc) in enumerate(zip(shots, list(way5_results.values()))):
        plt.annotate(f'{acc:.4f}',
                     (shot, acc),
                     textcoords="offset points",
                     xytext=(0, -15),
                     ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "few_shot_summary.png"), dpi=300)

    print(f"\nEvaluation complete. Results saved to {output_dir}")
    return results


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EEG Transformer Few-Shot Evaluation")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory containing model checkpoints")
    parser.add_argument("--data_dir", type=str, default="tokenized_data",
                        help="Directory containing tokenized EEG data")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--n_way", type=int, default=2,
                        help="Number of ways for few-shot evaluation")
    parser.add_argument("--n_shot", type=int, default=5,
                        help="Number of shots for few-shot evaluation")
    parser.add_argument("--n_trials", type=int, default=10,
                        help="Number of trials for few-shot evaluation")
    parser.add_argument("--codebook_size", type=int, default=130,
                        help="Size of the VQVAE codebook")
    parser.add_argument("--window_size", type=int, default=2304,
                        help="Size of each EEG window (72x32)")
    parser.add_argument("--d_model", type=int, default=768,
                        help="Hidden dimension of the model")
    parser.add_argument("--n_heads", type=int, default=12,
                        help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=6,
                        help="Number of transformer layers")
    parser.add_argument("--full_suite", action="store_true",
                        help="Run full evaluation suite")

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = EEGSimpleEvaluator(
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        codebook_size=args.codebook_size,
        window_size=args.window_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers
    )

    # Load checkpoint
    evaluator.load_checkpoint()

    if args.full_suite:
        # Run full evaluation suite
        run_evaluation_suite(evaluator, output_dir=args.output_dir)
    else:
        # Run single evaluation
        accuracy = evaluator.evaluate_few_shot(
            n_way=args.n_way,
            n_shot=args.n_shot,
            n_query=args.n_shot,  # Use same number for query
            n_trials=args.n_trials
        )
        print(f"\nFinal {args.n_way}-way {args.n_shot}-shot accuracy: {accuracy:.4f}")