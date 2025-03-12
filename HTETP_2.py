import os
import glob
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from bci_eval_2 import EEGSimpleEvaluator
import pandas as pd


# Model definition with special handling for pad tokens between windows
class HierarchicalEEGTransformer(nn.Module):
    def __init__(self,
                 codebook_size,  # Size of your VQAE codebook
                 window_size=2304,  # Flattened size (72x32)
                 d_model=768,  # Hidden dimension
                 n_heads=12,  # Attention heads
                 n_layers=6,  # Transformer layers
                 max_windows=50,  # Max sequence length in windows
                 pad_token_id=0):  # ID of the pad token
        super().__init__()
        self.d_model = d_model  # <--- Add this line

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
            TransformerLayer(d_model, n_heads) for _ in range(n_layers)
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
        # We know they should occur every window_size+1 positions
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
                    padded_window = torch.cat([window, torch.full((self.window_size - len(window),), self.pad_token_id,
                                                                  dtype=x.dtype, device=device)])
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
                            padded_window = torch.cat([window,
                                                       torch.full((self.window_size - len(window),), self.pad_token_id,
                                                                  dtype=x.dtype, device=device)])
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
                            padded_window = torch.cat([window,
                                                       torch.full((self.window_size - len(window),), self.pad_token_id,
                                                                  dtype=x.dtype, device=device)])
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

        Now creates a separate mask for each batch element
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


class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()

        # Self-attention block
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Feed-forward block
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attn(self.norm1(x), mask)
        x = x + self.dropout(attn_output)

        # Feed-forward with residual connection
        ff_output = self.ff(self.norm2(x))
        x = x + self.dropout(ff_output)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Project and reshape to [batch, heads, seq_len, head_dim]
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # Apply mask if provided - now handling batch-specific masks
        if mask is not None:
            # mask is now [batch_size, seq_len, seq_len]
            # we need to add a dimension for heads
            scores = scores + mask.unsqueeze(1)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)

        # Reshape back to [batch, seq_len, d_model]
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.out_proj(context)


# Custom DataLoader for EEG Tokens with pad handling
class EEGTokenDataLoader:
    def __init__(self, B, T, process_rank, num_processes, token_files, window_size=2304,
                 pad_token_id=0, eos_token_id=1, split='train'):
        """
        Args:
            B: Batch size
            T: Sequence length
            process_rank: Current process rank for DDP
            num_processes: Total number of processes for DDP
            token_files: List of token file paths
            window_size: Size of each EEG window (should be 2304 for 72x32)
            pad_token_id: ID of padding token
            eos_token_id: ID of end-of-sequence token
            split: 'train' or 'val'
        """
        self.B = B  # Batch size
        self.T = T  # Sequence length
        self.rank = process_rank
        self.world_size = num_processes
        self.split = split
        self.window_size = window_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        # Load all tokens from files
        all_tokens = []

        for file_path in token_files:
            tokens = torch.load(file_path, weights_only=False).to('cpu')

            # Verify the tokens contain expected pad tokens
            pad_mask = (tokens == pad_token_id)
            pad_count = pad_mask.sum().item()

            if self.rank == 0 and len(all_tokens) < 3:  # Only print for the first few files
                print(f"File: {file_path} - Tokens shape: {tokens.shape}, Pad tokens: {pad_count}")

                # Calculate theoretical window count (assuming 2304 tokens per window + 1 pad between)
                if pad_count > 0:
                    theoretical_windows = pad_count
                    print(f"  Approximate window count: {theoretical_windows}")

                    # Verification check for a few sample windows
                    for i in range(min(3, pad_count)):
                        pad_positions = torch.where(pad_mask)[0]
                        if i < len(pad_positions):
                            if i == 0:
                                start_idx = 0
                            else:
                                start_idx = pad_positions[i - 1].item() + 1

                            pad_pos = pad_positions[i].item()
                            window_size = pad_pos - start_idx
                            print(f"  Window {i + 1}: From idx {start_idx} to {pad_pos} (size: {window_size})")

            all_tokens.append(tokens)

        # Concatenate all tokens
        self.tokens = torch.cat(all_tokens, dim=0)
        self.total_length = self.tokens.shape[0]

        # Print token statistics
        if self.rank == 0:
            print(f"Total tokens loaded: {self.total_length}")
            pad_count = (self.tokens == pad_token_id).sum().item()
            eos_count = (self.tokens == eos_token_id).sum().item()
            print(f"Total PAD tokens: {pad_count}, Total EOS tokens: {eos_count}")

        # Initialize position counter
        self.pos = 0

    def next_batch(self):
        """Get next batch of tokens for training or validation"""
        # Each process gets a different section of data
        process_data_size = self.total_length // self.world_size
        process_start = self.rank * process_data_size
        process_end = process_start + process_data_size if self.rank < self.world_size - 1 else self.total_length

        # Check if we need to reset position
        if self.pos + self.B * self.T >= process_data_size:
            if self.split == 'train':
                self.pos = 0  # Reset for training
            else:
                return None  # End of validation data

        # Get batch starting positions (B different positions)
        batch_pos = self.pos + process_start + torch.arange(0, self.B, device=self.tokens.device) * self.T

        # Get sequences of length T from each position
        batch_seqs = []
        for p in batch_pos:
            # Make sure we don't go out of bounds
            if p + self.T < process_end:
                seq = self.tokens[p:p + self.T]
            else:
                # If we would go out of bounds, pad with zeros
                seq_len = process_end - p
                seq = torch.cat([self.tokens[p:process_end],
                                 torch.full((self.T - seq_len,), self.pad_token_id, dtype=self.tokens.dtype,
                                            device=self.tokens.device)])
            batch_seqs.append(seq)

        batch = torch.stack(batch_seqs)

        # Update position
        self.pos += self.B * self.T

        return batch

    def reset(self):
        """Reset position counter (for new epochs or validation rounds)"""
        self.pos = 0


def calculate_steps_per_epoch(dataloader, world_size):
    """
    Calculate actual steps needed for one epoch based on the total data and DDP setup.

    Args:
        dataloader: The EEGTokenDataLoader instance
        world_size: Number of processes/GPUs

    Returns:
        int: Number of steps needed for one epoch
    """
    # Total tokens across all processes
    total_tokens = dataloader.total_length

    # Tokens per batch across all processes
    tokens_per_step = dataloader.B * dataloader.T * world_size

    # Steps needed to process all tokens once
    steps_per_epoch = total_tokens // tokens_per_step

    # Add one more step if there's a remainder
    if total_tokens % tokens_per_step > 0:
        steps_per_epoch += 1

    return steps_per_epoch


def plot_losses(train_losses, val_losses, step, epoch, save_path, plot_every=100,
                moving_avg_window=20, title_suffix=""):
    """
    Plot training and validation losses together on the same scale and save to file.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses (can be empty)
        step: Current training step (for naming)
        epoch: Current epoch (for naming)
        save_path: Directory to save plot
        plot_every: How often to generate/save plots
        moving_avg_window: Window size for moving average
        title_suffix: Optional text to add to the plot title
    """
    # Only plot at specified intervals
    if step % plot_every != 0 and step != len(train_losses) - 1:
        return None

    plt.figure(figsize=(10, 6))

    # Plot training loss
    if train_losses:
        train_steps = np.arange(1, len(train_losses) + 1)
        plt.plot(train_steps, train_losses, 'b-', alpha=0.3, label='Train Loss')

        # Add moving average for training
        if len(train_losses) >= moving_avg_window:
            window = min(moving_avg_window, len(train_losses) // 2)
            if window > 0:
                weights = np.ones(window) / window
                train_ma = np.convolve(train_losses, weights, 'valid')
                plt.plot(train_steps[window - 1:], train_ma, 'b-',
                         label=f'Train Moving Avg ({window} steps)')

    # Plot validation losses if available
    if val_losses:
        val_steps = np.arange(1, len(val_losses) + 1)
        plt.plot(val_steps, val_losses, 'r-', alpha=0.3, label='Val Loss')

        # Add moving average for validation
        if len(val_losses) >= moving_avg_window // 2:
            window = min(moving_avg_window // 2, len(val_losses) // 2)
            if window > 0:
                weights = np.ones(window) / window
                val_ma = np.convolve(val_losses, weights, 'valid')
                plt.plot(val_steps[window - 1:], val_ma, 'r-',
                         label=f'Val Moving Avg ({window} steps)')

    # Add title and labels
    title = f'Training Progress - Epoch {epoch + 1}'
    if title_suffix:
        title += f" {title_suffix}"
    plt.title(title)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Same scale for train and val losses
    if train_losses and val_losses:
        all_losses = train_losses + val_losses
        plt.ylim(0, max(all_losses) * 1.1)

    # Save the plot
    os.makedirs(save_path, exist_ok=True)
    plot_filename = os.path.join(save_path, f'loss_epoch{epoch + 1}_step{step}.png')
    plt.savefig(plot_filename)
    plt.close()

    return plot_filename


def plot_full_training_history(all_epochs_data, save_path):
    """
    Create a comprehensive plot of the entire training history.

    Args:
        all_epochs_data: Dictionary containing training history
        save_path: Directory to save the plot
    """
    plt.figure(figsize=(12, 10))

    # Extract data from history
    epochs = all_epochs_data['epochs']
    all_train_losses = all_epochs_data['train_losses']
    all_val_losses = all_epochs_data['val_losses']
    all_eval_results = all_epochs_data['eval_results']

    # Top plot: Loss curves
    plt.subplot(2, 1, 1)

    # Flatten all losses for continuous curve
    all_steps = []
    flat_train_losses = []
    step_counter = 0

    for epoch, losses in zip(epochs, all_train_losses):
        steps = range(step_counter, step_counter + len(losses))
        plt.plot(steps, losses, 'b-', alpha=0.2)

        all_steps.extend(steps)
        flat_train_losses.extend(losses)
        step_counter += len(losses)

    # Add moving average for better visualization
    if flat_train_losses:
        window = min(100, len(flat_train_losses) // 10)
        if window > 0:
            weights = np.ones(window) / window
            train_ma = np.convolve(flat_train_losses, weights, 'valid')
            plt.plot(all_steps[window - 1:], train_ma, 'b-', linewidth=2,
                     label='Train Loss (Moving Avg)')

    # Add validation losses if available
    if all_val_losses:
        val_steps = []
        flat_val_losses = []
        val_step_counter = 0

        for epoch, losses in zip(epochs, all_val_losses):
            steps = range(val_step_counter, val_step_counter + len(losses))
            plt.plot(steps, losses, 'r-', alpha=0.2)

            val_steps.extend(steps)
            flat_val_losses.extend(losses)
            val_step_counter += len(losses)

        # Add moving average for validation losses
        if flat_val_losses:
            window = min(50, len(flat_val_losses) // 10)
            if window > 0:
                weights = np.ones(window) / window
                val_ma = np.convolve(flat_val_losses, weights, 'valid')
                plt.plot(val_steps[window - 1:], val_ma, 'r-', linewidth=2,
                         label='Val Loss (Moving Avg)')

    # Add epoch boundaries as vertical lines
    if epochs:
        step_markers = [0]
        for epoch, losses in zip(epochs, all_train_losses):
            step_markers.append(step_markers[-1] + len(losses))

        for i, marker in enumerate(step_markers[:-1]):
            plt.axvline(x=marker, color='gray', linestyle='--', alpha=0.5)
            # Add epoch number
            mid_x = (step_markers[i] + step_markers[i + 1]) / 2
            plt.text(mid_x, plt.ylim()[1] * 0.9, f'Epoch {epochs[i]}',
                     horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.5))

    plt.title('Training and Validation Loss History')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Bottom plot: Evaluation metrics
    plt.subplot(2, 1, 2)

    if all_eval_results and 'steps' in all_eval_results:
        eval_steps = all_eval_results['steps']

        if 'few_shot_accuracy' in all_eval_results:
            plt.plot(eval_steps, all_eval_results['few_shot_accuracy'], 'g-o',
                     label='Few-shot Accuracy')

        if 'classifier_accuracy' in all_eval_results:
            plt.plot(eval_steps, all_eval_results['classifier_accuracy'], 'm-o',
                     label='Classifier Accuracy')

        plt.title('Evaluation Metrics During Training')
        plt.xlabel('Training Steps')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(save_path, 'full_training_history.png')
    plt.savefig(plot_path)
    plt.close()

    return plot_path


def main():
    parser = argparse.ArgumentParser(description="Train Hierarchical EEG Transformer")

    # Data parameters
    parser.add_argument("--data_dir", type=str, default="training_data_shards",
                        help="Directory containing token files")
    parser.add_argument("--train_val_split", type=float, default=0.9,
                        help="Proportion of data to use for training")
    parser.add_argument("--shuffle_files", action="store_true",
                        help="Shuffle files before splitting")

    # Model parameters
    parser.add_argument("--codebook_size", type=int, default=130,
                        help="Size of the VQAE codebook")
    parser.add_argument("--window_size", type=int, default=2304,
                        help="Size of flattened EEG window (72x32)")
    parser.add_argument("--d_model", type=int, default=360,
                        help="Hidden dimension of the model")
    parser.add_argument("--n_heads", type=int, default=6,
                        help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=4,
                        help="Number of transformer layers")
    parser.add_argument("--max_windows", type=int, default=4,
                        help="Maximum number of windows in sequence")
    parser.add_argument("--pad_token_id", type=int, default=129,
                        help="ID of padding token")
    parser.add_argument("--eos_token_id", type=int, default=128,
                        help="ID of end-of-sequence token")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size per GPU")
    parser.add_argument("--seq_length", type=int, default=4608,
                        help="Sequence length for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-4,
                        help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=1000,
                        help="Steps per epoch")

    # Logging and evaluation parameters
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--save_every", type=int, default=1,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory to save loss plots")
    parser.add_argument("--plot_every", type=int, default=100,
                        help="Plot losses every N steps")
    parser.add_argument("--eval_every", type=int, default=500,
                        help="Run evaluation every N steps")

    # DDP parameters (for torchrun)
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (set by torchrun)")

    # Debug mode
    parser.add_argument("--debug", action="store_true",
                        help="Print debug information")

    args = parser.parse_args()

    # Initialize the distributed environment
    dist.init_process_group(backend="nccl")

    # Set the device
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Get global rank and world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Create save directory if it doesn't exist
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, "evaluations"), exist_ok=True)

    # File list management
    data_dir = args.data_dir
    train_val_split = args.train_val_split
    shuffle_files = args.shuffle_files

    if rank == 0:  # Master process only
        # Find all token files
        token_files = sorted(glob.glob(os.path.join(data_dir, "*_tokens.pt")))[0:500]
        if not token_files:
            raise ValueError(f"No token files found in {data_dir}")

        # Shuffle files with a fixed seed for consistency
        if shuffle_files:
            rng = random.Random(42)  # Fixed seed for determinism
            rng.shuffle(token_files)

        # Split into train and val
        split_idx = int(len(token_files) * train_val_split)
        train_files = token_files[:split_idx]
        val_files = token_files[split_idx:]

        if args.debug:
            print(f"Found {len(token_files)} token files")
            print(f"Using {len(train_files)} for training and {len(val_files)} for validation")
    else:
        train_files = None
        val_files = None

    # Broadcast file lists from master to all workers
    if rank == 0:
        objects = [train_files, val_files]
        dist.broadcast_object_list(objects, src=0)
    else:
        objects = [None, None]
        dist.broadcast_object_list(objects, src=0)
        train_files, val_files = objects[0], objects[1]

    # Synchronize processes
    dist.barrier()

    # Create data loaders with proper token handling
    train_loader = EEGTokenDataLoader(
        B=args.batch_size,
        T=args.seq_length,
        process_rank=rank,
        num_processes=world_size,
        token_files=train_files,
        window_size=args.window_size,
        pad_token_id=args.pad_token_id,
        eos_token_id=args.eos_token_id,
        split='train'
    )

    val_loader = EEGTokenDataLoader(
        B=args.batch_size,
        T=args.seq_length,
        process_rank=rank,
        num_processes=world_size,
        token_files=val_files,
        window_size=args.window_size,
        pad_token_id=args.pad_token_id,
        eos_token_id=args.eos_token_id,
        split='val'
    )

    # Calculate actual steps needed for one epoch
    steps_per_epoch = calculate_steps_per_epoch(train_loader, world_size)
    if rank == 0:
        print(f"Calculated steps per epoch: {steps_per_epoch} (based on dataset size)")

    # Create model with special pad token handling
    model = HierarchicalEEGTransformer(
        codebook_size=args.codebook_size,
        window_size=args.window_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_windows=args.max_windows,
        pad_token_id=args.pad_token_id
    ).to(device)

    # Wrap model with DDP
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr
    )

    # Initialize evaluation setup if we're the main process
    if rank == 0:
        try:
            eval_output_dir = os.path.join(args.log_dir, "evaluations")
            os.makedirs(eval_output_dir, exist_ok=True)

            # Path to tokenized data
            tokenized_data_dir = "tokenized_bci_data"  # Update to match your data path

            evaluator = EEGSimpleEvaluator(
                checkpoint_dir=args.save_dir,
                data_dir=tokenized_data_dir,
                device=f"cuda:{local_rank}",
                codebook_size=args.codebook_size,
                window_size=args.window_size,
                d_model=args.d_model,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
                max_windows=args.max_windows,
                pad_token_id=args.pad_token_id,
                eos_token_id=args.eos_token_id
            )

            # Initialize evaluation results tracking
            eval_results = {
                'steps': [],
                'epochs': [],
                'few_shot_accuracy': [],
                'classifier_accuracy': []
            }
        except Exception as e:
            print(f"Error initializing evaluator: {str(e)}")
            evaluator = None
            eval_results = None

    # Initialize storage for all epoch losses
    training_history = {
        'epochs': [],
        'train_losses': [],
        'val_losses': [],
        'eval_results': None
    }

    # Global step counter for evaluation and plotting
    global_step = 0

    # Training loop with tqdm progress bars
    if rank == 0:
        epoch_bar = tqdm(range(args.epochs), desc="Training", position=0)
    else:
        epoch_bar = range(args.epochs)

    for epoch in epoch_bar:
        model.train()
        train_loss = 0.0
        num_train_batches = 0

        # Reset training data loader for each epoch
        train_loader.reset()

        # Initialize step loss trackers for this epoch
        epoch_train_losses = []
        epoch_val_losses = []

        # Create step progress bar (only on rank 0)
        if rank == 0:
            step_bar = tqdm(range(steps_per_epoch),
                            desc=f"Epoch {epoch + 1}/{args.epochs}",
                            position=1,
                            leave=False)
        else:
            step_bar = range(steps_per_epoch)

        # Train for calculated number of steps per epoch
        for step in step_bar:
            # Get next batch
            batch = train_loader.next_batch()
            if batch is None:
                if rank == 0:
                    tqdm.write(f"Warning: Ran out of data at step {step}/{steps_per_epoch}")
                break

            batch = batch.to(device)

            # Debug info to check for invalid values
            if args.debug and rank == 0 and step == 0:
                print(f"Batch min: {batch.min().item()}, max: {batch.max().item()}")
                print(f"Codebook size: {args.codebook_size}")

            # Clamp values to valid range to avoid CUDA assertion
            batch = torch.clamp(batch, max=args.codebook_size - 1)

            # Forward pass: input is all but last token, target is all but first token
            x = batch[:, :-1]
            target = batch[:, 1:]

            # Get model predictions
            logits = model(x)

            # Make sure logits and target have same batch size
            min_length = min(logits.size(1), target.size(1))
            logits = logits[:, :min_length, :]
            target = target[:, :min_length]

            # Compute loss with padding token ignored
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
                ignore_index=args.pad_token_id  # Ignore padding tokens in loss calculation
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            # Update statistics
            train_loss += loss.item()
            num_train_batches += 1

            # Track step losses for plotting (only on rank 0)
            if rank == 0:
                epoch_train_losses.append(loss.item())

            # Update global step counter
            global_step += 1

            # Update step progress bar with current loss (only on rank 0)
            if rank == 0:
                step_bar.set_postfix({"loss": f"{loss.item():.4f}"})

                # Generate plot on the fly at specified intervals
                plot_path = plot_losses(
                    epoch_train_losses,
                    epoch_val_losses,
                    step,
                    epoch,
                    os.path.join(args.log_dir, "plots"),
                    plot_every=args.plot_every,
                    title_suffix="(Training in progress)"
                )

                if plot_path:
                    step_bar.set_postfix({"loss": f"{loss.item():.4f}", "plot": plot_path})

            # Run validation and evaluation at regular intervals during training
            if global_step % args.eval_every == 0:
                # Save a temporary checkpoint for evaluation
                if rank == 0:
                    temp_checkpoint = {
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }
                    temp_checkpoint_path = os.path.join(args.save_dir, f"temp_checkpoint_step_{global_step}.pt")
                    torch.save(temp_checkpoint, temp_checkpoint_path)

                # Run validation
                model.eval()
                val_loss = 0.0
                num_val_batches = 0

                # Reset validation data loader
                val_loader.reset()

                # Number of validation steps - use a smaller subset for mid-training evaluation
                val_steps = min(50, calculate_steps_per_epoch(val_loader, world_size) // 4)

                # Run validation
                with torch.no_grad():
                    for val_step in range(val_steps):
                        # Get next batch
                        batch = val_loader.next_batch()
                        if batch is None:
                            break

                        batch = batch.to(device)
                        batch = torch.clamp(batch, max=args.codebook_size - 1)

                        # Forward pass
                        x = batch[:, :-1]
                        target = batch[:, 1:]

                        logits = model(x)

                        # Make sure lengths match
                        min_length = min(logits.size(1), target.size(1))
                        logits = logits[:, :min_length, :]
                        target = target[:, :min_length]

                        # Compute loss
                        loss = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            target.reshape(-1),
                            ignore_index=args.pad_token_id
                        )

                        # Update statistics
                        val_loss += loss.item()
                        num_val_batches += 1

                        # Track validation losses (on rank 0)
                        if rank == 0:
                            epoch_val_losses.append(loss.item())

                # Calculate average validation loss
                if num_val_batches > 0:
                    val_loss = val_loss / num_val_batches
                    val_loss_tensor = torch.tensor(val_loss).to(device)
                    dist.all_reduce(val_loss_tensor)
                    val_loss = val_loss_tensor.item() / world_size

                # Run evaluation (only on rank 0)
                if rank == 0 and evaluator is not None:
                    try:
                        # Load the temporary checkpoint
                        evaluator.model.load_state_dict(temp_checkpoint['model_state_dict'])
                        evaluator.model.eval()

                        # Use fewer shots for faster evaluation during training
                        few_shot_acc = evaluator.evaluate_few_shot(n_shots=3, n_trials=5)
                        classifier_acc = 0  # Or run a real classifier if available

                        # Store results
                        eval_results['steps'].append(global_step)
                        eval_results['epochs'].append(epoch + 1)
                        eval_results['few_shot_accuracy'].append(few_shot_acc)
                        eval_results['classifier_accuracy'].append(classifier_acc)

                        # Save evaluation results to CSV
                        eval_df = pd.DataFrame(eval_results)
                        eval_df.to_csv(os.path.join(eval_output_dir, 'ongoing_evaluation.csv'), index=False)

                        # Update progress message
                        tqdm.write(f"Step {global_step} | Epoch {epoch + 1} | "
                                   f"Train loss: {train_loss / num_train_batches:.4f} | "
                                   f"Val loss: {val_loss:.4f} | "
                                   f"Few-shot acc: {few_shot_acc:.4f}")
                    except Exception as e:
                        tqdm.write(f"Error during evaluation: {str(e)}")

                # Delete temporary checkpoint if it exists
                if rank == 0 and os.path.exists(temp_checkpoint_path):
                    os.remove(temp_checkpoint_path)

                # Back to training mode
                model.train()

            # Make sure processes are synced after evaluation
            dist.barrier()

        # Average training loss for the epoch
        train_loss = train_loss / num_train_batches if num_train_batches > 0 else 0
        train_loss_tensor = torch.tensor(train_loss).to(device)
        dist.all_reduce(train_loss_tensor)
        train_loss = train_loss_tensor.item() / world_size

        # Run a full validation pass at the end of each epoch
        model.eval()
        val_loss = 0.0
        num_val_batches = 0

        # Reset validation data loader
        val_loader.reset()

        # Full validation steps
        val_steps = calculate_steps_per_epoch(val_loader, world_size)

        if rank == 0:
            val_bar = tqdm(range(val_steps), desc="Epoch Validation", position=1, leave=False)

        # Full validation losses for end-of-epoch
        full_val_losses = []

        with torch.no_grad():
            for val_step in range(val_steps) if rank != 0 else val_bar:
                batch = val_loader.next_batch()
                if batch is None:
                    break

                batch = batch.to(device)
                batch = torch.clamp(batch, max=args.codebook_size - 1)

                # Forward pass
                x = batch[:, :-1]
                target = batch[:, 1:]

                logits = model(x)

                # Make sure lengths match
                min_length = min(logits.size(1), target.size(1))
                logits = logits[:, :min_length, :]
                target = target[:, :min_length]

                # Compute loss
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target.reshape(-1),
                    ignore_index=args.pad_token_id
                )

                # Update statistics
                val_loss += loss.item()
                num_val_batches += 1

                # Track full validation losses (on rank 0)
                if rank == 0:
                    full_val_losses.append(loss.item())
                    val_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})

        # Average validation loss
        val_loss = val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        val_loss_tensor = torch.tensor(val_loss).to(device)
        dist.all_reduce(val_loss_tensor)
        val_loss = val_loss_tensor.item() / world_size

        # Update learning rate
        scheduler.step()

        # Store the losses for this epoch (only on rank 0)
        if rank == 0:
            training_history['epochs'].append(epoch + 1)
            training_history['train_losses'].append(epoch_train_losses)
            training_history['val_losses'].append(full_val_losses)
            training_history['eval_results'] = eval_results

            # Generate final plot for this epoch with both train and validation
            end_plot_path = plot_losses(
                epoch_train_losses,
                full_val_losses,
                steps_per_epoch - 1,  # Final step of epoch
                epoch,
                os.path.join(args.log_dir, "plots"),
                plot_every=1,  # Force plot
                title_suffix="(Epoch Complete)"
            )

            # Update epoch progress bar
            epoch_bar.set_postfix({
                "train_loss": f"{train_loss:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.6f}",
                "plot": os.path.basename(end_plot_path) if end_plot_path else "none"
            })

            # Save checkpoint
            if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
                checkpoint = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }

                checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                torch.save(checkpoint, checkpoint_path)
                tqdm.write(f"Saved checkpoint at epoch {epoch + 1}")

                # Run a full evaluation at checkpoint saving points
                if evaluator is not None:
                    try:
                        evaluator.model.load_state_dict(checkpoint['model_state_dict'])
                        evaluator.model.eval()

                        # Run more comprehensive evaluation at checkpoints
                        few_shot_acc = evaluator.evaluate_few_shot(n_shots=5, n_trials=20)
                        classifier_acc = 0  # Or run real classifier evaluation

                        # Record final evaluation for this epoch
                        tqdm.write(f"Checkpoint evaluation - Few-shot: {few_shot_acc:.4f}, "
                                   f"Classifier: {classifier_acc:.4f}")

                        # Create and save evaluation plots
                        plt.figure(figsize=(10, 6))
                        plt.plot(eval_results['steps'], eval_results['few_shot_accuracy'], 'b-o',
                                 label='Few-shot Accuracy')
                        plt.plot(eval_results['steps'], eval_results['classifier_accuracy'], 'r-o',
                                 label='Classifier Accuracy')
                        plt.title('Training Evaluation Metrics')
                        plt.xlabel('Training Step')
                        plt.ylabel('Accuracy')
                        plt.grid(True, alpha=0.3)
                        plt.legend()

                        eval_plot_path = os.path.join(eval_output_dir, f'eval_metrics_epoch_{epoch + 1}.png')
                        plt.savefig(eval_plot_path)
                        plt.close()
                    except Exception as e:
                        tqdm.write(f"Error during checkpoint evaluation: {str(e)}")

            # Periodically generate full training history plot
            if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
                history_plot_path = plot_full_training_history(
                    training_history,
                    os.path.join(args.log_dir, "plots")
                )
                tqdm.write(f"Generated full training history plot: {history_plot_path}")

    # Final cleanup
    if rank == 0:
        # Create comprehensive training plot
        final_plot_path = plot_full_training_history(
            training_history,
            os.path.join(args.log_dir, "plots")
        )
        print(f"Training complete. Final history plot: {final_plot_path}")

        # Save final training statistics
        final_stats = {
            'epochs': training_history['epochs'],
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'total_steps': global_step
        }

        # Convert to DataFrame for easy viewing
        stats_df = pd.DataFrame([final_stats])
        stats_path = os.path.join(args.log_dir, "final_training_stats.csv")
        stats_df.to_csv(stats_path, index=False)
        print(f"Final training statistics saved to {stats_path}")

    # Clean up distributed process group
    dist.destroy_process_group()


if __name__ == "__main__":
    main()