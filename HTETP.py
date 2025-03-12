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
            tokens = torch.load(file_path,weights_only=False)

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


def plot_and_save_losses(train_losses, val_losses, epoch, log_dir):
    """
    Plot training and validation losses for the current epoch and save to log directory.

    Args:
        train_losses: List of training losses for each step
        val_losses: List of validation losses for each step
        epoch: Current epoch number
        log_dir: Directory to save plots
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Plot training losses
    steps = np.arange(1, len(train_losses) + 1)
    ax1.plot(steps, train_losses, 'b-', alpha=0.3, label='Step Loss')

    # Calculate and plot moving average (window of 20 steps)
    if len(train_losses) >= 20:
        window_size = 20
        weights = np.ones(window_size) / window_size
        train_ma = np.convolve(train_losses, weights, 'valid')
        ax1.plot(steps[window_size - 1:], train_ma, 'r-', label='Moving Avg (20 steps)')

    ax1.set_title(f'Training Loss - Epoch {epoch + 1}')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot validation losses
    val_steps = np.arange(1, len(val_losses) + 1)
    ax2.plot(val_steps, val_losses, 'g-', alpha=0.3, label='Step Loss')

    # Calculate and plot moving average (window of 20 steps or fewer if not enough points)
    if len(val_losses) >= 10:
        window_size = min(20, len(val_losses) // 2)
        weights = np.ones(window_size) / window_size
        val_ma = np.convolve(val_losses, weights, 'valid')
        ax2.plot(val_steps[window_size - 1:], val_ma, 'm-', label=f'Moving Avg ({window_size} steps)')

    ax2.set_title(f'Validation Loss - Epoch {epoch + 1}')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    # Tight layout
    plt.tight_layout()

    # Save figure
    plot_path = os.path.join(log_dir, f'loss_plot_epoch_{epoch + 1}.png')
    plt.savefig(plot_path)
    plt.close(fig)  # Close figure to free memory

    return plot_path


def plot_training_progress(all_train_losses, all_val_losses, log_dir):
    """
    Create a single plot showing training and validation losses across all epochs.

    Args:
        all_train_losses: List of lists containing training losses for each epoch
        all_val_losses: List of lists containing validation losses for each epoch
        log_dir: Directory to save the plot
    """
    plt.figure(figsize=(15, 10))

    # Plot mean losses per epoch
    epochs = range(1, len(all_train_losses) + 1)
    train_means = [np.mean(losses) for losses in all_train_losses]
    val_means = [np.mean(losses) for losses in all_val_losses]

    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_means, 'bo-', label='Training Loss')
    plt.plot(epochs, val_means, 'ro-', label='Validation Loss')
    plt.title('Mean Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Plot all step losses as continuous line with epoch boundaries
    plt.subplot(2, 1, 2)

    # Prepare continuous step axis
    all_train_steps = []
    all_val_steps = []
    train_step_losses = []
    val_step_losses = []

    # Mark epoch boundaries
    epoch_boundaries = [0]
    step_count = 0

    for epoch_idx, (train_losses, val_losses) in enumerate(zip(all_train_losses, all_val_losses)):
        # Add train loss points
        epoch_steps = range(step_count, step_count + len(train_losses))
        all_train_steps.extend(epoch_steps)
        train_step_losses.extend(train_losses)

        step_count += len(train_losses)
        epoch_boundaries.append(step_count)

        # Add validation loss points at the end of each epoch
        # We'll offset these slightly for visualization
        val_steps = [step_count + i * 0.5 for i in range(len(val_losses))]
        all_val_steps.extend(val_steps)
        val_step_losses.extend(val_losses)

        step_count += len(val_losses)  # Move step count past validation

    # Plot step losses
    plt.plot(all_train_steps, train_step_losses, 'b-', alpha=0.3, label='Training Steps')
    plt.plot(all_val_steps, val_step_losses, 'r-', alpha=0.3, label='Validation Steps')

    # Calculate and plot moving averages
    window_size = 100
    if len(train_step_losses) >= window_size:
        weights = np.ones(window_size) / window_size
        train_ma = np.convolve(train_step_losses, weights, 'valid')
        plt.plot(all_train_steps[window_size - 1:], train_ma, 'g-', label=f'Train Moving Avg ({window_size} steps)')

    if len(val_step_losses) >= window_size // 2:
        val_window = min(window_size // 2, len(val_step_losses) // 2)
        if val_window > 0:
            weights = np.ones(val_window) / val_window
            val_ma = np.convolve(val_step_losses, weights, 'valid')
            plt.plot(all_val_steps[val_window - 1:], val_ma, 'm-', label=f'Val Moving Avg ({val_window} steps)')

    # Add epoch boundary lines
    for boundary in epoch_boundaries:
        plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.7)

    # Add epoch numbers
    for i in range(len(epoch_boundaries) - 1):
        mid = (epoch_boundaries[i] + epoch_boundaries[i + 1]) / 2
        plt.text(mid, plt.ylim()[1] * 0.9, f'Epoch {i + 1}',
                 horizontalalignment='center', verticalalignment='center',
                 bbox=dict(facecolor='white', alpha=0.7))

    plt.title('All Training Steps')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(log_dir, 'full_training_progress.png')
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
    parser.add_argument("--d_model", type=int, default=24,
                        help="Hidden dimension of the model")
    parser.add_argument("--n_heads", type=int, default=2,
                        help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=2,
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

    # Saving parameters
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--save_every", type=int, default=1,
                        help="Save checkpoint every N epochs")

    # DDP parameters (for torchrun)
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (set by torchrun)")

    # Debug mode - print shapes
    parser.add_argument("--debug", action="store_true",
                        help="Print debug information")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory to save loss plots")
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


    # File list management
    data_dir = args.data_dir
    train_val_split = args.train_val_split
    shuffle_files = args.shuffle_files

    if rank == 0:  # Master process only
        # Find all token files
        token_files = sorted(glob.glob(os.path.join(data_dir, "*_tokens.pt")))
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


########### eval
    # Add this import at the top of your file

    # Add this code right after your argument parser setup but before the main training loop
    # (around line 637, after creating log_dir and before initializing DDP)



    # Initialize storage for all epoch losses (right before the epoch loop)
    all_train_losses = []
    all_val_losses = []

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

        # Initialize step loss trackers
        step_train_losses = []

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
                step_train_losses.append(loss.item())

            # Update step progress bar with current loss (only on rank 0)
            if rank == 0:
                step_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Average training loss
        train_loss = train_loss / num_train_batches if num_train_batches > 0 else 0

        # All-reduce the training loss across processes
        train_loss_tensor = torch.tensor(train_loss).to(device)
        dist.all_reduce(train_loss_tensor)
        train_loss = train_loss_tensor.item() / world_size

        # Validation with progress bar
        model.eval()
        val_loss = 0.0
        num_val_batches = 0

        # Reset validation data loader
        val_loader.reset()

        # Initialize validation step loss trackers
        step_val_losses = []

        # Calculate validation steps
        val_steps = calculate_steps_per_epoch(val_loader, world_size)

        # Create validation progress bar (only on rank 0)
        if rank == 0:
            val_bar = tqdm(range(val_steps), desc="Validation", position=1, leave=False)

        with torch.no_grad():
            for val_step in range(val_steps) if rank != 0 else val_bar:
                # Get next batch
                batch = val_loader.next_batch()
                if batch is None:
                    break

                batch = batch.to(device)

                # Clamp values for validation as well
                batch = torch.clamp(batch, max=args.codebook_size - 1)

                # Forward pass
                x = batch[:, :-1]
                target = batch[:, 1:]

                logits = model(x)

                # Make sure logits and target have same batch size
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

                # Track step losses for plotting (only on rank 0)
                if rank == 0:
                    step_val_losses.append(loss.item())

                # Update validation progress bar (only on rank 0)
                if rank == 0:
                    val_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})

        # Average validation loss
        val_loss = val_loss / num_val_batches if num_val_batches > 0 else float('inf')

        # All-reduce the validation loss across processes
        val_loss_tensor = torch.tensor(val_loss).to(device)
        dist.all_reduce(val_loss_tensor)
        val_loss = val_loss_tensor.item() / world_size

        # Update learning rate
        scheduler.step()

        # Store the losses for this epoch (only after completing the entire epoch)
        if rank == 0:
            all_train_losses.append(step_train_losses)
            all_val_losses.append(step_val_losses)

            # Plot per-epoch losses
            plot_path = plot_and_save_losses(step_train_losses, step_val_losses, epoch, args.log_dir)
            tqdm.write(f"Saved loss plot to {plot_path}")

            # Update epoch progress bar
            epoch_bar.set_postfix({
                "train_loss": f"{train_loss:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.6f}"
            })

            # Save checkpoint
            if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }
                torch.save(checkpoint, os.path.join(args.save_dir, f"checkpoint_epoch_{epoch + 1}.pt"))
                # Print outside the progress bar
                tqdm.write(f"Saved checkpoint at epoch {epoch + 1}")

                checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch + 1}.pt")

                ###EVAL
                # Load the current checkpoint
                if step > 0:
                    if rank == 0:
                        # Initialize evaluator only on the main process
                        # Only evaluate every N epochs to save time
                        eval_every = args.save_every  # Or set a custom value
                        eval_output_dir = os.path.join(args.log_dir, "evaluations")
                        os.makedirs(eval_output_dir, exist_ok=True)

                        # Path to your tokenized data
                        tokenized_data_dir = "tokenized_bci_data"  # Update this to your data path

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

                        # Track evaluation results
                        eval_results = {
                            'epoch': [],
                            'few_shot_accuracy': [],
                            'classifier_accuracy': []
                        }
                    checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{local_rank}",weights_only=False)
                    evaluator.model.load_state_dict(checkpoint['model_state_dict'])
                    evaluator.model.eval()

                # Run evaluations
                try:
                    # Use fewer shots for faster evaluation during training
                    few_shot_acc = evaluator.evaluate_few_shot(n_shots=3, n_trials=10)
                    classifier_acc = 0
                    # Store results
                    eval_results['epoch'].append(epoch + 1)
                    eval_results['few_shot_accuracy'].append(few_shot_acc)
                    eval_results['classifier_accuracy'].append(classifier_acc)

                    # Create plot showing progress so far
                    plt.figure(figsize=(12, 8))

                    plt.plot(eval_results['epoch'], eval_results['few_shot_accuracy'], 'b-o',
                             label='Few-shot Learning Accuracy', linewidth=2)
                    plt.plot(eval_results['epoch'], eval_results['classifier_accuracy'], 'r-o',
                             label='Classifier Head Accuracy', linewidth=2)

                    plt.title('EEG Transformer Evaluation During Training', fontsize=16)
                    plt.xlabel('Epoch', fontsize=14)
                    plt.ylabel('Accuracy', fontsize=14)
                    plt.grid(True, alpha=0.3)
                    plt.legend(fontsize=12)

                    # Set y-axis limits with some padding
                    max_acc = max(
                        max(eval_results['few_shot_accuracy']) if eval_results['few_shot_accuracy'] else 0,
                        max(eval_results['classifier_accuracy']) if eval_results['classifier_accuracy'] else 0
                    )
                    plt.ylim(0, min(1.0, max_acc + 0.1))

                    # Save plot and results
                    plt.tight_layout()
                    plt.savefig(os.path.join(eval_output_dir, 'ongoing_evaluation.png'), dpi=300)
                    plt.close()

                    # Save results to CSV
                    pd.DataFrame(eval_results).to_csv(
                        os.path.join(eval_output_dir, 'evaluation_results.csv'), index=False)

                    tqdm.write(f"Evaluation complete - Few-shot: {few_shot_acc:.4f}, Classifier: {classifier_acc:.4f}")

                except Exception as e:
                    tqdm.write(f"Error during evaluation: {str(e)}")

    # Clean up
    if rank == 0:
        # Create comprehensive training plot
        plot_path = plot_training_progress(all_train_losses, all_val_losses, args.log_dir)
        print(f"Saved full training progress plot to {plot_path}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()