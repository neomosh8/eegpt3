import os
import glob
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math


# Model definition with handling for non-divisible sequence lengths
class HierarchicalEEGTransformer(nn.Module):
    def __init__(self,
                 codebook_size,  # Size of your VQAE codebook
                 window_size=2304,  # Flattened size (72x32)
                 d_model=768,  # Hidden dimension
                 n_heads=12,  # Attention heads
                 n_layers=6,  # Transformer layers
                 max_windows=50):  # Max sequence length in windows
        super().__init__()

        self.window_size = window_size

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

    def forward(self, x):
        """
        Args:
            x: Tensor of token indices [batch_size, seq_length]
        """
        batch_size, seq_length = x.shape
        device = x.device

        # Handle case where sequence length is not divisible by window size
        num_full_windows = seq_length // self.window_size
        remainder = seq_length % self.window_size

        if remainder > 0:
            # Either pad the sequence or truncate it to multiple of window_size
            if remainder < self.window_size // 2:
                # Truncate - remove the remainder tokens
                x = x[:, :num_full_windows * self.window_size]
                seq_length = x.shape[1]
            else:
                # Pad - add zero padding to complete the window
                padding_size = self.window_size - remainder
                padding = torch.zeros((batch_size, padding_size), dtype=x.dtype, device=device)
                x = torch.cat([x, padding], dim=1)
                seq_length = x.shape[1]

        # Now we know seq_length is divisible by window_size
        num_windows = seq_length // self.window_size

        # Get token embeddings
        x = self.token_embedding(x)  # [B, S, D]

        # Reshape to [batch, windows, tokens_per_window, dim]
        x = x.view(batch_size, num_windows, self.window_size, -1)

        # Add positional encodings
        # 1. Window-level positions
        x = x + self.window_pos_embed[:, :num_windows, :].unsqueeze(2)

        # 2. Token-level positions
        x = x + self.token_pos_embed[:, :self.window_size, :].unsqueeze(1)

        # Reshape back to sequence
        x = x.view(batch_size, seq_length, -1)

        # Create hierarchical attention mask
        mask = self._create_hierarchical_mask(num_windows, self.window_size, device)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        return self.output_head(x)

    def _create_hierarchical_mask(self, num_windows, window_size, device):
        """
        Create mask that allows:
        1. Full attention within each window (bidirectional)
        2. Causal attention between windows
        """
        seq_length = num_windows * window_size
        mask = torch.ones(seq_length, seq_length, device=device) * float('-inf')

        # Allow full attention within each window
        for i in range(num_windows):
            start_idx = i * window_size
            end_idx = (i + 1) * window_size
            mask[start_idx:end_idx, start_idx:end_idx] = 0

        # Allow causal attention between windows
        for i in range(num_windows):
            for j in range(i):
                i_start = i * window_size
                i_end = (i + 1) * window_size
                j_start = j * window_size
                j_end = (j + 1) * window_size
                mask[i_start:i_end, j_start:j_end] = 0

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

        # Apply mask if provided
        if mask is not None:
            scores = scores + mask.unsqueeze(0).unsqueeze(1)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)

        # Reshape back to [batch, seq_len, d_model]
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.out_proj(context)


# Modified DataLoader for EEG Tokens that ensures window divisibility
class EEGTokenDataLoader:
    def __init__(self, B, T, process_rank, num_processes, token_files, window_size=2304, split='train'):
        """
        Args:
            B: Batch size
            T: Sequence length
            process_rank: Current process rank for DDP
            num_processes: Total number of processes for DDP
            token_files: List of token file paths
            window_size: Size of each EEG window (should be 2304 for 72x32)
            split: 'train' or 'val'
        """
        self.B = B  # Batch size
        self.T = T  # Sequence length
        self.rank = process_rank
        self.world_size = num_processes
        self.split = split
        self.window_size = window_size

        # Load all tokens from files
        all_tokens = []
        for file_path in token_files:
            tokens = torch.load(file_path)
            all_tokens.append(tokens)

        # Concatenate all tokens
        self.tokens = torch.cat(all_tokens, dim=0)
        self.total_length = self.tokens.shape[0]

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
                seq = torch.cat([
                    self.tokens[p:process_end],
                    torch.zeros(self.T - seq_len, dtype=self.tokens.dtype, device=self.tokens.device)
                ])
            batch_seqs.append(seq)

        batch = torch.stack(batch_seqs)

        # Ensure sequence length is divisible by window_size
        T_adjusted = (self.T // self.window_size) * self.window_size
        if T_adjusted < self.T:
            batch = batch[:, :T_adjusted]

        # Update position
        self.pos += self.B * self.T

        return batch

    def reset(self):
        """Reset position counter (for new epochs or validation rounds)"""
        self.pos = 0


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
    parser.add_argument("--codebook_size", type=int, default=129,
                        help="Size of the VQAE codebook")
    parser.add_argument("--window_size", type=int, default=2304,
                        help="Size of flattened EEG window (72x32)")
    parser.add_argument("--d_model", type=int, default=768,
                        help="Hidden dimension of the model")
    parser.add_argument("--n_heads", type=int, default=12,
                        help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=6,
                        help="Number of transformer layers")
    parser.add_argument("--max_windows", type=int, default=50,
                        help="Maximum number of windows in sequence")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size per GPU")
    parser.add_argument("--seq_length", type=int, default=4608,
                        help="Sequence length for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping")
    parser.add_argument("--epochs", type=int, default=10,
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

    # Create data loaders with correct window size
    train_loader = EEGTokenDataLoader(
        B=args.batch_size,
        T=args.seq_length,
        process_rank=rank,
        num_processes=world_size,
        token_files=train_files,
        window_size=args.window_size,
        split='train'
    )

    val_loader = EEGTokenDataLoader(
        B=args.batch_size,
        T=args.seq_length,
        process_rank=rank,
        num_processes=world_size,
        token_files=val_files,
        window_size=args.window_size,
        split='val'
    )

    # Create model
    model = HierarchicalEEGTransformer(
        codebook_size=args.codebook_size,
        window_size=args.window_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_windows=args.max_windows
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

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        num_train_batches = 0

        # Train for a number of steps per epoch
        for step in range(args.steps_per_epoch):
            # Get next batch
            batch = train_loader.next_batch()
            if batch is None:
                break

            batch = batch.to(device)

            # Debug shapes
            if args.debug and rank == 0 and step == 0:
                print(f"Batch shape: {batch.shape}")
                print(f"Is divisible by window size: {batch.shape[1] % args.window_size == 0}")

            # Forward pass: input is all but last token, target is all but first token
            x = batch[:, :-1]
            target = batch[:, 1:]

            # Debug shapes
            if args.debug and rank == 0 and step == 0:
                print(f"Input shape: {x.shape}, Target shape: {target.shape}")
                print(f"Is input divisible by window size: {x.shape[1] % args.window_size == 0}")

            # Get model predictions
            logits = model(x)

            # Compute loss
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1)
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

        # Average training loss
        train_loss = train_loss / num_train_batches if num_train_batches > 0 else 0

        # All-reduce the training loss across processes
        train_loss_tensor = torch.tensor(train_loss).to(device)
        dist.all_reduce(train_loss_tensor)
        train_loss = train_loss_tensor.item() / world_size

        # Validation
        model.eval()
        val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            val_loader.reset()  # Reset validation data loader
            while True:
                # Get next batch
                batch = val_loader.next_batch()
                if batch is None:
                    break

                batch = batch.to(device)

                # Forward pass
                x = batch[:, :-1]
                target = batch[:, 1:]

                logits = model(x)

                # Compute loss
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target.reshape(-1)
                )

                # Update statistics
                val_loss += loss.item()
                num_val_batches += 1

        # Average validation loss
        val_loss = val_loss / num_val_batches if num_val_batches > 0 else float('inf')

        # All-reduce the validation loss across processes
        val_loss_tensor = torch.tensor(val_loss).to(device)
        dist.all_reduce(val_loss_tensor)
        val_loss = val_loss_tensor.item() / world_size

        # Update learning rate
        scheduler.step()

        # Print progress and save checkpoint (only on rank 0)
        if rank == 0:
            print(
                f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

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
                print(f"Saved checkpoint at epoch {epoch + 1}")

    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    main()