import os
import glob
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math

from tqdm import tqdm


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
            x: Tensor of token indices [batch_size, seq_length].
                 Contains multiple windows separated/padded with `pad_token_id`.
        Returns:
            logits: [batch_size, total_tokens, codebook_size]
        """

        # 1) Extract windows from x.
        #    - windows has shape [B, N, W],
        #      where N = number of windows, W = window_size.
        #    - This should be padded with self.pad_token_id (129), not zero!
        windows, _ = self._extract_windows_with_pad_tokens(x)  # shape [B, N, W]
        B, N, W = windows.shape
        device = windows.device

        # 2) Flatten windows so we can embed tokens directly
        seq_len = N * W  # total tokens per sample
        flat_windows = windows.reshape(B, seq_len)  # [B, seq_len]

        # 3) Token embedding
        embedded = self.token_embedding(flat_windows)  # [B, seq_len, d_model]

        # 4) Add hierarchical positional embeddings
        #    - Reshape to [B, N, W, d_model], so we can add window-level + token-level positions
        embedded = embedded.view(B, N, W, self.d_model)

        # 4.1) Add window-level positions (shape [1, N, d_model])
        #      We unsqueeze(2) so it can broadcast over the W dimension
        embedded = embedded + self.window_pos_embed[:, :N, :].unsqueeze(2)  # => [B, N, W, d_model]

        # 4.2) Add token-level positions (shape [1, W, d_model])
        #      We unsqueeze(1) so it can broadcast over the N dimension
        embedded = embedded + self.token_pos_embed[:, :W, :].unsqueeze(1)  # => [B, N, W, d_model]

        # 5) Reshape back to a simple sequence per batch element: [B, seq_len, d_model]
        embedded = embedded.view(B, seq_len, self.d_model)

        # 6) Build a [B, seq_len, seq_len] mask that encodes:
        #    (a) Full bidirectional attention within each window
        #    (b) Causal (one-direction) attention across windows
        #    (c) Pad tokens (129) are blocked entirely
        mask = self._create_hierarchical_mask(windows)  # => [B, seq_len, seq_len]

        # 7) Pass through the Transformer layers, each expecting [B, seq_len, d_model] plus mask
        x = embedded
        for layer in self.layers:
            x = layer(x, mask=mask)  # [B, seq_len, d_model]

        # 8) Final normalization + output projection
        x = self.norm(x)  # [B, seq_len, d_model]
        logits = self.output_head(x)  # [B, seq_len, codebook_size]

        return logits

    def _create_hierarchical_mask(self, windows):
        """
        windows: [B, N, W] (already padded with self.pad_token_id where needed).
        Returns:
            mask: [B, N*W, N*W] with float('-inf') where attention is disallowed,
                  and 0.0 where it's allowed.
        """
        B, N, W = windows.shape
        seq_len = N * W
        device = windows.device

        # 1) Build a single "hierarchical" mask for a sequence of length seq_len
        #    We'll replicate it for each batch element.
        base_mask = torch.full((seq_len, seq_len), float('-inf'), device=device)

        # "Full" attention inside the same window; "causal" across windows
        for i in range(N):
            start_i = i * W
            end_i = start_i + W
            # Full bidirectional inside window i
            base_mask[start_i:end_i, start_i:end_i] = 0.0

            # Allow causal from window j < i to window i
            for j in range(i):
                start_j = j * W
                end_j = start_j + W
                base_mask[start_i:end_i, start_j:end_j] = 0.0

        # 2) Replicate for each item in the batch -> [B, seq_len, seq_len]
        mask = base_mask.unsqueeze(0).expand(B, seq_len, seq_len).clone()

        # 3) Block out pad tokens entirely by setting both rows & columns to -inf
        #    Flatten windows to shape [B, seq_len]
        flat_w = windows.view(B, seq_len)
        for b in range(B):
            pad_positions = (flat_w[b] == self.pad_token_id).nonzero(as_tuple=True)[0]
            if len(pad_positions) > 0:
                mask[b, pad_positions, :] = float('-inf')
                mask[b, :, pad_positions] = float('-inf')

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
        B, seq_len, _ = x.shape

        # x has shape [B, seq_len, d_model]
        q = self.q_proj(x)  # [B, seq_len, d_model]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Then reshape for multi-heads: [B, seq_len, n_heads, head_dim]
        q = q.view(B, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # -> [B, n_heads, seq_len, head_dim]
        k = k.view(B, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute scores: shape [B, n_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # Add the mask (with shape [B, seq_len, seq_len]) as scores += mask.unsqueeze(1)
        if mask is not None:
            # IMPORTANT: Use mask.unsqueeze(1) instead of .unsqueeze(0).unsqueeze(1)
            # so each batch item gets its own mask.
            scores = scores + mask.unsqueeze(1)

        # Softmax, drop, etc.
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)

        # Reshape context back to [B, seq_len, d_model]
        context = context.transpose(1, 2).reshape(B, seq_len, self.d_model)
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
            tokens = torch.load(file_path)

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
    parser.add_argument("--n_heads", type=int, default=4,
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

    # Training loop
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

        # Print progress and save checkpoint (only on rank 0)
        if rank == 0:
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

    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    main()