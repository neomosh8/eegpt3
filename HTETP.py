import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import time
import random
import inspect
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class IntraEpochTransformer(nn.Module):
    """Processes tokens within a single epoch to capture spatial patterns"""

    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # No or minimal positional encoding since tokens aren't sequential in time
        self.pos_embedding = nn.Parameter(torch.zeros(1, 2304, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # # Global pooling to get epoch representation
        # self.pool = nn.Sequential(
        #     nn.Linear(d_model * 2304, d_model * 4),
        #     nn.GELU(),
        #     nn.Linear(d_model * 4, d_model)
        # )

    def forward(self, x):
        # x shape: [batch_size, epoch_length, d_model]
        x = self.embedding(x)  # [batch_size, epoch_length, d_model]
        x = x + self.pos_embedding  # Add learned positional encoding
        x = self.transformer(x)  # [batch_size, epoch_length, d_model]

        # Use global mean pooling
        epoch_embedding = torch.mean(x, dim=1)  # [batch_size, d_model]
        return epoch_embedding


class InterEpochTransformer(nn.Module):
    """Processes sequence of epochs to capture temporal patterns - with causal masking"""

    def __init__(self, d_model=512, nhead=8, num_layers=4, max_epochs=10):
        super().__init__()
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_epochs, d_model))

        # Create a causal mask for the transformer
        # This ensures each epoch only attends to itself and previous epochs
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_epochs, max_epochs) * float('-inf'), diagonal=1)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        # x shape: [batch_size, num_epochs, d_model]
        x = x + self.pos_encoding[:, :x.size(1), :]  # Add positional encoding

        # Apply causal mask to prevent attending to future epochs
        # Only use the portion of the mask that matches the current sequence length
        mask = self.causal_mask[:x.size(1), :x.size(1)]
        x = self.transformer(x, mask=mask)  # [batch_size, num_epochs, d_model]
        return x


class EpochPredictionHead(nn.Module):
    """Predicts the next epoch of tokens"""

    def __init__(self, d_model=512, vocab_size=1024, epoch_length=2400):
        super().__init__()
        self.epoch_length = epoch_length
        self.vocab_size = vocab_size

        # Expand the embedding into token predictions
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, epoch_length * vocab_size)
        )

    def forward(self, x):
        # x shape: [batch_size, d_model]
        logits = self.proj(x)  # [batch_size, epoch_length * vocab_size]
        logits = logits.view(-1, self.epoch_length, self.vocab_size)  # [batch_size, epoch_length, vocab_size]
        return logits

class HierarchicalEEGTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8,
                 intra_layers=4, inter_layers=4, max_epochs=10, epoch_length=2400):
        super().__init__()
        self.intra_transformer = IntraEpochTransformer(
            vocab_size, d_model, nhead, intra_layers
        )
        self.inter_transformer = InterEpochTransformer(
            d_model, nhead, inter_layers, max_epochs
        )
        self.prediction_head = EpochPredictionHead(
            d_model, vocab_size, epoch_length
        )
        self.epoch_length = epoch_length
        self.vocab_size = vocab_size
        self.max_epochs = max_epochs

    def forward(self, x, targets=None):
        # x shape: [batch_size, num_epochs * epoch_length]
        # targets shape: [batch_size, epoch_length] - contains exactly the next epoch
        batch_size, seq_length = x.shape

        # Ensure we have enough data for all epochs
        if seq_length < self.epoch_length * self.max_epochs:
            raise ValueError(
                f"Input sequence length {seq_length} is less than required {self.epoch_length * self.max_epochs}")

        # Reshape sequential tokens into epochs
        epochs = []
        for i in range(self.max_epochs):
            start_idx = i * self.epoch_length
            end_idx = start_idx + self.epoch_length
            epochs.append(x[:, start_idx:end_idx])

        x_epochs = torch.stack(epochs, dim=1)  # [batch_size, num_epochs, epoch_length]

        # Process each epoch to get embeddings
        epoch_embeddings = []
        for i in range(self.max_epochs):
            emb = self.intra_transformer(x_epochs[:, i])
            epoch_embeddings.append(emb)

        epoch_embeddings = torch.stack(epoch_embeddings, dim=1)  # [batch_size, num_epochs, d_model]

        # Process sequence of epochs with causal masking
        temporal_embeddings = self.inter_transformer(epoch_embeddings)  # [batch_size, num_epochs, d_model]

        # Predict next epoch from the last temporal embedding
        next_epoch_pred = self.prediction_head(temporal_embeddings[:, -1])  # [batch_size, epoch_length, vocab_size]

        loss = None
        if targets is not None:
            # With our new dataloader, targets already contains exactly the next epoch
            # No need for complex indexing, we can use targets directly
            loss = F.cross_entropy(
                next_epoch_pred.reshape(-1, self.vocab_size),
                targets.reshape(-1)
            )

        return next_epoch_pred, loss

    def configure_optimizer(self, weight_decay, learning_rate, device):
        """Configure optimizer with different learning rates for different components"""
        # Group parameters by component type
        embed_params = []
        attn_params = []
        other_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # Apply weight decay to 2D+ parameters only
            wd = weight_decay if param.dim() >= 2 else 0.0

            if 'embedding' in name:
                embed_params.append({'params': param, 'weight_decay': wd, 'lr': learning_rate * 0.5})
            elif 'transformer' in name:
                attn_params.append({'params': param, 'weight_decay': wd, 'lr': learning_rate * 5.0})
            else:
                other_params.append({'params': param, 'weight_decay': wd, 'lr': learning_rate})

        # Combine all parameter groups
        optim_groups = embed_params + attn_params + other_params

        # Create optimizer
        optimizer = torch.optim.AdamW(
            optim_groups,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused='cuda' in device and 'fused' in inspect.signature(torch.optim.AdamW).parameters
        )

        return optimizer

# Dataloader from the second document
class EEGTokenDataLoader:
    """
    A data loader that loads EEG token files, concatenates them,
    and provides properly aligned data for epoch-level prediction.
    """

    def __init__(self, B, epoch_length, max_epochs, process_rank, num_processes, token_files, split):
        self.B = B  # Batch size
        self.epoch_length = epoch_length  # Number of tokens per epoch
        self.max_epochs = max_epochs  # Number of epochs to process
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split

        # Calculate total sequence length needed for input and target
        # We need max_epochs epochs for input and 1 more epoch for target
        self.T = epoch_length * (max_epochs + 1)

        # Load tokens
        all_tokens = []
        for file_path in token_files:
            try:
                token_tensor = torch.load(file_path, map_location='cpu')
                all_tokens.append(token_tensor)
            except Exception as e:
                if process_rank == 0:
                    print(f"Error loading {file_path}: {e}")
                continue
        self.tokens = torch.cat(all_tokens, dim=0)
        print(f"Process {process_rank}: Loaded {len(self.tokens)} tokens for {split} split")

        # Ensure all processes reach this point
        if num_processes > 1:
            dist.barrier()
        print(f"Process {process_rank}: Post-barrier, tokens loaded successfully")

        self.current_position = self.B * self.T * self.process_rank
        self.total_len = len(self.tokens)
        print(f"Process {process_rank}: current_position = {self.current_position}, total_len = {self.total_len}")

    def next_batch(self):
        """
        Fetch next batch of tokens as (x, y) for training.
        x contains max_epochs complete epochs
        y contains the next complete epoch (aligned at epoch boundary)
        """
        B, T, epoch_length, max_epochs = self.B, self.T, self.epoch_length, self.max_epochs

        # We need enough tokens for max_epochs epochs (input) + 1 epoch (target)
        needed = B * T

        # Get tokens
        if self.current_position + needed <= self.total_len:
            buf_tokens = self.tokens[self.current_position: self.current_position + needed]
            self.current_position += B * T * self.num_processes
            if self.current_position + needed > self.total_len:
                self.current_position = self.process_rank * B * T
        else:
            # Handle wrapping around the dataset
            leftover = self.total_len - self.current_position
            wrap_amount = needed - leftover
            part1_tokens = self.tokens[self.current_position:]
            part2_tokens = self.tokens[:wrap_amount]
            buf_tokens = torch.cat([part1_tokens, part2_tokens], dim=0)
            self.current_position = wrap_amount + (B * T * (self.num_processes - 1))

        if len(buf_tokens) != needed:
            raise RuntimeError(f"Unexpected token count. Expected {needed}, got {len(buf_tokens)}")

        # Reshape to [B, T]
        buf_tokens = buf_tokens.view(B, T)

        # Input: first max_epochs epochs
        input_length = epoch_length * max_epochs
        x = buf_tokens[:, :input_length]

        # Target: the next epoch
        y = buf_tokens[:, input_length:input_length + epoch_length]

        return x, y

    def reset(self):
        self.current_position = self.B * self.T * self.process_rank

    def __len__(self):
        return self.total_len // (self.B * self.T)

def moving_average(values, window_size=10):
    """Compute the simple moving average of a list of values."""
    if len(values) < window_size:
        return values  # not enough data, just return as-is

    averaged = []
    for i in range(len(values)):
        start = max(0, i - window_size + 1)
        chunk = values[start: i + 1]
        averaged.append(sum(chunk) / len(chunk))
    return averaged


# Setup distributed training
def setup_distributed():
    ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
    if ddp:
        # use of DDP demands CUDA, set device according to rank
        assert torch.cuda.is_available(), "CUDA is required for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        # non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        # attempt to autodetect device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"Using device: {device}")

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device, device_type


# Main training function
def train_model():
    # Configuration
    config = {
        'data_dir': 'training_data_shards',
        'log_dir': 'logs',
        'vocab_size': 129,  # From second document
        'd_model': 400,
        'n_head': 4,
        'intra_layers': 4,
        'inter_layers': 4,
        'epoch_length': 2304,  # Length of each EEG epoch
        'max_epochs': 4,  # Number of epochs in sequence for model
        'num_epochs': 10,  # Number of training epochs
        'batch_size': 1,
        'learning_rate': 1e-3,
        'max_lr': 4e-3,
        'min_lr': 4e-4,
        'weight_decay': 0.05,
        'warmup_ratio': 0.01,
        'clip_grad_norm': 1.0,
        'train_val_split': 0.9,
        'shuffle_files': True,
        'seed': 9259,
        'resume': False
    }

    # Set up distributed training
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device, device_type = setup_distributed()

    # Set random seed for reproducibility
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])

    # Create log directory
    log_dir = config['log_dir']
    if master_process:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "training.log")
        with open(log_file, "w") as f:
            pass  # Initialize empty log file

    # Load data files
    if ddp_rank == 0:  # Master process
        # Find all token files
        token_files = sorted(glob.glob(os.path.join(config['data_dir'], "*_tokens.pt")))
        if not token_files:
            raise ValueError(f"No token files found in {config['data_dir']}")

        # Shuffle files with a fixed seed for consistency
        if config['shuffle_files']:
            rng = random.Random(config['seed'])
            rng.shuffle(token_files)

        # Split into train and val
        split_idx = int(len(token_files) * config['train_val_split'])
        train_files = token_files[:split_idx]
        val_files = token_files[split_idx:]

        if master_process:
            print(f"Found {len(token_files)} token files")
            print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")
    else:
        train_files = None
        val_files = None

    # Broadcast file lists for distributed training
    if ddp:
        objects = [train_files, val_files] if ddp_rank == 0 else [None, None]
        dist.broadcast_object_list(objects, src=0)
        train_files, val_files = objects[0], objects[1]
        dist.barrier()

    # Calculate required sequence length for model input
    # We need at least (max_epochs + 1) * epoch_length tokens
    required_seq_length = (config['max_epochs'] + 1) * config['epoch_length']

    # Initialize data loaders
    train_loader = EEGTokenDataLoader(
        B=config['batch_size'],
        epoch_length=config['epoch_length'],
        max_epochs=config['max_epochs'],
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        token_files=train_files,
        split='train'
    )

    val_loader = EEGTokenDataLoader(
        B=config['batch_size'],
        epoch_length=config['epoch_length'],
        max_epochs=config['max_epochs'],
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        token_files=val_files,
        split='val'
    )

    # Create model
    model = HierarchicalEEGTransformer(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        nhead=config['n_head'],
        intra_layers=config['intra_layers'],
        inter_layers=config['inter_layers'],
        max_epochs=config['max_epochs'],
        epoch_length=config['epoch_length']
    )
    model.to(device)

    # For distributed training, wrap model with DDP
    if ddp:
        # model = DDP(model, device_ids=[ddp_local_rank],find_unused_parameters=True)
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    # Set up optimizer
    optimizer = raw_model.configure_optimizer(
        weight_decay=config['weight_decay'],
        learning_rate=config['learning_rate'],
        device=device
    )

    # Set up learning rate scheduler
    steps_per_epoch = train_loader.total_len // (config['batch_size'] * required_seq_length * ddp_world_size)
    max_steps = config['num_epochs'] * steps_per_epoch
    warmup_steps = int(config['warmup_ratio'] * max_steps)

    # Set up different learning rates for different parameter groups
    max_lr_per_group = []
    for group in optimizer.param_groups:
        base_lr = group['lr']
        lr_ratio = base_lr / config['learning_rate'] if config['learning_rate'] > 0 else 1.0
        group_max_lr = config['max_lr'] * lr_ratio
        max_lr_per_group.append(group_max_lr)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr_per_group,
        total_steps=max_steps,
        pct_start=warmup_steps / max_steps,
        anneal_strategy='cos',
        cycle_momentum=False
    )

    if master_process:
        print(f"Configured OneCycleLR with {len(max_lr_per_group)} learning rates")
        print(f"LR ranges: Embeddings: {config['learning_rate'] * 0.5:.2e}-{config['max_lr'] * 0.5:.2e}, "
              f"Attention: {config['learning_rate'] * 5.0:.2e}-{config['max_lr'] * 5.0:.2e}, "
              f"Other: {config['learning_rate']:.2e}-{config['max_lr']:.2e}")

    # Load checkpoint if resuming
    start_epoch = 0
    if config['resume']:
        checkpoint_path = os.path.join(log_dir, "latest_checkpoint.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            raw_model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            if master_process:
                print(f"Resuming from checkpoint at epoch {start_epoch}")

    # Training metrics tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    # Main training loop
    for epoch in range(start_epoch, config['num_epochs']):
        if master_process:
            print(f"Starting epoch {epoch + 1}/{config['num_epochs']}")

        # Training phase
        model.train()
        train_loss = 0.0
        train_steps = 0

        # Use tqdm for progress visualization
        train_iter = range(steps_per_epoch)
        if master_process:
            train_iter = tqdm(train_iter, desc=f"Epoch {epoch + 1}/{config['num_epochs']} [Train]")

        for step in train_iter:
            # Zero gradients
            optimizer.zero_grad()

            # Get batch
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            # Forward pass with autocast for mixed precision if on CUDA
            if device_type == 'cuda':
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)

            if loss is not None:
                # Backward pass
                loss.backward()

                # Gradient clipping
                if config['clip_grad_norm'] > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])

                # Update weights
                optimizer.step()
                scheduler.step()

                # Collect statistics
                train_loss += loss.item()
                train_steps += 1

                # Update progress bar
                if master_process:
                    current_lr = optimizer.param_groups[0]['lr']
                    train_iter.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'avg_loss': f"{train_loss / train_steps:.4f}",
                        'lr': f"{current_lr:.2e}"
                    })

        # Average training loss
        train_loss = train_loss / train_steps if train_steps > 0 else float('inf')
        if ddp:
            # Gather training loss from all processes
            train_loss_tensor = torch.tensor([train_loss], device=device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
            train_loss = train_loss_tensor.item()

        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_steps = 0

        # Calculate validation steps (limited to reduce computation time)
        val_steps_per_epoch = min(100,
                                  val_loader.total_len // (config['batch_size'] * required_seq_length * ddp_world_size))

        # Use tqdm for progress visualization
        val_iter = range(val_steps_per_epoch)
        if master_process:
            val_iter = tqdm(val_iter, desc=f"Epoch {epoch + 1}/{config['num_epochs']} [Val]")

        with torch.no_grad():
            for step in val_iter:
                # Get batch
                # Get batch
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)

                # Forward pass with autocast for mixed precision if on CUDA
                if device_type == 'cuda':
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        _, loss = model(x, y)
                else:
                    _, loss = model(x, y)

                if loss is not None:
                    # Collect statistics
                    val_loss += loss.item()
                    val_steps += 1

                    # Update progress bar
                    if master_process:
                        val_iter.set_postfix({'loss': f"{loss.item():.4f}", 'avg_loss': f"{val_loss / val_steps:.4f}"})

                # Average validation loss
            val_loss = val_loss / val_steps if val_steps > 0 else float('inf')
            if ddp:
                # Gather validation loss from all processes
                val_loss_tensor = torch.tensor([val_loss], device=device)
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                val_loss = val_loss_tensor.item()

            val_losses.append(val_loss)

            # Log results for this epoch
            if master_process:
                print(
                    f"Epoch {epoch + 1}/{config['num_epochs']} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                # Write to log file
                with open(os.path.join(log_dir, "training.log"), "a") as f:
                    f.write(f"Epoch {epoch + 1} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}\n")

                # # Check if this is the best model
                # is_best = val_loss < best_val_loss
                # if is_best:
                #     best_val_loss = val_loss
                #     # Save best model
                #     torch.save({
                #         'epoch': epoch,
                #         'model': raw_model.state_dict(),
                #         'optimizer': optimizer.state_dict(),
                #         'scheduler': scheduler.state_dict(),
                #         'train_loss': train_loss,
                #         'val_loss': val_loss,
                #         'config': config,
                #     }, os.path.join(log_dir, "best_model.pt"))
                #
                # # Save checkpoint
                # torch.save({
                #     'epoch': epoch,
                #     'model': raw_model.state_dict(),
                #     'optimizer': optimizer.state_dict(),
                #     'scheduler': scheduler.state_dict(),
                #     'train_loss': train_loss,
                #     'val_loss': val_loss,
                #     'config': config,
                # }, os.path.join(log_dir, "latest_checkpoint.pt"))

                # Save checkpoint every 5 epochs
                if (epoch + 1) % 5 == 0:
                    torch.save({
                        'epoch': epoch,
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'config': config,
                    }, os.path.join(log_dir, f"checkpoint_epoch_{epoch + 1}.pt"))

                # Plot training and validation loss curves
                plt.figure(figsize=(10, 6))
                epochs_range = list(range(1, len(train_losses) + 1))
                plt.plot(epochs_range, train_losses, label='Train Loss', color='#63B8FF', alpha=0.6)
                plt.plot(epochs_range, val_losses, label='Val Loss', color='#1E56A0')

                # Add moving average for train loss
                if len(train_losses) >= 3:  # Need at least 3 points for meaningful moving average
                    ma_train_losses = moving_average(train_losses, window_size=3)
                    plt.plot(epochs_range, ma_train_losses, label='Train Loss (MA)', color='black', linestyle='--')

                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.title('Training and Validation Loss')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(log_dir, "loss_plot.png"))
                plt.close()

        # End of training
    if master_process:
        print(f"Training completed. Best validation loss: {best_val_loss:.4f}")

        # Clean up distributed training
    if ddp:
        destroy_process_group()

    return raw_model, best_val_loss

def main():
    # Enable higher precision for matrix multiplications
    torch.set_float32_matmul_precision('medium')

    # Train model
    model, best_val_loss = train_model()

    # Additional code for model evaluation, inference, etc. can be added here

    return model

if __name__ == "__main__":
    main()