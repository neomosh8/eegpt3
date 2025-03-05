#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import time


# -------------------------
# Custom LR Scheduler
# -------------------------
class CustomLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    A custom learning rate scheduler that:
      - Linearly increases the LR from base_lr to max_lr for a warmup phase.
      - Then exponentially decays from max_lr to min_lr over the remaining steps.
    """

    def __init__(self, optimizer, total_steps, warmup_steps, base_lr, max_lr, min_lr, last_epoch=-1):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1  # current step count (1-indexed)
        if step < self.warmup_steps:
            # Linear warmup from base_lr to max_lr.
            lr = self.base_lr + (self.max_lr - self.base_lr) * (step / self.warmup_steps)
        else:
            # Exponential decay from max_lr to min_lr.
            decay_steps = self.total_steps - self.warmup_steps
            progress = (step - self.warmup_steps) / decay_steps  # in [0, 1]
            # Compute decay factor so that at progress=0, lr = max_lr and at progress=1, lr = min_lr.
            lr = self.max_lr * ((self.min_lr / self.max_lr) ** progress)
        return [lr for _ in self.optimizer.param_groups]


# -------------------------
# Dataset and Model Definitions (as provided)
# -------------------------
class EEGNpyDataset(Dataset):
    def __init__(self, directory, normalize=False):
        self.files = [f for f in os.listdir(directory) if f.endswith('.npy')]
        if not self.files:
            raise ValueError("No .npy files found in {}".format(directory))
        self.files.sort()
        self.dir = directory
        self.normalize = normalize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = np.load(os.path.join(self.dir, self.files[idx]))
        if self.normalize:
            x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        return torch.from_numpy(x).float()


class VectorQuantizerEMA(nn.Module):
    def __init__(self, codebook_size, embedding_dim, decay=0.9, eps=1e-5):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.eps = eps

        # Initialize embedding on the default device (will move to GPU later)
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

        # Register buffers (these will be moved to the same device as model)
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_w", self.embedding.weight.clone())

    def forward(self, z):
        # Ensure input is on the correct device
        device = z.device

        flat_z = z.reshape(-1, self.embedding_dim)  # (B*H*W, C)

        # Ensure all tensors are on the same device
        dist = torch.cdist(flat_z, self.embedding.weight, p=2)  # (B*H*W, codebook_size)
        idxs = dist.argmin(dim=1)

        # Create one_hot encodings on the same device as inputs
        encodings = F.one_hot(idxs, self.codebook_size).to(flat_z.dtype).to(device)

        if self.training:
            self._update_ema(flat_z, encodings)

        z_q = self.embedding(idxs).reshape(z.shape)

        # Compute loss (everything stays on GPU)
        diff = (z_q.detach() - z).pow(2).mean() + (z_q - z.detach()).pow(2).mean()

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        return z_q, idxs.reshape(z.shape[:-1]), diff

    def _update_ema(self, flat_z, encodings):
        device = flat_z.device

        # Ensure all operations happen on the same device
        cluster_size = encodings.sum(dim=0).to(device)

        # Update cluster size EMA
        self.cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)

        # Update embedding EMA
        dw = torch.matmul(flat_z.t(), encodings).to(device)
        self.ema_w.data.mul_(self.decay).add_(dw.t(), alpha=1 - self.decay)

        # Normalize embeddings
        n = self.cluster_size.sum()

        # Handle empty clusters
        cluster_size = ((self.cluster_size + self.eps) / (n + self.codebook_size * self.eps)) * n

        # Update embedding weights - stay on same device
        embed_normalized = self.ema_w / cluster_size.unsqueeze(1)
        self.embedding.weight.data.copy_(embed_normalized)


class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, hidden_channels, 3, 2, 1), nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, out_channels=3, hidden_channels=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, 64, 3, 2, 1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, out_channels, 3, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class VQCAE(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=128, codebook_size=128, decay=0.9, commitment_beta=0.3):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_channels)
        self.vq = VectorQuantizerEMA(codebook_size, hidden_channels, decay=decay)
        self.decoder = Decoder(in_channels, hidden_channels)
        self.commitment_beta = commitment_beta

    def forward(self, x):
        z_e = self.encoder(x)
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        z_q, idxs, vq_loss = self.vq(z_e)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        x_rec = self.decoder(z_q)
        # Keep everything on the same device
        loss = F.mse_loss(x_rec, x) + self.commitment_beta * vq_loss
        return x_rec, loss

    def encode_indices(self, x):
        z_e = self.encoder(x)
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        _, idxs, _ = self.vq(z_e)
        return idxs


# -------------------------
# Distributed Training Setup
# -------------------------
def setup_ddp():
    """Set up distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])

        # Initialize the process group
        dist.init_process_group(backend='nccl')

        # Set GPU device
        torch.cuda.set_device(local_rank)
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    return rank, local_rank, world_size


def cleanup_ddp():
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


# -------------------------
# Visualization functions modified for DDP
# -------------------------
def plot_reconstructions(model, data_loader, device, save_path="output/reconstructions.png", n=8, rank=0):
    if rank != 0:  # Only generate plots on rank 0
        return

    model.eval()
    batch = next(iter(data_loader)).to(device)
    with torch.no_grad():
        # Handle DDP model by using model.module if needed
        if isinstance(model, DDP):
            recons, _ = model.module(batch)
        else:
            recons, _ = model(batch)

    batch = batch.cpu().numpy()
    recons = recons.cpu().numpy()
    n = min(n, batch.shape[0])
    fig, axs = plt.subplots(2, n, figsize=(2 * n, 4))
    for i in range(n):
        orig = batch[i]
        rec = recons[i]
        orig = (orig - orig.min()) / (orig.max() - orig.min() + 1e-8)
        rec = (rec - rec.min()) / (rec.max() - rec.min() + 1e-8)
        if orig.shape[0] == 1:
            axs[0, i].imshow(orig[0], cmap='gray')
            axs[1, i].imshow(rec[0], cmap='gray')
        else:
            axs[0, i].imshow(np.transpose(orig, (1, 2, 0)))
            axs[1, i].imshow(np.transpose(rec, (1, 2, 0)))
        axs[0, i].axis('off')
        axs[1, i].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Reconstruction plot saved to {save_path}")


def plot_latent_code_distribution(model, data_loader, device, save_path="output/latent_code_distribution.png", rank=0):
    model.eval()
    all_idxs = []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            # Handle DDP model
            if isinstance(model, DDP):
                idxs = model.module.encode_indices(batch)
            else:
                idxs = model.encode_indices(batch)
            all_idxs.append(idxs.view(-1).cpu())

    all_idxs = torch.cat(all_idxs, dim=0)

    # Gather indices from all processes
    if dist.is_initialized() and dist.get_world_size() > 1:
        gathered_idxs = [torch.zeros_like(all_idxs) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_idxs, all_idxs)
        all_idxs = torch.cat(gathered_idxs, dim=0)

    codebook_size = model.module.vq.codebook_size if isinstance(model, DDP) else model.vq.codebook_size
    counts = torch.bincount(all_idxs, minlength=codebook_size).float()

    # Only plot on rank 0
    if rank == 0:
        plt.figure(figsize=(10, 4))
        plt.bar(np.arange(len(counts)), counts.numpy())
        plt.xlabel("Codebook Index")
        plt.ylabel("Count")
        plt.title("Distribution of Latent Codes")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Latent code distribution plot saved to {save_path}")

    return counts


def evaluate_quality_metrics(model, data_loader, device, rank=0):
    model.eval()
    total_mse = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            # Handle DDP model
            if isinstance(model, DDP):
                x_rec, _ = model.module(batch)
            else:
                x_rec, _ = model(batch)
            mse = F.mse_loss(x_rec, batch, reduction='sum').item()
            total_mse += mse
            total_samples += batch.size(0)

    # Aggregate metrics across processes - use torch tensor on the correct device
    if dist.is_initialized() and dist.get_world_size() > 1:
        metrics = torch.tensor([total_mse, total_samples], dtype=torch.float64, device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_mse, total_samples = metrics[0].item(), metrics[1].item()

    avg_mse = total_mse / total_samples
    if rank == 0:
        print(f"Average Reconstruction MSE: {avg_mse:.4f}")
    return avg_mse


def evaluate_codebook_usage(model, data_loader, device, save_path="output/codebook_usage.png", rank=0):
    model.eval()
    all_idxs = []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            # Handle DDP model
            if isinstance(model, DDP):
                idxs = model.module.encode_indices(batch)
            else:
                idxs = model.encode_indices(batch)
            all_idxs.append(idxs.view(-1).cpu())

    all_idxs = torch.cat(all_idxs, dim=0)

    # Gather indices from all processes
    if dist.is_initialized() and dist.get_world_size() > 1:
        gathered_idxs = [torch.zeros_like(all_idxs) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_idxs, all_idxs)
        all_idxs = torch.cat(gathered_idxs, dim=0)

    codebook_size = model.module.vq.codebook_size if isinstance(model, DDP) else model.vq.codebook_size
    counts = torch.bincount(all_idxs, minlength=codebook_size).float()
    total = counts.sum().item()
    probs = counts / total
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
    perplexity = np.exp(entropy)

    # Only plot on rank 0
    if rank == 0:
        plt.figure(figsize=(8, 4))
        plt.bar(np.arange(codebook_size), counts.numpy())
        plt.xlabel("Codebook Index")
        plt.ylabel("Count")
        plt.title(f"Codebook Usage (Perplexity: {perplexity:.2f})")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Codebook usage histogram saved to {save_path}")
        print(f"Codebook usage perplexity: {perplexity:.2f}")

    return perplexity, counts


# -------------------------
# Main Training Loop with DDP and torch.compile
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="training_data/coeffs/")
    parser.add_argument("--batch_size", type=int, default=5 * 16)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4, help="Base learning rate")
    parser.add_argument("--max_lr", type=float, default=4e-3, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-4, help="Final learning rate")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set up distributed training
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create output directory on rank 0
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    # Ensure all processes have passed the directory creation
    if dist.is_initialized():
        dist.barrier()

    # Load dataset
    ds = EEGNpyDataset(args.data_dir, normalize=False)
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size

    # Use deterministic splitting for reproducibility
    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=generator)

    # Create distributed samplers
    train_sampler = DistributedSampler(train_ds,
                                       num_replicas=world_size,
                                       rank=rank,
                                       shuffle=True,
                                       seed=args.seed)

    val_sampler = DistributedSampler(val_ds,
                                     num_replicas=world_size,
                                     rank=rank,
                                     shuffle=False,
                                     seed=args.seed)

    # Create data loaders with distributed samplers
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )

    # Get input channels from sample
    sample = ds[0]
    in_channels = sample.shape[0]

    # Create model and explicitly move to device BEFORE wrapping with DDP
    model = VQCAE(in_channels=in_channels, hidden_channels=4096, codebook_size=64).to(device)

    # Verify all model parameters are on the correct device
    for param in model.parameters():
        assert param.device == device, f"Parameter on wrong device: {param.device} vs {device}"

    # Check buffer devices
    for buffer_name, buffer in model.named_buffers():
        assert buffer.device == device, f"Buffer {buffer_name} on wrong device: {buffer.device} vs {device}"

    # Wrap model with DDP after ensuring it's correctly on device
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # Apply torch.compile if requested and available
    if args.compile and hasattr(torch, 'compile'):
        if rank == 0:
            print("Using torch.compile to optimize model")
        model = torch.compile(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Compute total steps and warmup steps
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    scheduler = CustomLRScheduler(optimizer,
                                  total_steps=total_steps,
                                  warmup_steps=warmup_steps,
                                  base_lr=args.lr,
                                  max_lr=args.max_lr,
                                  min_lr=args.min_lr)

    # Lists to record loss history for plotting
    train_losses = []
    val_losses = []

    # Wait for all processes to reach this point before starting training
    if dist.is_initialized():
        dist.barrier()

    # Training loop
    start_time = time.time()

    for epoch in range(args.epochs):
        # Set epoch for the samplers
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        model.train()
        total_loss = 0
        count = 0

        # Create progress bar on rank 0 only
        if rank == 0:
            pbar = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}/{args.epochs}", leave=False)
        else:
            pbar = train_loader

        for batch in pbar:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

            _, loss = model(batch)
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update LR every batch

            # Keep all metrics on the correct device to avoid CPU/GPU transfers
            batch_loss = loss.item() * batch.size(0)
            total_loss += batch_loss
            count += batch.size(0)

            # Display progress only on rank 0
            if rank == 0 and isinstance(pbar, tqdm):
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({"loss": loss.item(), "lr": current_lr})

        # Aggregate losses across processes - use tensors on the correct device
        if dist.is_initialized() and world_size > 1:
            metrics = torch.tensor([total_loss, count], dtype=torch.float64, device=device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            total_loss, count = metrics[0].item(), metrics[1].item()

        avg_train_loss = total_loss / count
        train_losses.append(avg_train_loss)

        # Validation loop
        model.eval()
        val_loss = 0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device, non_blocking=True)
                _, loss = model(batch)
                val_loss += loss.item() * batch.size(0)
                val_count += batch.size(0)

        # Aggregate validation losses across processes
        if dist.is_initialized() and world_size > 1:
            metrics = torch.tensor([val_loss, val_count], dtype=torch.float64, device=device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            val_loss, val_count = metrics[0].item(), metrics[1].item()

        avg_val_loss = val_loss / val_count
        val_losses.append(avg_val_loss)

        # Wait for all processes to finish epoch evaluation
        if dist.is_initialized():
            dist.barrier()

        # Print status on rank 0
        if rank == 0:
            print(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # Periodically save checkpoint
            if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.module.state_dict(),  # Save unwrapped model
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                }
                torch.save(checkpoint, f"{args.output_dir}/vqcae_checkpoint_epoch{epoch + 1}.pt")

    # Training complete
    end_time = time.time()
    training_time = end_time - start_time

    if rank == 0:
        # Plot train and validation loss
        plt.figure(figsize=(10, 6))
        epochs_range = range(1, args.epochs + 1)
        plt.plot(epochs_range, train_losses, label='Train Loss')
        plt.plot(epochs_range, val_losses, label='Validation Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train and Validation Loss Over Epochs")
        plt.legend()
        plt.savefig(f"{args.output_dir}/loss.png")
        plt.close()
        print("Loss plot saved to", f"{args.output_dir}/loss.png")

        # Save final model
        final_checkpoint = {
            "epoch": args.epochs,
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
        }
        torch.save(final_checkpoint, f"{args.output_dir}/vqcae_final.pt")

        # Print training time
        print(f"Total training time: {training_time / 3600:.2f} hours")

        # Generate evaluation plots
        plot_reconstructions(model, val_loader, device=device,
                             save_path=f"{args.output_dir}/reconstructions.png", rank=rank)

        evaluate_quality_metrics(model, val_loader, device=device, rank=rank)

        evaluate_codebook_usage(model, val_loader, device=device,
                                save_path=f"{args.output_dir}/codebook_usage.png", rank=rank)

        plot_latent_code_distribution(model, val_loader, device=device,
                                      save_path=f"{args.output_dir}/latent_code_distribution.png", rank=rank)

    # Make sure all processes reach cleanup together
    if dist.is_initialized():
        dist.barrier()

    # Clean up DDP resources
    cleanup_ddp()

    if rank == 0:
        print("Training complete. Model and evaluation plots saved in", args.output_dir)


if __name__ == "__main__":
    main()