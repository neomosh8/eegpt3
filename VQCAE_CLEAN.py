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
        self.files = [f for f in os.listdir(directory) if f.endswith('.npy')][0:20]
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
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0/codebook_size, 1.0/codebook_size)
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_w", self.embedding.weight.clone())

    def forward(self, z):
        flat_z = z.reshape(-1, self.embedding_dim)  # (B*H*W, C)
        dist = torch.cdist(flat_z, self.embedding.weight, p=2)  # (B*H*W, codebook_size)
        idxs = dist.argmin(dim=1)
        encodings = F.one_hot(idxs, self.codebook_size).type(flat_z.dtype)
        if self.training:
            self._update_ema(flat_z, encodings)
        z_q = self.embedding(idxs).reshape(z.shape)
        diff = (z_q.detach() - z).pow(2).mean() + (z_q - z.detach()).pow(2).mean()
        z_q = z + (z_q - z).detach()  # Straight-through estimator
        return z_q, idxs.reshape(z.shape[:-1]), diff

    def _update_ema(self, flat_z, encodings):
        cluster_size = encodings.sum(dim=0)
        self.cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        dw = flat_z.t() @ encodings
        self.ema_w.data.mul_(self.decay).add_(dw.t(), alpha=1 - self.decay)
        n = self.cluster_size.sum()
        cluster_size = ((self.cluster_size + self.eps) / (n + self.codebook_size * self.eps)) * n
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
    def __init__(self, in_channels=3, hidden_channels=128, codebook_size=128, decay=0.9, commitment_beta=0.0):
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
        loss = F.mse_loss(x_rec, x) + self.commitment_beta * vq_loss
        return x_rec, loss

    def encode_indices(self, x):
        z_e = self.encoder(x)
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        _, idxs, _ = self.vq(z_e)
        return idxs

def plot_reconstructions(model, data_loader, device, save_path="output/reconstructions.png", n=8):
    model.eval()
    batch = next(iter(data_loader)).to(device)
    with torch.no_grad():
        recons, _ = model(batch)
    batch = batch.cpu().numpy()
    recons = recons.cpu().numpy()
    n = min(n, batch.shape[0])
    fig, axs = plt.subplots(2, n, figsize=(2*n, 4))
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

def plot_latent_code_distribution(model, data_loader, device, save_path="output/latent_code_distribution.png"):
    model.eval()
    all_idxs = []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            idxs = model.encode_indices(batch)
            all_idxs.append(idxs.view(-1).cpu())
    all_idxs = torch.cat(all_idxs, dim=0)
    counts = torch.bincount(all_idxs, minlength=model.vq.codebook_size).float()
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(model.vq.codebook_size), counts.numpy())
    plt.xlabel("Codebook Index")
    plt.ylabel("Count")
    plt.title("Distribution of Latent Codes")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Latent code distribution plot saved to {save_path}")
    return counts

def evaluate_quality_metrics(model, data_loader, device):
    model.eval()
    total_mse = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            x_rec, _ = model(batch)
            mse = F.mse_loss(x_rec, batch, reduction='sum').item()
            total_mse += mse
            total_samples += batch.size(0)
    avg_mse = total_mse / total_samples
    print(f"Average Reconstruction MSE: {avg_mse:.4f}")
    return avg_mse

def evaluate_codebook_usage(model, data_loader, device, save_path="output/codebook_usage.png"):
    model.eval()
    all_idxs = []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            idxs = model.encode_indices(batch)
            all_idxs.append(idxs.view(-1).cpu())
    all_idxs = torch.cat(all_idxs, dim=0)
    counts = torch.bincount(all_idxs, minlength=model.vq.codebook_size).float()
    total = counts.sum().item()
    probs = counts / total
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
    perplexity = np.exp(entropy)
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(model.vq.codebook_size), counts.numpy())
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
# Main Training Loop with Scheduler Integration
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="training_data/coeffs/")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2000)
    # Base learning rate used during warmup
    parser.add_argument("--lr", type=float, default=3e-4, help="Base learning rate")
    # Maximum learning rate after warmup
    parser.add_argument("--max_lr", type=float, default=4e-2, help="Peak learning rate")
    # Minimum (final) learning rate after decay
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Final learning rate")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    ds = EEGNpyDataset(args.data_dir, normalize=False)
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    sample = ds[0]
    in_channels = sample.shape[0]

    model = VQCAE(in_channels=in_channels, hidden_channels=512, codebook_size=128).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Compute total steps and warmup steps (using 10% of total steps for warmup)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    scheduler = CustomLRScheduler(optimizer,
                                  total_steps=total_steps,
                                  warmup_steps=warmup_steps,
                                  base_lr=args.lr,
                                  max_lr=args.max_lr,
                                  min_lr=args.min_lr)

    # Lists to record loss history for plotting.
    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        count = 0
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{args.epochs}", leave=False)
        for batch in pbar:
            batch = batch.to(args.device)
            optimizer.zero_grad()
            _, loss = model(batch)
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update LR every batch
            total_loss += loss.item() * batch.size(0)
            count += batch.size(0)
            # Optionally, display current learning rate in the progress bar
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({"loss": loss.item(), "lr": current_lr})
            # print(f"lr: {current_lr}")
        avg_train_loss = total_loss / count
        train_losses.append(avg_train_loss)

        # Validation loop
        model.eval()
        val_loss = 0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(args.device)
                _, loss = model(batch)
                val_loss += loss.item() * batch.size(0)
                val_count += batch.size(0)
        avg_val_loss = val_loss / val_count
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Plot train and validation loss over epochs.
    os.makedirs("output", exist_ok=True)
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, args.epochs+1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss Over Epochs")
    plt.legend()
    plt.savefig("output/loss.png")
    plt.close()
    print("Loss plot saved to output/loss.png")

    plot_reconstructions(model, val_loader, device=args.device, save_path="output/reconstructions.png")
    evaluate_quality_metrics(model, val_loader, device=args.device)
    evaluate_codebook_usage(model, val_loader, device=args.device, save_path="output/codebook_usage.png")
    plot_latent_code_distribution(model, val_loader, device=args.device, save_path="output/latent_code_distribution.png")
    checkpoint = {
        "epoch": epoch + 1,  # the next epoch to run
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    torch.save(checkpoint, "output/vqcae_checkpoint.pt")
    print("Training complete. Model and evaluation plots saved in 'output/'.")

if __name__ == "__main__":
    main()
