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
    def __init__(self, codebook_size, embedding_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.eps = eps
        self.embedding = nn.Embedding(self.codebook_size, self.embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0/self.codebook_size, 1.0/self.codebook_size)
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_w", self.embedding.weight.clone())

    def forward(self, z):
        with torch.no_grad():
            flat_z = z.view(-1, self.embedding_dim)
            dist = torch.cdist(flat_z, self.embedding.weight, p=2)
            idxs = dist.argmin(dim=1)
            encodings = F.one_hot(idxs, self.codebook_size).type(z.dtype)
            idxs = idxs.view(*z.shape[:-1])
        z_q = self.embedding(idxs)
        z_q = z_q.view_as(z)
        diff = (z_q.detach() - z).pow(2).mean() + (z_q - z.detach()).pow(2).mean()
        if self.training:
            self._update_ema(flat_z, encodings)
        z_q = z + (z_q - z).detach()
        return z_q, idxs, diff

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
            nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)

class VQCAE(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=128, codebook_size=128, decay=0.99, commitment_beta=0.25):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_channels)
        self.vq = VectorQuantizerEMA(codebook_size, hidden_channels, decay=decay)
        self.decoder = Decoder(in_channels, hidden_channels)
        self.commitment_beta = commitment_beta
    def forward(self, x):
        z_e = self.encoder(x)
        z_q, idxs, vq_loss = self.vq(z_e.permute(0,2,3,1))
        z_q = z_q.permute(0,3,1,2)
        x_rec = self.decoder(z_q)
        loss = F.mse_loss(x_rec, x) + self.commitment_beta * vq_loss
        return x_rec, loss

def plot_reconstructions(model, data_loader, device, save_path="recon.png", n=8):
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
            axs[0, i].imshow(np.transpose(orig, (1,2,0)))
            axs[1, i].imshow(np.transpose(rec, (1,2,0)))
        axs[0, i].axis('off')
        axs[1, i].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="training_data/coeffs/")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
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

    model = VQCAE(in_channels=in_channels, hidden_channels=128, codebook_size=128).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

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
            total_loss += loss.item() * batch.size(0)
            count += batch.size(0)
            pbar.set_postfix({"loss": loss.item()})
        avg_train_loss = total_loss / count

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
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    os.makedirs("output", exist_ok=True)
    plot_reconstructions(model, val_loader, device=args.device, save_path="output/recon.png")
    torch.save(model.state_dict(), "output/vqcae.pt")

if __name__ == "__main__":
    main()
