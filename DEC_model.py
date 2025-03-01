#!/usr/bin/env python3
"""
A single-file PyTorch implementation of:
1) Convolutional Autoencoder (CAE) for 2D EEG images (CWT, etc.)
2) Deep Embedded Clustering (DEC) to get discrete cluster tokens

Addresses:
- Decoder reshape issues (no blind out.view(...,64,...))
- Freezing decoder during DEC fine-tuning
- Minimal validation metrics (reconstruction error, silhouette)
- Safer latent dimension extraction
- More robust shape handling & numerical stability

Author: [Your Name]
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm


# =========================
#   1) Dataset Definition
# =========================
class EEGNpyDataset(Dataset):
    """
    Example dataset that reads .npy files (C,H,W) from a directory.
    Adjust for your actual data pipeline.
    """

    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        self.files = [f for f in os.listdir(directory) if f.endswith('.npy')]
        self.files.sort()
        # Optional: read one file to infer shape
        if len(self.files) == 0:
            raise ValueError("No .npy files found in directory: ", directory)
        sample_path = os.path.join(directory, self.files[0])
        sample_data = np.load(sample_path)
        self.image_shape = sample_data.shape  # (C,H,W) expected

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.directory, self.files[idx])
        arr = np.load(path)
        # Potential check: arr.shape == self.image_shape
        return torch.from_numpy(arr).float()


# =========================
#    2) CAE Definition
# =========================
class ConvEncoder(nn.Module):
    """Encoder part of a Convolutional Autoencoder."""

    def __init__(self, input_shape=(3, 64, 64), latent_dim=64):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        C, H, W = input_shape
        # Define the conv layers
        self.conv_net = nn.Sequential(
            nn.Conv2d(C, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        # Figure out post-conv dimensions
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            out = self.conv_net(dummy)
            self.after_conv_shape = out.shape  # (1, 64, H', W')
            self.flatten_dim = out.view(1, -1).shape[1]

        # FC layer from flattened conv output to latent
        self.fc = nn.Linear(self.flatten_dim, self.latent_dim)

    def forward(self, x):
        out = self.conv_net(x)
        out = out.view(out.size(0), -1)
        z = self.fc(out)
        return z


class ConvDecoder(nn.Module):
    """Decoder part of a Convolutional Autoencoder."""

    def __init__(self, after_conv_shape, latent_dim=64):
        """
        after_conv_shape: shape after the encoder conv stack, e.g. (batch, 64, H', W')
                          We only need (64, H', W') part to properly reshape.
        latent_dim: dimension of the latent space
        """
        super().__init__()
        self.after_conv_shape = after_conv_shape  # (1, 64, H', W') for reference
        _, out_channels, out_h, out_w = after_conv_shape
        self.latent_dim = latent_dim

        # FC to go from latent to the shape needed by the deconv
        self.fc = nn.Linear(latent_dim, out_channels * out_h * out_w)

        # Mirror the conv_net in reverse
        self.deconv_net = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        B = z.size(0)
        _, out_channels, out_h, out_w = self.after_conv_shape
        # FC
        out = self.fc(z)  # shape (B, out_channels*out_h*out_w)
        # Reshape to feed into deconv
        out = out.view(B, out_channels, out_h, out_w)
        x_recon = self.deconv_net(out)
        return x_recon


class CAE(nn.Module):
    """Full Convolutional AutoEncoder for 2D images."""

    def __init__(self, input_shape=(3, 64, 64), latent_dim=64):
        super().__init__()
        self.encoder = ConvEncoder(input_shape, latent_dim)
        # Use the shape info from encoder
        after_conv_shape = self.encoder.after_conv_shape
        self.decoder = ConvDecoder(after_conv_shape, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


# =========================
#  3) CAE Training
# =========================
def pretrain_cae(cae_model, train_loader, val_loader=None,
                 epochs=20, lr=1e-3, device='cuda'):
    """
    Train the CAE for reconstruction. 
    Optionally validate on val_loader to monitor overfitting.
    """
    cae_model.to(device)
    optimizer = optim.Adam(cae_model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        cae_model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, _ = cae_model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # Validation
        if val_loader is not None:
            cae_model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for vbatch in val_loader:
                    vbatch = vbatch.to(device)
                    vrecon, _ = cae_model(vbatch)
                    vloss = criterion(vrecon, vbatch)
                    val_running_loss += vloss.item() * vbatch.size(0)
            val_loss = val_running_loss / len(val_loader.dataset)
            print(f"[CAE] Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        else:
            print(f"[CAE] Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}")

    return cae_model


# =========================
#   4) DEC Definition
# =========================
class DEC(nn.Module):
    """
    Deep Embedded Clustering: 
      - Takes a pretrained encoder 
      - Learns cluster centers in latent space 
      - Refines assignments using KL divergence with a target distribution.
    """

    def __init__(self, encoder, n_clusters=10, alpha=1.0):
        super().__init__()
        self.encoder = encoder
        self.n_clusters = n_clusters
        self.alpha = alpha

        # Extract latent_dim from encoder in a robust manner:
        if hasattr(encoder, 'latent_dim'):
            latent_dim = encoder.latent_dim
        else:
            raise ValueError("Encoder must define a .latent_dim attribute.")

        # Cluster centers as learnable parameters: shape (n_clusters, latent_dim)
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, latent_dim))

    @torch.no_grad()
    def initialize_centers(self, data_loader, device='cuda'):
        """
        Use K-means on the encoder's latent vectors to initialize cluster centers.
        """
        self.encoder.to(device)
        self.encoder.eval()

        latent_list = []
        for batch in data_loader:
            batch = batch.to(device)
            z = self.encoder(batch)
            latent_list.append(z.cpu().numpy())
        all_z = np.concatenate(latent_list, axis=0)

        print("[DEC] Initializing cluster centers via K-means...")
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
        preds = kmeans.fit_predict(all_z)
        centers = kmeans.cluster_centers_
        self.cluster_centers.data = torch.from_numpy(centers).to(device)
        return preds

    def forward(self, x):
        """ Forward pass: get latent z and soft cluster assignment q. """
        z = self.encoder(x)  # shape (B, latent_dim)
        q = self._student_t_distribution(z, self.cluster_centers, self.alpha)
        return z, q

    @staticmethod
    def _student_t_distribution(z, centers, alpha=1.0, eps=1e-10):
        """
        Student's t-distribution used by DEC to measure similarity.
         q_{ij} ~ (1 + ||z_i - mu_j||^2 / alpha)^(- (alpha+1)/2 )
        Returns soft assignments q of shape (B, n_clusters).
        """
        B = z.size(0)
        K = centers.size(0)

        # Expand for distance computation
        z_expand = z.unsqueeze(1).expand(B, K, -1)  # (B, K, latent_dim)
        centers_expand = centers.unsqueeze(0).expand(B, K, -1)  # (B, K, latent_dim)

        dist_sq = torch.sum((z_expand - centers_expand) ** 2, dim=2)  # (B, K)
        numerator = (1.0 + dist_sq / alpha).pow(- (alpha + 1.0) / 2.0)
        denominator = torch.sum(numerator, dim=1, keepdim=True) + eps
        q = numerator / denominator
        return q

    @staticmethod
    def target_distribution(q, eps=1e-10):
        """
        Compute the target distribution p from q (soft assignments).
         p_{ij} = q_{ij}^2 / f_j  / sum_j(q_{ij}^2 / f_j)
         where f_j = sum_i q_{ij}.
        """
        weight = (q ** 2) / torch.sum(q, dim=0, keepdim=True)
        # Add small eps to avoid div-by-zero
        weight = weight + eps
        p = (weight.t() / torch.sum(weight, dim=1, keepdim=True).t()).t()
        return p


def train_dec(dec_model, train_loader, val_loader=None,
              epochs=20, update_interval=50, device='cuda'):
    """
    Fine-tune the encoder (and cluster centers) for DEC using KL divergence.
    Optionally freeze the decoder (if you have a full CAE) to focus only on encoder.
    """
    dec_model.to(device)
    dec_model.train()

    # OPTIONAL: freeze the decoder if part of a bigger model
    # If dec_model.encoder is a CAE's encoder, you can do:
    # for param in your_cae_model.decoder.parameters():
    #     param.requires_grad = False

    optimizer = optim.Adam(dec_model.parameters(), lr=1e-4)
    loss_fn = nn.KLDivLoss(reduction='batchmean')

    # We'll track cluster assignments for silhouette checking if val_loader is given
    for epoch in range(1, epochs + 1):
        batch_count = 0
        for batch in train_loader:
            batch_count += 1
            x = batch.to(device)
            # Forward
            z, q = dec_model(x)

            # Update target distribution p periodically
            if (batch_count % update_interval) == 1:
                with torch.no_grad():
                    p = dec_model.target_distribution(q)

            # KL Divergence: KL(P||Q) = sum p * log(p/q)
            # We use F.kl_div(log_q, p, reduction='batchmean') => kl_div(p || q)
            # DEC typically uses sum(p_{ij} log(p_{ij}/q_{ij})).
            # We'll invert it: kl_div(log_q, p) => ∑ p * log(p/q).
            # Or we can do kl_div(log_p, q), but must be consistent:
            log_q = torch.log(q + 1e-10)
            loss = loss_fn(log_q, p)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # End of epoch, print info
        print(f"[DEC] Epoch {epoch}/{epochs}, KL Loss: {loss.item():.4f}")

        # Optional: validation silhouette
        if val_loader is not None and (epoch % 5 == 0):
            dec_model.eval()
            all_z = []
            all_labels = []
            with torch.no_grad():
                for vbatch in val_loader:
                    vbatch = vbatch.to(device)
                    vz, vq = dec_model(vbatch)
                    # Hard assignment
                    labels = torch.argmax(vq, dim=1).cpu().numpy()
                    all_labels.extend(labels)
                    all_z.append(vz.cpu().numpy())

            all_z = np.concatenate(all_z, axis=0)
            # If there's at least 2 clusters
            if len(np.unique(all_labels)) > 1:
                sscore = silhouette_score(all_z, all_labels)
                print(f"   [Val] Silhouette Score: {sscore:.4f}")
            else:
                print("   [Val] Only one cluster present; cannot compute silhouette.")
            dec_model.train()

    return dec_model


# =========================
#  5) Example Main Script
# =========================
if __name__ == "__main__":
    import argparse
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Subset

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="training_data/coeffs/",
                        help="Directory containing .npy EEG wavelet images")
    parser.add_argument("--epochs_cae", type=int, default=10)
    parser.add_argument("--epochs_dec", type=int, default=10)
    parser.add_argument("--n_clusters", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--lr_cae", type=float, default=1e-3)
    parser.add_argument("--lr_dec", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to run on")
    args = parser.parse_args()

    # Load dataset
    dataset = EEGNpyDataset(args.data_dir)
    full_size = len(dataset)
    indices = list(range(full_size))
    # Simple train/val split
    train_indices, val_indices = train_test_split(indices, test_size=0.2, shuffle=True, random_state=42)
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Infer data shape from first item
    sample_data = dataset[0]  # shape: (C,H,W)
    input_shape = tuple(sample_data.shape)  # e.g. (3, 64, 64)

    # 1) Pretrain CAE
    cae_model = CAE(input_shape=input_shape, latent_dim=args.latent_dim)
    cae_model = pretrain_cae(cae_model, train_loader, val_loader=val_loader,
                             epochs=args.epochs_cae, lr=args.lr_cae, device=args.device)

    # 2) Construct DEC with pretrained encoder
    #    We "detach" the encoder from the CAE, but it's the same weights.
    dec_model = DEC(encoder=cae_model.encoder,
                    n_clusters=args.n_clusters, alpha=1.0)

    # 3) Initialize cluster centers with K-means
    print("[MAIN] Initializing DEC cluster centers.")
    dec_model.initialize_centers(train_loader, device=args.device)

    # OPTIONAL: Freeze decoder if you want to keep reconstruction stable
    # for param in cae_model.decoder.parameters():
    #     param.requires_grad = False

    # 4) Train DEC
    print("[MAIN] Starting DEC training/fine-tuning.")
    dec_model = train_dec(dec_model, train_loader, val_loader=val_loader,
                          epochs=args.epochs_dec, update_interval=50,
                          device=args.device)

    print("Training complete.")

    # 5) Using the final DEC for tokenizing new data
    # Example: get cluster tokens for the validation set
    dec_model.eval()
    all_tokens = []
    with torch.no_grad():
        for vb in val_loader:
            vb = vb.to(args.device)
            vz, vq = dec_model(vb)
            cluster_indices = torch.argmax(vq, dim=1)
            all_tokens.extend(cluster_indices.cpu().numpy())
    print(f"Sample of final cluster tokens for val_data: {all_tokens[:50]}")
