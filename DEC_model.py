#!/usr/bin/env python3

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend if on a headless server
import matplotlib.pyplot as plt

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
        if not self.files:
            raise ValueError(f"No .npy files found in directory: {directory}")
        self.files.sort()
        sample = np.load(os.path.join(directory, self.files[0]))
        self.image_shape = sample.shape  # e.g., (C, H, W)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.directory, self.files[idx])
        arr = np.load(path)
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
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU()
        )
        # Determine shape after convs
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
        super().__init__()
        _, out_channels, out_h, out_w = after_conv_shape
        self.fc = nn.Linear(latent_dim, out_channels * out_h * out_w)

        self.deconv_net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1), nn.Tanh()
        )
        self.out_channels = out_channels
        self.out_h = out_h
        self.out_w = out_w

    def forward(self, z):
        B = z.size(0)
        out = self.fc(z)  # shape (B, out_channels * out_h * out_w)
        out = out.view(B, self.out_channels, self.out_h, self.out_w)
        x_recon = self.deconv_net(out)
        return x_recon


class CAE(nn.Module):
    """Full Convolutional AutoEncoder for 2D images."""
    def __init__(self, input_shape=(3,64,64), latent_dim=64):
        super().__init__()
        self.encoder = ConvEncoder(input_shape, latent_dim)
        self.decoder = ConvDecoder(self.encoder.after_conv_shape, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


# =========================
#  3) CAE Training
# =========================
def pretrain_cae(cae_model, train_loader, val_loader=None,
                 epochs=10, lr=1e-3, device='cuda'):
    """
    Train the CAE for reconstruction.
    Optionally validate on val_loader to monitor overfitting.
    """
    cae_model.to(device)
    optimizer = optim.Adam(cae_model.parameters(), lr=lr)
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    for epoch in range(1, epochs+1):
        cae_model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, _ = cae_model(batch)
            loss = 0.5 * criterion_mse(recon, batch) + 0.5 * criterion_l1(recon, batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        if val_loader is not None:
            cae_model.eval()
            val_loss_accum = 0.0
            with torch.no_grad():
                for vbatch in val_loader:
                    vbatch = vbatch.to(device)
                    vrecon, _ = cae_model(vbatch)
                    vloss = 0.5 * criterion_mse(recon, batch) + 0.5 * criterion_l1(recon, batch)
                    val_loss_accum += vloss.item() * vbatch.size(0)
            val_loss = val_loss_accum / len(val_loader.dataset)
            print(f"[CAE] Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        else:
            print(f"[CAE] Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}")

    return cae_model


# =========================
#   4) DEC Definition
# =========================
class DEC(nn.Module):
    """
    Deep Embedded Clustering (DEC):
      - Takes a pretrained encoder
      - Learns cluster centers in latent space
      - Refines assignments using KL divergence.
    """
    def __init__(self, encoder, n_clusters=10, alpha=1.0):
        super().__init__()
        self.encoder = encoder
        self.n_clusters = n_clusters
        self.alpha = alpha

        if hasattr(encoder, 'latent_dim'):
            latent_dim = encoder.latent_dim
        else:
            raise ValueError("Encoder must define a .latent_dim attribute.")

        # cluster centers (n_clusters, latent_dim)
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

        print("[DEC] K-means initialization...")
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
        preds = kmeans.fit_predict(all_z)
        centers = kmeans.cluster_centers_
        self.cluster_centers.data = torch.from_numpy(centers).to(device)
        return preds

    def forward(self, x):
        z = self.encoder(x)  # shape (B, latent_dim)
        q = self._student_t_distribution(z, self.cluster_centers, self.alpha)
        return z, q

    @staticmethod
    def _student_t_distribution(z, centers, alpha=1.0, eps=1e-10):
        """
        q_{ij} ~ (1 + ||z_i - mu_j||^2 / alpha)^(- (alpha+1)/2)
        """
        B, latent_dim = z.size()
        K, _ = centers.size()

        z_expand = z.unsqueeze(1).expand(B, K, latent_dim)
        centers_expand = centers.unsqueeze(0).expand(B, K, latent_dim)
        dist_sq = torch.sum((z_expand - centers_expand)**2, dim=2)
        numerator = (1.0 + dist_sq / alpha).pow(- (alpha + 1.0)/2)
        denominator = torch.sum(numerator, dim=1, keepdim=True) + eps
        q = numerator / denominator
        return q

    @staticmethod
    def target_distribution(q, eps=1e-10):
        """
        p = q^2 / f_j  (normalized over j)
        f_j = sum_i q_{ij}
        """
        weight = q**2 / torch.sum(q, dim=0, keepdim=True)
        weight = weight + eps
        p = (weight.t() / torch.sum(weight, dim=1, keepdim=True).t()).t()
        return p


def train_dec_full_pass(dec_model, train_loader, val_loader=None,
                        epochs=10, device='cuda'):
    """
    Classic DEC approach (two-pass each epoch):
      1) On each epoch, compute q for entire dataset in eval mode.
      2) Compute p (target distribution) for entire dataset.
      3) Train in mini-batches, slicing p for each batch.
    """
    dec_model.to(device)

    optimizer = optim.Adam([
        {'params': dec_model.encoder.parameters(), 'lr': 0},  # Encoder frozen initially (lr=0)
        {'params': [dec_model.cluster_centers], 'lr': 1e-4}  # Cluster centers trained with lr=1e-4
    ])
    loss_fn = nn.KLDivLoss(reduction='batchmean')

    train_data_size = len(train_loader.dataset)

    for epoch in range(1, epochs+1):
        # ---- 1) Compute q for entire dataset ----
        dec_model.eval()
        all_q = []
        with torch.no_grad():
            for batch in train_loader:
                x = batch.to(device)
                _, q = dec_model(x)
                all_q.append(q.cpu())
        all_q = torch.cat(all_q, dim=0)  # shape (N, n_clusters)

        # ---- 2) Compute p (target distribution) for entire dataset ----
        p = dec_model.target_distribution(all_q)

        if epoch == 10:
            optimizer.param_groups[0]['lr'] = 1e-5
            print("[DEC] Switching to fine-tuning mode with encoder lr=1e-5")

        # ---- 3) Train in mini-batches, slicing p for each batch ----
        dec_model.train()
        idx_offset = 0
        running_loss = 0.0

        for batch in train_loader:
            b_size = batch.size(0)
            x = batch.to(device)
            p_batch = p[idx_offset: idx_offset + b_size, :].to(device)
            idx_offset += b_size

            _, q_batch = dec_model(x)
            log_q = torch.log(q_batch + 1e-10)
            kl_loss = loss_fn(log_q, p_batch)

            optimizer.zero_grad()
            kl_loss.backward()
            optimizer.step()
            running_loss += kl_loss.item() * b_size

        epoch_loss = running_loss / train_data_size
        print(f"[DEC] Epoch {epoch}/{epochs}, KL Loss: {epoch_loss:.4f}")

        # Optional: measure cluster quality on val set
        if val_loader is not None and epoch % 10 == 0:
            dec_model.eval()
            z_vals = []
            labels = []
            with torch.no_grad():
                for vbatch in val_loader:
                    vbatch = vbatch.to(device)
                    vz, vq = dec_model(vbatch)
                    cluster_idx = torch.argmax(vq, dim=1)
                    z_vals.append(vz.cpu().numpy())
                    labels.extend(cluster_idx.cpu().numpy())
            z_vals = np.concatenate(z_vals, axis=0)
            # e.g., silhouette
            if len(np.unique(labels)) > 1:
                from sklearn.metrics import silhouette_score
                sscore = silhouette_score(z_vals, labels)
                print(f"   [Val] Silhouette Score: {sscore:.4f}")
            else:
                print("   [Val] Only one cluster => silhouette not defined.")

    return dec_model


# =========================
#  5) QA Plotting Helpers
# =========================

def plot_ae_reconstructions(cae_model, dataloader, device='cuda', n=8, out_path='QA/DEC/ae_recons.png'):
    """
    Saves a figure showing a few examples of original vs. reconstructed images from the CAE.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cae_model.eval()
    batch = next(iter(dataloader))  # get a single batch
    batch = batch.to(device)
    with torch.no_grad():
        recons, _ = cae_model(batch)

    # Convert to CPU for plotting
    originals = batch.cpu().numpy()
    recons = recons.cpu().numpy()

    # We'll plot up to n examples
    n = min(n, originals.shape[0])

    fig, axes = plt.subplots(2, n, figsize=(2*n, 4))
    for i in range(n):
        # Original
        ax_orig = axes[0, i]
        # Recon
        ax_recon = axes[1, i]

        orig_img = originals[i]
        recon_img = recons[i]

        ## for not normalize data
        orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-8)
        recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min() + 1e-8)

        # If C==1, or C==3
        if orig_img.shape[0] == 1:
            ax_orig.imshow(orig_img[0], cmap='gray')
            ax_recon.imshow(recon_img[0], cmap='gray')
        else:
            # Channels-first => transpose to H,W,C
            ax_orig.imshow(np.transpose(orig_img, (1, 2, 0)))
            ax_recon.imshow(np.transpose(recon_img, (1, 2, 0)))

        ax_orig.set_title("Original")
        ax_recon.set_title("Reconstructed")
        ax_orig.axis('off')
        ax_recon.axis('off')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[QA] Saved AE reconstruction examples to {out_path}")


def plot_cluster_distribution(labels, out_path='QA/DEC/cluster_hist.png'):
    """
    Saves a histogram of cluster frequencies.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    unique_labels, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(6,4))
    plt.bar(unique_labels, counts, color='steelblue')
    plt.xlabel('Cluster ID')
    plt.ylabel('Count')
    plt.title("Cluster Distribution")
    plt.savefig(out_path)
    plt.close()
    print(f"[QA] Saved cluster histogram to {out_path}")


def plot_most_frequent_cluster_images(dec_model, dataset, device='cuda',
                                      top_n=8, out_path='QA/DEC/most_freq_cluster.png'):
    """
    Finds which cluster is most frequent (on the entire dataset),
    collects 'top_n' examples from that cluster, and plots them.

    Note: This is just original images, not reconstructions.
          If you want reconstructions, run them through the CAE decoder.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    dec_model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    all_labels = []
    # We'll store indices -> cluster
    idx_to_cluster = []

    with torch.no_grad():
        idx_counter = 0
        for batch in loader:
            batch = batch.to(device)
            _, q = dec_model(batch)
            lbls = torch.argmax(q, dim=1).cpu().numpy()
            for l in lbls:
                idx_to_cluster.append(l)
            idx_counter += len(lbls)

    all_labels = np.array(idx_to_cluster)  # shape (N,)

    # find the most frequent cluster
    unique_lbls, counts = np.unique(all_labels, return_counts=True)
    most_freq_cluster = unique_lbls[np.argmax(counts)]
    print(f"[QA] Most frequent cluster is {most_freq_cluster} with count={np.max(counts)}")

    # Collect top_n examples from that cluster (just pick the first top_n we encounter)
    selected_indices = np.where(all_labels == most_freq_cluster)[0][:top_n]
    if len(selected_indices) == 0:
        print("[QA] No samples found for most frequent cluster? Something is off.")
        return

    # Now retrieve those images from the dataset
    fig, axes = plt.subplots(1, len(selected_indices), figsize=(3*len(selected_indices), 3))
    if len(selected_indices) == 1:
        axes = [axes]  # ensure it's iterable

    for i, idx_val in enumerate(selected_indices):
        img = dataset[idx_val].numpy()  # shape (C,H,W)
        ax = axes[i]
        if img.shape[0] == 1:
            ax.imshow(img[0], cmap='gray')
        else:
            ax.imshow(np.transpose(img, (1, 2, 0)))
        ax.set_title(f"Cluster {most_freq_cluster}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[QA] Saved images from most frequent cluster to {out_path}")


# =========================
#  6) Example Main Script
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="training_data/coeffs/")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--n_clusters", type=int, default=100)
    parser.add_argument("--epochs_cae", type=int, default=100)
    parser.add_argument("--epochs_dec", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # 1) Dataset
    dataset = EEGNpyDataset(args.data_dir)
    idxs = list(range(len(dataset)))
    # Simple train/val split
    train_idx, val_idx = train_test_split(idxs, test_size=0.2, random_state=42)
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 2) CAE Pretraining
    sample_data = dataset[0]
    input_shape = sample_data.shape  # e.g., (C, H, W)
    cae_model = CAE(input_shape, args.latent_dim)
    cae_model = pretrain_cae(cae_model, train_loader, val_loader=val_loader,
                             epochs=args.epochs_cae, lr=3e-4, device=args.device)

    # ---- QA: Check reconstructions from validation set ----
    plot_ae_reconstructions(cae_model, val_loader, device=args.device,
                            n=8, out_path='QA/DEC/ae_recons.png')

    # 3) DEC Setup
    dec_model = DEC(encoder=cae_model.encoder, n_clusters=args.n_clusters, alpha=1)
    # Initialize cluster centers
    print("[MAIN] Initializing DEC cluster centers...")
    dec_model.initialize_centers(train_loader, device=args.device)

    # 4) DEC Training (two-pass each epoch)
    print("[MAIN] Starting DEC training/fine-tuning...")
    dec_model = train_dec_full_pass(dec_model, train_loader, val_loader=val_loader,
                                    epochs=args.epochs_dec, device=args.device)

    # 5) Evaluate final cluster distribution on validation
    dec_model.eval()
    all_val_labels = []
    with torch.no_grad():
        for vbatch in val_loader:
            vb = vbatch.to(args.device)
            _, vq = dec_model(vb)
            labels = torch.argmax(vq, dim=1).cpu().numpy()
            all_val_labels.extend(labels)
    all_val_labels = np.array(all_val_labels)

    # Plot cluster distribution
    plot_cluster_distribution(all_val_labels, out_path='QA/DEC/cluster_hist.png')

    # Plot images from the most frequent cluster
    plot_most_frequent_cluster_images(dec_model, val_ds, device=args.device,
                                      top_n=8, out_path='QA/DEC/most_freq_cluster.png')

    print("[MAIN] Done. QA plots saved in QA/DEC/")
