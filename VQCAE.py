#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
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
from tqdm import tqdm

# =========================
#   1) Dataset Definition
# =========================
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class EEGNpyDataset(Dataset):
    """
    Example dataset that reads .npy files (C,H,W) from a directory.
    Adjust for your actual data pipeline.
    """
    def __init__(self, directory, normalize=False):
        super().__init__()
        self.directory = directory
        self.files = [f for f in os.listdir(directory) if f.endswith('.npy')]
        if not self.files:
            raise ValueError(f"No .npy files found in directory: {directory}")
        self.files.sort()
        sample = np.load(os.path.join(directory, self.files[0]))
        self.image_shape = sample.shape  # e.g., (C, H, W)
        self.normalize = normalize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.directory, self.files[idx])
        arr = np.load(path)
        if self.normalize:
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return torch.from_numpy(arr).float()

# =========================
#    2) CAE Definition
# =========================

class VQCAE(nn.Module):
    def __init__(self, input_shape=(3, 136, 512), codebook_size=128, beta=0.25):
        super().__init__()
        # Encoder
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU()
        )
        # Codebook
        self.codebook = nn.Parameter(torch.randn(codebook_size, 128))
        # Decoder
        self.deconv_net = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1), nn.Tanh()
        )
        self.beta = beta

    def forward(self, x):
        z_e = self.conv_net(x)  # (B, 128, 17, 64)
        B, C, H, W = z_e.shape
        z_e_reshaped = z_e.permute(0, 2, 3, 1).reshape(B, -1, C)  # (B, 17*64, 128)
        distances = torch.sum((z_e_reshaped.unsqueeze(2) - self.codebook.unsqueeze(0))**2, dim=-1)  # (B, 17*64, K)
        indices = torch.argmin(distances, dim=2)  # (B, 17*64)
        z_q_reshaped = self.codebook[indices]  # (B, 17*64, 128)
        z_q = z_q_reshaped.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, 128, 17, 64)
        recon = self.deconv_net(z_q)
        return recon, z_e, z_q



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
        output = cae_model(batch)
        if isinstance(output, tuple) and len(output) >= 1:
            recons = output[0]
        else:
            recons = output

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
            x_recon, z, q = dec_model(batch)
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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--n_clusters", type=int, default=500)
    parser.add_argument("--epochs_cae", type=int, default=200)
    parser.add_argument("--epochs_dec", type=int, default=30)
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

    # # 2) CAE Pretraining
    sample_data = dataset[0]
    input_shape = sample_data.shape #(C, H, W)
    cae_model = VQCAE().to(args.device)
    optimizer = torch.optim.Adam(cae_model.parameters(), lr=1e-3)
    for epoch in range(args.epochs_cae):
        print(f"Epoch {epoch + 1}/{args.epochs_cae}")
        for batch in tqdm(train_loader, desc="Training batches"):
            batch = batch.to(args.device)
            optimizer.zero_grad()
            recon, z_e, z_q = cae_model(batch)
            reconstruction_loss = 0.5 * F.mse_loss(recon, batch) + 0.5 * F.l1_loss(recon, batch)
            vq_loss = ((z_e.detach() - z_q) ** 2).mean()
            commitment_loss = ((z_e - z_q.detach()) ** 2).mean()
            total_loss = reconstruction_loss + vq_loss + 0.25 * commitment_loss
            total_loss.backward()
            optimizer.step()
    # ---- QA: Check reconstructions from validation set ----
    plot_ae_reconstructions(cae_model, val_loader, device=args.device,
                            n=8, out_path='QA/DEC/ae_recons.png')
    # After CAE training is done
    torch.save(cae_model.state_dict(), "QA/DEC/cae_model.pt")


    cae_model.eval()
    all_val_labels = []
    with torch.no_grad():
        for vbatch in val_loader:
            vb = vbatch.to(args.device)
            x_recon, z, vq = cae_model(vb)
            labels = torch.argmax(vq, dim=1).cpu().numpy()
            all_val_labels.extend(labels)
    all_val_labels = np.array(all_val_labels)

    # Plot cluster distribution
    plot_cluster_distribution(all_val_labels, out_path='QA/DEC/cluster_hist.png')

    # Plot images from the most frequent cluster
    plot_most_frequent_cluster_images(cae_model, val_ds, device=args.device,
                                      top_n=8, out_path='QA/DEC/most_freq_cluster.png')

    print("[MAIN] Done. QA plots saved in QA/DEC/")

    # After DEC training is done
    torch.save(cae_model.state_dict(), "QA/DEC/dec_model.pt")
    # ---- QA: Check reconstructions from validation set ----
    plot_ae_reconstructions(cae_model, val_loader, device=args.device,
                            n=8, out_path='QA/DEC/ae_recons_final_IDEC.png')
