import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import math

from DEC_model import EEGNpyDataset

import os
import matplotlib.pyplot as plt


def plot_ae_reconstructions(model, data_loader, device, n=8, out_path='recon.png'):
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            x = batch.to(device)
            x_recon, _, _, _ = model(x)
            break
    x = x.cpu()[:n]
    x_recon = x_recon.cpu()[:n]

    # Get input shape: x has shape (batch_size, C, H, W)
    C, H, W = x.shape[1], x.shape[2], x.shape[3]

    # Normalize x_recon to [0, 1] for visualization
    x_recon = (x_recon - x_recon.min()) / (
                x_recon.max() - x_recon.min() + 1e-8)  # Add small epsilon to avoid division by zero

    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    for i in range(n):
        if C == 1:
            # Grayscale: permute to (H, W, C) and squeeze to (H, W)
            axes[0, i].imshow(x[i].permute(1, 2, 0).squeeze(), cmap='gray')
            axes[1, i].imshow(x_recon[i].permute(1, 2, 0).squeeze(), cmap='gray')
        elif C == 3:
            # RGB: permute to (H, W, 3)
            axes[0, i].imshow(x[i].permute(1, 2, 0))
            axes[1, i].imshow(x_recon[i].permute(1, 2, 0))
        else:
            # For C > 3 or other cases, plot the first channel as grayscale
            axes[0, i].imshow(x[i][0], cmap='gray')
            axes[1, i].imshow(x_recon[i][0], cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].axis('off')
    plt.savefig(out_path)
    plt.close()# Create QA/VaDE directory if it doesn't exist
os.makedirs('QA/VaDE', exist_ok=True)
# Assuming EEGNpyDataset and plot_ae_reconstructions are defined elsewhere
# from your_module import EEGNpyDataset, plot_ae_reconstructions

def predict_clusters(model, data_loader, device):
    model.eval()
    clusters = []
    with torch.no_grad():
        for batch in data_loader:
            x = batch.to(device)
            _, mu_q, log_var_q, z = model(x)
            # Compute q(c|x)
            diff = mu_q.unsqueeze(1) - model.mu_c  # (batch_size, K, latent_dim)
            log_likelihood = (
                -0.5 * model.latent_dim * math.log(2 * math.pi)
                - 0.5 * model.latent_dim * model.log_var_c
                - 0.5 / model.log_var_c.exp() * diff.pow(2).sum(2)
            )  # (batch_size, K)
            log_p_c = F.log_softmax(model.log_p_c, dim=0)  # (K,)
            log_q_c_x_unnorm = log_p_c + log_likelihood
            log_q_c_x = F.log_softmax(log_q_c_x_unnorm, dim=1)
            q_c_x = log_q_c_x.exp()  # (batch_size, K)
            cluster_assignments = q_c_x.argmax(dim=1)  # (batch_size,)
            clusters.append(cluster_assignments.cpu().numpy())
    clusters = np.concatenate(clusters)
    return clusters

class VaDE(nn.Module):
    """
    Variational Deep Embedding (VaDE) model with convolutional encoder and decoder.
    """
    def __init__(self, input_shape, latent_dim=10, n_clusters=10):
        """
        Args:
            input_shape (tuple): Input shape (C, H, W).
            latent_dim (int): Dimension of the latent space.
            n_clusters (int): Number of clusters for GMM.
        """
        super(VaDE, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        C, H, W = input_shape

        # Encoder: Conv layers to reduce spatial dimensions, then linear to latent space
        self.encoder = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (H // 4) * (W // 4), 256),
            nn.ReLU(),
            nn.Linear(256, 2 * latent_dim)  # Outputs mu and log_var
        )

        # Decoder: Linear to feature map, then transposed conv to reconstruct input
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * (H // 4) * (W // 4)),
            nn.ReLU(),
            nn.Unflatten(1, (64, H // 4, W // 4)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, C, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

        # GMM parameters
        self.mu_c = nn.Parameter(torch.randn(n_clusters, latent_dim))
        self.log_var_c = nn.Parameter(torch.zeros(n_clusters))
        self.log_p_c = nn.Parameter(torch.zeros(n_clusters))

    def encode(self, x):
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu_q, log_var_q = h.chunk(2, dim=1)
        return mu_q, log_var_q

    def reparameterize(self, mu_q, log_var_q):
        """Reparameterization trick to sample z from q(z|x)."""
        std = torch.exp(0.5 * log_var_q)
        eps = torch.randn_like(std)
        return mu_q + eps * std

    def decode(self, z):
        """Decode latent z back to input space."""
        return self.decoder(z)

    def forward(self, x):
        """Forward pass: encode, sample, decode."""
        mu_q, log_var_q = self.encode(x)
        z = self.reparameterize(mu_q, log_var_q)
        x_recon = self.decode(z)
        return x_recon, mu_q, log_var_q, z

def vae_loss(x, x_recon, mu_q, log_var_q):
    """
    Standard VAE loss: reconstruction loss + KL divergence to N(0, I).
    """
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    kl_div = -0.5 * (1 + log_var_q - mu_q.pow(2) - log_var_q.exp()).sum(1).mean()
    return recon_loss + kl_div

def vade_loss(x, x_recon, mu_q, log_var_q, model):
    """
    VaDE loss: reconstruction loss + KL divergence with GMM prior.
    """
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    d = model.latent_dim

    # GMM parameters
    var_c = model.log_var_c.exp()           # (K,)
    log_var_c = model.log_var_c            # (K,)
    mu_c = model.mu_c                      # (K, latent_dim)
    log_p_c = F.log_softmax(model.log_p_c, dim=0)  # (K,)

    # Compute q(c|x)
    diff = mu_q.unsqueeze(1) - mu_c        # (batch_size, K, latent_dim)
    log_likelihood = (
        -0.5 * d * math.log(2 * math.pi)
        - 0.5 * d * log_var_c
        - 0.5 / var_c * diff.pow(2).sum(2)
    )                                      # (batch_size, K)
    log_q_c_x_unnorm = log_p_c + log_likelihood
    log_q_c_x = F.log_softmax(log_q_c_x_unnorm, dim=1)
    q_c_x = log_q_c_x.exp()                # (batch_size, K)

    # Compute KL(q(z|x) || p(z|c)) for each cluster
    sum_var_q = log_var_q.exp().sum(1)     # (batch_size,)
    log_det_q = log_var_q.sum(1)           # (batch_size,)
    diff_sq = diff.pow(2).sum(2)           # (batch_size, K)
    inv_var_c = 1 / var_c                  # (K,)
    kl_per_cluster = 0.5 * (
        inv_var_c.unsqueeze(0) * sum_var_q.unsqueeze(1) +
        inv_var_c.unsqueeze(0) * diff_sq +
        d * log_var_c.unsqueeze(0) - d -
        log_det_q.unsqueeze(1)
    )                                      # (batch_size, K)

    # Expected KL over clusters
    expected_kl = (q_c_x * kl_per_cluster).sum(1)  # (batch_size,)

    # KL(q(c|x) || p(c))
    kl_categorical = (q_c_x * (log_q_c_x - log_p_c)).sum(1)  # (batch_size,)

    total_kl = expected_kl + kl_categorical
    return recon_loss + total_kl.mean()

def initialize_gmm_params(model, train_loader, device):
    """Initialize GMM parameters using K-means on latent means after pretraining."""
    with torch.no_grad():
        all_mu_q = []
        for batch in train_loader:
            x = batch.to(device)
            mu_q, _ = model.encode(x)
            all_mu_q.append(mu_q.cpu())
        all_mu_q = torch.cat(all_mu_q, dim=0)

        # Run K-means
        kmeans = KMeans(n_clusters=model.n_clusters, random_state=42)
        kmeans.fit(all_mu_q.numpy())
        model.mu_c.data = torch.from_numpy(kmeans.cluster_centers_).to(device)

        # Compute total variance in latent space
        mean_mu = all_mu_q.mean(0)
        var_total = ((all_mu_q - mean_mu)**2).mean()
        model.log_var_c.data = torch.log(var_total) * torch.ones(model.n_clusters).to(device)

        # Initialize p(c) as uniform (log_p_c = 0)
        model.log_p_c.data = torch.zeros(model.n_clusters).to(device)
def evaluate_vae(model, data_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            x = batch.to(device)
            x_recon, mu_q, log_var_q, z = model(x)
            loss = vae_loss(x, x_recon, mu_q, log_var_q)  # Assume vae_loss is defined
            total_loss += loss.item()
    return total_loss / len(data_loader)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="training_data/coeffs/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pretrain_epochs", type=int, default=50)
    parser.add_argument("--cluster_epochs", type=int, default=100)
    args = parser.parse_args()

    # 1) Dataset
    dataset = EEGNpyDataset(args.data_dir,normalize=True)
    idxs = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(idxs, test_size=0.2, random_state=42)
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    sample_data = dataset[0]
    input_shape = sample_data.shape  # (C, H, W)
    print(input_shape)

    # 2) Model
    model = VaDE(input_shape=input_shape, latent_dim=1024, n_clusters=100).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 3) Pretraining as VAE
    print("Starting VAE pretraining...")
    pretrain_train_losses = []
    pretrain_val_losses = []

    for epoch in range(args.pretrain_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            x = batch.to(args.device)
            x_recon, mu_q, log_var_q, z = model(x)
            loss = vae_loss(x, x_recon, mu_q, log_var_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        pretrain_train_losses.append(train_loss)

        val_loss = evaluate_vae(model, val_loader, args.device)
        pretrain_val_losses.append(val_loss)
        print(f"Pretraining Epoch {epoch + 1}/{args.pretrain_epochs}, "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    plt.plot(pretrain_train_losses, label='Train')
    plt.plot(pretrain_val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Pretraining Loss')
    plt.legend()
    plt.savefig('QA/VaDE/pretrain_loss.png')
    plt.close()
    # 4) Initialize GMM parameters
    print("Initializing GMM parameters...")
    initialize_gmm_params(model, train_loader, args.device)
    plot_ae_reconstructions(model, val_loader, device=args.device, n=8, out_path='QA/VaDE/pretrain_ae_recons.png')
    # 5) Training with VaDE loss
    print("Starting VaDE clustering training...")


    def evaluate_vade(model, data_loader, device):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in data_loader:
                x = batch.to(device)
                x_recon, mu_q, log_var_q, z = model(x)
                loss = vade_loss(x, x_recon, mu_q, log_var_q, model)  # Assume vade_loss is defined
                total_loss += loss.item()
        return total_loss / len(data_loader)


    cluster_train_losses = []
    cluster_val_losses = []

    for epoch in range(args.cluster_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            x = batch.to(args.device)
            x_recon, mu_q, log_var_q, z = model(x)
            loss = vade_loss(x, x_recon, mu_q, log_var_q, model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        cluster_train_losses.append(train_loss)

        val_loss = evaluate_vade(model, val_loader, args.device)
        cluster_val_losses.append(val_loss)
        print(f"Clustering Epoch {epoch + 1}/{args.cluster_epochs}, "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    plt.plot(cluster_train_losses, label='Train')
    plt.plot(cluster_val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Clustering Training Loss')
    plt.legend()
    plt.savefig('QA/VaDE/cluster_loss.png')
    plt.close()
    clusters = predict_clusters(model, val_loader, args.device)

    # Plot histogram
    plt.hist(clusters, bins=np.arange(model.n_clusters + 1) - 0.5, rwidth=0.8)
    plt.xlabel('Cluster')
    plt.ylabel('Frequency')
    plt.title('Cluster Assignments Histogram')
    plt.savefig('QA/VaDE/clusters_histogram.png')
    plt.close()
    plot_ae_reconstructions(model, val_loader, device=args.device, n=8, out_path='QA/VaDE/final_ae_recons.png')