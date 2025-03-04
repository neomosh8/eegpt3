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
class ConvEncoder(nn.Module):
    """Encoder part of a Convolutional Autoencoder."""
    def __init__(self, input_shape=(3, 64, 64), latent_dim=64):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        C, H, W = input_shape
        # Define the conv layers
        self.conv_net = nn.Sequential(
            nn.Conv2d(C, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),nn.BatchNorm2d(128),nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(128),nn.ReLU(),
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
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),nn.BatchNorm2d(64), nn.ReLU(),
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
                 epochs=10, lr=1e-3, device='cuda',max_lr=1e-3):
    """
    Train the CAE for reconstruction.
    Optionally validate on val_loader to monitor overfitting.
    """
    cae_model.to(device)
    optimizer = optim.Adam(cae_model.parameters(), lr=lr,weight_decay=1e-5)
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()

    # OneCycle parameters
    total_steps = epochs * len(train_loader)
    pct_start = 0.3  # Use 30% of steps for increasing LR
    div_factor = 10  # initial_lr = max_lr/div_factor
    final_div_factor = 100  # final_lr = initial_lr/final_div_factor

    # Step-wise LR scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor
    )

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
            scheduler.step()
            running_loss += loss.item() * batch.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        if val_loader is not None:
            cae_model.eval()
            val_loss_accum = 0.0
            with torch.no_grad():
                for vbatch in val_loader:
                    vbatch = vbatch.to(device)
                    vrecon, _ = cae_model(vbatch)
                    vloss = 0.5 * criterion_mse(vrecon, vbatch) + 0.5 * criterion_l1(vrecon, vbatch)
                    val_loss_accum += vloss.item() * vbatch.size(0)
            val_loss = val_loss_accum / len(val_loader.dataset)
            print(f"[CAE] Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        else:
            print(f"[CAE] Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}")

    return cae_model


# =========================
#   4) DEC Definition
# =========================

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
            x_recon, z, q = idec_model(batch)
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



class IDEC(nn.Module):
    """
    Improved Deep Embedded Clustering (IDEC) combines clustering with a reconstruction loss.
    It uses:
      - An encoder-decoder (autoencoder) architecture
      - A clustering branch in the latent space with KL divergence loss
      - A balancing term on the cluster frequencies
    """
    def __init__(self, encoder, decoder, n_clusters=10, alpha=1.0, target_beta=0.1):
        """
        :param encoder: Pretrained encoder that should have a .latent_dim attribute.
        :param decoder: Decoder network for reconstruction.
        :param n_clusters: Number of clusters.
        :param alpha: Degrees of freedom for the Student's t-distribution.
        :param target_beta: Mixing factor for target distribution (0 means pure DEC target, >0 mixes with uniform).
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.target_beta = target_beta  # for refining the target distribution

        if hasattr(encoder, 'latent_dim'):
            latent_dim = encoder.latent_dim
        else:
            raise ValueError("Encoder must define a .latent_dim attribute.")

        # Initialize cluster centers (n_clusters, latent_dim)
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, latent_dim))

    @torch.no_grad()
    def initialize_centers(self, data_loader, device='cuda'):
        """
        Uses KMeans++ (with more initializations) on the latent representations to initialize cluster centers.
        """
        self.encoder.to(device)
        self.encoder.eval()

        latent_list = []
        for batch in data_loader:
            batch = batch.to(device)
            z = self.encoder(batch)
            latent_list.append(z.cpu().numpy())
        all_z = np.concatenate(latent_list, axis=0)

        print("[IDEC] Initializing cluster centers with KMeans++...")
        # Use KMeans++ for robust initialization
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, init='k-means++')
        kmeans.fit(all_z)
        centers = kmeans.cluster_centers_
        self.cluster_centers.data = torch.from_numpy(centers).to(device)
        return kmeans.labels_

    def forward(self, x):
        """
        Forward pass returns:
          - x_recon: Reconstruction of x from the decoder.
          - z: Latent representation from the encoder.
          - q: Soft cluster assignments from the Student's t-distribution.
        """
        z = self.encoder(x)
        q = self._student_t_distribution(z, self.cluster_centers, self.alpha)
        x_recon = self.decoder(z)
        return x_recon, z, q

    @staticmethod
    def _student_t_distribution(z, centers, alpha=1.0, eps=1e-10):
        """
        Compute soft assignments using the Student's t-distribution:
          q_{ij} = (1 + ||z_i - μ_j||²/alpha)^(- (alpha+1)/2) / (normalization)
        """
        B, latent_dim = z.size()
        K, _ = centers.size()

        # Expand dimensions for broadcasting
        z_expand = z.unsqueeze(1).expand(B, K, latent_dim)
        centers_expand = centers.unsqueeze(0).expand(B, K, latent_dim)
        dist_sq = torch.sum((z_expand - centers_expand) ** 2, dim=2)
        numerator = (1.0 + dist_sq / alpha).pow(- (alpha + 1.0) / 2)
        denominator = torch.sum(numerator, dim=1, keepdim=True) + eps
        q = numerator / denominator
        return q

    @staticmethod
    def target_distribution(q, eps=1e-10, beta=0.1):
        """
        Compute the refined target distribution:
          1. Compute the standard DEC target: p' = (q^2 / f) normalized over clusters (with f_j = sum_i q_{ij}).
          2. Mix with a uniform distribution: p = (1-beta)*p' + beta*(1/n_clusters).
        """
        weight = q ** 2 / (torch.sum(q, dim=0, keepdim=True) + eps)
        p_std = weight / torch.sum(weight, dim=1, keepdim=True)
        uniform = torch.ones_like(p_std) / p_std.size(1)
        p = (1 - beta) * p_std + beta * uniform
        return p


def center_separation_loss(cluster_centers, eps=1e-10):
    """
    Computes a loss that encourages cluster centers to be far apart.
    It calculates the average pairwise Euclidean distance and returns
    its negative (since we minimize loss, but want to maximize distance).
    """
    n = cluster_centers.size(0)
    if n <= 1:
        return torch.tensor(0.0, device=cluster_centers.device)

    # Compute pairwise distances between cluster centers
    centers_expanded = cluster_centers.unsqueeze(0).expand(n, n, -1)
    centers_transposed = cluster_centers.unsqueeze(1).expand(n, n, -1)
    distances = torch.sqrt(torch.sum((centers_expanded - centers_transposed) ** 2, dim=2) + eps)

    # Exclude the diagonal (self-distance)
    mask = 1 - torch.eye(n, device=cluster_centers.device)
    avg_distance = torch.sum(distances * mask) / (mask.sum() + eps)

    # Loss is negative average distance (we want to maximize the distance)
    loss = -avg_distance
    return loss


def train_idec_full_pass(model, train_loader, val_loader=None, epochs=10,
                         device='cuda', lambda_recon=0.05, lambda_bal=0.3, lambda_entropy=0.1,lambda_sep = 0.01,lambda_kl = 1.5):
    """
    Training loop for IDEC with an additional entropy penalty regularizer.
      - Computes the soft assignments q over the entire training set.
      - Computes the refined target distribution p.
      - Trains the model by combining:
           * KL divergence loss (between log q and p)
           * Reconstruction loss (MSE between x and its reconstruction)
           * A balancing loss to encourage uniform cluster frequencies.
           * An entropy penalty to explicitly discourage cluster collapse.
    """
    model.to(device)
    optimizer = optim.Adam([
        {'params': model.encoder.parameters(), 'lr': 1e-4},
        {'params': model.decoder.parameters(), 'lr': 1e-4},
        {'params': [model.cluster_centers], 'lr': 1e-3}
    ])
    loss_fn_kl = nn.KLDivLoss(reduction='batchmean')
    train_data_size = len(train_loader.dataset)
    model.target_beta = 0.3  # Increase beta to reduce over-sharpening

    for epoch in range(1, epochs + 1):
        # Step 1: Compute q for the entire dataset in evaluation mode
        model.eval()
        all_q = []
        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(device)
                _, _, q = model(batch)
                all_q.append(q.cpu())
        all_q = torch.cat(all_q, dim=0)

        # Step 2: Compute refined target distribution p using beta to mix with uniform
        p = model.target_distribution(all_q, beta=model.target_beta)

        # Step 3: Train in mini-batches
        model.train()
        idx_offset = 0
        running_loss = 0.0
        for batch in train_loader:
            b_size = batch.size(0)
            batch = batch.to(device)
            p_batch = p[idx_offset: idx_offset + b_size, :].to(device)
            idx_offset += b_size

            x_recon, z, q_batch = model(batch)
            log_q = torch.log(q_batch + 1e-10)
            kl_loss = loss_fn_kl(log_q, p_batch)

            # Reconstruction loss (MSE)
            recon_loss = F.mse_loss(x_recon, batch)

            # Balancing loss: KL divergence from uniform to the average cluster frequency f
            f = torch.mean(q_batch, dim=0)
            uniform = torch.ones_like(f) / f.numel()
            balance_loss = torch.sum(uniform * (torch.log(uniform + 1e-10) - torch.log(f + 1e-10)))

            # Entropy penalty: encourage high entropy in f (i.e. a uniform distribution)
            entropy = -torch.sum(f * torch.log(f + 1e-10))

              # Adjust this hyperparameter as needed
            sep_loss = center_separation_loss(model.cluster_centers)

            # Combine losses: subtract the entropy penalty to encourage higher entropy (more balanced clusters)
            total_loss = (lambda_kl * kl_loss +
                          lambda_recon * recon_loss +
                          lambda_bal * balance_loss -
                          lambda_entropy * entropy +
                          lambda_sep * sep_loss)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item() * b_size

        epoch_loss = running_loss / train_data_size
        print(f"[IDEC] Epoch {epoch}/{epochs}, Total Loss: {epoch_loss:.4f}, "
              f"KL: {kl_loss.item():.4f}, Recon: {recon_loss.item():.4f}, "
              f"Balance: {balance_loss.item():.4f}, Entropy: {entropy.item():.4f}")

        # Optional: Evaluate on validation data (e.g., using silhouette score)
        if val_loader is not None and epoch % 10 == 0:
            model.eval()
            z_vals = []
            labels = []
            with torch.no_grad():
                for vbatch in val_loader:
                    vbatch = vbatch.to(device)
                    _, vz, vq = model(vbatch)
                    cluster_idx = torch.argmax(vq, dim=1)
                    z_vals.append(vz.cpu().numpy())
                    labels.extend(cluster_idx.cpu().numpy())
            z_vals = np.concatenate(z_vals, axis=0)
            if len(np.unique(labels)) > 1:
                from sklearn.metrics import silhouette_score
                sscore = silhouette_score(z_vals, labels)
                print(f"   [Val] Silhouette Score: {sscore:.4f}")
            else:
                print("   [Val] Only one cluster => silhouette not defined.")

    return model


def train_idec_uniform_clusters(model, train_loader, val_loader=None, epochs=20,
                                device='cuda', initial_lambdas=None):
    """
    Training loop for IDEC with a specific focus on:
    1. Achieving uniform distribution of cluster assignments
    2. Creating well-separated clusters
    3. Maintaining good reconstruction quality

    This version fixes the negative/positive loss discrepancy and provides
    better metrics for monitoring cluster quality.
    """
    model.to(device)

    # Initialize lambdas with defaults if not provided
    if initial_lambdas is None:
        lambdas = {
            'lambda_kl': 1.5,
            'lambda_recon': 0.05,
            'lambda_bal': 0.3,
            'lambda_entropy': 0.1,
            'lambda_sep': 0.01
        }
    else:
        lambdas = initial_lambdas.copy()

    optimizer = optim.Adam([
        {'params': model.encoder.parameters(), 'lr': 1e-4},
        {'params': model.decoder.parameters(), 'lr': 1e-4},
        {'params': [model.cluster_centers], 'lr': 1e-3}
    ])

    loss_fn_kl = nn.KLDivLoss(reduction='batchmean')
    train_data_size = len(train_loader.dataset)

    # Track uniformity and separation metrics
    uniformity_history = []
    separation_history = []
    silhouette_history = []

    # For adaptive mixing factor based on uniformity
    model.target_beta = 0.3

    for epoch in range(1, epochs + 1):
        # Step 1: Compute q for the entire dataset in evaluation mode
        model.eval()
        all_q = []
        all_z = []
        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(device)
                _, z, q = model(batch)
                all_q.append(q.cpu())
                all_z.append(z.cpu())
        all_q = torch.cat(all_q, dim=0)  # (N, n_clusters)
        all_z = torch.cat(all_z, dim=0)  # (N, latent_dim)

        # Compute cluster assignments
        cluster_assignments = torch.argmax(all_q, dim=1).numpy()

        # Compute uniformity metric (how evenly distributed are the clusters)
        cluster_counts = np.bincount(cluster_assignments, minlength=model.n_clusters)
        cluster_freqs = cluster_counts / len(cluster_assignments)
        uniformity = -np.sum(cluster_freqs * np.log(cluster_freqs + 1e-10))  # Entropy of distribution
        uniformity_ratio = uniformity / np.log(model.n_clusters)  # Normalized to [0,1]
        uniformity_history.append(uniformity_ratio)

        # Compute average separation between clusters (simple metric)
        centers = model.cluster_centers.detach().cpu().numpy()
        center_dists = []
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                center_dists.append(dist)
        avg_separation = np.mean(center_dists) if center_dists else 0
        separation_history.append(avg_separation)

        # Step 2: Compute refined target distribution p with adaptive beta
        # Increase beta if uniformity is low to encourage more balanced clusters
        if epoch > 1 and uniformity_ratio < 0.7:
            model.target_beta = min(0.6, model.target_beta + 0.05)
            print(f"[IDEC] Adjusting target_beta to {model.target_beta:.2f} to improve uniformity")
        elif epoch > 1 and uniformity_ratio > 0.85:
            model.target_beta = max(0.1, model.target_beta - 0.05)
            print(f"[IDEC] Reducing target_beta to {model.target_beta:.2f} (uniformity is good)")

        p = model.target_distribution(all_q, beta=model.target_beta)

        # Step 3: Train in mini-batches
        model.train()
        idx_offset = 0
        epoch_losses = {
            'recon': 0.0,
            'kl': 0.0,
            'balance': 0.0,
            'entropy': 0.0,
            'sep': 0.0,
            'total_with_entropy': 0.0,  # Including negative entropy term
            'total_positive': 0.0  # Excluding negative entropy term
        }

        for batch in train_loader:
            b_size = batch.size(0)
            batch = batch.to(device)
            p_batch = p[idx_offset: idx_offset + b_size, :].to(device)
            idx_offset += b_size

            x_recon, z, q_batch = model(batch)
            log_q = torch.log(q_batch + 1e-10)

            # Individual loss components
            kl_loss = loss_fn_kl(log_q, p_batch)
            recon_loss = F.mse_loss(x_recon, batch)

            # Balancing loss
            f = torch.mean(q_batch, dim=0)  # Average cluster assignment frequency
            uniform = torch.ones_like(f) / f.numel()
            balance_loss = torch.sum(uniform * (torch.log(uniform + 1e-10) - torch.log(f + 1e-10)))

            # Entropy penalty (to maximize entropy of assignments)
            entropy = -torch.sum(f * torch.log(f + 1e-10))

            # Separation loss (to encourage well-separated clusters)
            sep_loss = center_separation_loss(model.cluster_centers)

            # IMPORTANT: Two separate loss calculations for reporting vs optimization
            # Loss for optimization (including negative entropy term)
            total_loss = (lambdas['lambda_kl'] * kl_loss +
                          lambdas['lambda_recon'] * recon_loss +
                          lambdas['lambda_bal'] * balance_loss -
                          lambdas['lambda_entropy'] * entropy +
                          lambdas['lambda_sep'] * sep_loss)

            # Positive-only loss for reporting (excluding negative entropy term)
            positive_loss = (lambdas['lambda_kl'] * kl_loss +
                             lambdas['lambda_recon'] * recon_loss +
                             lambdas['lambda_bal'] * balance_loss +
                             lambdas['lambda_sep'] * sep_loss)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Accumulate losses
            epoch_losses['kl'] += kl_loss.item() * b_size
            epoch_losses['recon'] += recon_loss.item() * b_size
            epoch_losses['balance'] += balance_loss.item() * b_size
            epoch_losses['entropy'] += entropy.item() * b_size
            epoch_losses['sep'] += sep_loss.item() * b_size
            epoch_losses['total_with_entropy'] += total_loss.item() * b_size
            epoch_losses['total_positive'] += positive_loss.item() * b_size

        # Calculate average losses
        for k in epoch_losses:
            epoch_losses[k] /= train_data_size

        # Adapt hyperparameters based on uniformity and separation goals
        if epoch > 1:
            # If uniformity is low, increase balance and entropy lambdas
            if uniformity_ratio < 0.6:
                lambdas['lambda_bal'] = min(1.0, lambdas['lambda_bal'] * 1.2)
                lambdas['lambda_entropy'] = min(0.3, lambdas['lambda_entropy'] * 1.2)
                print(f"[IDEC] Increasing balance and entropy lambdas to improve uniformity")

            # If separation is low, increase separation lambda
            if avg_separation < 0.5 and separation_history[-2] >= avg_separation:
                lambdas['lambda_sep'] = min(0.05, lambdas['lambda_sep'] * 1.3)
                print(f"[IDEC] Increasing separation lambda to improve cluster separation")

            # If KL loss is very small, slightly reduce its importance
            if epoch_losses['kl'] < 0.01 and lambdas['lambda_kl'] > 0.5:
                lambdas['lambda_kl'] = max(0.5, lambdas['lambda_kl'] * 0.9)
                print(f"[IDEC] Reducing KL lambda to balance with other objectives")

            # If reconstruction is degrading, increase its importance
            if epoch > 2 and epoch_losses['recon'] > 1.05 * epoch_losses['recon']:
                lambdas['lambda_recon'] = min(0.2, lambdas['lambda_recon'] * 1.2)
                print(f"[IDEC] Increasing reconstruction lambda to maintain quality")

        # Print consistent training metrics
        print(f"[IDEC] Epoch {epoch}/{epochs}, "
              f"Positive Loss: {epoch_losses['total_positive']:.4f}, "
              f"Optimization Loss: {epoch_losses['total_with_entropy']:.4f}, "
              f"Uniformity: {uniformity_ratio:.4f}, "
              f"Avg Separation: {avg_separation:.4f}")
        print(f"      KL: {epoch_losses['kl']:.4f}, Recon: {epoch_losses['recon']:.4f}, "
              f"Balance: {epoch_losses['balance']:.4f}, Entropy: {epoch_losses['entropy']:.4f}")

        # Evaluate on validation set if provided
        if val_loader is not None:
            model.eval()
            val_recon_loss = 0.0
            val_z = []
            val_labels = []

            with torch.no_grad():
                for vbatch in val_loader:
                    vbatch = vbatch.to(device)
                    vx_recon, vz, vq = model(vbatch)

                    # Compute validation reconstruction loss
                    vrecon_loss = F.mse_loss(vx_recon, vbatch)
                    val_recon_loss += vrecon_loss.item() * vbatch.size(0)

                    # Get cluster assignments
                    cluster_idx = torch.argmax(vq, dim=1).cpu().numpy()
                    val_z.append(vz.cpu().numpy())
                    val_labels.extend(cluster_idx)

            val_recon_loss /= len(val_loader.dataset)
            val_z = np.concatenate(val_z, axis=0)
            val_labels = np.array(val_labels)

            # Compute silhouette score if we have multiple clusters
            unique_labels = np.unique(val_labels)
            silhouette = None
            if len(unique_labels) > 1:
                try:
                    from sklearn.metrics import silhouette_score
                    silhouette = silhouette_score(val_z, val_labels)
                    silhouette_history.append(silhouette)
                    print(f"   [Val] Recon Loss: {val_recon_loss:.4f}, Silhouette: {silhouette:.4f}")
                except Exception as e:
                    print(f"   [Val] Recon Loss: {val_recon_loss:.4f}, Silhouette error: {e}")
            else:
                print(f"   [Val] Recon Loss: {val_recon_loss:.4f}, Only one cluster detected")

            # Compute and report validation uniformity
            val_counts = np.bincount(val_labels, minlength=model.n_clusters)
            val_freqs = val_counts / len(val_labels)
            val_uniformity = -np.sum(val_freqs * np.log(val_freqs + 1e-10))
            val_uniformity_ratio = val_uniformity / np.log(model.n_clusters)
            print(f"   [Val] Uniformity: {val_uniformity_ratio:.4f}")

            # Generate visual report on cluster distribution every 5 epochs
            if epoch % 5 == 0 or epoch == epochs:
                # Create directory if needed
                os.makedirs("QA/DEC", exist_ok=True)

                # Plot cluster distribution
                plt.figure(figsize=(10, 6))
                non_zero_indices = np.where(val_counts > 0)[0]
                plt.bar(non_zero_indices, val_counts[non_zero_indices])
                plt.xlabel('Cluster ID')
                plt.ylabel('Number of samples')
                plt.title(f'Cluster Distribution (Epoch {epoch}) - Uniformity: {val_uniformity_ratio:.4f}')
                plt.savefig(f"QA/DEC/cluster_dist_epoch_{epoch}.png")
                plt.close()

                # If we have a 2D latent space, visualize the clusters
                if model.encoder.latent_dim == 2:
                    plt.figure(figsize=(10, 8))
                    plt.scatter(val_z[:, 0], val_z[:, 1], c=val_labels, cmap='tab20', alpha=0.6)
                    plt.colorbar(label='Cluster ID')
                    plt.title(f'2D Latent Space Visualization (Epoch {epoch})')
                    plt.savefig(f"QA/DEC/latent_vis_epoch_{epoch}.png")
                    plt.close()

    # Save final model
    torch.save(model.state_dict(), "QA/DEC/idec_final_model.pt")

    # Plot training progress
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(uniformity_history)
    plt.xlabel('Epoch')
    plt.ylabel('Uniformity Ratio')
    plt.title('Cluster Uniformity')

    plt.subplot(1, 3, 2)
    plt.plot(separation_history)
    plt.xlabel('Epoch')
    plt.ylabel('Average Separation')
    plt.title('Cluster Separation')

    if silhouette_history:
        plt.subplot(1, 3, 3)
        plt.plot(silhouette_history)
        plt.xlabel('Epoch')
        plt.ylabel('Silhouette Score')
        plt.title('Clustering Quality')

    plt.tight_layout()
    plt.savefig("QA/DEC/training_progress.png")
    plt.close()

    return model
def adapt_hyperparameters(lambdas, loss_history, epoch):
    """
    Adapt hyperparameters based on loss trends

    Strategy:
    1. If KL loss is very small, reduce its lambda to prevent other losses from being ignored
    2. If reconstruction loss is growing, increase its lambda
    3. If entropy is decreasing, increase its lambda to prevent mode collapse
    4. If balance loss is high, increase its lambda
    5. If separation loss is not improving, increase its lambda
    """
    new_lambdas = lambdas.copy()

    # Get current and previous losses
    curr_losses = {k: loss_history[k][-1] for k in loss_history}

    # Only adapt if we have enough history
    if epoch >= 3:
        prev_losses = {k: loss_history[k][-2] for k in loss_history}
        trend_losses = {k: loss_history[k][-3:] for k in loss_history}

        # 1. KL-divergence lambda adaptation
        if curr_losses['kl'] < 0.05:
            # KL loss is very small, reduce its impact slightly
            new_lambdas['lambda_kl'] = max(0.5, lambdas['lambda_kl'] * 0.9)
        elif curr_losses['kl'] > 0.5 and lambdas['lambda_kl'] < 2.0:
            # KL loss is significant, might need more weight
            new_lambdas['lambda_kl'] = min(2.0, lambdas['lambda_kl'] * 1.1)

        # 2. Reconstruction loss adaptation
        recon_trend = trend_losses['recon']
        if recon_trend[-1] > recon_trend[0] and recon_trend[-1] > 0.2:
            # Reconstruction loss is increasing, give it more weight
            new_lambdas['lambda_recon'] = min(0.2, lambdas['lambda_recon'] * 1.2)
        elif recon_trend[-1] < 0.1 and lambdas['lambda_recon'] > 0.02:
            # Reconstruction is good, can reduce weight
            new_lambdas['lambda_recon'] = max(0.01, lambdas['lambda_recon'] * 0.9)

        # 3. Entropy adaptation (prevent mode collapse)
        entropy_trend = trend_losses['entropy']
        if entropy_trend[-1] < entropy_trend[0] * 0.9:
            # Entropy decreasing significantly, increase its importance
            new_lambdas['lambda_entropy'] = min(0.5, lambdas['lambda_entropy'] * 1.5)

        # 4. Balance loss adaptation
        if curr_losses['balance'] > 0.1:
            # High imbalance, increase its weight
            new_lambdas['lambda_bal'] = min(1.0, lambdas['lambda_bal'] * 1.2)
        elif curr_losses['balance'] < 0.01:
            # Good balance, can reduce weight
            new_lambdas['lambda_bal'] = max(0.05, lambdas['lambda_bal'] * 0.9)

        # 5. Separation loss adaptation
        sep_trend = trend_losses['sep']
        if sep_trend[-1] >= sep_trend[0] and lambdas['lambda_sep'] < 0.05:
            # Separation is not improving, increase weight
            new_lambdas['lambda_sep'] = min(0.05, lambdas['lambda_sep'] * 1.2)

    return new_lambdas

def train_idec_with_simpler_adapt(model, train_loader, val_loader, epochs=50, device='cuda'):
    # 1) Initialize your lambda dictionary and beta
    from copy import deepcopy
    scheduler = SimpleAdaptiveScheduler(
        lambda_kl=1.5,
        lambda_recon=0.05,
        lambda_bal=0.3,
        lambda_entropy=0.1,
        lambda_sep=0.01,
        max_beta=0.3
    )
    lambdas = deepcopy(scheduler.lambdas)  # or just refer to scheduler.lambdas
    target_beta = 0.1

    # 2) Setup your optimizer, etc.
    optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters(), 'lr': 1e-4},
        {'params': model.decoder.parameters(), 'lr': 1e-4},
        {'params': [model.cluster_centers], 'lr': 1e-3}
    ])
    loss_fn_kl = nn.KLDivLoss(reduction='batchmean')
    train_data_size = len(train_loader.dataset)

    for epoch in range(1, epochs + 1):
        # 2a) Compute q over entire dataset
        model.eval()
        all_q = []
        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(device)
                _, z, q = model(batch)
                all_q.append(q.cpu())
        all_q = torch.cat(all_q, dim=0)

        # 2b) Compute uniformity, cluster assignments, etc.
        cluster_assignments = torch.argmax(all_q, dim=1).numpy()
        cluster_counts = np.bincount(cluster_assignments, minlength=model.n_clusters)
        freqs = cluster_counts / len(cluster_assignments)
        uniformity = -np.sum(freqs * np.log(freqs + 1e-10))
        uniformity_ratio = uniformity / np.log(model.n_clusters)  # normalized

        # 2c) Compute refined target distribution p
        p = model.target_distribution(all_q, beta=target_beta)

        # 3) Train in mini-batches
        model.train()
        idx_offset = 0

        epoch_losses = {
            'kl': 0.0,
            'recon': 0.0,
            'balance': 0.0,
            'entropy': 0.0,
            'sep': 0.0,
            'total_with_entropy': 0.0
        }

        for batch in train_loader:
            b_size = batch.size(0)
            batch = batch.to(device)
            p_batch = p[idx_offset: idx_offset + b_size, :].to(device)
            idx_offset += b_size

            x_recon, z, q_batch = model(batch)
            log_q = torch.log(q_batch + 1e-10)

            kl_loss = loss_fn_kl(log_q, p_batch)
            recon_loss = F.mse_loss(x_recon, batch)

            # Balanced uniform
            f = torch.mean(q_batch, dim=0)
            uniform = torch.ones_like(f) / f.numel()
            balance_loss = torch.sum(uniform * (torch.log(uniform + 1e-10) - torch.log(f + 1e-10)))

            # Negative Entropy
            entropy = -torch.sum(f * torch.log(f + 1e-10))

            # Separation
            sep_loss = center_separation_loss(model.cluster_centers)

            # Weighted sum
            total_loss = (lambdas['lambda_kl']  * kl_loss   +
                          lambdas['lambda_recon'] * recon_loss +
                          lambdas['lambda_bal']  * balance_loss -
                          lambdas['lambda_entropy'] * entropy +
                          lambdas['lambda_sep']  * sep_loss)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Accumulate
            epoch_losses['kl'] += kl_loss.item() * b_size
            epoch_losses['recon'] += recon_loss.item() * b_size
            epoch_losses['balance'] += balance_loss.item() * b_size
            epoch_losses['entropy'] += entropy.item() * b_size
            epoch_losses['sep'] += sep_loss.item() * b_size
            epoch_losses['total_with_entropy'] += total_loss.item() * b_size

        # Average over dataset
        for k in epoch_losses:
            epoch_losses[k] /= train_data_size

        # 4) Print or log
        print(f"Epoch {epoch}/{epochs}, KL: {epoch_losses['kl']:.4f}, Recon: {epoch_losses['recon']:.4f}, "
              f"Bal: {epoch_losses['balance']:.4f}, Ent: {epoch_losses['entropy']:.4f}, "
              f"Sep: {epoch_losses['sep']:.4f}, Uniformity: {uniformity_ratio:.4f}, "
              f"Beta: {target_beta:.2f}")

        # 5) After epoch, pass metrics to your adaptive scheduler
        updated_lambdas, updated_beta = scheduler.update(
            epoch=epoch,
            losses={
                'kl': epoch_losses['kl'],
                'recon': epoch_losses['recon'],
                'balance': epoch_losses['balance'],
                'entropy': epoch_losses['entropy'],
                'sep': epoch_losses['sep']
            },
            uniformity_ratio=uniformity_ratio,
            target_beta=target_beta
        )

        # Update local lambdas + beta
        lambdas = updated_lambdas
        target_beta = updated_beta

        # Optionally do validation checks here

    # End of training
    return model


# =========================
#  6) Example Main Script
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="training_data/coeffs/")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=2048)
    parser.add_argument("--n_clusters", type=int, default=100)
    parser.add_argument("--epochs_cae", type=int, default=200)
    parser.add_argument("--epochs_dec", type=int, default=100)
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
    # cae_model = CAE(input_shape, args.latent_dim)
    # cae_model = pretrain_cae(cae_model, train_loader, val_loader=val_loader,
    #                          epochs=args.epochs_cae, lr=3e-4, device=args.device)
    #
    # # ---- QA: Check reconstructions from validation set ----
    # plot_ae_reconstructions(cae_model, val_loader, device=args.device,
    #                         n=8, out_path='QA/DEC/ae_recons.png')
    # # After CAE training is done
    # torch.save(cae_model.state_dict(), "QA/DEC/cae_model.pt")

    # Create CAE instance and load pretrained weights
    cae_model = CAE(input_shape, args.latent_dim)
    cae_model.load_state_dict(torch.load("QA/DEC/cae_model.pt"))
    cae_model.to(args.device)

    # Extract encoder and decoder
    encoder = cae_model.encoder
    decoder = cae_model.decoder

    # 3) DEC Setup
    idec_model = IDEC(encoder=cae_model.encoder, decoder=cae_model.decoder,
                      n_clusters=args.n_clusters, alpha=1.0, target_beta=0.1)    # Initialize cluster centers
    print("[MAIN] Initializing DEC cluster centers...")
    idec_model.initialize_centers(train_loader, device=args.device)

    # 4) DEC Training (two-pass each epoch)
    print("[MAIN] Starting DEC training/fine-tuning...")

    # Initial lambda values
    # initial_lambdas = {
    #     'lambda_kl': 1.5,
    #     'lambda_recon': 0.05,
    #     'lambda_bal': 0.1,
    #     'lambda_entropy': 0.001,
    #     'lambda_sep': 0.001
    # }

    # idec_model = train_idec_uniform_clusters(
    #     idec_model,
    #     train_loader,
    #     val_loader=val_loader,
    #     epochs=args.epochs_dec,
    #     device=args.device,
    #     initial_lambdas=initial_lambdas,
    # )

    idec_model = train_idec_with_simpler_adapt(idec_model,
                                               train_loader,
                                               val_loader,
                                               epochs=args.epochs_dec,
                                               device=args.device)
    # 5) Evaluate final cluster distribution on validation
    idec_model.eval()
    all_val_labels = []
    with torch.no_grad():
        for vbatch in val_loader:
            vb = vbatch.to(args.device)
            x_recon, z, vq = idec_model(vb)
            labels = torch.argmax(vq, dim=1).cpu().numpy()
            all_val_labels.extend(labels)
    all_val_labels = np.array(all_val_labels)

    # Plot cluster distribution
    plot_cluster_distribution(all_val_labels, out_path='QA/DEC/cluster_hist.png')

    # Plot images from the most frequent cluster
    plot_most_frequent_cluster_images(idec_model, val_ds, device=args.device,
                                      top_n=8, out_path='QA/DEC/most_freq_cluster.png')

    print("[MAIN] Done. QA plots saved in QA/DEC/")

    # After DEC training is done
    torch.save(idec_model.state_dict(), "QA/DEC/dec_model.pt")
    # ---- QA: Check reconstructions from validation set ----
    plot_ae_reconstructions(idec_model, val_loader, device=args.device,
                            n=8, out_path='QA/DEC/ae_recons_final_IDEC.png')
