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
        {'params': [dec_model.cluster_centers], 'lr': 1e-5}  # Cluster centers trained with lr=1e-4
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

        if epoch == 100:
            optimizer.param_groups[0]['lr'] = 1e-6
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


def train_idec_full_pass_adaptive(model, train_loader, val_loader=None, epochs=10,
                                  device='cuda', initial_lambdas=None, adaptation_frequency=1):
    """
    Training loop for IDEC with adaptive hyperparameters that adjust based on loss values.

    Args:
        model: The IDEC model
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation data
        epochs: Number of training epochs
        device: Device to run training on
        initial_lambdas: Dictionary with initial lambda values
        adaptation_frequency: How often to adapt parameters (in epochs)
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

    # Store history of losses for adaptation
    loss_history = {
        'kl': [],
        'recon': [],
        'balance': [],
        'entropy': [],
        'sep': [],
        'total': []
    }

    optimizer = optim.Adam([
        {'params': model.encoder.parameters(), 'lr': 1e-4},
        {'params': model.decoder.parameters(), 'lr': 1e-4},
        {'params': [model.cluster_centers], 'lr': 1e-3}
    ])

    loss_fn_kl = nn.KLDivLoss(reduction='batchmean')
    train_data_size = len(train_loader.dataset)
    model.target_beta = 0.3  # Mixing factor for target distribution

    # For early stopping
    best_loss = float('inf')
    patience = 50
    patience_counter = 0

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

        # Step 2: Compute refined target distribution p
        p = model.target_distribution(all_q, beta=model.target_beta)

        # Step 3: Train in mini-batches
        model.train()
        idx_offset = 0
        running_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_balance_loss = 0.0
        epoch_entropy = 0.0
        epoch_sep_loss = 0.0

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
            f = torch.mean(q_batch, dim=0)
            uniform = torch.ones_like(f) / f.numel()
            balance_loss = torch.sum(uniform * (torch.log(uniform + 1e-10) - torch.log(f + 1e-10)))

            # Entropy penalty
            entropy = -torch.sum(f * torch.log(f + 1e-10))

            # Separation loss
            sep_loss = center_separation_loss(model.cluster_centers)

            # Combine losses with current lambdas
            total_loss = (lambdas['lambda_kl'] * kl_loss +
                          lambdas['lambda_recon'] * recon_loss +
                          lambdas['lambda_bal'] * balance_loss -
                          lambdas['lambda_entropy'] * entropy +
                          lambdas['lambda_sep'] * sep_loss)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Accumulate losses for epoch statistics
            running_loss += total_loss.item() * b_size
            epoch_kl_loss += kl_loss.item() * b_size
            epoch_recon_loss += recon_loss.item() * b_size
            epoch_balance_loss += balance_loss.item() * b_size
            epoch_entropy += entropy.item() * b_size
            epoch_sep_loss += sep_loss.item() * b_size

        # Calculate epoch average losses
        epoch_loss = running_loss / train_data_size
        epoch_kl_loss /= train_data_size
        epoch_recon_loss /= train_data_size
        epoch_balance_loss /= train_data_size
        epoch_entropy /= train_data_size
        epoch_sep_loss /= train_data_size

        # Store losses for adaptation
        loss_history['total'].append(epoch_loss)
        loss_history['kl'].append(epoch_kl_loss)
        loss_history['recon'].append(epoch_recon_loss)
        loss_history['balance'].append(epoch_balance_loss)
        loss_history['entropy'].append(epoch_entropy)
        loss_history['sep'].append(epoch_sep_loss)

        # Adapt hyperparameters every adaptation_frequency epochs
        if epoch % adaptation_frequency == 0 and epoch > 1:
            lambdas = adapt_hyperparameters(lambdas, loss_history, epoch)
            print(f"[IDEC] Adapted lambdas: {lambdas}")

            # Also adjust target_beta based on entropy trends
            if epoch > 3 and loss_history['entropy'][-1] < loss_history['entropy'][-3]:
                # Entropy decreasing - increase beta to encourage more uniform distribution
                model.target_beta = min(0.6, model.target_beta + 0.05)
                print(f"[IDEC] Increased target_beta to {model.target_beta}")

        print(f"[IDEC] Epoch {epoch}/{epochs}, Total Loss: {epoch_loss:.4f}, "
              f"KL: {epoch_kl_loss:.4f}, Recon: {epoch_recon_loss:.4f}, "
              f"Balance: {epoch_balance_loss:.4f}, Entropy: {epoch_entropy:.4f}")

        # Optional: Evaluate on validation data
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            silhouette = None

            with torch.no_grad():
                z_vals = []
                labels = []
                for vbatch in val_loader:
                    vbatch = vbatch.to(device)
                    vx_recon, vz, vq = model(vbatch)
                    cluster_idx = torch.argmax(vq, dim=1)
                    z_vals.append(vz.cpu().numpy())
                    labels.extend(cluster_idx.cpu().numpy())

                    # Compute validation loss components (same as training)
                    vlog_q = torch.log(vq + 1e-10)
                    vkl_loss = torch.zeros(1, device=device)  # Can't compute KL without target
                    vrecon_loss = F.mse_loss(vx_recon, vbatch)

                    # Simple validation loss (just reconstruction for monitoring)
                    val_loss += vrecon_loss.item() * vbatch.size(0)

                val_loss /= len(val_loader.dataset)

                # Compute silhouette if we have enough samples and unique clusters
                z_vals = np.concatenate(z_vals, axis=0)
                unique_labels = np.unique(labels)
                if len(unique_labels) > 1:
                    try:
                        silhouette = silhouette_score(z_vals, labels)
                        print(f"   [Val] Loss: {val_loss:.4f}, Silhouette: {silhouette:.4f}")
                    except:
                        print(f"   [Val] Loss: {val_loss:.4f}, Silhouette: Error")
                else:
                    print(f"   [Val] Loss: {val_loss:.4f}, Only one cluster (silhouette not defined)")

                # Early stopping based on validation loss
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), "QA/DEC/idec_model_best.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"[IDEC] Early stopping at epoch {epoch} due to no improvement in validation loss.")
                        # Load best model
                        model.load_state_dict(torch.load("QA/DEC/idec_model_best.pt"))
                        return model

    return model


def train_idec_uniform_clusters(model, train_loader, val_loader=None, epochs=20,
                    device='cuda', initial_lambdas=None):
    """
    Fixed IDEC training with properly calculated loss values and enhanced
    cluster uniformity strategies.
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

    # Start with moderate target_beta
    model.target_beta = 0.3

    # For cluster assignment tracking
    previous_assignments = None

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
        all_q = torch.cat(all_q, dim=0)
        all_z = torch.cat(all_z, dim=0)

        # Get cluster assignments
        cluster_assignments = torch.argmax(all_q, dim=1).numpy()

        # Compute uniformity metrics (two different ways)
        cluster_counts = np.bincount(cluster_assignments, minlength=model.n_clusters)
        active_clusters = np.sum(cluster_counts > 0)

        # Normalized entropy (0-1 scale)
        cluster_freqs = cluster_counts / len(cluster_assignments)
        uniformity_entropy = -np.sum(cluster_freqs * np.log(cluster_freqs + 1e-10))
        max_entropy = np.log(model.n_clusters)
        uniformity_ratio = uniformity_entropy / max_entropy

        # Alternative uniformity metric: coefficient of variation (lower is more uniform)
        non_empty_counts = cluster_counts[cluster_counts > 0]
        if len(non_empty_counts) > 1:
            cv = non_empty_counts.std() / non_empty_counts.mean()
            uniformity_cv = 1.0 / (1.0 + cv)  # Transform to 0-1 scale, higher is better
        else:
            uniformity_cv = 0.0

        # Track stability (how many assignments changed since last epoch)
        stability = 1.0
        if previous_assignments is not None:
            changes = np.sum(previous_assignments != cluster_assignments)
            stability = 1.0 - (changes / len(cluster_assignments))
        previous_assignments = cluster_assignments.copy()

        # Compute cluster separation (normalized by latent dimension)
        centers = model.cluster_centers.detach().cpu().numpy()
        latent_dim = centers.shape[1]
        center_dists = []
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                # Normalize by sqrt(dimension) to make it more interpretable
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2)) / np.sqrt(latent_dim)
                center_dists.append(dist)
        avg_separation = np.mean(center_dists) if center_dists else 0

        # Step 2: Compute target distribution with adaptive beta
        # Only adjust beta if we have imbalanced clusters
        if uniformity_ratio < 0.7:
            # More aggressive beta adjustment based on uniformity
            model.target_beta = min(0.8, model.target_beta + (0.7 - uniformity_ratio) * 0.2)
            print(f"[IDEC] Target_beta -> {model.target_beta:.2f} to improve uniformity ({uniformity_ratio:.3f})")

        p = model.target_distribution(all_q, beta=model.target_beta)

        # Compute a balanced p-target to more strongly encourage uniformity
        # This creates a target that pushes toward both the standard DEC target AND uniform assignment
        if uniformity_ratio < 0.6:
            # Create a more aggressive uniformity-enhancing target
            # Start with standard p, then blend with uniform target for low-count clusters
            uniform_target = torch.ones((len(p), model.n_clusters), device=p.device) / model.n_clusters

            # Calculate adaptive mixing weights based on cluster sizes
            # Less frequent clusters get more uniform influence
            norm_counts = torch.from_numpy(cluster_counts).float().to(p.device)
            norm_counts = norm_counts / torch.max(norm_counts)
            # Invert: small clusters get high weights
            weight_adjustment = 1.0 - norm_counts

            # Create a more balanced p that boosts probability for smaller clusters
            p_uniform_mix = p.clone()

            # Apply per-cluster uniformity mixing
            uniformity_strength = 0.5 * (0.6 - uniformity_ratio) / 0.6  # 0-0.5 range based on uniformity
            for c in range(model.n_clusters):
                mix_factor = uniformity_strength * weight_adjustment[c]
                p_uniform_mix[:, c] = (1 - mix_factor) * p[:, c] + mix_factor * uniform_target[:, c]

            # Renormalize
            p_uniform_mix = p_uniform_mix / p_uniform_mix.sum(dim=1, keepdim=True)
            p = p_uniform_mix
            print(f"[IDEC] Applied uniformity-enhancing target distribution")

        # Step 3: Train in mini-batches
        model.train()
        idx_offset = 0

        # Initialize loss accumulators correctly
        loss_components = {
            'kl': 0.0,
            'recon': 0.0,
            'balance': 0.0,
            'entropy': 0.0,
            'sep': 0.0,
            'optimization_loss': 0.0,  # The full loss with negative entropy
            'monitoring_loss': 0.0     # The positive sum without negative entropy
        }

        for batch in train_loader:
            b_size = batch.size(0)
            batch = batch.to(device)
            p_batch = p[idx_offset: idx_offset + b_size, :].to(device)
            idx_offset += b_size

            x_recon, z, q_batch = model(batch)
            log_q = torch.log(q_batch + 1e-10)

            # Calculate individual loss components
            kl_loss = loss_fn_kl(log_q, p_batch)
            recon_loss = F.mse_loss(x_recon, batch)

            # Balancing loss
            f = torch.mean(q_batch, dim=0)
            uniform = torch.ones_like(f) / f.numel()
            balance_loss = F.kl_div(torch.log(f + 1e-10), uniform, reduction='sum')

            # Entropy term (note: higher entropy is better, so we use negative in the loss)
            entropy = -torch.sum(f * torch.log(f + 1e-10))

            # Separation loss
            sep_loss = center_separation_loss(model.cluster_centers)

            # Compute final loss values

            # 1. Optimization Loss: Used for backpropagation (includes negative entropy)
            optimization_loss = (
                lambdas['lambda_kl'] * kl_loss +
                lambdas['lambda_recon'] * recon_loss +
                lambdas['lambda_bal'] * balance_loss -
                lambdas['lambda_entropy'] * entropy +
                lambdas['lambda_sep'] * sep_loss
            )

            # 2. Monitoring Loss: Strictly positive sum for reporting progress
            # Important: we EXCLUDE the negative entropy term
            monitoring_loss = (
                lambdas['lambda_kl'] * kl_loss +
                lambdas['lambda_recon'] * recon_loss +
                lambdas['lambda_bal'] * balance_loss +
                lambdas['lambda_sep'] * sep_loss
            )

            # Optimize based on the optimization loss
            optimizer.zero_grad()
            optimization_loss.backward()
            optimizer.step()

            # Accumulate loss components for reporting
            loss_components['kl'] += kl_loss.item() * b_size
            loss_components['recon'] += recon_loss.item() * b_size
            loss_components['balance'] += balance_loss.item() * b_size
            loss_components['entropy'] += entropy.item() * b_size
            loss_components['sep'] += sep_loss.item() * b_size
            loss_components['optimization_loss'] += optimization_loss.item() * b_size
            loss_components['monitoring_loss'] += monitoring_loss.item() * b_size

        # Calculate average loss components
        for k in loss_components:
            loss_components[k] /= train_data_size

        # Adapt hyperparameters based on loss components and uniformity
        adapt_message = []

        # Adjust KL lambda based on KL loss magnitude
        if loss_components['kl'] < 0.01 and lambdas['lambda_kl'] > 0.5:
            lambdas['lambda_kl'] = max(0.5, lambdas['lambda_kl'] * 0.9)
            adapt_message.append(f"KL lambda -> {lambdas['lambda_kl']:.3f}")

        # Adjust balance lambda based on uniformity
        if uniformity_ratio < 0.5:
            # More aggressive for low uniformity
            increase_factor = 1.0 + max(0.5, (0.5 - uniformity_ratio) * 2.0)
            lambdas['lambda_bal'] = min(2.0, lambdas['lambda_bal'] * increase_factor)
            adapt_message.append(f"balance lambda -> {lambdas['lambda_bal']:.3f}")

        # Adjust entropy lambda based on uniformity trend
        if uniformity_ratio < 0.5:
            lambdas['lambda_entropy'] = min(0.5, lambdas['lambda_entropy'] * 1.2)
            adapt_message.append(f"entropy lambda -> {lambdas['lambda_entropy']:.3f}")

        # Adjust separation lambda if needed
        if avg_separation < 1.0:
            lambdas['lambda_sep'] = min(0.05, lambdas['lambda_sep'] * 1.2)
            adapt_message.append(f"sep lambda -> {lambdas['lambda_sep']:.3f}")

        # Print adaptation message if any parameters changed
        if adapt_message:
            print(f"[IDEC] Adapted: {', '.join(adapt_message)}")

        # Report training metrics in a clear way
        print(f"[IDEC] Epoch {epoch}/{epochs}")
        print(f"  Monitoring Loss: {loss_components['monitoring_loss']:.4f} (positive sum)")
        print(f"  Optimization Loss: {loss_components['optimization_loss']:.4f} (with neg entropy)")
        print(f"  Uniformity: {uniformity_ratio:.4f} (entropy), {uniformity_cv:.4f} (CV)")
        print(f"  Active Clusters: {active_clusters}/{model.n_clusters}, Avg Separation: {avg_separation:.4f}")
        print(f"  Components: KL={loss_components['kl']:.4f}, Recon={loss_components['recon']:.4f}, " +
              f"Balance={loss_components['balance']:.4f}, Entropy={loss_components['entropy']:.4f}")

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

            # Compute validation uniformity
            val_counts = np.bincount(val_labels, minlength=model.n_clusters)
            val_active_clusters = np.sum(val_counts > 0)

            val_freqs = val_counts / len(val_labels)
            val_uniformity_entropy = -np.sum(val_freqs * np.log(val_freqs + 1e-10))
            val_uniformity_ratio = val_uniformity_entropy / max_entropy

            # Compute silhouette score if we have multiple clusters
            silhouette = None
            if val_active_clusters > 1:
                try:
                    from sklearn.metrics import silhouette_score
                    silhouette = silhouette_score(val_z, val_labels)
                    print(f"  [Val] Recon Loss: {val_recon_loss:.4f}, Silhouette: {silhouette:.4f}")
                except Exception as e:
                    print(f"  [Val] Recon Loss: {val_recon_loss:.4f}, Silhouette error: {e}")
            else:
                print(f"  [Val] Recon Loss: {val_recon_loss:.4f}, Only one active cluster")

            print(f"  [Val] Active Clusters: {val_active_clusters}/{model.n_clusters}, " +
                  f"Uniformity: {val_uniformity_ratio:.4f}")

            # Generate cluster distribution visualization every 5 epochs or at the end
            if epoch % 5 == 0 or epoch == epochs:
                try:
                    # Plot cluster distribution
                    plt.figure(figsize=(10, 6))
                    plt.bar(range(model.n_clusters), val_counts)
                    plt.xlabel('Cluster ID')
                    plt.ylabel('Number of samples')
                    plt.title(f'Cluster Distribution (Epoch {epoch})\n' +
                             f'Uniformity: {val_uniformity_ratio:.3f}, Active: {val_active_clusters}/{model.n_clusters}')
                    plt.savefig(f"QA/DEC/cluster_dist_epoch_{epoch}.png")
                    plt.close()
                    print(f"  [QA] Saved cluster distribution to QA/DEC/cluster_dist_epoch_{epoch}.png")
                except Exception as e:
                    print(f"  [QA] Could not save visualization: {e}")

    # Save final model
    try:
        torch.save(model.state_dict(), "QA/DEC/idec_final_model.pt")
        print(f"[IDEC] Saved final model to QA/DEC/idec_final_model.pt")
    except Exception as e:
        print(f"[IDEC] Could not save model: {e}")

    return model


def uniformity_loss(q_batch, target_uniformity=0.8):
    """
    Calculate a loss that directly encourages uniform cluster assignment frequencies.

    Parameters:
        q_batch: Soft cluster assignments from the model
        target_uniformity: Target uniformity ratio (0-1), higher means more uniform

    Returns:
        Loss value (higher when distribution is less uniform)
    """
    # Average assignment frequency for each cluster
    f = torch.mean(q_batch, dim=0)

    # Perfect uniform distribution
    uniform = torch.ones_like(f) / f.numel()

    # KL divergence from f to uniform (how far is f from being uniform)
    kl_div = torch.sum(f * (torch.log(f + 1e-10) - torch.log(uniform + 1e-10)))

    return kl_div
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
    parser.add_argument("--epochs_cae", type=int, default=5)
    parser.add_argument("--epochs_dec", type=int, default=20)
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
    input_shape = sample_data.shape #(C, H, W)
    cae_model = CAE(input_shape, args.latent_dim)
    cae_model = pretrain_cae(cae_model, train_loader, val_loader=val_loader,
                             epochs=args.epochs_cae, lr=3e-4, device=args.device)

    # ---- QA: Check reconstructions from validation set ----
    plot_ae_reconstructions(cae_model, val_loader, device=args.device,
                            n=8, out_path='QA/DEC/ae_recons.png')
    # After CAE training is done
    torch.save(cae_model.state_dict(), "QA/DEC/cae_model.pt")

    # 3) DEC Setup
    idec_model = IDEC(encoder=cae_model.encoder, decoder=cae_model.decoder,
                      n_clusters=args.n_clusters, alpha=1.0, target_beta=0.1)    # Initialize cluster centers
    print("[MAIN] Initializing DEC cluster centers...")
    idec_model.initialize_centers(train_loader, device=args.device)

    # 4) DEC Training (two-pass each epoch)
    print("[MAIN] Starting DEC training/fine-tuning...")
    # dec_model = train_dec_full_pass(dec_model, train_loader, val_loader=val_loader,
    #                                 epochs=args.epochs_dec, device=args.device)

    # idec_model = train_idec_full_pass(idec_model, train_loader, val_loader=val_loader,
    #                                 epochs=args.epochs_dec, device=args.device)

    # Initial lambda values
    initial_lambdas = {
        'lambda_kl': 1.5,
        'lambda_recon': 0.05,
        'lambda_bal': 0.3,
        'lambda_entropy': 0.1,
        'lambda_sep': 0.01
    }

    idec_model = train_idec_uniform_clusters(
        idec_model,
        train_loader,
        val_loader=val_loader,
        epochs=args.epochs_dec,
        device=args.device,
        initial_lambdas=initial_lambdas,
    )
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
