import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Subset, DistributedSampler
from sklearn.model_selection import train_test_split
import math
import os
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset
from torch.amp import GradScaler, autocast

from torch.cuda.amp import GradScaler, autocast


class EEGNpyDataset(Dataset):
    """
    Memory-optimized dataset that preloads all .npy files into memory
    and transfers data to GPU during __getitem__ calls. If normalization is enabled,
    it computes the global mean and std over the entire dataset and normalizes
    each sample accordingly.
    """

    def __init__(self, directory, normalize=False):
        super().__init__()
        self.directory = directory
        self.normalize = normalize

        # Find all .npy files
        self.files = [f for f in os.listdir(directory) if f.endswith('.npy')]
        if not self.files:
            raise ValueError(f"No .npy files found in directory: {directory}")
        self.files.sort()

        # Preload all data into memory
        print(f"Preloading {len(self.files)} files into memory...")
        self.data = []
        for i, file in enumerate(self.files):
            if i % 1000 == 0:
                print(f"Loading file {i}/{len(self.files)}...")
            path = os.path.join(directory, file)
            arr = np.load(path)
            # Convert to torch tensor (kept on CPU)
            self.data.append(torch.from_numpy(arr).float())

        print(f"Finished loading {len(self.data)} samples into memory")
        self.image_shape = self.data[0].shape

        if self.normalize:
            # Stack all samples to compute global statistics
            all_data = torch.stack(self.data, dim=0)
            self.global_mean = all_data.mean()
            self.global_std = all_data.std()
            print(f"Global mean: {self.global_mean.item():.4f}, Global std: {self.global_std.item():.4f}")

            # Normalize each sample using the global mean and std
            epsilon = 1e-8  # To prevent division by zero
            self.data = [(sample - self.global_mean) / (self.global_std + epsilon)
                         for sample in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return the preloaded (and possibly normalized) tensor
        return self.data[idx]


def setup(rank, world_size):
    """Initialize the distributed environment."""
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(rank)
    print(
        f"Process {rank}/{world_size} initialized with MASTER_ADDR={os.environ.get('MASTER_ADDR', 'unknown')}, MASTER_PORT={os.environ.get('MASTER_PORT', 'unknown')}")


def cleanup():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def plot_ae_reconstructions(model, data_loader, device, n=8, out_path='recon.png', rank=0, inverse_norm=None):
    """
    Plot autoencoder reconstructions.

    Args:
        model: The trained model.
        data_loader: DataLoader providing input samples.
        device: Device to run inference on.
        n: Number of samples to plot.
        out_path: File path to save the plot.
        rank: Process rank (only rank 0 performs plotting).
        inverse_norm: Tuple (global_mean, global_std) to invert standardization for display.
    """
    # Only perform plotting on rank 0
    if rank != 0:
        return

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            x = batch.to(device)
            x_recon, _, _, _ = model(x)
            break

    # Get first n samples and move to CPU
    x = x.cpu()[:n]
    x_recon = x_recon.cpu()[:n]

    # Optionally inverse the standardization if global stats are provided
    if inverse_norm is not None:
        global_mean, global_std = inverse_norm
        x = x * global_std + global_mean
        x_recon = x_recon * global_std + global_mean

    # Normalize both for display (per batch normalization for visualization)
    x_disp = (x - x.min()) / (x.max() - x.min() + 1e-8)
    x_recon_disp = (x_recon - x_recon.min()) / (x_recon.max() - x_recon.min() + 1e-8)

    # Determine image shape (assumes (batch_size, C, H, W))
    C, H, W = x_disp.shape[1], x_disp.shape[2], x_disp.shape[3]

    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    for i in range(n):
        if C == 1:
            axes[0, i].imshow(x_disp[i].permute(1, 2, 0).squeeze(), cmap='gray')
            axes[1, i].imshow(x_recon_disp[i].permute(1, 2, 0).squeeze(), cmap='gray')
        elif C == 3:
            axes[0, i].imshow(x_disp[i].permute(1, 2, 0))
            axes[1, i].imshow(x_recon_disp[i].permute(1, 2, 0))
        else:
            # For multi-channel data where channels > 3, display the first channel as grayscale
            axes[0, i].imshow(x_disp[i][0], cmap='gray')
            axes[1, i].imshow(x_recon_disp[i][0], cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].axis('off')

    plt.savefig(out_path)
    plt.close()


def predict_clusters(model, data_loader, device):
    model.eval()
    clusters = []
    with torch.no_grad():
        for batch in data_loader:
            x = batch.to(device)
            _, mu_q, log_var_q, z = model(x)
            # Compute q(c|x)
            diff = mu_q.unsqueeze(1) - model.module.mu_c  # Access params through module for DDP
            log_likelihood = (
                    -0.5 * model.module.latent_dim * math.log(2 * math.pi)
                    - 0.5 * model.module.latent_dim * model.module.log_var_c
                    - 0.5 / model.module.log_var_c.exp() * diff.pow(2).sum(2)
            )  # (batch_size, K)
            log_p_c = F.log_softmax(model.module.log_p_c, dim=0)  # (K,)
            log_q_c_x_unnorm = log_p_c + log_likelihood
            log_q_c_x = F.log_softmax(log_q_c_x_unnorm, dim=1)
            q_c_x = log_q_c_x.exp()  # (batch_size, K)
            cluster_assignments = q_c_x.argmax(dim=1)  # (batch_size,)
            clusters.append(cluster_assignments.cpu().numpy())

    # Concatenate local predictions first
    local_clusters = np.concatenate(clusters) if clusters else np.array([])

    # Get size information from all processes
    local_size = torch.tensor([len(local_clusters)], device=device)
    sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, local_size)

    # Convert sizes to integers
    sizes = [size.item() for size in sizes]

    # Gather all cluster data using object gathering (handles variable sizes)
    all_clusters = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(all_clusters, local_clusters)

    # Concatenate all gathered clusters
    result = np.concatenate([cluster for cluster in all_clusters if len(cluster) > 0])

    return result

def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Apply before training

class VaDE(nn.Module):
    """
    Variational Deep Embedding (VaDE) model with a revised encoder/decoder architecture.
    The encoder and decoder are made symmetric by using three downsampling/upsampling layers.
    This structure can help prevent overly flattened reconstructions.
    """

    def __init__(self, input_shape, latent_dim=10, n_clusters=10):
        super(VaDE, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        C, H, W = input_shape

        # --- Encoder ---
        # Three convolutional layers with stride 2 downsample the input by a factor of 8.
        self.encoder = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, stride=2, padding=1),  # Output: 32 x (H/2) x (W/2)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: 64 x (H/4) x (W/4)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: 128 x (H/8) x (W/8)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.flatten = nn.Flatten()
        # Fully connected layers produce both the mean and log variance of the latent space.
        self.fc_mu_logvar = nn.Sequential(
            nn.Linear(128 * (H // 8) * (W // 8), 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2 * latent_dim)  # First half: mu, second half: log_var
        )

        # --- Decoder ---
        # First project the latent vector to a feature map.
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128 * (H // 8) * (W // 8)),
            nn.ReLU(inplace=True)
        )
        self.unflatten = nn.Unflatten(1, (128, H // 8, W // 8))
        # Mirror the encoder with transposed convolutions to upsample back to the original size.
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # Output: 64 x (H/4) x (W/4)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # Output: 32 x (H/2) x (W/2)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, C, kernel_size=3, stride=2,
                               padding=1, output_padding=1)  # Output: C x H x W
            # For standardized data, we often omit a bounded activation here.
            # If you want to constrain outputs (and adjust your targets), you could use nn.Tanh() or nn.Sigmoid()
        )

        # GMM parameters (unchanged)
        self.mu_c = nn.Parameter(torch.randn(n_clusters, latent_dim))
        self.log_var_c = nn.Parameter(torch.zeros(n_clusters))
        self.log_p_c = nn.Parameter(torch.zeros(n_clusters))

    def encode(self, x):
        h = self.encoder(x)
        h_flat = self.flatten(h)
        h_out = self.fc_mu_logvar(h_flat)
        mu_q, log_var_q = h_out.chunk(2, dim=1)
        return mu_q, log_var_q

    def reparameterize(self, mu_q, log_var_q):
        std = torch.exp(0.5 * log_var_q)
        eps = torch.randn_like(std)
        return mu_q + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        h = self.unflatten(h)
        x_recon = self.decoder_conv(h)
        # Add bounded activation since data is normalized with mean 0, std ~0.96
        # Tanh outputs values in [-1, 1] which matches your normalized data range well
        x_recon = torch.tanh(x_recon)
        return x_recon

    def forward(self, x):
        mu_q, log_var_q = self.encode(x)
        z = self.reparameterize(mu_q, log_var_q)
        x_recon = self.decode(z)
        return x_recon, mu_q, log_var_q, z


def vae_loss(x, x_recon, mu_q, log_var_q, beta=1.0):
    # Clip reconstruction to prevent extreme values
    x_recon = torch.clamp(x_recon, min=-10.0, max=10.0)

    mse_loss = F.mse_loss(x_recon, x, reduction='mean')
    l1_loss = F.l1_loss(x_recon, x, reduction='mean')
    recon_loss = 0.5 * mse_loss + 0.5 * l1_loss

    # More robust KL divergence calculation
    eps = 1e-8
    log_var_q = torch.clamp(log_var_q, min=-20.0, max=20.0)  # Prevent extreme values
    mu_q = torch.clamp(mu_q, min=-20.0, max=20.0)  # Prevent extreme values

    var_q = torch.exp(log_var_q) + eps
    kl_div = -0.5 * torch.mean(1 + log_var_q - mu_q.pow(2) - var_q)
    kl_div = torch.clamp(kl_div, max=100.0)  # Prevent extreme values

    return recon_loss + beta * kl_div


def vade_loss(x, x_recon, mu_q, log_var_q, model, beta=0.01):
    # Clip reconstructions
    x_recon = torch.clamp(x_recon, min=-10.0, max=10.0)

    # For DDP, access parameters through model.module
    if isinstance(model, DDP):
        d = model.module.latent_dim
        var_c = torch.exp(model.module.log_var_c) + 1e-6  # Add epsilon for stability
        log_var_c = torch.clamp(model.module.log_var_c, min=-20.0, max=20.0)
        mu_c = model.module.mu_c
        log_p_c = F.log_softmax(model.module.log_p_c, dim=0)
    else:
        d = model.latent_dim
        var_c = torch.exp(model.log_var_c) + 1e-6  # Add epsilon for stability
        log_var_c = torch.clamp(model.log_var_c, min=-20.0, max=20.0)
        mu_c = model.mu_c
        log_p_c = F.log_softmax(model.log_p_c, dim=0)

    # Clamp inputs for stability
    mu_q = torch.clamp(mu_q, min=-20.0, max=20.0)
    log_var_q = torch.clamp(log_var_q, min=-20.0, max=20.0)

    mse_loss = F.mse_loss(x_recon, x, reduction='mean')
    l1_loss = F.l1_loss(x_recon, x, reduction='mean')
    recon_loss = 0.5 * mse_loss + 0.5 * l1_loss

    # Compute difference with numerical stability
    diff = mu_q.unsqueeze(1) - mu_c

    # Compute log likelihood with numerical safeguards
    log_likelihood = (
            -0.5 * d * math.log(2 * math.pi)
            - 0.5 * d * log_var_c
            - 0.5 * torch.div(diff.pow(2).sum(2) + 1e-8, var_c + 1e-8)  # Avoid division by zero
    )

    # More stable softmax computation
    log_likelihood = torch.clamp(log_likelihood, min=-100, max=100)
    log_q_c_x = F.log_softmax(log_p_c + log_likelihood, dim=1)
    q_c_x = torch.exp(log_q_c_x)

    # Add numerical stability to variance calculation
    var_q = torch.exp(log_var_q) + 1e-6
    sum_var_q = var_q.sum(1)
    log_det_q = log_var_q.sum(1)

    diff_sq = diff.pow(2).sum(2)
    inv_var_c = 1.0 / (var_c + 1e-8)  # Add epsilon to denominator

    # More stable KL computation
    kl_per_cluster = 0.5 * (
            inv_var_c.unsqueeze(0) * sum_var_q.unsqueeze(1) +
            inv_var_c.unsqueeze(0) * diff_sq +
            d * log_var_c.unsqueeze(0) - d -
            log_det_q.unsqueeze(1)
    )

    # Clip extremely large values to prevent numerical overflow
    kl_per_cluster = torch.clamp(kl_per_cluster, max=1e6)

    expected_kl = (q_c_x * kl_per_cluster).sum(1)
    kl_categorical = (q_c_x * (log_q_c_x - log_p_c)).sum(1)

    total_kl = expected_kl + kl_categorical

    # Clamp the final loss to prevent numerical issues
    loss = recon_loss + beta * torch.clamp(total_kl.mean(), max=1e6)

    # Early detection of NaN or Inf
    if torch.isnan(loss) or torch.isinf(loss):
        print("WARNING: Loss is NaN or Inf, using fallback loss")
        return recon_loss  # Fallback to just reconstruction loss

    return loss
def initialize_gmm_params(model, train_loader, device, rank):
    """Initialize GMM parameters using K-means on latent means after pretraining."""
    # Only collect latent representations on current device
    with torch.no_grad():
        all_mu_q = []
        for batch in train_loader:
            x = batch.to(device)
            mu_q, _ = model.module.encode(x)
            all_mu_q.append(mu_q.cpu())
        local_mu_q = torch.cat(all_mu_q, dim=0)

    # Gather all latent representations across devices
    local_size = torch.tensor([local_mu_q.shape[0]], device=device)
    size_list = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(dist.get_world_size())]
    dist.all_gather(size_list, local_size)

    max_size = max(size.item() for size in size_list)

    # Pad tensor to same size for all_gather
    if local_mu_q.shape[0] < max_size:
        padding = torch.zeros(max_size - local_mu_q.shape[0], local_mu_q.shape[1], device='cpu')
        padded_mu_q = torch.cat([local_mu_q, padding], dim=0)
    else:
        padded_mu_q = local_mu_q

    # Move to the device for gathering
    padded_mu_q = padded_mu_q.to(device)

    # Gather tensors from all processes
    gathered_mu_q = [torch.zeros_like(padded_mu_q) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_mu_q, padded_mu_q)

    # Process on rank 0 and then broadcast
    if rank == 0:
        # Unpad and concatenate
        all_mu_q = []
        for i, size in enumerate(size_list):
            all_mu_q.append(gathered_mu_q[i][:size.item()])
        all_mu_q = torch.cat(all_mu_q, dim=0)

        mu_mean = all_mu_q.mean()
        mu_std = all_mu_q.std()
        print(f"Latent mu_q mean: {mu_mean.item():.4f}, std: {mu_std.item():.4f}")
        print(f"Collected {all_mu_q.shape[0]} total latent vectors for clustering")

        # Run K-means
        kmeans = KMeans(n_clusters=model.module.n_clusters, random_state=42)
        labels = kmeans.fit_predict(all_mu_q.cpu().numpy())
        model.module.mu_c.data = torch.from_numpy(kmeans.cluster_centers_).to(device)

        # Compute per-cluster variance
        for k in range(model.module.n_clusters):
            cluster_points = all_mu_q[labels == k]
            if len(cluster_points) > 1:
                var_k = torch.var(cluster_points, dim=0).mean().cpu()  # Mean variance across dimensions
                model.module.log_var_c.data[k] = torch.log(var_k + 1e-6)  # Add epsilon for stability
            else:
                # Fallback to total variance if cluster has too few points
                var_total = torch.var(all_mu_q, dim=0).mean().cpu()
                model.module.log_var_c.data[k] = torch.log(var_total + 1e-6)

        # Initialize p(c) as uniform
        model.module.log_p_c.data = torch.zeros(model.module.n_clusters).to(device)

    # Broadcast the updated parameters to all processes
    dist.broadcast(model.module.mu_c, 0)
    dist.broadcast(model.module.log_var_c, 0)
    dist.broadcast(model.module.log_p_c, 0)


def evaluate_vae(model, data_loader, device, beta):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for batch in data_loader:
            x = batch.to(device)
            batch_size = x.size(0)
            x_recon, mu_q, log_var_q, z = model(x)
            loss = vae_loss(x, x_recon, mu_q, log_var_q, beta=beta)

            if torch.isnan(loss):
                print("Loss is NaN! Debugging values:")
                print("x:", x.min().item(), x.max().item())
                print("x_recon:", x_recon.min().item(), x_recon.max().item())
                print("mu_q:", mu_q.min().item(), mu_q.max().item())
                print("log_var_q:", log_var_q.min().item(), log_var_q.max().item())
                # Possibly break or skip the update
                continue
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    # Sum losses across all processes
    total_loss_tensor = torch.tensor([total_loss], device=device)
    total_samples_tensor = torch.tensor([total_samples], device=device)

    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

    avg_loss = total_loss_tensor.item() / total_samples_tensor.item() if total_samples_tensor.item() > 0 else 0
    return avg_loss


def evaluate_vade(model, data_loader, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for batch in data_loader:
            x = batch.to(device)
            batch_size = x.size(0)
            x_recon, mu_q, log_var_q, z = model(x)
            loss = vade_loss(x, x_recon, mu_q, log_var_q, model)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    # Sum losses across all processes
    total_loss_tensor = torch.tensor([total_loss], device=device)
    total_samples_tensor = torch.tensor([total_samples], device=device)

    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

    avg_loss = total_loss_tensor.item() / total_samples_tensor.item() if total_samples_tensor.item() > 0 else 0
    return avg_loss


def save_checkpoint(model, optimizer, epoch, filename, rank=0):
    if rank == 0:  # Only save on rank 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, filename)


def train_vae(model, train_loader, optimizer, device, epoch, args):  # removed scaler
    model.train()
    total_train_loss = 0
    total_train_samples = 0

    # Use more gradual KL annealing - start with small beta and increase slowly
    beta = min(0.1, epoch / (args.pretrain_epochs * 0.75) * 0.1)

    for batch_idx, batch in enumerate(train_loader):
        x = batch.to(device)
        batch_size = x.size(0)

        optimizer.zero_grad()

        try:
            # Remove autocast context manager
            x_recon, mu_q, log_var_q, z = model(x)

            # Check for NaNs early
            if torch.isnan(x_recon).any():
                print(f"NaN detected in x_recon on batch {batch_idx}, skipping")
                continue

            loss = vae_loss(x, x_recon, mu_q, log_var_q, beta=beta)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Loss is NaN/Inf on batch {batch_idx}, skipping")
                print(f"Debug values: x: [{x.min().item()}, {x.max().item()}], "
                      f"x_recon: [{x_recon.min().item()}, {x_recon.max().item()}], "
                      f"mu_q: [{mu_q.min().item()}, {mu_q.max().item()}], "
                      f"log_var_q: [{log_var_q.min().item()}, {log_var_q.max().item()}]")
                continue

            # Replace scaler.scale().backward() with normal backward
            loss.backward()

            # Keep gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            # Check for NaN gradients
            has_nan_grad = False
            for param in model.parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    has_nan_grad = True
                    break

            if has_nan_grad:
                print(f"NaN/Inf gradient detected on batch {batch_idx}, skipping update")
                continue

            # Replace scaler.step() and scaler.update() with normal step
            optimizer.step()

            total_train_loss += loss.item() * batch_size
            total_train_samples += batch_size

        except Exception as e:
            print(f"Exception in training: {e}")
            continue

    return total_train_loss, total_train_samples


def train_vade(model, train_loader, optimizer, device, epoch, args):  # removed scaler
    model.train()
    total_train_loss = 0
    total_train_samples = 0

    for batch_idx, batch in enumerate(train_loader):
        x = batch.to(device)
        batch_size = x.size(0)

        optimizer.zero_grad()

        try:
            # Remove autocast
            x_recon, mu_q, log_var_q, z = model(x)

            # Check for NaNs early
            if torch.isnan(x_recon).any():
                print(f"NaN detected in x_recon on batch {batch_idx}, skipping")
                continue

            loss = vade_loss(x, x_recon, mu_q, log_var_q, model)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Loss is NaN/Inf on batch {batch_idx}, skipping")
                continue

            # Normal backward instead of scaled
            loss.backward()

            # Keep gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            # Check for NaN gradients
            has_nan_grad = False
            for param in model.parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    has_nan_grad = True
                    break

            if has_nan_grad:
                print(f"NaN/Inf gradient detected on batch {batch_idx}, skipping update")
                continue

            # Normal step instead of scaled
            optimizer.step()

            total_train_loss += loss.item() * batch_size
            total_train_samples += batch_size

        except Exception as e:
            print(f"Exception in training: {e}")
            continue

    return total_train_loss, total_train_samples

def main():
    # Get rank and world size from environment (set by torchrun)
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Set the TORCH_DISTRIBUTED_DEBUG environment variable to get more info if needed
    if rank == 0:
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="training_data/coeffs/")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)  # Reduced learning rate
    parser.add_argument("--pretrain_epochs", type=int, default=20)
    parser.add_argument("--cluster_epochs", type=int, default=120)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--latent_dim", type=int, default=2048)
    parser.add_argument("--n_clusters", type=int, default=50)
    args = parser.parse_args()

    try:
        # Initialize process group
        setup(rank, world_size)

        # Create directory for outputs on rank 0
        if rank == 0:
            os.makedirs('QA/VaDE', exist_ok=True)
            print(f"Starting training with {world_size} GPUs")

        # Load dataset
        if rank == 0:
            print(f"Loading dataset from {args.data_dir}")

        dataset = EEGNpyDataset(args.data_dir, normalize=True)

        # Split dataset
        idxs = list(range(len(dataset)))
        train_idx, val_idx = train_test_split(idxs, test_size=0.2, random_state=42)
        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)

        if rank == 0:
            print(f"Dataset split: {len(train_ds)} training, {len(val_ds)} validation")

        # Use DistributedSampler
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

        # Use DataLoader with the samplers
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )

        # Get input shape
        sample_data = dataset[0]
        input_shape = sample_data.shape  # (C, H, W)
        if rank == 0:
            print(f"Input shape: {input_shape}")

        # Create model
        model = VaDE(input_shape=input_shape, latent_dim=args.latent_dim, n_clusters=args.n_clusters).to(rank)
        model.apply(weights_init)

        # Initialize GMM means with smaller values for stability
        with torch.no_grad():
            model.mu_c.data = torch.randn_like(model.mu_c.data) * 0.1

        # Wrap model in DDP with find_unused_parameters=False for better performance
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)
        if hasattr(model, "_set_static_graph"):
            model._set_static_graph()  # Optional optimization

        # Use Adam optimizer with smaller learning rate and increased epsilon
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5, eps=1e-8)

        # Add LR scheduler for adaptive learning rate
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=(rank == 0)
        )

        # Begin VAE pretraining
        if rank == 0:
            print("Starting VAE pretraining...")

        pretrain_train_losses = []
        pretrain_val_losses = []

        for epoch in range(args.pretrain_epochs):
            # Set epoch for sampler
            train_sampler.set_epoch(epoch)

            total_train_loss, total_train_samples = train_vae(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                device=rank,
                epoch=epoch,
                args=args
            )

            # Aggregate losses across all processes
            train_loss_tensor = torch.tensor([total_train_loss], device=rank)
            train_samples_tensor = torch.tensor([total_train_samples], device=rank)

            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_samples_tensor, op=dist.ReduceOp.SUM)

            train_loss = train_loss_tensor.item() / train_samples_tensor.item() if train_samples_tensor.item() > 0 else 0

            # Use annealing for KL divergence weight
            beta = 0.0 if epoch < args.warmup_epochs else min(0.1, (epoch - args.warmup_epochs + 1) /
                                                              (args.pretrain_epochs - args.warmup_epochs) * 0.1)

            # Evaluate on validation set
            val_loss = evaluate_vae(model, val_loader, rank, beta)

            # Update learning rate based on validation loss
            if rank == 0:
                scheduler.step(val_loss)

            if rank == 0:
                pretrain_train_losses.append(train_loss)
                pretrain_val_losses.append(val_loss)
                print(f"Pretraining Epoch {epoch + 1}/{args.pretrain_epochs}, "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Beta: {beta:.4f}")

                # Save checkpoint periodically
                if (epoch + 1) % 10 == 0 or epoch + 1 == args.pretrain_epochs:
                    save_checkpoint(model, optimizer, epoch, f'QA/VaDE/pretrain_checkpoint_epoch_{epoch + 1}.pt', rank)

        # Plot pretraining losses (only on rank 0)
        if rank == 0:
            plt.figure(figsize=(10, 6))
            plt.plot(pretrain_train_losses, label='Train')
            plt.plot(pretrain_val_losses, label='Val')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Pretraining Loss')
            plt.legend()
            plt.savefig('QA/VaDE/pretrain_loss.png')
            plt.close()
            print("Pretraining completed successfully.")

        # Initialize GMM parameters
        if rank == 0:
            print("Initializing GMM parameters...")

        initialize_gmm_params(model, train_loader, rank, rank)
        dist.barrier()  # Ensure all processes have updated parameters

        # Plot reconstructions (only on rank 0)
        plot_ae_reconstructions(model, val_loader, device=rank, n=2, out_path='QA/VaDE/pretrain_ae_recons.png',
                                rank=rank)

        # Begin VaDE clustering training
        if rank == 0:
            print("Starting VaDE clustering training...")

        cluster_train_losses = []
        cluster_val_losses = []

        # Reset scheduler for clustering phase
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=(rank == 0)
        )

        for epoch in range(args.cluster_epochs):
            # Set epoch for sampler
            train_sampler.set_epoch(epoch + args.pretrain_epochs)

            # Use the train_vade function
            total_train_loss, total_train_samples = train_vade(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                device=rank,
                epoch=epoch,
                args=args
            )

            # Aggregate losses across all processes
            train_loss_tensor = torch.tensor([total_train_loss], device=rank)
            train_samples_tensor = torch.tensor([total_train_samples], device=rank)

            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_samples_tensor, op=dist.ReduceOp.SUM)

            train_loss = train_loss_tensor.item() / train_samples_tensor.item() if train_samples_tensor.item() > 0 else 0

            # Evaluate on validation set
            val_loss = evaluate_vade(model, val_loader, rank)

            # Update learning rate based on validation loss
            if rank == 0:
                scheduler.step(val_loss)

            if rank == 0:
                cluster_train_losses.append(train_loss)
                cluster_val_losses.append(val_loss)
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Clustering Epoch {epoch + 1}/{args.cluster_epochs}, "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")

                # Save checkpoint periodically
                if (epoch + 1) % 10 == 0 or epoch + 1 == args.cluster_epochs:
                    save_checkpoint(model, optimizer, epoch, f'QA/VaDE/cluster_checkpoint_epoch_{epoch + 1}.pt', rank)

        # Save final model (only on rank 0)
        save_checkpoint(model, optimizer, args.cluster_epochs, 'QA/VaDE/final_model.pt', rank)

        # Plot clustering losses (only on rank 0)
        if rank == 0:
            plt.figure(figsize=(10, 6))
            plt.plot(cluster_train_losses, label='Train')
            plt.plot(cluster_val_losses, label='Val')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Clustering Training Loss')
            plt.legend()
            plt.savefig('QA/VaDE/cluster_loss.png')
            plt.close()

        # Predict clusters
        if rank == 0:
            print("Predicting clusters...")
        clusters = predict_clusters(model, val_loader, rank)

        # Plot histogram (only on rank 0)
        if rank == 0 and clusters is not None:
            plt.figure(figsize=(12, 6))
            plt.hist(clusters, bins=np.arange(model.module.n_clusters + 1) - 0.5, rwidth=0.8)
            plt.xlabel('Cluster')
            plt.ylabel('Frequency')
            plt.title('Cluster Assignments Histogram')
            plt.savefig('QA/VaDE/clusters_histogram.png')
            plt.close()

        # Plot final reconstructions (only on rank 0)
        plot_ae_reconstructions(model, val_loader, device=rank, n=2, out_path='QA/VaDE/final_ae_recons.png', rank=rank)

        if rank == 0:
            print("Training completed successfully!")

    except Exception as e:
        print(f"Error on rank {rank}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        cleanup()

if __name__ == "__main__":
    main()