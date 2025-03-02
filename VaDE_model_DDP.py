import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import math
import os
import matplotlib.pyplot as plt
from DEC_model import EEGNpyDataset


# Utility Functions
def plot_ae_reconstructions(model, data_loader, device, n=8, out_path='recon.png'):
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            x = batch.to(device)
            x_recon, _, _, _ = model(x)
            break
    x = x.cpu()[:n]
    x_recon = x_recon.cpu()[:n]
    C, H, W = x.shape[1], x.shape[2], x.shape[3]
    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    for i in range(n):
        if C == 1:
            axes[0, i].imshow(x[i].permute(1, 2, 0).squeeze(), cmap='gray')
            axes[1, i].imshow(x_recon[i].permute(1, 2, 0).squeeze(), cmap='gray')
        elif C == 3:
            axes[0, i].imshow(x[i].permute(1, 2, 0))
            axes[1, i].imshow(x_recon[i].permute(1, 2, 0))
        else:
            axes[0, i].imshow(x[i][0], cmap='gray')
            axes[1, i].imshow(x_recon[i][0], cmap='gray')
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
            diff = mu_q.unsqueeze(1) - model.module.mu_c
            log_likelihood = (
                    -0.5 * model.module.latent_dim * math.log(2 * math.pi)
                    - 0.5 * model.module.latent_dim * model.module.log_var_c
                    - 0.5 / model.module.log_var_c.exp() * diff.pow(2).sum(2)
            )
            log_p_c = F.log_softmax(model.module.log_p_c, dim=0)
            log_q_c_x_unnorm = log_p_c + log_likelihood
            log_q_c_x = F.log_softmax(log_q_c_x_unnorm, dim=1)
            q_c_x = log_q_c_x.exp()
            cluster_assignments = q_c_x.argmax(dim=1)
            clusters.append(cluster_assignments.cpu().numpy())
    return np.concatenate(clusters)


# Model Definition
class VaDE(nn.Module):
    def __init__(self, input_shape, latent_dim=10, n_clusters=10):
        super(VaDE, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        C, H, W = input_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(C, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * (H // 8) * (W // 8), 256),
            nn.ReLU(),
            nn.Linear(256, 2 * latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128 * (H // 8) * (W // 8)),
            nn.ReLU(),
            nn.Unflatten(1, (128, H // 8, W // 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, C, kernel_size=3, stride=1, padding=1),
        )
        self.mu_c = nn.Parameter(torch.randn(n_clusters, latent_dim))
        self.log_var_c = nn.Parameter(torch.zeros(n_clusters))
        self.log_p_c = nn.Parameter(torch.zeros(n_clusters))

    def encode(self, x):
        h = self.encoder(x)
        mu_q, log_var_q = h.chunk(2, dim=1)
        return mu_q, log_var_q

    def reparameterize(self, mu_q, log_var_q):
        std = torch.exp(0.5 * log_var_q)
        eps = torch.randn_like(std)
        return mu_q + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu_q, log_var_q = self.encode(x)
        z = self.reparameterize(mu_q, log_var_q)
        x_recon = self.decode(z)
        return x_recon, mu_q, log_var_q, z


# Loss Functions
def vae_loss(x, x_recon, mu_q, log_var_q, beta=1.0):
    mse_loss = F.mse_loss(x_recon, x, reduction='mean')
    l1_loss = F.l1_loss(x_recon, x, reduction='mean')
    recon_loss = 0.5 * mse_loss + 0.5 * l1_loss
    kl_div = -0.5 * (1 + log_var_q - mu_q.pow(2) - log_var_q.exp()).sum(1).mean()
    return recon_loss + beta * kl_div


def vade_loss(x, x_recon, mu_q, log_var_q, model, beta=0.01):
    mse_loss = F.mse_loss(x_recon, x, reduction='mean')
    l1_loss = F.l1_loss(x_recon, x, reduction='mean')
    recon_loss = 0.5 * mse_loss + 0.5 * l1_loss
    d = model.module.latent_dim
    var_c = model.module.log_var_c.exp()
    log_var_c = model.module.log_var_c
    mu_c = model.module.mu_c
    log_p_c = F.log_softmax(model.module.log_p_c, dim=0)
    diff = mu_q.unsqueeze(1) - mu_c
    log_likelihood = (
            -0.5 * d * math.log(2 * math.pi)
            - 0.5 * d * log_var_c
            - 0.5 / var_c * diff.pow(2).sum(2)
    )
    log_q_c_x_unnorm = log_p_c + log_likelihood
    log_q_c_x = F.log_softmax(log_q_c_x_unnorm, dim=1)
    q_c_x = log_q_c_x.exp()
    sum_var_q = log_var_q.exp().sum(1)
    log_det_q = log_var_q.sum(1)
    diff_sq = diff.pow(2).sum(2)
    inv_var_c = 1 / var_c
    kl_per_cluster = 0.5 * (
            inv_var_c.unsqueeze(0) * sum_var_q.unsqueeze(1) +
            inv_var_c.unsqueeze(0) * diff_sq +
            d * log_var_c.unsqueeze(0) - d -
            log_det_q.unsqueeze(1)
    )
    expected_kl = (q_c_x * kl_per_cluster).sum(1)
    kl_categorical = (q_c_x * (log_q_c_x - log_p_c)).sum(1)
    total_kl = expected_kl + kl_categorical
    return recon_loss + beta * total_kl.mean()


# GMM Initialization
def initialize_gmm_params(model, train_loader, device, local_rank, world_size):
    model.eval()
    with torch.no_grad():
        all_mu_q = []
        for batch in train_loader:
            x = batch.to(device)
            mu_q, _ = model.module.encode(x)
            all_mu_q.append(mu_q.cpu())
        all_mu_q_local = torch.cat(all_mu_q, dim=0)
        all_mu_q_list = [torch.zeros_like(all_mu_q_local) for _ in range(world_size)]
        dist.gather(all_mu_q_local, gather_list=all_mu_q_list if local_rank == 0 else None, dst=0)
        if local_rank == 0:
            all_mu_q = torch.cat(all_mu_q_list, dim=0)
            kmeans = KMeans(n_clusters=model.module.n_clusters, random_state=42)
            labels = kmeans.fit_predict(all_mu_q.numpy())
            mu_c = torch.from_numpy(kmeans.cluster_centers_).to(device)
            log_var_c = torch.zeros(model.module.n_clusters, device=device)
            for k in range(model.module.n_clusters):
                cluster_points = all_mu_q[labels == k]
                if len(cluster_points) > 1:
                    var_k = torch.var(cluster_points, dim=0).mean()
                    log_var_c[k] = torch.log(var_k + 1e-6)
                else:
                    var_total = torch.var(all_mu_q, dim=0).mean()
                    log_var_c[k] = torch.log(var_total + 1e-6)
            log_p_c = torch.zeros(model.module.n_clusters, device=device)
        else:
            mu_c = torch.zeros(model.module.n_clusters, model.module.latent_dim, device=device)
            log_var_c = torch.zeros(model.module.n_clusters, device=device)
            log_p_c = torch.zeros(model.module.n_clusters, device=device)
        dist.broadcast(mu_c, src=0)
        dist.broadcast(log_var_c, src=0)
        dist.broadcast(log_p_c, src=0)
        model.module.mu_c.data = mu_c
        model.module.log_var_c.data = log_var_c
        model.module.log_p_c.data = log_p_c


# Evaluation Functions
def evaluate_vae(model, data_loader, device, beta):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for batch in data_loader:
            x = batch.to(device)
            batch_size = x.size(0)
            x_recon, mu_q, log_var_q, _ = model(x)
            loss = vae_loss(x, x_recon, mu_q, log_var_q, beta=beta)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    return total_loss / total_samples if total_samples > 0 else 0


def evaluate_vade(model, data_loader, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for batch in data_loader:
            x = batch.to(device)
            batch_size = x.size(0)
            x_recon, mu_q, log_var_q, _ = model(x)
            loss = vade_loss(x, x_recon, mu_q, log_var_q, model)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    return total_loss / total_samples if total_samples > 0 else 0


# Main Worker Function for DDP
def main_worker(local_rank, args, train_idx, val_idx):
    # Initialize distributed process group
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=args.world_size, rank=local_rank)

    # Set device for this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    # Create output directory on rank 0
    if local_rank == 0:
        os.makedirs('QA/VaDE', exist_ok=True)

    # Load dataset
    dataset = EEGNpyDataset(args.data_dir, normalize=True)

    # Training subset with DistributedSampler
    train_subset = Subset(dataset, train_idx)
    train_sampler = DistributedSampler(train_subset, num_replicas=args.world_size,
                                       rank=local_rank, shuffle=True)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                              sampler=train_sampler)

    # Validation subset on rank 0 only
    val_loader = None
    if local_rank == 0:
        val_subset = Subset(dataset, val_idx)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

    # Initialize model and wrap with DDP
    model = VaDE(input_shape=args.input_shape, latent_dim=args.latent_dim,
                 n_clusters=args.n_clusters).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Pretraining Phase
    if local_rank == 0:
        print("Starting VAE pretraining...")
    pretrain_train_losses = []
    pretrain_val_losses = []

    for epoch in range(args.pretrain_epochs):
        model.train()
        train_sampler.set_epoch(epoch)  # Ensure different shuffling each epoch
        total_train_loss = 0
        total_train_samples = 0
        for batch in train_loader:
            x = batch.to(device)
            batch_size = x.size(0)
            x_recon, mu_q, log_var_q, _ = model(x)
            beta = (0.0 if epoch < args.warmup_epochs else
                    (epoch - args.warmup_epochs + 1) / (args.pretrain_epochs - args.warmup_epochs))
            loss = vae_loss(x, x_recon, mu_q, log_var_q, beta=beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch_size
            total_train_samples += batch_size

        # Aggregate training loss across all GPUs
        loss_tensor = torch.tensor(total_train_loss, device=device)
        samples_tensor = torch.tensor(total_train_samples, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)
        train_loss = loss_tensor.item() / samples_tensor.item()

        # Validation and logging on rank 0
        if local_rank == 0:
            val_loss = evaluate_vae(model, val_loader, device, beta)
            pretrain_train_losses.append(train_loss)
            pretrain_val_losses.append(val_loss)
            print(f"Pretraining Epoch {epoch + 1}/{args.pretrain_epochs}, "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Plot pretraining losses on rank 0
    if local_rank == 0:
        plt.plot(pretrain_train_losses, label='Train')
        plt.plot(pretrain_val_losses, label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Pretraining Loss')
        plt.legend()
        plt.savefig('QA/VaDE/pretrain_loss.png')
        plt.close()

    # Initialize GMM parameters
    if local_rank == 0:
        print("Initializing GMM parameters...")
    initialize_gmm_params(model, train_loader, device, local_rank, args.world_size)

    # Plot reconstructions after pretraining on rank 0
    if local_rank == 0:
        plot_ae_reconstructions(model, val_loader, device, n=8,
                                out_path='QA/VaDE/pretrain_ae_recons.png')

    # Clustering Phase
    if local_rank == 0:
        print("Starting VaDE clustering training...")
    cluster_train_losses = []
    cluster_val_losses = []

    for epoch in range(args.cluster_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        total_train_loss = 0
        total_train_samples = 0
        for batch in train_loader:
            x = batch.to(device)
            batch_size = x.size(0)
            x_recon, mu_q, log_var_q, _ = model(x)
            loss = vade_loss(x, x_recon, mu_q, log_var_q, model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch_size
            total_train_samples += batch_size

        # Aggregate training loss
        loss_tensor = torch.tensor(total_train_loss, device=device)
        samples_tensor = torch.tensor(total_train_samples, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)
        train_loss = loss_tensor.item() / samples_tensor.item()

        # Validation and logging on rank 0
        if local_rank == 0:
            val_loss = evaluate_vade(model, val_loader, device)
            cluster_train_losses.append(train_loss)
            cluster_val_losses.append(val_loss)
            print(f"Clustering Epoch {epoch + 1}/{args.cluster_epochs}, "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Post-training visualization on rank 0
    if local_rank == 0:
        plt.plot(cluster_train_losses, label='Train')
        plt.plot(cluster_val_losses, label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Clustering Training Loss')
        plt.legend()
        plt.savefig('QA/VaDE/cluster_loss.png')
        plt.close()

        clusters = predict_clusters(model, val_loader, device)
        plt.hist(clusters, bins=np.arange(model.module.n_clusters + 1) - 0.5, rwidth=0.8)
        plt.xlabel('Cluster')
        plt.ylabel('Frequency')
        plt.title('Cluster Assignments Histogram')
        plt.savefig('QA/VaDE/clusters_histogram.png')
        plt.close()

        plot_ae_reconstructions(model, val_loader, device, n=8,
                                out_path='QA/VaDE/final_ae_recons.png')

    # Cleanup
    dist.destroy_process_group()


# Main Entry Point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VaDE Training with DDP")
    parser.add_argument("--data_dir", type=str, default="training_data/coeffs/",
                        help="Directory containing EEG data")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=6e-4,
                        help="Learning rate")
    parser.add_argument("--pretrain_epochs", type=int, default=200,
                        help="Number of pretraining epochs")
    parser.add_argument("--cluster_epochs", type=int, default=100,
                        help="Number of clustering epochs")
    parser.add_argument("--warmup_epochs", type=int, default=100,
                        help="Number of warmup epochs with beta=0")
    parser.add_argument("--latent_dim", type=int, default=2048,
                        help="Latent dimension size")
    parser.add_argument("--n_clusters", type=int, default=100,
                        help="Number of clusters")
    args = parser.parse_args()

    # Prepare dataset and splits
    dataset = EEGNpyDataset(args.data_dir, normalize=True)
    idxs = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(idxs, test_size=0.2, random_state=42)
    args.world_size = torch.cuda.device_count()
    args.input_shape = dataset[0].shape

    # Launch DDP training
    mp.spawn(main_worker, args=(args, train_idx, val_idx),
             nprocs=args.world_size, join=True)