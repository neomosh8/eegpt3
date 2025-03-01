#!/usr/bin/env python3

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

#####################
#   Example Dataset
#####################
class EEGNpyDataset(Dataset):
    """Simple dataset that loads .npy files (C,H,W)."""
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        self.files = [f for f in os.listdir(directory) if f.endswith('.npy')]
        if not self.files:
            raise ValueError("No .npy files found in directory: " + directory)
        self.files.sort()
        # Check shape
        sample = np.load(os.path.join(directory, self.files[0]))
        self.image_shape = sample.shape

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.directory, self.files[idx])
        arr = np.load(path)
        return torch.from_numpy(arr).float()


#####################
#  Convolutional AE
#####################
class ConvEncoder(nn.Module):
    def __init__(self, input_shape=(3,64,64), latent_dim=64):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        C, H, W = input_shape

        self.conv_net = nn.Sequential(
            nn.Conv2d(C, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            out = self.conv_net(dummy)
            self.after_conv_shape = out.shape  # (1, 64, H', W')
            self.flatten_dim = out.view(1, -1).shape[1]

        self.fc = nn.Linear(self.flatten_dim, self.latent_dim)

    def forward(self, x):
        out = self.conv_net(x)
        out = out.view(out.size(0), -1)
        z = self.fc(out)
        return z


class ConvDecoder(nn.Module):
    def __init__(self, after_conv_shape, latent_dim=64):
        super().__init__()
        # after_conv_shape ~ (1, 64, H', W')
        _, out_ch, out_h, out_w = after_conv_shape
        self.fc = nn.Linear(latent_dim, out_ch * out_h * out_w)

        self.deconv_net = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        self.out_ch = out_ch
        self.out_h = out_h
        self.out_w = out_w

    def forward(self, z):
        B = z.size(0)
        out = self.fc(z)
        out = out.view(B, self.out_ch, self.out_h, self.out_w)
        return self.deconv_net(out)


class CAE(nn.Module):
    def __init__(self, input_shape=(3,64,64), latent_dim=64):
        super().__init__()
        self.encoder = ConvEncoder(input_shape, latent_dim)
        self.decoder = ConvDecoder(self.encoder.after_conv_shape, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


##############################
#   Train the CAE
##############################
def pretrain_cae(cae_model, train_loader, val_loader=None, epochs=10, lr=4e-3, device='cuda'):
    cae_model.to(device)
    optimizer = optim.Adam(cae_model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs+1):
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

        if val_loader is not None:
            cae_model.eval()
            val_loss_accum = 0.0
            with torch.no_grad():
                for vbatch in val_loader:
                    vbatch = vbatch.to(device)
                    vrecon, _ = cae_model(vbatch)
                    vloss = criterion(vrecon, vbatch)
                    val_loss_accum += vloss.item() * vbatch.size(0)
            val_loss = val_loss_accum / len(val_loader.dataset)
            print(f"[CAE] Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        else:
            print(f"[CAE] Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}")
    return cae_model


##############################
#     DEC Definition
##############################
class DEC(nn.Module):
    """
    Deep Embedded Clustering:
    - Takes a pretrained encoder
    - Learns cluster centers
    - Refines assignments via KL divergence
    """
    def __init__(self, encoder, n_clusters=10, alpha=1.0):
        super().__init__()
        self.encoder = encoder
        self.n_clusters = n_clusters
        self.alpha = alpha

        # If your encoder has .latent_dim:
        if hasattr(encoder, 'latent_dim'):
            latent_dim = encoder.latent_dim
        else:
            raise ValueError("Encoder must define a .latent_dim attribute.")
        # Cluster centers as learnable
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, latent_dim))

    @torch.no_grad()
    def initialize_centers(self, loader, device='cuda'):
        """Initialize cluster centers via K-means on the whole dataset's latent."""
        self.encoder.to(device)
        self.encoder.eval()
        all_z = []
        for batch in loader:
            x = batch.to(device)
            z = self.encoder(x)
            all_z.append(z.cpu().numpy())
        all_z = np.concatenate(all_z, axis=0)

        print("[DEC] K-means initialization...")
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
        preds = kmeans.fit_predict(all_z)
        self.cluster_centers.data = torch.from_numpy(kmeans.cluster_centers_).to(device)
        return preds

    def forward(self, x):
        z = self.encoder(x)
        q = self._student_t_distribution(z, self.cluster_centers, self.alpha)
        return z, q

    @staticmethod
    def _student_t_distribution(z, centers, alpha=1.0, eps=1e-10):
        B, latent_dim = z.size()
        K, _ = centers.size()
        # expand
        z_expand = z.unsqueeze(1).expand(B, K, latent_dim)
        c_expand = centers.unsqueeze(0).expand(B, K, latent_dim)
        dist_sq = torch.sum((z_expand - c_expand)**2, dim=2)
        numerator = (1.0 + dist_sq / alpha).pow(- (alpha + 1.0)/2)
        denominator = torch.sum(numerator, dim=1, keepdim=True) + eps
        return numerator / denominator

    @staticmethod
    def target_distribution(q, eps=1e-10):
        """
        p_{ij} = q_{ij}^2 / f_j, then normalize over j
        f_j = sum_i(q_{ij})
        """
        weight = q**2 / torch.sum(q, dim=0, keepdim=True)
        weight = weight + eps
        p = (weight.t() / torch.sum(weight, dim=1, keepdim=True).t()).t()
        return p


##############################
#   Revised DEC Training
##############################
def train_dec_full_pass(dec_model, train_loader, val_loader=None, epochs=10, device='cuda'):
    """
    Classic DEC approach:
      1) On each epoch, compute q for entire dataset in eval mode.
      2) Compute p (target distribution) from that entire q.
      3) Train in mini-batches, slicing the relevant portion of p for each batch.
    """
    for param in cae_model.decoder.parameters():
        param.requires_grad = False

    dec_model.to(device)
    optimizer = optim.Adam(dec_model.parameters(), lr=1e-4)
    loss_fn = nn.KLDivLoss(reduction='batchmean')

    train_data_size = len(train_loader.dataset)

    for epoch in range(1, epochs+1):
        # 1) Compute q for entire dataset
        dec_model.eval()
        all_q = []
        with torch.no_grad():
            for batch in train_loader:
                x = batch.to(device)
                _, q = dec_model(x)
                all_q.append(q.cpu())
        all_q = torch.cat(all_q, dim=0)  # shape (N, n_clusters)

        # 2) Compute p
        p = dec_model.target_distribution(all_q)
        # p also shape (N, n_clusters)

        # 3) Train in mini-batches, slicing p correspondingly
        dec_model.train()
        idx_offset = 0
        running_loss = 0.0
        for batch in train_loader:
            b_size = batch.size(0)
            x = batch.to(device)
            p_batch = p[idx_offset : idx_offset + b_size, :].to(device)
            idx_offset += b_size

            _, q_batch = dec_model(x)
            log_q = torch.log(q_batch + 1e-10)  # shape (b_size, n_clusters)
            # Ensure shapes match
            # p_batch is also (b_size, n_clusters)
            kl_loss = loss_fn(log_q, p_batch)

            optimizer.zero_grad()
            kl_loss.backward()
            optimizer.step()

            running_loss += kl_loss.item() * b_size

        epoch_loss = running_loss / train_data_size
        print(f"[DEC] Epoch {epoch}/{epochs}, KL Loss: {epoch_loss:.4f}")

        # Optional: measure cluster quality on val set (e.g., Silhouette)
        if val_loader is not None and epoch % 5 == 0:
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
            if len(np.unique(labels)) > 1:
                sscore = silhouette_score(z_vals, labels)
                print(f"   [Val] Silhouette Score: {sscore:.4f}")
            else:
                print("   [Val] Only one cluster => silhouette not defined.")

    return dec_model


##############################
#   Example Main
##############################
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="training_data/coeffs/")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--n_clusters", type=int, default=10)
    parser.add_argument("--epochs_cae", type=int, default=50)
    parser.add_argument("--epochs_dec", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # 1) Dataset
    dataset = EEGNpyDataset(args.data_dir)
    idxs = list(range(len(dataset)))
    # Simple train/val split
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(idxs, test_size=0.2, random_state=42)
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 2) CAE Pretraining
    sample_data = dataset[0]
    input_shape = sample_data.shape  # (C, H, W)
    cae_model = CAE(input_shape, args.latent_dim)
    cae_model = pretrain_cae(cae_model, train_loader, val_loader=val_loader,
                             epochs=args.epochs_cae, device=args.device)

    # 3) DEC Setup
    dec_model = DEC(encoder=cae_model.encoder, n_clusters=args.n_clusters, alpha=1.0)
    # Initialize cluster centers
    dec_model.initialize_centers(train_loader, device=args.device)

    # 4) DEC Training (two-pass each epoch)
    dec_model = train_dec_full_pass(dec_model, train_loader, val_loader,
                                    epochs=args.epochs_dec,
                                    device=args.device)

    # 5) Tokenize new data: cluster assignment
    dec_model.eval()
    tokens = []
    with torch.no_grad():
        for batch in val_loader:
            x = batch.to(args.device)
            z, q = dec_model(x)
            # discrete token is argmax
            cluster_ids = torch.argmax(q, dim=1).cpu().numpy()
            tokens.extend(cluster_ids)
    print("Sample tokens from val data:", tokens[:50])
