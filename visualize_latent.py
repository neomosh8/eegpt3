#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse


# =========================
#   Dataset Definition
# =========================
class EEGNpyDataset(Dataset):
    """
    Example dataset that reads .npy files (C, H, W) from a directory.
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
#      CAE Definition
# =========================
class ConvEncoder(nn.Module):
    def __init__(self, input_shape=(3, 64, 64), latent_dim=64):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        C, H, W = input_shape
        self.conv_net = nn.Sequential(
            nn.Conv2d(C, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        # Determine shape after convs
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            out = self.conv_net(dummy)
            self.after_conv_shape = out.shape  # (1, out_channels, H', W')
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
        _, out_channels, out_h, out_w = after_conv_shape
        self.fc = nn.Linear(latent_dim, out_channels * out_h * out_w)
        self.deconv_net = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1), nn.Tanh()
        )
        self.out_channels = out_channels
        self.out_h = out_h
        self.out_w = out_w

    def forward(self, z):
        B = z.size(0)
        out = self.fc(z)
        out = out.view(B, self.out_channels, self.out_h, self.out_w)
        x_recon = self.deconv_net(out)
        return x_recon


class CAE(nn.Module):
    def __init__(self, input_shape=(3, 64, 64), latent_dim=64):
        super().__init__()
        self.encoder = ConvEncoder(input_shape, latent_dim)
        self.decoder = ConvDecoder(self.encoder.after_conv_shape, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


# =========================
#   Visualization Function
# =========================
def visualize_latent_space(cae_model, data_loader, device='cuda', max_samples=1000, out_path='latent_space_pca.png'):
    """
    Extracts latent vectors from the CAE encoder, performs PCA to reduce them to 2D, and saves a scatter plot.

    Args:
        cae_model (nn.Module): Trained CAE model.
        data_loader (DataLoader): DataLoader for your dataset.
        device (str): Device to run the model on.
        max_samples (int): Maximum number of samples to visualize.
        out_path (str): File path to save the output plot.
    """
    cae_model.eval()  # set model to evaluation mode
    latent_list = []
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            # Get latent representations from the CAE.
            _, latent = cae_model(batch)
            latent_np = latent.cpu().numpy()
            latent_list.append(latent_np)
            total_samples += latent_np.shape[0]
            if total_samples >= max_samples:
                break

    latents = np.concatenate(latent_list, axis=0)
    if latents.shape[0] > max_samples:
        indices = np.random.choice(latents.shape[0], max_samples, replace=False)
        latents = latents[indices]

    # Reduce latent dimensions to 2D using PCA.
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latents)

    # Create a scatter plot of the 2D latent space.
    plt.figure(figsize=(8, 6))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.7)
    plt.title("Latent Space Visualization (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.savefig(out_path)
    plt.close()
    print(f"Latent space visualization saved to {out_path}")


# =========================
#   Helper to Load Model
# =========================
def load_pretrained_cae(model, model_path, device='cuda'):
    """
    Loads a pretrained CAE model state from the given file path.
    """
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded pretrained model from {model_path}")
    else:
        print(f"Model file {model_path} not found. Please check the path.")
    return model


# =========================
#        Main Script
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize CAE Latent Space with PCA")
    parser.add_argument("--data_dir", type=str, default="training_data/coeffs/",
                        help="Directory containing .npy files")
    parser.add_argument("--model_path", type=str, default="QA/DEC/cae_model.pt",
                        help="Path to the pretrained CAE model")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for DataLoader")
    parser.add_argument("--latent_dim", type=int, default=512,
                        help="Latent dimension of the CAE")
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Maximum number of samples for visualization")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--out_path", type=str, default="latent_space_pca.png",
                        help="Output file path for the latent space plot")
    args = parser.parse_args()

    # Set device.
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Create dataset and DataLoader.
    dataset = EEGNpyDataset(args.data_dir, normalize=True)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Get a sample to determine the input shape.
    sample = dataset[0]
    input_shape = sample.shape  # e.g., (C, H, W)

    # Initialize and load the CAE model.
    cae_model = CAE(input_shape=input_shape, latent_dim=args.latent_dim)
    cae_model = cae_model.to(device)
    cae_model = load_pretrained_cae(cae_model, args.model_path, device=device)

    # Visualize the latent space.
    visualize_latent_space(cae_model, data_loader, device=device,
                           max_samples=args.max_samples, out_path=args.out_path)
