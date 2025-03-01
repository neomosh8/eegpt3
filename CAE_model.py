import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import torch
from sklearn.mixture import BayesianGaussianMixture
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
class CAE(nn.Module):
    def __init__(self, input_shape, latent_dim=128):
        """
        Convolutional AutoEncoder.

        Args:
            input_shape (tuple): Shape of the input image as (C, H, W).
                                 For our RGB wavelet images, C should be 3.
            latent_dim (int): Dimension of the latent (bottleneck) vector.
        """
        super(CAE, self).__init__()
        self.input_shape = input_shape  # (C, H, W)
        C, H, W = input_shape

        # -------- Encoder --------
        self.encoder = nn.Sequential(
            nn.Conv2d(C, 16, kernel_size=3, stride=2, padding=1),  # (16, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (32, H/4, W/4)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (64, H/8, W/8)
            nn.ReLU()
        )
        # Determine flattened size after convolutional layers
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            enc_out = self.encoder(dummy)
            self.enc_shape = enc_out.shape  # e.g., (1, 64, H_enc, W_enc)
            self.flattened_size = enc_out.view(1, -1).shape[1]

        # Bottleneck fully-connected layers
        self.fc1 = nn.Linear(self.flattened_size, latent_dim)
        self.fc2 = nn.Linear(latent_dim, self.flattened_size)

        # -------- Decoder --------
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (32, H/4, W/4)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # (16, H/2, W/2)
            nn.ReLU(),
            nn.ConvTranspose2d(16, C, kernel_size=3, stride=2, padding=1, output_padding=1),  # (C, H, W)
            nn.Sigmoid()  # Assuming the images are normalized between 0 and 1.
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input batch tensor of shape (batch_size, C, H, W).

        Returns:
            reconstruction (torch.Tensor): Reconstructed images.
            latent (torch.Tensor): Latent representations.
        """
        # Encoder
        encoded = self.encoder(x)
        encoded_flat = encoded.view(x.size(0), -1)
        latent = self.fc1(encoded_flat)
        # Decoder
        decoded_flat = self.fc2(latent)
        decoded = decoded_flat.view(x.size(0), *self.enc_shape[1:])
        reconstruction = self.decoder(decoded)
        return reconstruction, latent


def train_cae(model, dataloader, epochs=20, lr=1e-3, device=torch.device("cpu")):
    """
    Trains the CAE model on the given dataloader.

    Args:
        model (nn.Module): The CAE model.
        dataloader (DataLoader): PyTorch DataLoader yielding batches of images.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        device (torch.device): Device on which to train.

    Returns:
        model (nn.Module): Trained CAE model.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            batch = batch.float().to(device)
            optimizer.zero_grad()
            reconstruction, _ = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataloader.dataset)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")
    return model


# Function to extract latent representations from the trained model
def get_latent_space(model, dataloader, device=torch.device("cpu")):
    model.eval()
    latent_list = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.float().to(device)
            _, latent = model(batch)
            latent_list.append(latent.cpu())
    return torch.cat(latent_list, dim=0).numpy()


class NpyDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.file_list = []
        files = [f for f in os.listdir(directory) if f.endswith('.npy')]
        self.image_shape = None

        # Loop over files to set the expected shape and filter out mismatches
        for f in files:
            file_path = os.path.join(directory, f)
            try:
                image = np.load(file_path)
                # If we haven't set the expected shape yet, do it now.
                if self.image_shape is None:
                    self.image_shape = image.shape
                # Check if the current file's shape matches the expected shape.
                if image.shape == self.image_shape:
                    self.file_list.append(f)
                else:
                    print(f"Skipping file {file_path} due to shape mismatch. Expected {self.image_shape}, got {image.shape}")
            except Exception as e:
                print(f"Skipping file {file_path} due to error: {e}")

    def __len__(self):
        # Return the total number of images
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.file_list[idx])
        try:
            image = np.load(file_path)
        except EOFError:
            print(f"Skipping file {file_path} due to EOFError")
            # Option 1: Return a default image (e.g., an array of zeros)
            image = np.zeros(self.image_shape) if self.image_shape is not None else None
            # Option 2: You could also raise a custom exception or handle it in a way that fits your training loop.
        return torch.from_numpy(image)

    def get_image_shape(self):
        """
        Returns the shape of the images in the dataset as a tuple (C, H, W).
        Assumes all images have the same shape.

        Returns:
            tuple: Shape of the images in the format (C, H, W), where
                   C is the number of channels, H is the height, and W is the width.

        Raises:
            ValueError: If no images are found in the directory.
        """
        if self.image_shape is None:
            raise ValueError("No images found in the directory.")
        return self.image_shape


def train_dp_gmm(latent_data, n_components=10):
    """
    Trains a Dirichlet Process Gaussian Mixture Model on the provided latent space.

    Args:
        latent_data (np.ndarray): Array of shape (n_samples, latent_dim).
        n_components (int): The maximum number of mixture components.

    Returns:
        dp_gmm (BayesianGaussianMixture): Trained DP-GMM.
    """
    dp_gmm = BayesianGaussianMixture(
        n_components=n_components,
        covariance_type='full',
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=1e-2,  # Adjust based on your dataset
        max_iter=10000,
        random_state=42,
    )
    dp_gmm.fit(latent_data)
    return dp_gmm

if __name__ == "__main__":

    # Set the training data directory
    training_data_directory = "training_data/coeffs/"

    # Create the dataset
    dataset = NpyDataset(training_data_directory)

    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Define input shape (C, H, W) from the dataset
    try:
        C, H, W = dataset.get_image_shape()
        input_shape = (C, H, W)
        print(f"Input shape: {input_shape}")
    except ValueError as e:
        print(f"Error: {e}")

    # Initialize and train the CAE
    cae_model = CAE(input_shape, latent_dim=32)
    device = torch.device("cuda" if torch.cuda.is_available() else  "mps" if torch.mps.is_available() else "cpu")
    cae_model = train_cae(cae_model, dataloader, epochs=100, lr=3e-4, device=device)

    # After training, extract the latent representations
    latent_reps = get_latent_space(cae_model, dataloader, device=device)
    dp_gmm = train_dp_gmm(latent_reps, n_components=400)

    print("Latent space shape:", latent_reps.shape)
    # Optionally, predict cluster labels for the latent representations
    cluster_labels = dp_gmm.predict(latent_reps)
    print("Cluster labels:", cluster_labels)

    # Calculate effective number of clusters used by DP-GMM
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    print("Unique clusters:", unique_clusters)
    print("Samples per cluster:", counts)

    # Compute Silhouette Score: higher is better (range -1 to 1)
    sil_score = silhouette_score(latent_reps, cluster_labels)
    print("Silhouette Score:", sil_score)

    # Compute Davies-Bouldin Index: lower values indicate better clustering
    db_index = davies_bouldin_score(latent_reps, cluster_labels)
    print("Davies-Bouldin Index:", db_index)

    # -------------------------
    # Plotting the Silhouette Analysis
    # -------------------------
    n_clusters = len(unique_clusters)
    sample_silhouette_values = silhouette_samples(latent_reps, cluster_labels)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_lower = 10  # starting position for the first silhouette plot

    for i in range(n_clusters):
        cluster_silhouette_vals = sample_silhouette_values[cluster_labels == unique_clusters[i]]
        cluster_silhouette_vals.sort()

        size_cluster_i = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                     0, cluster_silhouette_vals,
                     facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(unique_clusters[i]))

        y_lower = y_upper + 10  # gap between clusters

    ax.set_title("Silhouette Plot for DP-GMM Clusters")
    ax.set_xlabel("Silhouette Coefficient Values")
    ax.set_ylabel("Cluster Label")
    ax.axvline(x=sil_score, color="red", linestyle="--", label=f"Average Silhouette Score: {sil_score:.2f}")
    ax.legend()
    plt.savefig("QA/Silhouette.png")
    plt.close()

    # -------------------------
    # Visualizing the Latent Space using t-SNE
    # -------------------------
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_reps)

    plt.figure(figsize=(8, 6))
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], marker='o', s=50, lw=0, alpha=0.7, c=colors)
    plt.title("t-SNE Projection of Latent Space Colored by DP-GMM Clusters")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.savefig("QA/latent_space_tsne.png")
    plt.close()