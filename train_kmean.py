import random
import numpy as np
import boto3
import pandas as pd
import asyncio
import aioboto3
from concurrent.futures import ProcessPoolExecutor
import os
import io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import joblib

from create_text_files_from_csv_2 import create_regional_bipolar_channels
from utils import call_gpt_for_instructions, preprocess_data, wavelet_decompose_window, list_csv_files_in_folder, \
    list_s3_folders

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import joblib


def generate_qa_plots(all_coeffs_unified, cae, gmm, device):
    """
    Generate quality assurance plots for latent space and autoencoder performance.

    Parameters:
    - all_coeffs_unified: List of 2D wavelet coefficient arrays (e.g., [(25, 512), ...])
    - cae: Trained Convolutional Autoencoder model
    - gmm: Trained Gaussian Mixture Model
    - device: PyTorch device (e.g., 'cuda' or 'cpu')
    """
    # Create QA folder if it doesn’t exist
    os.makedirs('QA', exist_ok=True)

    # Prepare the data: standardize each image individually
    standardized_coeffs = []
    for img in all_coeffs_unified:
        mean_img = np.mean(img)
        std_img = np.std(img) if np.std(img) > 0 else 1e-8  # Avoid division by zero
        standardized_img = (img - mean_img) / std_img
        standardized_coeffs.append(standardized_img)

    coeffs_array = np.stack(standardized_coeffs, axis=0)  # Shape: (N, 25, 512)
    coeffs_array = coeffs_array[:, np.newaxis, :, :]  # Shape: (N, 1, 25, 512)
    coeffs_tensor = torch.tensor(coeffs_array, dtype=torch.float32).to(device)

    # Generate latent representations using the encoder
    cae.eval()  # Set to evaluation mode
    with torch.no_grad():
        latent_reps = cae.encoder(coeffs_tensor).cpu().numpy()  # Shape: (N, latent_dim)

    # Get GMM cluster labels
    labels = gmm.predict(latent_reps)

    # **Plot 1: PCA of Latent Space**
    pca = PCA(n_components=2)
    latent_2d_pca = pca.fit_transform(latent_reps)
    explained_var = pca.explained_variance_ratio_

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d_pca[:, 0], latent_2d_pca[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(
        f'PCA of Latent Space with GMM Clusters\nExplained Variance: {explained_var[0]:.2f}, {explained_var[1]:.2f}'
    )
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig('QA/latent_space_pca.png')
    plt.close()

    # **Plot 2: t-SNE of Latent Space**
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d_tsne = tsne.fit_transform(latent_reps)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d_tsne[:, 0], latent_2d_tsne[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('t-SNE of Latent Space with GMM Clusters')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.savefig('QA/latent_space_tsne.png')
    plt.close()

    # **Plot 3: Encoder/Decoder Reconstructions**
    num_samples = 5
    indices = np.random.choice(len(all_coeffs_unified), num_samples, replace=False)

    for i, idx in enumerate(indices):
        # Prepare original sample with per-image standardization
        original = all_coeffs_unified[idx]  # Shape: (25, 512)
        mean_img = np.mean(original)
        std_img = np.std(original) if np.std(original) > 0 else 1e-8
        original_norm = (original - mean_img) / std_img  # Standardized version
        original_tensor = torch.tensor(original_norm[np.newaxis, np.newaxis, :, :], dtype=torch.float32).to(device)

        # Get reconstruction
        with torch.no_grad():
            reconstructed, _ = cae(original_tensor)
            reconstructed = reconstructed.cpu().numpy()[0, 0]  # Shape: (25, 512)

        # Calculate Mean Squared Error
        mse = np.mean((original_norm - reconstructed) ** 2)

        # Create side-by-side plot
        plt.figure(figsize=(12, 4))
        # Original heatmap
        plt.subplot(1, 2, 1)
        plt.imshow(original_norm, aspect='auto', cmap='viridis')
        plt.title(f'Original Sample {i + 1} (Standardized)')
        plt.xlabel('Time')
        plt.ylabel('Frequency Scale')
        plt.colorbar()

        # Reconstructed heatmap
        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed, aspect='auto', cmap='viridis')
        plt.title(f'Reconstructed Sample {i + 1}\nMSE: {mse:.6f}')
        plt.xlabel('Time')
        plt.ylabel('Frequency Scale')
        plt.colorbar()

        plt.tight_layout()
        plt.savefig(f'QA/reconstruction_{i + 1}.png')
        plt.close()


def calculate_sps_from_df(df):
    # Check if 'timestamp' exists in the DataFrame's columns
    if 'timestamp' in df.columns:
        time_col = df['timestamp']
    else:
        time_col = df[df.columns[0]]  # Use the first column's data
    # Convert to a numpy array (in case it's not already)
    time_col = np.array(time_col)
    dt = np.diff(time_col).mean()
    return 1.0 / dt if dt != 0 else 1.0



# Process CSV to collect 2D CWT coefficients
def process_csv_for_coeffs(csv_key, bucket, window_length_sec=2, num_samples_per_file=10, z_threshold=3.0):
    """
    Process a CSV file from S3, normalize per regional channel, window the signal, and reject artifacts.

    Args:
        csv_key (str): S3 key to the CSV file.
        bucket (str): S3 bucket name.
        window_length_sec (float): Length of each window in seconds.
        num_samples_per_file (int): Number of windows to sample after artifact rejection.
        z_threshold (float): Z-score threshold for artifact rejection.

    Returns:
        dict: Dictionary with regional keys ("frontal", "motor_temporal", "parietal_occipital")
              mapping to lists of CWT coefficients.
    """
    # Initialize S3 client and load CSV
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=csv_key)
    df = pd.read_csv(response['Body'])

    base_name = os.path.splitext(os.path.basename(csv_key))[0]
    all_columns = list(df.columns)
    instructions = call_gpt_for_instructions(channel_names=all_columns, dataset_id=base_name)

    # Skip if instructed
    if instructions.get("action") == "skip":
        print(f"Skipping dataset '{base_name}'.")
        return {"frontal": [], "motor_temporal": [], "parietal_occipital": []}

    channels_to_drop = instructions.get("channels_to_drop", [])
    print(f"Processing '{base_name}'. Dropping: {channels_to_drop}")

    # Create regional bipolar channels
    regional_bipolar = create_regional_bipolar_channels(df, channels_to_drop)
    original_sps = calculate_sps_from_df(df)
    regional_preprocessed = {}
    new_sps_val = None

    # Preprocess each regional channel
    for key, signal_array in regional_bipolar.items():
        signal_2d = signal_array[np.newaxis, :]
        preprocessed_signal, new_sps = preprocess_data(signal_2d, original_sps)
        regional_preprocessed[key] = preprocessed_signal[0, :]
        if new_sps_val is None:
            new_sps_val = new_sps

    # Step 1: Global Normalization per Regional Channel (Standardization)
    for key in regional_preprocessed:
        signal = regional_preprocessed[key]
        mean_signal = np.mean(signal)
        std_signal = np.std(signal) if np.std(signal) > 0 else 1e-8  # Avoid division by zero
        regional_preprocessed[key] = (signal - mean_signal) / std_signal

    # Check minimum length across regions
    min_length = min(len(regional_preprocessed[region]) for region in regional_preprocessed
                     if len(regional_preprocessed[region]) > 0)
    if min_length == 0:
        return {"frontal": [], "motor_temporal": [], "parietal_occipital": []}

    # Step 2: Windowing and Artifact Rejection
    n_window_samples = int(window_length_sec * new_sps_val)
    num_windows = min_length // n_window_samples

    # Calculate window-level statistics (e.g., mean) for Z-score computation
    window_stats = []
    for i in range(num_windows):
        window_start = i * n_window_samples
        window_end = window_start + n_window_samples
        window_data = np.vstack([regional_preprocessed[region][window_start:window_end]
                                 for region in regional_preprocessed
                                 if len(regional_preprocessed[region]) > 0])
        if window_data.size == 0:
            continue
        window_mean = np.mean(window_data)  # Summary statistic for the window
        window_stats.append(window_mean)

    # Compute Z-scores across all windows
    window_stats = np.array(window_stats)
    window_mu = np.mean(window_stats)
    window_sigma = np.std(window_stats) if np.std(window_stats) > 0 else 1e-8
    z_scores = (window_stats - window_mu) / window_sigma

    # Identify windows to keep (Z-score <= threshold)
    keep_indices = np.where(np.abs(z_scores) <= z_threshold)[0]
    discarded_count = num_windows - len(keep_indices)
    print(f"Discarded {discarded_count} windows out of {num_windows} due to artifact rejection (|Z| > {z_threshold}).")

    # Sample from retained windows
    selected_indices = np.random.choice(keep_indices, min(num_samples_per_file, len(keep_indices)), replace=False)

    # Step 3: Process retained windows into CWT coefficients
    coeffs_dict = {"frontal": [], "motor_temporal": [], "parietal_occipital": []}
    for i in selected_indices:
        window_start = i * n_window_samples
        window_end = window_start + n_window_samples
        window_data = np.vstack([regional_preprocessed[region][window_start:window_end]
                                 for region in coeffs_dict
                                 if len(regional_preprocessed[region]) > 0])
        if window_data.size == 0:
            continue
        decomposed_channels, _, _, _ = wavelet_decompose_window(
            window_data,
            wavelet='cmor1.5-1.0',
            scales=None,  # Adjust scales as needed
            normalization=False,
            sampling_period=1.0 / new_sps_val
        )
        for idx, region in enumerate(coeffs_dict.keys()):
            if idx < len(decomposed_channels):
                coeffs_2d = decomposed_channels[idx]  # Shape: (scales, time_points)
                coeffs_dict[region].append(coeffs_2d)

    return coeffs_dict
# Async helper to download files
async def download_file(s3_client, bucket, csv_key):
    response = await s3_client.get_object(Bucket=bucket, Key=csv_key)
    body = await response['Body'].read()
    return csv_key, body

# Collect coefficients from S3
# Collect coefficients from S3
async def collect_coeffs_from_s3_async(csv_files, bucket, num_samples_per_file=10, window_length_sec=2, z_threshold=3.0):
    all_coeffs = {"frontal": [], "motor_temporal": [], "parietal_occipital": []}
    async with aioboto3.Session().client("s3") as s3_client:
        tasks = [download_file(s3_client, bucket, csv_file) for csv_file in csv_files]
        results = await asyncio.gather(*tasks)
        csv_data = {key: pd.read_csv(io.BytesIO(body)) for key, body in results}

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_csv_for_coeffs, key, bucket, window_length_sec, num_samples_per_file, z_threshold)
                       for key in csv_data.keys()]
            for future in futures:
                coeffs = future.result()
                for region in all_coeffs:
                    all_coeffs[region].extend(coeffs[region])
    return all_coeffs

def collect_coeffs_from_s3(csv_files, bucket, num_samples_per_file=10, window_length_sec=2, z_threshold=3.0):
    return asyncio.run(collect_coeffs_from_s3_async(csv_files, bucket, num_samples_per_file, window_length_sec, z_threshold))
# Define the CAE model with corrected dimensions
import torch
import torch.nn as nn

class CAE(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(CAE, self).__init__()
        self.input_shape = input_shape

        # Define convolutional part of the encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True)
        )

        # Calculate the size after convolution for the fully connected layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *input_shape)
            conv_output = self.encoder_conv(dummy_input)
            self.encoder_height = conv_output.shape[2]
            self.encoder_width = conv_output.shape[3]
            flattened_size = 16 * self.encoder_height * self.encoder_width

        # Define fully connected part
        self.encoder_fc = nn.Linear(flattened_size, latent_dim)

        # Combine into a single encoder module
        self.encoder = nn.Sequential(
            self.encoder_conv,
            nn.Flatten(),  # Flatten the output of convolutions
            self.encoder_fc
        )

        # Define decoder (example)
        self.decoder_fc = nn.Linear(latent_dim, flattened_size)
        self.decoder_conv = nn.Sequential(
            nn.Unflatten(1, (16, self.encoder_height, self.encoder_width)),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=(0, 1))
        )

    def forward(self, x):
        # Pass input through encoder
        encoded = self.encoder(x)
        # Decode from the latent representation
        x = self.decoder_fc(encoded)
        x = self.decoder_conv(x)
        return x, encoded  # Return reconstructed output and encoded representation


def train_cae(coeffs_2d_list, latent_dim=64, epochs=100, batch_size=32, output_folder="/tmp", region="unknown"):
    """
    Train a Convolutional Autoencoder on 2D CWT coefficients with per-image standardization.

    Args:
        coeffs_2d_list (list): List of 2D CWT coefficient arrays.
        latent_dim (int): Size of the latent representation.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        output_folder (str): Folder to save the model.
        region (str): Region name for logging.

    Returns:
        tuple: (model_path, encoder, None, None)
    """
    if not coeffs_2d_list or len(coeffs_2d_list) == 0:
        print(f"No data for CAE training in region '{region}'.")
        return None, None, None, None

    input_shape = coeffs_2d_list[0].shape
    print(f"Region '{region}' - Input shape for CAE: {input_shape}")

    # Step: Standardize each 2D CWT image individually
    standardized_coeffs = []
    for img in coeffs_2d_list:
        mean_img = np.mean(img)
        std_img = np.std(img) if np.std(img) > 0 else 1e-8  # Avoid division by zero
        standardized_img = (img - mean_img) / std_img
        standardized_coeffs.append(standardized_img)

    # Stack standardized images and add channel dimension
    coeffs_array = np.stack(standardized_coeffs, axis=0)
    coeffs_array = coeffs_array[:, np.newaxis, :, :]  # Shape: (samples, 1, height, width)

    # Convert to tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coeffs_tensor = torch.tensor(coeffs_array, dtype=torch.float32).to(device)

    # Initialize CAE
    cae = CAE(input_shape, latent_dim).to(device)
    dataset = TensorDataset(coeffs_tensor, coeffs_tensor)  # Input and target are the same
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training setup
    optimizer = optim.Adam(cae.parameters())
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        for inputs, _ in dataloader:
            optimizer.zero_grad()
            outputs, _ = cae(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
        print(f"Region '{region}' - Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    # Save model
    model_path = os.path.join(output_folder, f"cae_{region}.pt")
    torch.save(cae.state_dict(), model_path)
    encoder = cae.encoder  # This now works!
    return model_path, encoder, None, None
# Get latent representations
def get_latent_reps(encoder, coeffs_2d_list, device):
    """
    Generate latent representations from 2D CWT coefficients using the trained encoder.

    Args:
        encoder (nn.Module): Trained CAE encoder.
        coeffs_2d_list (list): List of 2D CWT coefficient arrays.
        device (torch.device): Device to perform computation on.

    Returns:
        np.ndarray: Latent representations.
    """
    # Standardize each 2D CWT image individually
    standardized_coeffs = []
    for img in coeffs_2d_list:
        mean_img = np.mean(img)
        std_img = np.std(img) if np.std(img) > 0 else 1e-8  # Avoid division by zero
        standardized_img = (img - mean_img) / std_img
        standardized_coeffs.append(standardized_img)

    # Stack and add channel dimension
    coeffs_array = np.stack(standardized_coeffs, axis=0)
    coeffs_array = coeffs_array[:, np.newaxis, :, :]  # Shape: (samples, 1, height, width)

    # Convert to tensor and compute latent representations
    coeffs_tensor = torch.tensor(coeffs_array, dtype=torch.float32).to(device)
    with torch.no_grad():
        latent_reps = encoder(coeffs_tensor).cpu().numpy()

    return latent_reps


import os
import numpy as np
import joblib
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def train_gmm(latent_reps, max_components=100, output_folder="/tmp", region="unknown", use_gpu=False):
    if len(latent_reps) < 2:
        print(f"Not enough data for GMM in region '{region}'.")
        return None, None, None

    if use_gpu:
        try:
            from cuml import GaussianMixture as cuGMM
            import cudf
            latent_reps_gpu = cudf.DataFrame(latent_reps)
            gmm = cuGMM(n_components=max_components, random_state=0)
            gmm.fit(latent_reps_gpu)
            labels = gmm.predict(latent_reps_gpu)
            effective_components = max_components  # Adjust based on BIC if needed
        except ImportError:
            print("cuML not installed; falling back to CPU.")
            use_gpu = False

    if not use_gpu:
        bgmm = BayesianGaussianMixture(
            n_components=max_components,
            weight_concentration_prior=1e-2,
            n_init=5,
            random_state=0
        )
        bgmm.fit(latent_reps)
        weights = bgmm.weights_
        effective_components = sum(weights > 1e-6)
        labels = bgmm.predict(latent_reps)
        print(f"Region '{region}' - Effective GMM components: {effective_components}")

    if len(set(labels)) > 1:
        silhouette = silhouette_score(latent_reps, labels)
        ch_score = calinski_harabasz_score(latent_reps, labels)
        db_score = davies_bouldin_score(latent_reps, labels)
        print(f"Region '{region}' - GMM Silhouette Score: {silhouette:.3f}")
        print(f"Region '{region}' - CH Score: {ch_score:.2f}, DB Score: {db_score:.2f}")
    else:
        silhouette = None
        print(f"Region '{region}' - GMM Silhouette Score: N/A (single cluster)")

    model_path = os.path.join(output_folder, f"gmm_{region}.pkl")
    joblib.dump(bgmm if not use_gpu else gmm, model_path)
    return model_path, effective_components, silhouette

# Main execution
if __name__ == "__main__":
    # Select random folders from S3
    all_folders = list_s3_folders()
    random.shuffle(all_folders)
    selected_folders = all_folders[:10]
    # selected_folders = ['ds002336']

    # Collect CSV files from selected folders
    csv_files = []
    for i, folder in enumerate(selected_folders):
        print(f"{i+1}/{len(selected_folders)}: Folder: {folder}")
        all_files = list_csv_files_in_folder(folder)
        selected_files = random.sample(all_files, min(5, len(all_files)))
        csv_files.extend(selected_files)
        print(f"Selected {len(selected_files)} files")

    print(f"Total files: {len(csv_files)}")

    # Collect wavelet coefficients from S3
    all_coeffs = collect_coeffs_from_s3(
        csv_files,
        "dataframes--use1-az6--x-s3",
        num_samples_per_file=400,
        window_length_sec=2
    )

    # Step 1: Combine coefficients from all regions into one list
    all_coeffs_unified = []
    for region in all_coeffs:  # e.g., ["frontal", "motor_temporal", "parietal_occipital"]
        all_coeffs_unified.extend(all_coeffs[region])
    print(f"Total coefficients across all regions: {len(all_coeffs_unified)}")

    # Set up S3 client and device
    s3 = boto3.client("s3")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 2: Train a unified CAE
    cae_path, encoder, min_val, max_val = train_cae(
        all_coeffs_unified,
        region="unified"  # For logging purposes
    )

    if cae_path is not None:
        # Step 3: Get latent representations
        latent_reps = get_latent_reps(
            encoder,
            all_coeffs_unified,
            device
        )
        print(f"Latent space shape: {latent_reps.shape}")

        gmm_path, n_components, silhouette = train_gmm(
            latent_reps,
            region="unified",  # For logging purposes
        )

        # Step 5: Generate QA plots if both models trained successfully
        if gmm_path is not None:
            # Load the full CAE model
            cae = CAE(input_shape=(25, 512), latent_dim=64)  # Adjust parameters as per your CAE definition
            cae.load_state_dict(torch.load(cae_path))
            cae.to(device)

            # Load the GMM model
            gmm = joblib.load(gmm_path)

            # Generate quality assurance plots
            generate_qa_plots(all_coeffs_unified, cae, gmm, device)

        # Step 6: Upload models to S3 and clean up
        if cae_path:
            s3.upload_file(cae_path, "dataframes--use1-az6--x-s3", "cae_models/cae_unified.pt")
            os.remove(cae_path)
            print("Uploaded unified CAE model")
        if gmm_path:
            s3.upload_file(gmm_path, "dataframes--use1-az6--x-s3", "gmm_models/gmm_unified.pkl")
            os.remove(gmm_path)
            print("Uploaded unified GMM model")
    else:
        print("CAE training failed, skipping GMM training")