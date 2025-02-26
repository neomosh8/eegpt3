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


def generate_qa_plots(all_coeffs_unified, cae, gmm, min_val, max_val, device):
    """
    Generate quality assurance plots for latent space and autoencoder performance.

    Parameters:
    - all_coeffs_unified: List of 2D wavelet coefficient arrays (e.g., [(25, 512), ...])
    - cae: Trained Convolutional Autoencoder model
    - gmm: Trained Gaussian Mixture Model
    - min_val: Minimum value used for normalization during training
    - max_val: Maximum value used for normalization during training
    - device: PyTorch device (e.g., 'cuda' or 'cpu')
    """
    # Create QA folder if it doesn’t exist
    os.makedirs('QA', exist_ok=True)

    # Prepare the data: stack coefficients and normalize
    coeffs_array = np.stack(all_coeffs_unified, axis=0)  # Shape: (N, 25, 512)
    coeffs_array = coeffs_array[:, np.newaxis, :, :]  # Shape: (N, 1, 25, 512)
    coeffs_array = (coeffs_array - min_val) / (max_val - min_val)  # Normalize to [0,1]
    coeffs_tensor = torch.tensor(coeffs_array, dtype=torch.float32).to(device)

    # Generate latent representations using the encoder
    cae.eval()  # Set to evaluation mode
    with torch.no_grad():
        latent_reps = cae.encoder(coeffs_tensor).cpu().numpy()  # Shape: (N, 32)

    # Get GMM cluster labels
    labels = gmm.predict(latent_reps)

    # **Plot 1: PCA of Latent Space**
    pca = PCA(n_components=2)
    latent_2d_pca = pca.fit_transform(latent_reps)
    explained_var = pca.explained_variance_ratio_  # How much variance is explained

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d_pca[:, 0], latent_2d_pca[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(
        f'PCA of Latent Space with GMM Clusters\nExplained Variance: {explained_var[0]:.2f}, {explained_var[1]:.2f}')
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
        # Prepare original sample
        original = all_coeffs_unified[idx]  # Shape: (25, 512)
        original_norm = (original - min_val) / (max_val - min_val)  # Normalize
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
        plt.imshow(original_norm, aspect='auto', cmap='viridis', vmin=0, vmax=1)
        plt.title(f'Original Sample {i + 1} (Normalized)')
        plt.xlabel('Time')
        plt.ylabel('Frequency Scale')
        plt.colorbar()

        # Reconstructed heatmap
        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed, aspect='auto', cmap='viridis', vmin=0, vmax=1)
        plt.title(f'Reconstructed Sample {i + 1}\nMSE: {mse:.6f}')
        plt.xlabel('Time')
        plt.ylabel('Frequency Scale')
        plt.colorbar()

        plt.tight_layout()
        plt.savefig(f'QA/reconstruction_{i + 1}.png')
        plt.close()



def calculate_sps_from_df(df):
    time_col = df.get('timestamp', df.columns[0])
    dt = np.diff(time_col).mean()
    return 1.0 / dt if dt != 0 else 1.0



# Process CSV to collect 2D CWT coefficients
def process_csv_for_coeffs(csv_key, bucket, window_length_sec=2, num_samples_per_file=10):
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=csv_key)
    df = pd.read_csv(response['Body'])

    base_name = os.path.splitext(os.path.basename(csv_key))[0]
    all_columns = list(df.columns)
    instructions = call_gpt_for_instructions(channel_names=all_columns, dataset_id=base_name)

    if instructions.get("action") == "skip":
        print(f"Skipping dataset '{base_name}'.")
        return {"frontal": [], "motor_temporal": [], "parietal_occipital": []}

    channels_to_drop = instructions.get("channels_to_drop", [])
    print(f"Processing '{base_name}'. Dropping: {channels_to_drop}")

    regional_bipolar = create_regional_bipolar_channels(df, channels_to_drop)
    original_sps = calculate_sps_from_df(df)
    regional_preprocessed = {}
    new_sps_val = None
    for key, signal_array in regional_bipolar.items():
        signal_2d = signal_array[np.newaxis, :]
        preprocessed_signal, new_sps = preprocess_data(signal_2d, original_sps)
        regional_preprocessed[key] = preprocessed_signal[0, :]
        if new_sps_val is None:
            new_sps_val = new_sps

    all_data = np.concatenate([regional_preprocessed[region] for region in regional_preprocessed if len(regional_preprocessed[region]) > 0])
    if len(all_data) == 0:
        return {"frontal": [], "motor_temporal": [], "parietal_occipital": []}
    global_mean = np.mean(all_data)
    global_std = np.std(all_data) if np.std(all_data) > 0 else 1e-8
    for key in regional_preprocessed:
        regional_preprocessed[key] = (regional_preprocessed[key] - global_mean) / global_std

    min_length = min(len(regional_preprocessed[region]) for region in regional_preprocessed if len(regional_preprocessed[region]) > 0)
    if min_length == 0:
        return {"frontal": [], "motor_temporal": [], "parietal_occipital": []}
    n_window_samples = int(window_length_sec * new_sps_val)
    num_windows = min_length // n_window_samples

    selected_indices = np.random.choice(num_windows, min(num_samples_per_file, num_windows), replace=False)

    coeffs_dict = {"frontal": [], "motor_temporal": [], "parietal_occipital": []}
    for i in selected_indices:
        window_start = i * n_window_samples
        window_end = window_start + n_window_samples
        window_data = np.vstack([regional_preprocessed[region][window_start:window_end]
                                 for region in coeffs_dict if len(regional_preprocessed[region]) > 0])
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
async def collect_coeffs_from_s3_async(csv_files, bucket, num_samples_per_file=10, window_length_sec=2):
    all_coeffs = {"frontal": [], "motor_temporal": [], "parietal_occipital": []}
    async with aioboto3.Session().client("s3") as s3_client:
        tasks = [download_file(s3_client, bucket, csv_file) for csv_file in csv_files]
        results = await asyncio.gather(*tasks)
        csv_data = {key: pd.read_csv(io.BytesIO(body)) for key, body in results}

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_csv_for_coeffs, key, bucket, window_length_sec, num_samples_per_file)
                       for key in csv_data.keys()]
            for future in futures:
                coeffs = future.result()
                for region in all_coeffs:
                    all_coeffs[region].extend(coeffs[region])
    return all_coeffs

def collect_coeffs_from_s3(csv_files, bucket, num_samples_per_file=10, window_length_sec=2):
    return asyncio.run(collect_coeffs_from_s3_async(csv_files, bucket, num_samples_per_file, window_length_sec))


# Define the CAE model with corrected dimensions
class CAE(nn.Module):
    def __init__(self, input_shape, latent_dim):
        """
        input_shape: tuple (height, width) of each 2D coefficient sample (e.g., (25, 512))
        latent_dim: dimensionality of the latent representation
        """
        super(CAE, self).__init__()
        self.input_shape = input_shape  # e.g. (25, 512)
        # For the encoder, we use a pooling layer with ceil_mode=True so that odd dimensions are rounded up.
        encoder_height = int(np.ceil(self.input_shape[0] / 2))  # For 25, this becomes 13.
        encoder_width = self.input_shape[1] // 2  # For 512, this becomes 256.
        flattened_size = 4 * encoder_height * encoder_width  # 4 channels * 13 * 256

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # (B, 8, 25, 512)
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True),  # (B, 8, 13, 256)
            nn.Conv2d(8, 4, kernel_size=3, padding=1),  # (B, 4, 13, 256)
            nn.ReLU(),
            nn.Flatten(),  # (B, 4*13*256)
            nn.Linear(flattened_size, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, flattened_size),
            nn.Unflatten(1, (4, encoder_height, encoder_width)),  # (B, 4, 13, 256)
            nn.ConvTranspose2d(4, 8, kernel_size=3, padding=1),  # (B, 8, 13, 256)
            nn.ReLU(),
            # This transposed convolution upsamples with stride 2.
            # With kernel_size=3, padding=1, and output_padding=(0,1),
            # the height is recovered as: (13-1)*2 - 2*1 + 3 + 0 = 25
            # and the width as: (256-1)*2 - 2*1 + 3 + 1 = 512.
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


# Train CAE function
def train_cae(coeffs_2d_list, latent_dim=32, epochs=5, batch_size=32, output_folder="/tmp", region="unknown"):
    """
    Trains a convolutional autoencoder (CAE) on a list of 2D coefficient arrays.
    coeffs_2d_list: list of numpy arrays, each with shape (height, width) (e.g., (25,512))
    Returns the model file path, extracted encoder, and normalization values.
    """
    if not coeffs_2d_list or len(coeffs_2d_list) == 0:
        print(f"No data for CAE training in region '{region}'.")
        return None, None, None, None

    # Determine the input shape from the first sample (e.g., (25,512))
    input_shape = coeffs_2d_list[0].shape
    print(f"Region '{region}' - Input shape for CAE: {input_shape}")

    # Stack coefficients: (N, height, width)
    coeffs_array = np.stack(coeffs_2d_list, axis=0)
    # Add a channel dimension: (N, 1, height, width)
    coeffs_array = coeffs_array[:, np.newaxis, :, :]
    print(f"Region '{region}' - Input tensor shape before normalization: {coeffs_array.shape}")

    # Normalize the data to [0,1]
    min_val = coeffs_array.min()
    max_val = coeffs_array.max()
    if max_val > min_val:
        coeffs_array = (coeffs_array - min_val) / (max_val - min_val)

    # Convert to PyTorch tensor and move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coeffs_tensor = torch.tensor(coeffs_array, dtype=torch.float32).to(device)
    print(f"Region '{region}' - Final input tensor shape: {coeffs_tensor.shape}")

    # Initialize the CAE model
    cae = CAE(input_shape, latent_dim).to(device)

    # Create DataLoader
    dataset = TensorDataset(coeffs_tensor, coeffs_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define optimizer and loss function
    optimizer = optim.Adam(cae.parameters())
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        for inputs, _ in dataloader:
            optimizer.zero_grad()
            outputs, _ = cae(inputs)
            # If by any chance the output shape mismatches the input, crop them
            if outputs.shape != inputs.shape:
                min_h = min(outputs.shape[2], inputs.shape[2])
                min_w = min(outputs.shape[3], inputs.shape[3])
                inputs_cropped = inputs[:, :, :min_h, :min_w]
                outputs = outputs[:, :, :min_h, :min_w]
                loss = criterion(outputs, inputs_cropped)
            else:
                loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
        print(f"Region '{region}' - Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    # Save the trained model
    model_path = os.path.join(output_folder, f"cae_{region}.pt")
    torch.save(cae.state_dict(), model_path)

    # Extract the encoder portion for later use
    encoder = nn.Sequential(*list(cae.encoder.children()))
    return model_path, encoder, min_val, max_val
# Get latent representations
def get_latent_reps(encoder, coeffs_2d_list, min_val, max_val, device):
    # Stack coefficients to shape (N, 25, 512)
    coeffs_array = np.stack(coeffs_2d_list, axis=0)
    # Add the channel dimension at axis 1 to get (N, 1, 25, 512)
    coeffs_array = np.expand_dims(coeffs_array, axis=1)
    # Normalize the data
    if max_val > min_val:
        coeffs_array = (coeffs_array - min_val) / (max_val - min_val)
    # Convert to a PyTorch tensor and move to device
    coeffs_tensor = torch.tensor(coeffs_array, dtype=torch.float32).to(device)
    with torch.no_grad():
        latent_reps = encoder(coeffs_tensor).cpu().numpy()
    return latent_reps


# Train GMM on latent space
def train_gmm(latent_reps, max_components=10, output_folder="/tmp", region="unknown"):
    if len(latent_reps) < 2:
        print(f"Not enough data for GMM in region '{region}'.")
        return None, None, None

    best_n = 2
    best_bic = float('inf')
    for n in range(2, min(max_components + 1, len(latent_reps))):
        gmm = GaussianMixture(n_components=n, random_state=0)
        gmm.fit(latent_reps)
        bic = gmm.bic(latent_reps)
        print(f"Region '{region}' - GMM with {n} components - BIC: {bic:.2f}")
        if bic < best_bic:
            best_bic = bic
            best_n = n

    gmm = GaussianMixture(n_components=best_n, random_state=0)
    gmm.fit(latent_reps)
    labels = gmm.predict(latent_reps)
    print(f"Region '{region}' - Optimal GMM components: {best_n}")

    if len(set(labels)) > 1:
        silhouette = silhouette_score(latent_reps, labels)
        print(f"Region '{region}' - GMM Silhouette Score: {silhouette:.3f}")
    else:
        silhouette = None
        print(f"Region '{region}' - GMM Silhouette Score: N/A (single cluster)")

    model_path = os.path.join(output_folder, f"gmm_{region}.pkl")
    joblib.dump(gmm, model_path)
    return model_path, best_n, silhouette

# Main execution
if __name__ == "__main__":
    # Select random folders from S3
    all_folders = list_s3_folders()
    random.shuffle(all_folders)
    selected_folders = all_folders[:5]

    # Collect CSV files from selected folders
    csv_files = []
    for i, folder in enumerate(selected_folders):
        print(f"{i+1}/{len(selected_folders)}: Folder: {folder}")
        all_files = list_csv_files_in_folder(folder)
        selected_files = random.sample(all_files, min(10, len(all_files)))
        csv_files.extend(selected_files)
        print(f"Selected {len(selected_files)} files")

    print(f"Total files: {len(csv_files)}")

    # Collect wavelet coefficients from S3
    all_coeffs = collect_coeffs_from_s3(
        csv_files,
        "dataframes--use1-az6--x-s3",
        num_samples_per_file=250,
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
            min_val,
            max_val,
            device
        )
        print(f"Latent space shape: {latent_reps.shape}")

        # Step 4: Train a shared GMM
        gmm_path, n_components, silhouette = train_gmm(
            latent_reps,
            region="unified"  # For logging purposes
        )

        # Step 5: Generate QA plots if both models trained successfully
        if gmm_path is not None:
            # Load the full CAE model
            cae = CAE(input_shape=(25, 512), latent_dim=32)  # Adjust parameters as per your CAE definition
            cae.load_state_dict(torch.load(cae_path))
            cae.to(device)

            # Load the GMM model
            gmm = joblib.load(gmm_path)

            # Generate quality assurance plots
            generate_qa_plots(all_coeffs_unified, cae, gmm, min_val, max_val, device)

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