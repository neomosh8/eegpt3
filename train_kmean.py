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

# Define the CAE model in PyTorch
class CAE(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(CAE, self).__init__()
        self.input_shape = input_shape
        flattened_size = 4 * (self.input_shape[0] // 2) * (self.input_shape[1] // 2)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(flattened_size, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, flattened_size),
            nn.Unflatten(1, (4, self.input_shape[0] // 2, self.input_shape[1] // 2)),
            nn.ConvTranspose2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

# Train the CAE using PyTorch
def train_cae(coeffs_2d_list, latent_dim=32, epochs=5, batch_size=32, output_folder="/tmp", region="unknown"):
    if not coeffs_2d_list or len(coeffs_2d_list) == 0:
        print(f"No data for CAE training in region '{region}'.")
        return None, None, None, None

    # Set the input shape based on the first sample, e.g. (25, 512)
    input_shape = coeffs_2d_list[0].shape
    print(f"Region '{region}' - Input shape for CAE: {input_shape}")

    # Stack coefficients into an array of shape (N, 25, 512)
    coeffs_array = np.stack(coeffs_2d_list, axis=0)
    # Add a channel dimension: (N, 1, 25, 512)
    coeffs_array = coeffs_array[:, np.newaxis, :, :]
    print(f"Region '{region}' - Input tensor shape before normalization: {coeffs_array.shape}")

    # Normalize the data
    min_val = coeffs_array.min()
    max_val = coeffs_array.max()
    if max_val > min_val:
        coeffs_array = (coeffs_array - min_val) / (max_val - min_val)

    # Convert the numpy array to a PyTorch tensor and send to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coeffs_tensor = torch.tensor(coeffs_array, dtype=torch.float32).to(device)
    print(f"Region '{region}' - Final input tensor shape: {coeffs_tensor.shape}")

    # Initialize the CAE model with the correct input shape
    cae = CAE(input_shape, latent_dim).to(device)

    # Create a DataLoader for training
    dataset = TensorDataset(coeffs_tensor, coeffs_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Set up the optimizer and loss function
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
        print(f"Region '{region}' - Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Save the trained CAE model
    model_path = os.path.join(output_folder, f"cae_{region}.pt")
    torch.save(cae.state_dict(), model_path)

    # Extract the encoder part from the CAE
    encoder = nn.Sequential(*list(cae.encoder.children()))
    return model_path, encoder, min_val, max_val

# Get latent representations
def get_latent_reps(encoder, coeffs_2d_list, min_val, max_val, device):
    coeffs_array = np.stack(coeffs_2d_list, axis=0)[..., np.newaxis]
    coeffs_array = (coeffs_array - min_val) / (max_val - min_val) if max_val > min_val else coeffs_array
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
    all_folders = list_s3_folders()
    random.shuffle(all_folders)
    selected_folders = all_folders[:2]

    csv_files = []
    for i, folder in enumerate(selected_folders):
        print(f"{i+1}/{len(selected_folders)}: Folder: {folder}")
        all_files = list_csv_files_in_folder(folder)
        selected_files = random.sample(all_files, min(2, len(all_files)))
        csv_files.extend(selected_files)
        print(f"Selected {len(selected_files)} files")

    print(f"Total files: {len(csv_files)}")

    all_coeffs = collect_coeffs_from_s3(
        csv_files,
        "dataframes--use1-az6--x-s3",
        num_samples_per_file=100,
        window_length_sec=2
    )

    s3 = boto3.client("s3")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for region, coeffs_2d_list in all_coeffs.items():
        print(f"\n--- Region: {region} ---")
        if not coeffs_2d_list:
            print("No data. Skipping.")
            continue

        cae_path, encoder, min_val, max_val = train_cae(coeffs_2d_list, region=region)
        if cae_path is None:
            continue

        latent_reps = get_latent_reps(encoder, coeffs_2d_list, min_val, max_val, device)
        print(f"Latent space shape: {latent_reps.shape}")

        gmm_path, n_components, silhouette = train_gmm(latent_reps, region=region)

        if cae_path:
            s3.upload_file(cae_path, "dataframes--use1-az6--x-s3", f"cae_models/cae_{region}.pt")
            os.remove(cae_path)
        if gmm_path:
            s3.upload_file(gmm_path, "dataframes--use1-az6--x-s3", f"gmm_models/gmm_{region}.pkl")
            os.remove(gmm_path)