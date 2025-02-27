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
from botocore.exceptions import ClientError
from torch.utils.data import DataLoader, Dataset
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from create_text_files_from_csv_2 import create_regional_bipolar_channels
from utils import call_gpt_for_instructions, preprocess_data, wavelet_decompose_window, list_csv_files_in_folder, \
    list_s3_folders, calculate_sps


# Custom Dataset for loading coefficients from disk
class CoeffsDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = np.load(self.file_paths[idx])
        # Standardize per image
        mean_img = np.mean(img)
        std_img = np.std(img) if np.std(img) > 0 else 1e-8
        standardized_img = (img - mean_img) / std_img
        # Add channel dimension
        standardized_img = standardized_img[np.newaxis, :, :]
        return torch.tensor(standardized_img, dtype=torch.float32)

# Process CSV and save coefficients to disk
def process_csv_for_coeffs(df, csv_key, window_length_sec=2, num_samples_per_file=10, z_threshold=3.0):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    base_name = os.path.splitext(os.path.basename(csv_key))[0]
    all_columns = list(df.columns)

    # Calculate original sampling rate before dropping any columns
    original_sps = calculate_sps(df)

    # Get instructions from GPT
    instructions = call_gpt_for_instructions(channel_names=all_columns, dataset_id=base_name)

    if instructions.get("action") == "skip":
        print(f"Skipping dataset '{base_name}'.")
        return []

    channels_to_drop = instructions.get("channels_to_drop", [])
    print(f"Processing '{base_name}'. Dropping: {channels_to_drop}")

    # Step 1: Drop the specified channels from the DataFrame
    df = df.drop(columns=channels_to_drop, errors='ignore')

    # Identify EEG channels (excluding timestamp columns)
    eeg_channels = [col for col in df.columns if col not in ['timestamp', 'TimeStamp']]

    # Extract signal data as a 2D array (channels x samples)
    signal_data = df[eeg_channels].values.T  # Shape: (num_channels, num_samples)

    # Step 2: Preprocess the entire file
    preprocessed_signal, new_sps = preprocess_data(signal_data, original_sps)

    # Create a preprocessed DataFrame to maintain channel names
    preprocessed_df = pd.DataFrame(preprocessed_signal.T, columns=eeg_channels)

    # Step 3: Create regional bipolar channels from the preprocessed data
    regional_bipolar = create_regional_bipolar_channels(preprocessed_df)

    # Step 4: Normalize each regional bipolar signal
    for key in regional_bipolar:
        signal = regional_bipolar[key]
        mean_signal = np.mean(signal)
        std_signal = np.std(signal) if np.std(signal) > 0 else 1e-8
        regional_bipolar[key] = (signal - mean_signal) / std_signal

    # Continue with the remaining processing
    min_length = min(len(regional_bipolar[region]) for region in regional_bipolar if len(regional_bipolar[region]) > 0)
    if min_length == 0:
        return []

    n_window_samples = int(window_length_sec * new_sps)
    num_windows = min_length // n_window_samples

    window_stats = []
    for i in range(num_windows):
        window_start = i * n_window_samples
        window_end = window_start + n_window_samples
        window_data = np.vstack([regional_bipolar[region][window_start:window_end] for region in regional_bipolar if
                                 len(regional_bipolar[region]) > 0])
        if window_data.size == 0:
            continue
        window_mean = np.mean(window_data)
        window_stats.append(window_mean)

    window_stats = np.array(window_stats)
    window_mu = np.mean(window_stats)
    window_sigma = np.std(window_stats) if np.std(window_stats) > 0 else 1e-8
    z_scores = (window_stats - window_mu) / window_sigma

    keep_indices = np.where(np.abs(z_scores) <= z_threshold)[0]
    rejected_indices = np.where(np.abs(z_scores) > z_threshold)[0]
    discarded_count = len(rejected_indices)
    print(f"Discarded {discarded_count} windows out of {num_windows} due to artifact rejection (|Z| > {z_threshold}).")

    # Plotting three channels with rejected windows highlighted
    regions = ["frontal", "motor_temporal", "parietal_occipital"]
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    time = np.arange(min_length) / new_sps
    for ax, region in zip(axes, regions):
        if region in regional_bipolar:
            signal = regional_bipolar[region]
            ax.plot(time, signal, label=region)
            ax.set_ylabel(region)
            for i in rejected_indices:
                start_time = (i * n_window_samples) / new_sps
                end_time = ((i + 1) * n_window_samples) / new_sps
                ax.axvspan(start_time, end_time, color='red', alpha=0.3)
        else:
            ax.set_visible(False)
    axes[0].set_title(f"EEG Signals with Rejected Windows - {base_name}")
    axes[2].set_xlabel("Time (s)")
    plt.tight_layout()
    safe_csv_key = csv_key.replace('/', '_').replace('.', '_')
    plot_path = os.path.join('QA', f"{safe_csv_key}_rejected_windows.png")
    os.makedirs('QA', exist_ok=True)
    plt.savefig(plot_path)
    plt.close()

    selected_indices = np.random.choice(keep_indices, min(num_samples_per_file, len(keep_indices)), replace=False)

    saved_files = []
    for i in selected_indices:
        window_start = i * n_window_samples
        window_end = window_start + n_window_samples
        window_data = np.vstack([regional_bipolar[region][window_start:window_end] for region in regional_bipolar if
                                 len(regional_bipolar[region]) > 0])
        if window_data.size == 0:
            continue
        decomposed_channels, _, _, _ = wavelet_decompose_window(
            window_data,
            wavelet='cmor1.5-1.0',
            scales=None,
            normalization=False,  # Already normalized earlier
            sampling_period=1.0 / new_sps
        )
        for idx, region in enumerate(["frontal", "motor_temporal", "parietal_occipital"]):
            if idx < len(decomposed_channels):
                coeffs_2d = decomposed_channels[idx]
                for i in range(coeffs_2d.shape[0]):
                    mean_scale = np.mean(coeffs_2d[i, :])
                    std_scale = np.std(coeffs_2d[i, :]) if np.std(coeffs_2d[i, :]) > 0 else 1e-8
                    coeffs_2d[i, :] = (coeffs_2d[i, :] - mean_scale) / std_scale
                safe_csv_key = csv_key.replace('/', '_').replace('.', '_')
                file_name = f"{safe_csv_key}_{region}_window_{i}.npy"
                file_path = os.path.join('training_data', 'coeffs', file_name)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                np.save(file_path, coeffs_2d)
                saved_files.append(file_path)
    return saved_files

# Async helper to download files
async def download_file(s3_client, bucket, csv_key, max_retries=3):
    tmp_path = os.path.join("/dev/shm", f"{csv_key.replace('/', '_')}.csv")
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    for attempt in range(max_retries):
        try:
            response = await s3_client.get_object(Bucket=bucket, Key=csv_key)
            body = await response["Body"].read()
            with open(tmp_path, "wb") as f:
                f.write(body)
            return csv_key, tmp_path
        except ClientError as e:
            if e.response["Error"]["Code"] == "AccessDenied":
                print(f"Access Denied for {csv_key} (Attempt {attempt + 1}/{max_retries})")
                if attempt + 1 == max_retries:
                    print(f"Failed to download {csv_key} after {max_retries} attempts: {e}")
                    return csv_key, None
            else:
                print(f"Error downloading {csv_key}: {e}")
                return csv_key, None
            await asyncio.sleep(1)  # Brief delay before retry
    return csv_key, None# Collect coefficients from S3 and save to disk


async def collect_coeffs_from_s3_async(csv_files, bucket, num_samples_per_file=10, window_length_sec=2, z_threshold=3.0,
                                       batch_size=10):
    all_saved_files = []
    total_files = len(csv_files)
    downloaded_files = 0
    processed_files = 0

    async with aioboto3.Session().client("s3") as s3_client:
        for i in range(0, total_files, batch_size):
            batch = csv_files[i:i + batch_size]
            # Download batch concurrently
            tasks = [download_file(s3_client, bucket, csv_file) for csv_file in batch]
            results = await asyncio.gather(*tasks)

            # Prepare downloaded files for processing
            csv_data = {}
            tmp_paths = []
            for key, tmp_path in results:
                if tmp_path is not None:
                    csv_data[key] = pd.read_csv(tmp_path)
                    tmp_paths.append(tmp_path)
                    downloaded_files += 1
                else:
                    print(f"Skipping {key} due to download failure.")
            print(f"Downloaded {downloaded_files}/{total_files} files")

            # Process downloaded files in parallel
            try:
                with ProcessPoolExecutor() as executor:
                    futures = [
                        executor.submit(
                            process_csv_for_coeffs,
                            csv_data[key],
                            key,
                            window_length_sec,
                            num_samples_per_file,
                            z_threshold
                        )
                        for key in csv_data.keys()
                    ]
                    batch_processed = 0
                    for future in futures:
                        try:
                            saved_files = future.result()
                            all_saved_files.extend(saved_files)
                            batch_processed += 1
                        except Exception as e:
                            print(f"Error processing file: {e}")
                    processed_files += batch_processed
                    print(f"Processed {processed_files}/{total_files} files")
            finally:
                # Clean up temporary files
                for tmp_path in tmp_paths:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

    return all_saved_files
def collect_coeffs_from_s3(csv_files, bucket, num_samples_per_file=10, window_length_sec=2, z_threshold=3.0, batch_size=10):
    return asyncio.run(collect_coeffs_from_s3_async(csv_files, bucket, num_samples_per_file, window_length_sec, z_threshold, batch_size))
# Define the CAE model
class CAE(nn.Module):
    def __init__(self, input_shape, latent_dim=256):
        """
        Args:
            input_shape: Tuple (H, W) of the input (without the channel dimension).
            latent_dim: Dimension of the latent (bottleneck) vector.
        """
        super(CAE, self).__init__()
        self.input_shape = input_shape  # Save original shape for later cropping
        self.latent_dim = latent_dim

        # ----- Encoder -----
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True),  # Halves spatial size (using ceil_mode)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True)  # Halves spatial size again
        )

        # Dynamically determine the output size of encoder_conv
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *input_shape)  # shape: (1, 1, H, W)
            conv_output = self.encoder_conv(dummy_input)
            self.out_channels = conv_output.shape[1]
            self.out_height = conv_output.shape[2]
            self.out_width = conv_output.shape[3]
            flattened_size = self.out_channels * self.out_height * self.out_width

        # Fully-connected bottleneck layer
        self.encoder_fc = nn.Linear(flattened_size, latent_dim)

        self.encoder = nn.Sequential(
            self.encoder_conv,
            nn.Flatten(),
            self.encoder_fc
        )

        # ----- Decoder -----
        self.decoder_fc = nn.Linear(latent_dim, flattened_size)
        self.decoder_conv = nn.Sequential(
            # Expand back to (channels, height, width)
            nn.Unflatten(1, (self.out_channels, self.out_height, self.out_width)),
            nn.ConvTranspose2d(self.out_channels, 16,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 1, H, W)
        Returns:
            reconstructed: Tensor (batch_size, 1, H, W) after decoding
            encoded:       Latent vector of shape (batch_size, latent_dim)
        """
        encoded = self.encoder(x)  # (batch_size, latent_dim)
        x = self.decoder_fc(encoded)  # (batch_size, flattened_size)
        reconstructed = self.decoder_conv(x)  # (batch_size, 1, H_decoded, W_decoded)

        # --- Dynamic adjustment: Crop (or pad) to match the original input shape ---
        target_h, target_w = self.input_shape
        _, _, h_out, w_out = reconstructed.shape

        # Crop if the decoder output is larger than the input dimensions
        if h_out > target_h or w_out > target_w:
            reconstructed = reconstructed[:, :, :target_h, :target_w]
        # Optionally, pad if output is smaller (unlikely with current params)
        elif h_out < target_h or w_out < target_w:
            pad_h = target_h - h_out
            pad_w = target_w - w_out
            reconstructed = nn.functional.pad(reconstructed, (0, pad_w, 0, pad_h))

        return reconstructed, encoded


# Train CAE using data from disk
def train_cae(file_paths,
              latent_dim=256,
              epochs=30,
              batch_size=16,
              output_folder="./",
              region="unknown",
              early_stop_patience=5):
    """
    Train the convolutional autoencoder on the given file paths.

    Args:
        file_paths: A list of .npy file paths containing your data arrays.
                    Each .npy file is assumed to shape like (1, H, W).
        latent_dim: Dimension of the latent bottleneck.
        epochs:     Number of epochs to train.
        batch_size: Training batch size.
        output_folder: Where to save the trained .pt file.
        region:     A label/string for logging or naming the model.
        early_stop_patience: Number of epochs allowed without improvement
                             before early stopping.

    Returns:
        model_path: Path to the saved CAE model.
        encoder:    The encoder portion of the trained model.
        None, None: Just placeholders, if you want to keep the same structure.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    # If no files, skip
    if not file_paths:
        print(f"No data for CAE training in region '{region}'.")
        return None, None, None, None

    # Load one example to get the shape (excluding the batch dimension)
    sample = np.load(file_paths[0])  # shape e.g. (1, H, W)
    if sample.ndim == 3:
        # shape is (1, H, W): remove the batch dimension for CAE init
        input_shape = sample.shape[1:]  # (H, W)
    elif sample.ndim == 2:
        # shape is (H, W): this means your dataset doesn't add a channel
        # so we'll treat the input channel as 1
        input_shape = sample.shape  # (H, W)
    else:
        raise ValueError(f"Sample data shape must be 2D or 3D, got {sample.shape}.")

    # Initialize CAE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cae = CAE(input_shape, latent_dim=latent_dim).to(device)

    # Build dataset/dataloader
    dataset = CoeffsDataset(file_paths)  # Make sure your dataset returns shape (batch, 1, H, W)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer and Loss
    optimizer = optim.Adam(cae.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3, verbose=True
    )

    # Training loop
    best_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(epochs):
        cae.train()
        total_loss = 0.0
        num_batches = 0

        for inputs in dataloader:
            # inputs shape: (batch_size, 1, H, W)
            inputs = inputs.to(device)

            optimizer.zero_grad()
            outputs, _ = cae(inputs)
            loss = criterion(outputs, inputs)  # Autoencoder MSE
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        epoch_loss = total_loss / num_batches
        print(f"[Region: {region}] Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")

        # Learning rate scheduler
        scheduler.step(epoch_loss)

        # Early stopping logic
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

    # Save the whole CAE model
    model_path = os.path.join(output_folder, f"cae_{region}.pt")
    torch.save(cae.state_dict(), model_path)
    print(f"Saved CAE model to {model_path}")

    # Return the path and the encoder portion
    return model_path, cae.encoder, None, None

# Get latent representations using data from disk
def get_latent_reps(encoder, file_paths, device, batch_size=32):
    dataset = CoeffsDataset(file_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latent_reps = []
    for batch in dataloader:
        batch = batch.to(device)
        with torch.no_grad():
            encoded = encoder(batch)
            latent_reps.append(encoded.cpu().numpy())
    latent_reps = np.concatenate(latent_reps, axis=0)
    return latent_reps

# Train GMM
def train_gmm(latent_reps, max_components=100, output_folder="/tmp", region="unknown", use_gpu=False):
    if len(latent_reps) < 2:
        print(f"Not enough data for GMM in region '{region}'.")
        return None, None, None

    # GPU support (optional, requires cuML)
    if use_gpu:
        try:
            from cuml import GaussianMixture as cuGMM
            import cudf
            latent_reps_gpu = cudf.DataFrame(latent_reps)
            bic_scores = []
            for n in range(1, max_components + 1):
                gmm = cuGMM(n_components=n, random_state=0)
                gmm.fit(latent_reps_gpu)
                bic = gmm.bic(latent_reps_gpu)
                bic_scores.append(bic)
            optimal_n = np.argmin(bic_scores) + 1
            gmm = cuGMM(n_components=optimal_n, random_state=0)
            gmm.fit(latent_reps_gpu)
            labels = gmm.predict(latent_reps_gpu)
            effective_components = optimal_n
        except ImportError:
            print("cuML not installed; falling back to CPU.")
            use_gpu = False

    # CPU implementation (default)
    if not use_gpu:
        bic_scores = []
        for n in range(1, max_components + 1):
            gmm = GaussianMixture(n_components=n, random_state=0)
            gmm.fit(latent_reps)
            bic = gmm.bic(latent_reps)
            bic_scores.append(bic)
        optimal_n = np.argmin(bic_scores) + 1  # Select the number of components with lowest BIC
        gmm = GaussianMixture(n_components=optimal_n, random_state=0)
        gmm.fit(latent_reps)
        labels = gmm.predict(latent_reps)
        effective_components = optimal_n

    # Evaluate clustering quality with silhouette score
    if len(set(labels)) > 1:
        silhouette = silhouette_score(latent_reps, labels)
        print(f"Region '{region}' - GMM Silhouette Score: {silhouette:.3f}")
    else:
        silhouette = None
        print(f"Region '{region}' - GMM Silhouette Score: N/A (single cluster)")

    # Save the trained GMM model
    model_path = os.path.join(output_folder, f"gmm_{region}.pkl")
    joblib.dump(gmm, model_path)
    return model_path, effective_components, silhouette
# Generate QA plots using a subset of data
def generate_qa_plots(file_paths, cae, gmm, device, num_samples=100):
    if len(file_paths) > num_samples:
        selected_files = random.sample(file_paths, num_samples)
    else:
        selected_files = file_paths

    all_coeffs_unified = [np.load(fp) for fp in selected_files]
    standardized_coeffs = []
    for img in all_coeffs_unified:
        mean_img = np.mean(img)
        std_img = np.std(img) if np.std(img) > 0 else 1e-8
        standardized_img = (img - mean_img) / std_img
        standardized_coeffs.append(standardized_img)

    coeffs_array = np.stack(standardized_coeffs, axis=0)
    coeffs_array = coeffs_array[:, np.newaxis, :, :]
    coeffs_tensor = torch.tensor(coeffs_array, dtype=torch.float32).to(device)

    cae.eval()
    with torch.no_grad():
        latent_reps = cae.encoder(coeffs_tensor).cpu().numpy()

    labels = gmm.predict(latent_reps)
    os.makedirs('QA', exist_ok=True)

    pca = PCA(n_components=2)
    latent_2d_pca = pca.fit_transform(latent_reps)
    explained_var = pca.explained_variance_ratio_

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d_pca[:, 0], latent_2d_pca[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(f'PCA of Latent Space with GMM Clusters\nExplained Variance: {explained_var[0]:.2f}, {explained_var[1]:.2f}')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig('QA/latent_space_pca.png')
    plt.close()

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

    num_samples_plot = min(5, len(all_coeffs_unified))
    indices = np.random.choice(len(all_coeffs_unified), num_samples_plot, replace=False)

    for i, idx in enumerate(indices):
        original = all_coeffs_unified[idx]
        mean_img = np.mean(original)
        std_img = np.std(original) if np.std(original) > 0 else 1e-8
        original_norm = (original - mean_img) / std_img
        original_tensor = torch.tensor(original_norm[np.newaxis, np.newaxis, :, :], dtype=torch.float32).to(device)

        with torch.no_grad():
            reconstructed, _ = cae(original_tensor)
            reconstructed = reconstructed.cpu().numpy()[0, 0]

        mse = np.mean((original_norm - reconstructed) ** 2)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(original_norm, aspect='auto', cmap='viridis')
        plt.title(f'Original Sample {i + 1} (Standardized)')
        plt.xlabel('Time')
        plt.ylabel('Frequency Scale')
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed, aspect='auto', cmap='viridis')
        plt.title(f'Reconstructed Sample {i + 1}\nMSE: {mse:.6f}')
        plt.xlabel('Time')
        plt.ylabel('Frequency Scale')
        plt.colorbar()

        plt.tight_layout()
        plt.savefig(f'QA/reconstruction_{i + 1}.png')
        plt.close()



# Main execution
if __name__ == "__main__":
    all_folders = list_s3_folders()[0:5]
    random.shuffle(all_folders)
    selected_folders = all_folders
    csv_files = []
    for i, folder in enumerate(selected_folders):
        print(f"{i+1}/{len(selected_folders)}: Folder: {folder}")
        all_files = list_csv_files_in_folder(folder)
        selected_files = random.sample(all_files, min(3, len(all_files)))
        csv_files.extend(selected_files)
        print(f"Selected {len(selected_files)} files")

    print(f"Total files: {len(csv_files)}")

    all_saved_files = collect_coeffs_from_s3(
        csv_files,
        "dataframes--use1-az6--x-s3",
        num_samples_per_file=400,
        window_length_sec=2
    )

    s3 = boto3.client("s3")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cae_path, encoder, min_val, max_val = train_cae(
        all_saved_files,
        region="unified"
    )

    if cae_path is not None:
        latent_reps = get_latent_reps(
            encoder,
            all_saved_files,
            device
        )
        print(f"Latent space shape: {latent_reps.shape}")

        gmm_path, n_components, silhouette = train_gmm(
            latent_reps,
            region="unified"
        )

        if gmm_path is not None:
            cae = CAE(input_shape=(25, 512), latent_dim=256)
            cae.load_state_dict(torch.load(cae_path))
            cae.to(device)
            gmm = joblib.load(gmm_path)
            generate_qa_plots(all_saved_files, cae, gmm, device, num_samples=100)

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