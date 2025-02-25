import numpy as np
from sklearn.cluster import KMeans
import joblib
import boto3
from concurrent.futures import ThreadPoolExecutor
import os
import pandas as pd

from create_text_files_from_csv_2 import create_regional_bipolar_channels
from utils import list_csv_files_in_folder, list_s3_folders, call_gpt_for_instructions, calculate_sps, preprocess_data, \
    wavelet_decompose_window


def process_csv_for_coeffs(csv_key, bucket, local_dir, window_length_sec=1.96, wvlet='db2', level=4, num_samples_per_file=10):
    """
    Process a single CSV file to extract wavelet coefficient vectors for K-means training.

    Args:
        csv_key (str): S3 key for the CSV file.
        bucket (str): S3 bucket name.
        local_dir (str): Local directory for temporary storage.
        window_length_sec (float): Length of each window in seconds.
        wvlet (str): Wavelet type for decomposition.
        level (int): Wavelet decomposition level.
        num_samples_per_file (int): Number of windows to sample per file.

    Returns:
        dict: Coefficient vectors for each region.
    """
    s3 = boto3.client("s3")
    local_csv = os.path.join(local_dir, os.path.basename(csv_key))
    s3.download_file(bucket, csv_key, local_csv)

    base_name = os.path.splitext(os.path.basename(csv_key))[0]
    df = pd.read_csv(local_csv)
    all_columns = list(df.columns)
    instructions = call_gpt_for_instructions(channel_names=all_columns, dataset_id=base_name)

    if instructions.get("action") == "skip":
        print(f"Skipping dataset '{base_name}' as instructed by GPT.")
        os.remove(local_csv)
        return {"frontal": [], "motor_temporal": [], "parietal_occipital": []}

    channels_to_drop = instructions.get("channels_to_drop", [])
    print(f"Processing dataset '{base_name}'. Dropping channels: {channels_to_drop}")

    # Create regional bipolar channels
    regional_bipolar = create_regional_bipolar_channels(df, channels_to_drop)

    # Preprocess the regional signals
    original_sps = calculate_sps(local_csv)
    regional_preprocessed = {}
    new_sps_val = None
    for key, signal_array in regional_bipolar.items():
        signal_2d = signal_array[np.newaxis, :]
        preprocessed_signal, new_sps = preprocess_data(signal_2d, original_sps)
        regional_preprocessed[key] = preprocessed_signal[0, :]
        if new_sps_val is None:
            new_sps_val = new_sps

    # Joint global normalization
    all_data = np.concatenate([
        regional_preprocessed["frontal"],
        regional_preprocessed["motor_temporal"],
        regional_preprocessed["parietal_occipital"]
    ])
    global_mean = np.mean(all_data)
    global_std = np.std(all_data)
    if np.isclose(global_std, 0):
        global_std = 1e-8
    for key in regional_preprocessed:
        regional_preprocessed[key] = (regional_preprocessed[key] - global_mean) / global_std

    # Determine the number of windows
    min_length = min(len(regional_preprocessed[region]) for region in regional_preprocessed)
    n_window_samples = int(window_length_sec * new_sps_val)
    num_windows = min_length // n_window_samples

    # Randomly select window indices
    if num_windows <= num_samples_per_file:
        selected_indices = list(range(num_windows))
    else:
        selected_indices = np.random.choice(num_windows, num_samples_per_file, replace=False)

    # Collect coefficients
    coeffs_dict = {"frontal": [], "motor_temporal": [], "parietal_occipital": []}
    for i in selected_indices:
        window_start = i * n_window_samples
        window_end = window_start + n_window_samples
        window_data = np.vstack([
            regional_preprocessed["frontal"][window_start:window_end],
            regional_preprocessed["motor_temporal"][window_start:window_end],
            regional_preprocessed["parietal_occipital"][window_start:window_end]
        ])
        # With this:
        (decomposed_channels,
         scales,
         num_samples,
         normalized_data) = wavelet_decompose_window(
            window_data,
            wavelet='cmor1.5-1.0',  # Morlet wavelet
            scales=None,  # Auto-compute scales
            normalization=False,
            sampling_period=1.0 / new_sps_val  # Pass the sampling period
        )
        for idx, region in enumerate(coeffs_dict.keys()):
            coeffs_for_channel = decomposed_channels[idx].flatten()
            coeffs_dict[region].append(coeffs_for_channel)

    os.remove(local_csv)
    return coeffs_dict

def collect_coeffs_from_s3(csv_files, bucket, num_samples_per_file=10, window_length_sec=1.96, wvlet='db2', level=4):
    """
    Collect wavelet coefficient vectors from a subset of CSV files on S3.

    Args:
        csv_files (list): List of S3 keys for CSV files.
        bucket (str): S3 bucket name.
        num_samples_per_file (int): Number of samples to collect per file.
        window_length_sec (float): Length of each window in seconds.
        wvlet (str): Wavelet type for decomposition.
        level (int): Wavelet decomposition level.

    Returns:
        dict: Dictionary with coefficient vectors for each region.
    """
    all_coeffs = {"frontal": [], "motor_temporal": [], "parietal_occipital": []}
    s3 = boto3.client("s3")
    local_dir = "/tmp"

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_csv_for_coeffs, csv_file, bucket, local_dir, window_length_sec, wvlet, level, num_samples_per_file)
                   for csv_file in csv_files]
        for future in futures:
            coeffs = future.result()
            for region in all_coeffs:
                all_coeffs[region].extend(coeffs[region])
    return all_coeffs
def train_kmeans_models(all_coeffs, num_clusters=256, output_folder="/tmp"):
    """
    Train K-means models for each region and save them locally.

    Args:
        all_coeffs (dict): Coefficient vectors for each region.
        num_clusters (int): Number of clusters for K-means.
        output_folder (str): Local directory to save models.

    Returns:
        dict: Paths to saved K-means models.
    """
    kmeans_models = {}
    for region, coeffs_list in all_coeffs.items():
        coeffs_array = np.vstack(coeffs_list)
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(coeffs_array)
        model_path = os.path.join(output_folder, f"kmeans_{region}.pkl")
        joblib.dump(kmeans, model_path)
        kmeans_models[region] = model_path
    return kmeans_models

if __name__ == "__main__":
    folders = list_s3_folders()
    csv_files = []
    i = 0
    for folder in folders:
        print(f"{i}/{len(folders)}: Looking into folder: {folder}")
        files = list_csv_files_in_folder(folder)
        csv_files.extend(files)
        i += 1
    print(f"Done listing. Total files: {len(csv_files)}")
    all_coeffs = collect_coeffs_from_s3(csv_files, "dataframes--use1-az6--x-s3", num_samples_per_file=200, window_length_sec=2, wvlet='db2', level=4)
    kmeans_model_paths = train_kmeans_models(all_coeffs, num_clusters=2048)
    s3 = boto3.client("s3")
    for region, path in kmeans_model_paths.items():
        s3.upload_file(path, "dataframes--use1-az6--x-s3", f"kmeans_models/kmeans_{region}.pkl")
        os.remove(path)