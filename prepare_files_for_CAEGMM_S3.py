import pandas as pd
import numpy as np
import pywt
import os
import matplotlib.pyplot as plt
from create_text_files_from_csv_3 import create_regional_bipolar_channels
from utils import wavelet_decompose_window, preprocess_data, calculate_sps, call_gpt_for_instructions
import boto3
import random
from multiprocessing import Pool, cpu_count

# Replace with your actual S3 bucket name
BUCKET_NAME = 'dataframes--use1-az6--x-s3'


def download_from_s3(s3_path, bucket_name=BUCKET_NAME, local_base='/tmp'):
    """Download a file from S3 to a local path, preserving the folder structure."""
    s3 = boto3.client('s3')
    local_path = os.path.join(local_base, s3_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(bucket_name, s3_path, local_path)
    return local_path


def process_s3_csv(s3_csv_path):
    """Process a single CSV file from S3: download, process, and clean up."""
    local_csv_path = None
    try:
        # Download the CSV file from S3
        local_csv_path = download_from_s3(s3_csv_path)
        print(f"Starting processing for {s3_csv_path}")

        # Process the CSV file and get the list of saved coefficient files
        saved_files = process_csv_for_coeffs(local_csv_path)

        # Log the result
        print(f"Finished {s3_csv_path}: saved {len(saved_files)} coefficient files")
    except Exception as e:
        # Log any errors
        print(f"Error with {s3_csv_path}: {e}")
    finally:
        # Clean up thetemporary file
        if local_csv_path and os.path.exists(local_csv_path):
            os.remove(local_csv_path)
            print(f"Cleaned up temporary file for {s3_csv_path}")


def process_csv_for_coeffs(csv_path, window_length_sec=0.5, overlap_percent=50, num_samples_per_file=200,
                           z_threshold=2):
    """
    Process a local CSV file containing EEG signal data to compute and save wavelet decomposition coefficients.

    Parameters:
        csv_path (str): Path to the local CSV file.
        window_length_sec (float): Window length in seconds (default: 0.5).
        overlap_percent (float): Percentage of overlap between consecutive windows (default: 50).
        num_samples_per_file (int): Number of windows to select per file (default: 200).
        z_threshold (float): Z-score threshold for artifact rejection (default: 2).

    Returns:
        saved_files (list): List of file paths where coefficients were saved.
    """
    # Read CSV into a DataFrame
    df = pd.read_csv(csv_path)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    all_columns = list(df.columns)
    instructions = call_gpt_for_instructions(channel_names=all_columns, dataset_id=base_name)
    original_sps = calculate_sps(csv_path)

    if instructions.get("action") == "skip":
        print(f"Skipping dataset '{base_name}'.")
        return []

    channels_to_drop = instructions.get("channels_to_drop", [])
    print(f"Processing '{base_name}'. Dropping: {channels_to_drop}")

    regional_bipolar = create_regional_bipolar_channels(df, channels_to_drop)
    channels = list(regional_bipolar.keys())
    data_2d = np.vstack([regional_bipolar[ch] for ch in channels])

    preprocessed_data, new_sps = preprocess_data(data_2d, original_sps)
    regional_preprocessed = {ch: preprocessed_data[i, :] for i, ch in enumerate(channels)}
    new_sps_val = new_sps

    # Standardize the signals
    for key in regional_preprocessed:
        signal = regional_preprocessed[key]
        mean_signal = np.mean(signal)
        std_signal = np.std(signal) if np.std(signal) > 0 else 1e-8
        regional_preprocessed[key] = (signal - mean_signal) / std_signal

    min_length = min(len(regional_preprocessed[region]) for region in regional_preprocessed if
                     len(regional_preprocessed[region]) > 0)
    if min_length == 0:
        return []

    n_window_samples = int(window_length_sec * new_sps_val)

    # Calculate step size based on overlap percentage
    step_size = int(n_window_samples * (1 - overlap_percent / 100))

    # Calculate number of windows with overlap
    num_windows = (min_length - n_window_samples) // step_size + 1 if step_size > 0 else 1

    # Compute window statistics for artifact rejection
    window_stats = []
    for i in range(num_windows):
        window_start = i * step_size
        window_end = window_start + n_window_samples
        window_data = np.vstack([regional_preprocessed[region][window_start:window_end]
                                 for region in regional_preprocessed if len(regional_preprocessed[region]) > 0])
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

    # Plotting for QA (save only, no display)
    regions = ["frontal", "motor_temporal", "parietal_occipital"]
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    time = np.arange(min_length) / new_sps_val
    for ax, region in zip(axes, regions):
        if region in regional_preprocessed:
            signal = regional_preprocessed[region]
            ax.plot(time, signal, label=region)
            ax.set_ylabel(region)
            for i in rejected_indices:
                start_time = (i * step_size) / new_sps_val
                end_time = (i * step_size + n_window_samples) / new_sps_val
                ax.axvspan(start_time, end_time, color='red', alpha=0.3)
        else:
            ax.set_visible(False)
    axes[0].set_title(f"EEG Signals with Rejected Windows - {base_name}")
    axes[2].set_xlabel("Time (s)")
    plt.tight_layout()
    safe_csv_key = csv_path.replace('/', '_').replace('.', '_')
    plot_path = os.path.join('QA', f"{safe_csv_key}_rejected_windows.png")
    os.makedirs('QA', exist_ok=True)
    plt.savefig(plot_path)
    plt.close()

    # Select windows to process
    selected_indices = np.random.choice(keep_indices, min(num_samples_per_file, len(keep_indices)), replace=False)
    saved_files = []

    # Process selected windows
    for i in selected_indices:
        window_start = i * step_size
        window_end = window_start + n_window_samples
        window_data = np.vstack([
            regional_preprocessed[region][window_start:window_end]
            for region in regional_preprocessed if len(regional_preprocessed[region]) > 0
        ])
        if window_data.size == 0:
            continue

        decomposed_channels, _, _, _ = wavelet_decompose_window(
            window_data,
            wavelet='cmor1.5-1.0',
            scales=None,
            normalization=True,
            sampling_period=1.0 / new_sps_val,
            verbose=False
        )

        if len(decomposed_channels) < 3:
            continue
        image = np.stack([decomposed_channels[0], decomposed_channels[1], decomposed_channels[2]], axis=0)

        # Save the coefficient image
        safe_csv_key = csv_path.replace('/', '_').replace('.', '_')
        file_name = f"{safe_csv_key}_image_{i}.npy"
        file_path = os.path.join('training_data', 'coeffs', file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, image)
        saved_files.append(file_path)

    return saved_files


if __name__ == "__main__":
    # Assuming these functions are defined in a module named s3_utils
    from utils import list_s3_folders, list_csv_files_in_folder

    # Step 1: Get the list of CSV files from S3
    all_folders = list_s3_folders()
    random.shuffle(all_folders)
    selected_folders = all_folders
    csv_files = []
    for i, folder in enumerate(selected_folders):
        print(f"{i + 1}/{len(selected_folders)}: Folder: {folder}")
        all_files = list_csv_files_in_folder(folder)
        selected_files = random.sample(all_files, min(4, len(all_files)))
        csv_files.extend(selected_files)
        print(f"Selected {len(selected_files)} files")

    print(f"Total files to process: {len(csv_files)}")

    # Step 2: Process files using multiprocessing
    with Pool(processes=cpu_count()) as pool:
        pool.map(process_s3_csv, csv_files)