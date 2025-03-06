import pandas as pd
import numpy as np
import pywt

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from create_text_files_from_csv_3 import create_regional_bipolar_channels
from utils import wavelet_decompose_window, preprocess_data, calculate_sps, call_gpt_for_instructions


def process_csv_for_coeffs(csv_path, window_length_sec=0.5, num_samples_per_file=10, z_threshold=2):
    """
    Processes a local CSV file containing EEG signal data to compute and save wavelet decomposition coefficients.

    This function reads the CSV file from csv_path, uses external helper functions to obtain processing instructions,
    creates regional bipolar channels, standardizes the data, segments it into windows, performs artifact rejection,
    plots the rejected windows for three regions, computes wavelet decomposition on selected windows, and saves the
    resulting standardized coefficient arrays to disk.

    External functions required:
        - call_gpt_for_instructions(channel_names, dataset_id)
        - create_regional_bipolar_channels(df, channels_to_drop)
        - preprocess_data(signal_2d, original_sps)
        - calculate_sps_from_df(df)
        - wavelet_decompose_window(window_data, wavelet, scales, normalization, sampling_period)

    Parameters:
        csv_path (str): Path to the local CSV file.
        window_length_sec (float): Window length in seconds (default: 2).
        num_samples_per_file (int): Number of windows (samples) to select per file (default: 10).
        z_threshold (float): Z-score threshold for artifact rejection (default: 3.0).

    Returns:
        saved_files (list): List of file paths where the computed coefficients were saved.
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
    # Stack all channels to create a 2D array (channels x samples)
    data_2d = np.vstack([regional_bipolar[ch] for ch in channels])

    # Call the preprocess function once
    preprocessed_data, new_sps = preprocess_data(data_2d, original_sps)

    # Convert back to a dictionary mapping channel names to their processed signals
    regional_preprocessed = {ch: preprocessed_data[i, :] for i, ch in enumerate(channels)}
    new_sps_val = new_sps

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
    num_windows = min_length // n_window_samples

    window_stats = []
    for i in range(num_windows):
        window_start = i * n_window_samples
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

    # Plotting three channels with rejected windows highlighted
    regions = ["frontal", "motor_temporal", "parietal_occipital"]
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    time = np.arange(min_length) / new_sps_val
    for ax, region in zip(axes, regions):
        if region in regional_preprocessed:
            signal = regional_preprocessed[region]
            ax.plot(time, signal, label=region)
            ax.set_ylabel(region)
            # Highlight rejected windows
            for i in rejected_indices:
                start_time = (i * n_window_samples) / new_sps_val
                end_time = ((i + 1) * n_window_samples) / new_sps_val
                ax.axvspan(start_time, end_time, color='red', alpha=0.3)
        else:
            ax.set_visible(False)
    axes[0].set_title(f"EEG Signals with Rejected Windows - {base_name}")
    axes[2].set_xlabel("Time (s)")
    plt.tight_layout()
    safe_csv_key = csv_path.replace('/', '_').replace('.', '_')
    plot_path = os.path.join('QA', f"{safe_csv_key}_rejected_windows.png")
    os.makedirs('QA', exist_ok=True)
    plt.show()
    # plt.savefig(plot_path)
    # plt.close()

    selected_indices = np.random.choice(keep_indices, min(num_samples_per_file, len(keep_indices)), replace=False)

    saved_files = []
    batch_images = []  # List to collect each window's RGB image

    for i in selected_indices:
        window_start = i * n_window_samples
        window_end = window_start + n_window_samples
        # Build window_data from all available channels
        window_data = np.vstack([
            regional_preprocessed[region][window_start:window_end]
            for region in regional_preprocessed if len(regional_preprocessed[region]) > 0
        ])
        if window_data.size == 0:
            continue

        # Perform wavelet decomposition (which already normalizes per scale if requested)
        decomposed_channels, _, _, _ = wavelet_decompose_window(
            window_data,
            wavelet='cmor1.5-1.0',
            scales=None,
            normalization=True,
            sampling_period=1.0 / new_sps_val,
            verbose=True
        )

        # Ensure we have at least 3 channels (one per region)
        if len(decomposed_channels) < 3:
            continue
        # Stack the first three channels along a new axis to form an RGB image.
        # Each decomposed channel is a 2D array with shape (num_scales, time_points).
        # The resulting image will have shape (num_scales, time_points, 3).
        image = np.stack([
            decomposed_channels[0],
            decomposed_channels[1],
            decomposed_channels[2]
        ], axis=0)

        # Generate a unique file name for this image
        safe_csv_key = csv_path.replace('/', '_').replace('.', '_')
        file_name = f"{safe_csv_key}_image_{i}.npy"
        file_path = os.path.join('training_data', 'coeffs', file_name)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save the individual image (shape: 3, num_scales, time_points)
        np.save(file_path, image)

        # Append the file path to the list of saved files
        saved_files.append(file_path)


    return saved_files


if __name__ == "__main__":
    # Example usage:
    csv_file =  "/Volumes/Untitled/sub-090_ses-03_task-VisualWorkingMemory_eeg.csv"  # Update with your CSV file path
    try:
        coefficients = process_csv_for_coeffs(csv_file)
    except Exception as e:
        print(f"Error processing CSV: {e}")
