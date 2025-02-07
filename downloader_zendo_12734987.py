#!/usr/bin/env python
"""
This script processes an EEG dataset recorded during motor imagery experiments.
Each of the 32 participants has an EEGLAB (.set) file.
Recordings were acquired from 17 channels:
  F3, Fz, F4, FC5, FC1, FC2, FC6, C3, Cz, C4, CP5, CP1, CP2, CP6, P3, Pz, and P4
at a sampling frequency of 250 Hz.

For hemispheric averaging, only lateral channels (i.e. those with lateralized names)
are used. Here we discard midline channels (Fz, Cz, Pz) and group the remaining channels:
  - Left hemisphere: channels whose names end in 1, 3, or 5.
  - Right hemisphere: channels whose names end in 2, 4, or 6.

The script downloads the ZIP file (raw_data.zip) from S3,
extracts the .set files, and for each subject:
  1. Loads the .set file using MNE’s EEGLAB reader.
  2. Picks only the expected lateral EEG channels.
  3. Computes the hemispheric averages to yield a 2-channel signal.
  4. Applies preprocessing.
  5. Segments the data into windows, performs wavelet decomposition and quantization.
  6. Saves the quantized coefficients and channel labels to text files.
"""

import matplotlib

matplotlib.use('Agg')
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import pywt
import zipfile
import tempfile
import boto3

from utils import preprocess_data, wavelet_decompose_window, quantize_number


def average_hemispheric_channels(data, ch_names):
    """
    Averages EEG channels into two hemispheric signals based on their labels.
    Channels with names ending in 1, 3, or 5 are considered left hemisphere,
    while channels with names ending in 2, 4, or 6 are considered right hemisphere.
    Channels that do not end with a digit (e.g. Fz, Cz, Pz) are ignored.

    Args:
        data: NumPy array of shape (n_channels, n_samples)
        ch_names: List of channel names corresponding to rows in data.

    Returns:
        A 2-channel NumPy array where:
          - Channel 0 is the average of left hemisphere channels.
          - Channel 1 is the average of right hemisphere channels.
    """
    left_indices = []
    right_indices = []
    for i, ch in enumerate(ch_names):
        # Only consider channels that end with a digit (ignore midline channels)
        if ch[-1].isdigit():
            if ch[-1] in ['1', '3', '5']:
                left_indices.append(i)
            elif ch[-1] in ['2', '4', '6']:
                right_indices.append(i)
    if not left_indices:
        raise ValueError("No left hemisphere channels found.")
    if not right_indices:
        raise ValueError("No right hemisphere channels found.")

    left_avg = np.mean(data[left_indices, :], axis=0)
    right_avg = np.mean(data[right_indices, :], axis=0)

    return np.stack([left_avg, right_avg])


def plot_window(window_data, sps, window_index=None):
    """
    Plots a single window of EEG data (2 channels).

    Args:
        window_data: NumPy array of shape (2, n_samples) for the window.
        sps: Sampling rate in Hz.
        window_index: (Optional) window number for the title.
    """
    n_samples = window_data.shape[1]
    time_axis = np.arange(n_samples) / sps
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, window_data[0, :], label='Left hemisphere average')
    plt.plot(time_axis, window_data[1, :], label='Right hemisphere average')
    if window_index is not None:
        plt.title(f"Window {window_index}")
    else:
        plt.title("EEG Window")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"window_{window_index}.png")
    plt.close()


def process_and_save(data, sps, coeffs_path, chans_path,
                     wavelet='db2', level=4, window_len_sec=1.8,
                     plot_windows=False, plot_random_n=1):
    """
    Segments a 2-channel signal into non-overlapping windows,
    performs wavelet decomposition and quantization on each window,
    and writes the results to text files.

    Args:
        data: NumPy array of shape (2, total_samples)
        sps: Sampling rate (Hz)
        coeffs_path: Output file path for quantized coefficients.
        chans_path: Output file path for channel labels.
        wavelet: Wavelet type (default 'db2').
        level: Decomposition level (default 4).
        window_len_sec: Window length in seconds (e.g., 1.8 sec).
        plot_windows: If True, plots every window.
        plot_random_n: If an integer and less than total windows, randomly selects that many windows to plot.
    """
    n_window_samples = int(window_len_sec * sps)
    total_samples = data.shape[1]
    total_windows = len(range(0, total_samples - n_window_samples + 1, n_window_samples))

    if plot_random_n is not None and plot_random_n < total_windows:
        selected_windows = np.random.choice(range(1, total_windows + 1), size=plot_random_n, replace=False)
    else:
        selected_windows = None

    os.makedirs(os.path.dirname(coeffs_path), exist_ok=True)
    window_counter = 0

    with open(coeffs_path, 'w') as f_coeffs, open(chans_path, 'w') as f_chans:
        for start_idx in range(0, total_samples - n_window_samples + 1, n_window_samples):
            window_counter += 1
            end_idx = start_idx + n_window_samples
            window_data = data[:, start_idx:end_idx]  # shape: (2, n_window_samples)

            if selected_windows is not None:
                if window_counter in selected_windows:
                    plot_window(window_data, sps, window_index=window_counter)
            elif plot_windows:
                plot_window(window_data, sps, window_index=window_counter)

            (decomposed_channels,
             coeffs_lengths,
             num_samples,
             normalized_data) = wavelet_decompose_window(
                window_data,
                wavelet=wavelet,
                level=level,
                normalization=True
            )

            all_channel_coeffs = []
            all_channel_names = []
            for ch_idx in range(decomposed_channels.shape[0]):
                ch_name = str(ch_idx)
                coeffs_flat = decomposed_channels[ch_idx].flatten()
                q_ids = [str(quantize_number(c)) for c in coeffs_flat]
                all_channel_coeffs.extend(q_ids)
                all_channel_names.extend([ch_name] * len(q_ids))

            f_coeffs.write(" ".join(all_channel_coeffs) + " ")
            f_chans.write(" ".join(all_channel_names) + " ")


if __name__ == "__main__":
    # --- S3 and Dataset Configuration for the Motor Imagery EEG Data ---
    s3_bucket = "dataframes--use1-az6--x-s3"
    s3_folder = "attention fintune/12734987"
    zip_filename = "raw_data.zip"
    s3_key = f"{s3_folder}/{zip_filename}"

    # Local output directory for processed files.
    output_base = "output-12734987"
    os.makedirs(output_base, exist_ok=True)

    # Initialize boto3 S3 client.
    s3 = boto3.client("s3")

    # Create a temporary directory for download and extraction.
    with tempfile.TemporaryDirectory() as temp_dir:
        local_zip_path = os.path.join(temp_dir, zip_filename)
        print(f"Downloading s3://{s3_bucket}/{s3_key} ...")
        try:
            s3.download_file(s3_bucket, s3_key, local_zip_path)
            print(f"Downloaded {zip_filename} to {local_zip_path}")
        except Exception as e:
            print(f"Error downloading {zip_filename}: {e}")
            exit(1)

        # Extract the ZIP file.
        extract_path = os.path.join(temp_dir, "extracted")
        os.makedirs(extract_path, exist_ok=True)
        try:
            with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print(f"Extracted {zip_filename} to {extract_path}")
        except Exception as e:
            print(f"Error extracting {zip_filename}: {e}")
            exit(1)

        # List all .set files in the extracted folder.
        set_files = [f for f in os.listdir(extract_path) if f.endswith(".set")]
        if not set_files:
            print("No .set files found in the extracted data.")
            exit(1)

        # Define expected lateral EEG channels (excluding midline channels).
        expected_lateral_channels = ["F3", "F4", "FC5", "FC1", "FC2", "FC6",
                                     "C3", "C4", "CP5", "CP1", "CP2", "CP6",
                                     "P3", "P4"]

        for set_file in sorted(set_files):
            subject_id = os.path.splitext(set_file)[0]  # e.g., "sub-01"
            file_path = os.path.join(extract_path, set_file)
            print(f"\nProcessing subject file: {set_file}")
            try:
                raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
            except Exception as e:
                print(f"Error loading {set_file}: {e}")
                continue

            print("Channel names:", raw.info["ch_names"])
            print("Raw data shape:", raw.get_data().shape)

            # Pick only the expected lateral channels.
            available_channels = [ch for ch in expected_lateral_channels if ch in raw.info["ch_names"]]
            if not available_channels:
                print(f"No expected lateral channels found in {set_file}. Skipping.")
                continue
            raw.pick_channels(available_channels)
            print("After picking lateral channels, data shape:", raw.get_data().shape)

            # Get data and sampling frequency.
            eeg_data = raw.get_data()  # shape: (n_channels, n_samples)
            fs = raw.info["sfreq"]  # expected to be 250 Hz

            # Compute hemispheric averages.
            twoch_data = average_hemispheric_channels(eeg_data, raw.info["ch_names"])
            print("Hemispheric averaged data shape:", twoch_data.shape)

            # Preprocess the two-channel data.
            prep_data, new_fs = preprocess_data(twoch_data, fs)

            # Define output file paths.
            coeffs_path = os.path.join(output_base, f"{subject_id}_combined_coeffs.txt")
            chans_path = os.path.join(output_base, f"{subject_id}_combined_channels.txt")

            # Process (windowing, wavelet decomposition, quantization) and save.
            process_and_save(prep_data, new_fs, coeffs_path, chans_path,
                             wavelet='db2', level=4, window_len_sec=1.8, plot_windows=True)
            print(f"Finished processing subject: {subject_id}")

    print("Done!")
