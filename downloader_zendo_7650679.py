#!/usr/bin/env python
"""
This script processes the raw EEG data from the project
"Neural signatures of linguistic predictions and listener's attention to speaker's communication intention".

The raw EEG data for Experiment 1 are provided as a ZIP file (DATA-EXP1.zip) stored in S3:
    s3://dataframes--use1-az6--x-s3/attention fintune/7650679/DATA-EXP1.zip

When extracted, the ZIP file creates a folder structure like:
    <temp_dir>/DATA-EXP1/Data/EEGraw/
        P01.bdf
        P02.bdf
        ...
        P46.bdf

For each subject’s BDF file, the processing pipeline is:
  1. Load the EEG data using mne.io.read_raw_bdf.
  2. Retain only EEG channels.
  3. Compute a two‑channel hemispheric average:
       - Left hemisphere: average of lateral channels whose names end with an odd digit.
       - Right hemisphere: average of lateral channels whose names end with an even digit.
       (Midline channels are ignored.)
  4. Preprocess the two‑channel data using a custom function preprocess_data.
  5. Segment the data into non‑overlapping windows, perform wavelet decomposition and quantization,
     and write the quantized coefficients and channel labels to text files.
  6. Optionally, plot selected windows.
"""

import os
import boto3
import tempfile
import zipfile
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import pywt

# Import your custom processing functions.
from utils import preprocess_data, wavelet_decompose_window, quantize_number

# S3 configuration
S3_BUCKET = "dataframes--use1-az6--x-s3"
S3_KEY = "attention fintune/7650679/DATA-EXP1.zip"

# Output directory for processed files.
OUTPUT_BASE = "output-7650679"
os.makedirs(OUTPUT_BASE, exist_ok=True)


def average_hemispheric_channels(data, ch_names):
    """
    Averages EEG channels into two hemispheric signals.

    Channels whose names end with an odd digit are assigned to the left hemisphere,
    while channels whose names end with an even digit are assigned to the right hemisphere.
    Channels that do not end with a digit (e.g., midline channels like Fz, Cz, Pz) are ignored.

    Args:
        data: NumPy array of shape (n_channels, n_samples)
        ch_names: List of channel names corresponding to rows in data.

    Returns:
        A 2-channel NumPy array: [left_avg, right_avg]
    """
    left_indices = []
    right_indices = []
    for i, ch in enumerate(ch_names):
        if ch and ch[-1].isdigit():
            if ch[-1] in ['1', '3', '5', '7', '9']:
                left_indices.append(i)
            elif ch[-1] in ['2', '4', '6', '8', '0']:
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
    Plots one window of two-channel EEG data.

    Args:
        window_data: NumPy array of shape (2, n_samples) for the window.
        sps: Sampling rate in Hz.
        window_index: Optional window number (used in title and filename).
    """
    n_samples = window_data.shape[1]
    time_axis = np.arange(n_samples) / sps
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, window_data[0, :], label="Left Hemisphere")
    plt.plot(time_axis, window_data[1, :], label="Right Hemisphere")
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
    Segments a two-channel EEG signal into non-overlapping windows,
    performs wavelet decomposition and quantization on each window,
    and writes the resulting quantized coefficients and channel labels to text files.

    Args:
        data: NumPy array of shape (2, total_samples)
        sps: Sampling rate in Hz.
        coeffs_path: Output file path for quantized coefficients.
        chans_path: Output file path for channel labels.
        wavelet: Wavelet type (default 'db2').
        level: Decomposition level (default 4).
        window_len_sec: Window length in seconds (e.g., 1.8 sec).
        plot_windows: If True, plots windows (or a random subset).
        plot_random_n: If an integer (and less than total windows), randomly selects that many windows to plot.
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
            window_data = data[:, start_idx:end_idx]

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


# --- Main Script ---
if __name__ == "__main__":
    # Initialize boto3 S3 client.
    s3 = boto3.client("s3")

    # Create a temporary directory.
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        local_zip_path = os.path.join(temp_dir, "DATA-EXP1.zip")
        print(f"Downloading DATA-EXP1.zip from S3: s3://{S3_BUCKET}/{S3_KEY} ...")
        try:
            s3.download_file(S3_BUCKET, S3_KEY, local_zip_path)
            print(f"Downloaded DATA-EXP1.zip to {local_zip_path}")
        except Exception as e:
            print(f"Error downloading DATA-EXP1.zip: {e}")
            exit(1)

        # Extract the ZIP file.
        try:
            with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            print(f"Extracted DATA-EXP1.zip into {temp_dir}")
        except Exception as e:
            print(f"Error extracting DATA-EXP1.zip: {e}")
            exit(1)

        # Locate the EEG raw data folder.
        eegraw_folder = os.path.join(temp_dir, "DATA-EXP1", "Data", "EEGraw")
        if not os.path.isdir(eegraw_folder):
            print(f"EEGraw folder not found in extracted data at {eegraw_folder}")
            exit(1)

        # List all BDF files in the EEGraw folder.
        bdf_files = [f for f in os.listdir(eegraw_folder) if f.lower().endswith(".bdf")]
        if not bdf_files:
            print("No BDF files found in the EEGraw folder.")
            exit(1)

        print(f"Found {len(bdf_files)} BDF files. Processing each subject...")
        for bdf in sorted(bdf_files):
            subject_id = os.path.splitext(bdf)[0]  # e.g., "P01"
            bdf_path = os.path.join(eegraw_folder, bdf)
            print(f"\nProcessing subject: {subject_id} from {bdf_path}")
            try:
                raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=False)
            except Exception as e:
                print(f"Error loading {bdf_path}: {e}")
                continue
            print("Channel names:", raw.info["ch_names"])
            print("Raw data shape:", raw.get_data().shape)

            # Retain only EEG channels.
            raw.pick_types(eeg=True)
            print("After picking EEG channels, shape:", raw.get_data().shape)

            eeg_data = raw.get_data()  # shape: (n_channels, n_samples)
            fs = raw.info["sfreq"]

            try:
                twoch_data = average_hemispheric_channels(eeg_data, raw.info["ch_names"])
            except Exception as e:
                print(f"Error computing hemispheric averages for subject {subject_id}: {e}")
                continue
            print("Hemispheric averaged data shape:", twoch_data.shape)

            # Preprocess the two-channel data.
            prep_data, new_fs = preprocess_data(twoch_data, fs)

            # Define output file paths.
            coeffs_path = os.path.join(OUTPUT_BASE, f"{subject_id}_combined_coeffs.txt")
            chans_path = os.path.join(OUTPUT_BASE, f"{subject_id}_combined_channels.txt")

            # Process (windowing, wavelet decomposition, quantization) and save.
            process_and_save(prep_data, new_fs, coeffs_path, chans_path,
                             wavelet='db2', level=4, window_len_sec=1.8, plot_windows=True)
            print(f"Finished processing subject: {subject_id}")

    print("Done!")
