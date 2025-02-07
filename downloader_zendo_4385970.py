#!/usr/bin/env python
"""
This script processes an EEG dataset recorded during a language comprehension experiment.
Participants listened to isochronous streams of monosyllabic words while EEG was recorded.
The data are provided in BrainVision format (.eeg, .vhdr, .vmrk) and are packaged in a ZIP archive.
For each subject the processing pipeline is as follows:
  1. Load the subject’s BrainVision EEG data using MNE.
  2. Pick only EEG channels.
  3. Compute hemispheric averages by:
       - Averaging channels whose names end with an odd digit → left hemisphere.
       - Averaging channels whose names end with an even digit → right hemisphere.
     (Channels without a trailing digit—typically midline channels—are ignored.)
  4. Preprocess the resulting two‐channel signal (using a custom preprocess_data function).
  5. Segment the preprocessed data into non‐overlapping windows.
  6. For each window, perform wavelet decomposition and quantization.
  7. Save the quantized coefficients and their channel labels to text files.
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

# These functions are assumed to be defined in your custom utils module.
from utils import preprocess_data, wavelet_decompose_window, quantize_number


def average_hemispheric_channels(data, ch_names):
    """
    Averages EEG channels into two hemispheric signals.

    Channels whose names end with an odd digit are assigned to the left hemisphere,
    while channels whose names end with an even digit are assigned to the right hemisphere.
    Channels that do not end with a digit (e.g. midline channels such as Fz, Cz, Pz) are ignored.

    Args:
        data: NumPy array of shape (n_channels, n_samples)
        ch_names: List of channel names corresponding to the rows in data.

    Returns:
        A 2-channel NumPy array:
          - Channel 0: left hemisphere average.
          - Channel 1: right hemisphere average.
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
        window_index: Optional window number (used for title and filename).
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
        sps: Sampling rate (Hz)
        coeffs_path: Output file path for quantized coefficients.
        chans_path: Output file path for channel labels.
        wavelet: Wavelet type (default 'db2').
        level: Decomposition level (default 4).
        window_len_sec: Window length in seconds (e.g., 1.8 sec).
        plot_windows: If True, plots windows (or a random subset).
        plot_random_n: If an integer (less than the total number of windows), randomly select that many windows to plot.
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


if __name__ == "__main__":
    # --- S3 and Dataset Configuration for the Language Processing EEG Data ---
    s3_bucket = "dataframes--use1-az6--x-s3"
    s3_folder = "attention fintune/4385970"
    # We assume the dataset is provided as a ZIP archive containing BrainVision files.
    zip_filename = "EEG_language.zip"
    s3_key = f"{s3_folder}/{zip_filename}"

    # Local output directory for processed files.
    output_base = "output-4385970"
    os.makedirs(output_base, exist_ok=True)

    # Initialize the boto3 S3 client.
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

        # Extract the ZIP archive.
        extract_path = os.path.join(temp_dir, "extracted")
        os.makedirs(extract_path, exist_ok=True)
        try:
            with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print(f"Extracted {zip_filename} to {extract_path}")
        except Exception as e:
            print(f"Error extracting {zip_filename}: {e}")
            exit(1)

        # Recursively find all BrainVision header files (.vhdr) in the extracted folder.
        vhdr_files = []
        for root, dirs, files in os.walk(extract_path):
            for file in files:
                if file.endswith(".vhdr"):
                    vhdr_files.append(os.path.join(root, file))
        if not vhdr_files:
            print("No BrainVision header (.vhdr) files found in the extracted data.")
            exit(1)

        # Process each subject's file.
        for vhdr_file in sorted(vhdr_files):
            subject_id = os.path.splitext(os.path.basename(vhdr_file))[0]
            print(f"\nProcessing subject: {subject_id}")
            try:
                raw = mne.io.read_raw_brainvision(vhdr_file, preload=True, verbose=False)
            except Exception as e:
                print(f"Error loading {vhdr_file}: {e}")
                continue

            print("Channel names:", raw.info["ch_names"])
            print("Raw data shape:", raw.get_data().shape)

            # Pick only EEG channels.
            raw.pick_types(eeg=True)
            print("After picking EEG channels, shape:", raw.get_data().shape)

            eeg_data = raw.get_data()  # shape: (n_channels, n_samples)
            fs = raw.info["sfreq"]  # sampling rate (e.g., 500, 1000 Hz, etc.)

            # Compute hemispheric averages.
            try:
                twoch_data = average_hemispheric_channels(eeg_data, raw.info["ch_names"])
            except Exception as e:
                print(f"Error in hemispheric averaging for {subject_id}: {e}")
                continue
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
