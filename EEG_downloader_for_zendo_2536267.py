#!/usr/bin/env python
"""
This script processes BIDS-formatted EEG meditation experiment data.
It downloads a ZIP file from S3 (the BIDS_EEG_meditation_experiment_v2.zip),
extracts the dataset, iterates over each subject’s sessions, loads the BDF files,
and concatenates them. Only EEG channels whose names start with "A" or "B"
are kept. The channels are then grouped into two signals by averaging all channels:
  - Channel 0: average of all channels starting with "A"
  - Channel 1: average of all channels starting with "B"
The two-channel data is then preprocessed, segmented into windows,
decomposed using wavelets, quantized, and saved to text files.
"""

import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import matplotlib.pyplot as plt
import mne  # MNE-Python
import pywt
import zipfile
import tempfile
import boto3

from utils import preprocess_data, wavelet_decompose_window, quantize_number


def average_AB_channels(data, ch_names):
    """
    Averages EEG channels based on their labels.

    Args:
        data: NumPy array of shape (n_channels, n_samples)
        ch_names: List of channel names corresponding to rows in data.

    Returns:
        A 2-channel NumPy array where:
          - Channel 0 is the average of all channels whose names start with "A"
          - Channel 1 is the average of all channels whose names start with "B"
    """
    # Find indices for channels starting with "A" and "B"
    a_indices = [i for i, ch in enumerate(ch_names) if ch.startswith("A")]
    b_indices = [i for i, ch in enumerate(ch_names) if ch.startswith("B")]

    if not a_indices:
        raise ValueError("No channels starting with 'A' were found.")
    if not b_indices:
        raise ValueError("No channels starting with 'B' were found.")

    a_avg = np.mean(data[a_indices, :], axis=0)
    b_avg = np.mean(data[b_indices, :], axis=0)
    return np.stack([a_avg, b_avg])


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
    plt.plot(time_axis, window_data[0, :], label='A channels average')
    plt.plot(time_axis, window_data[1, :], label='B channels average')
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
        coeffs_path: Output file path for quantized coefficients
        chans_path: Output file path for channel labels
        wavelet: Wavelet type (default 'db2')
        level: Decomposition level (default 4)
        window_len_sec: Window length in seconds (e.g., 1.8 sec)
        plot_windows: If True, plots every window.
        plot_random_n: If an integer and less than total windows, randomly selects that many windows to plot.
    """
    n_window_samples = int(window_len_sec * sps)
    total_samples = data.shape[1]
    total_windows = len(range(0, total_samples - n_window_samples + 1, n_window_samples))

    # Select random windows to plot if desired.
    if plot_random_n is not None and plot_random_n < total_windows:
        selected_windows = np.random.choice(range(1, total_windows + 1), size=plot_random_n, replace=False)
    else:
        selected_windows = None

    os.makedirs(os.path.dirname(coeffs_path), exist_ok=True)
    window_counter = 0

    with open(coeffs_path, 'w') as f_coeffs, open(chans_path, 'w') as f_chans:
        # Process each non-overlapping window.
        for start_idx in range(0, total_samples - n_window_samples + 1, n_window_samples):
            window_counter += 1
            end_idx = start_idx + n_window_samples
            window_data = data[:, start_idx:end_idx]  # Shape: (2, n_window_samples)

            # Plot the window if requested.
            if selected_windows is not None:
                if window_counter in selected_windows:
                    plot_window(window_data, sps, window_index=window_counter)
            elif plot_windows:
                plot_window(window_data, sps, window_index=window_counter)

            # Perform wavelet decomposition on the 2-channel window.
            (decomposed_channels,
             coeffs_lengths,
             num_samples,
             normalized_data) = wavelet_decompose_window(
                window_data,
                wavelet=wavelet,
                level=level,
                normalization=True
            )

            # Process each channel's decomposed coefficients.
            all_channel_coeffs = []
            all_channel_names = []
            for ch_idx in range(decomposed_channels.shape[0]):  # Should be 2 channels.
                ch_name = str(ch_idx)
                coeffs_flat = decomposed_channels[ch_idx].flatten()
                q_ids = [str(quantize_number(c)) for c in coeffs_flat]
                all_channel_coeffs.extend(q_ids)
                all_channel_names.extend([ch_name] * len(q_ids))

            f_coeffs.write(" ".join(all_channel_coeffs) + " ")
            f_chans.write(" ".join(all_channel_names) + " ")


def load_subject_raw_data_bids(subject_folder):
    """
    Loads raw EEG data for a given subject from a BIDS-formatted dataset.
    Expects each subject folder (e.g., "sub-001") to contain one or more session
    subfolders (e.g., "ses-01", "ses-02", etc.). Within each session folder, an "eeg"
    directory is expected that contains one or more BDF files (filenames ending with "_eeg.bdf").
    The function concatenates the raw recordings along the time axis.

    Args:
        subject_folder: Path to the subject folder (e.g., "path/to/sub-001").

    Returns:
        raw_combined: An MNE Raw object with concatenated EEG data.
    """
    raw_list = []
    # Iterate over session directories (e.g., "ses-01", "ses-02", …)
    for ses in sorted(os.listdir(subject_folder)):
        ses_path = os.path.join(subject_folder, ses)
        if os.path.isdir(ses_path) and ses.startswith("ses-"):
            eeg_path = os.path.join(ses_path, "eeg")
            if os.path.isdir(eeg_path):
                # Find BDF files (e.g., "*_eeg.bdf")
                bdf_files = [f for f in os.listdir(eeg_path) if f.endswith("_eeg.bdf")]
                if not bdf_files:
                    print(f"No BDF files found in {eeg_path}.")
                for bdf_file in bdf_files:
                    full_path = os.path.join(eeg_path, bdf_file)
                    try:
                        raw = mne.io.read_raw_bdf(full_path, preload=True, verbose=False)
                        raw_list.append(raw)
                    except Exception as e:
                        print(f"Error reading {full_path}: {e}")
            else:
                print(f"EEG folder not found in session folder: {ses_path}")
    if not raw_list:
        raise ValueError(f"No BDF files found in subject folder: {subject_folder}")
    if len(raw_list) > 1:
        raw_combined = mne.concatenate_raws(raw_list)
    else:
        raw_combined = raw_list[0]
    return raw_combined


if __name__ == "__main__":
    # --- S3 and Dataset Configuration for BIDS Data ---
    s3_bucket = "dataframes--use1-az6--x-s3"
    s3_folder = "attention fintune/2536267"
    # Choose the version 2 ZIP file.
    zip_filename = "BIDS_EEG_meditation_experiment.zip"
    s3_key = f"{s3_folder}/{zip_filename}"

    # Local output directory for processed files.
    output_base = "output-2536267"
    os.makedirs(output_base, exist_ok=True)

    # Initialize the boto3 S3 client.
    s3 = boto3.client("s3")

    # Create a temporary directory for downloading and extraction.
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

        # Determine the dataset root. If the ZIP contains a single top-level folder, use it.
        extracted_contents = os.listdir(extract_path)
        if len(extracted_contents) == 1 and os.path.isdir(os.path.join(extract_path, extracted_contents[0])):
            dataset_root = os.path.join(extract_path, extracted_contents[0])
        else:
            dataset_root = extract_path

        # --- Process Each Subject ---
        # In BIDS, subject folders are usually named with the prefix "sub-"
        subject_dirs = sorted([d for d in os.listdir(dataset_root)
                               if d.startswith("sub-") and os.path.isdir(os.path.join(dataset_root, d))])
        for subject in subject_dirs:
            subject_folder = os.path.join(dataset_root, subject)
            print(f"\nProcessing subject: {subject}")
            try:
                raw = load_subject_raw_data_bids(subject_folder)
            except Exception as e:
                print(f"Error loading subject {subject}: {e}")
                continue

            # For debugging: print channel names and shape.
            print("Channel names:", raw.info["ch_names"])
            print("Raw data shape:", raw.get_data().shape)

            # Select only EEG channels (those starting with "A" or "B").
            eeg_ch_names = [ch for ch in raw.info["ch_names"] if ch.startswith("A") or ch.startswith("B")]
            raw_eeg = raw.copy().pick_channels(eeg_ch_names)
            eeg_data = raw_eeg.get_data()  # Expected shape: (number of A+B channels, n_samples)

            # Average channels based on labels: A channels together and B channels together.
            twoch_data = average_AB_channels(eeg_data, eeg_ch_names)
            print("A/B-averaged data shape:", twoch_data.shape)

            # Preprocess the two-channel data.
            prep_data, new_fs = preprocess_data(twoch_data, raw_eeg.info["sfreq"])

            # Define output file paths.
            coeffs_path = os.path.join(output_base, f"{subject}_combined_coeffs.txt")
            chans_path = os.path.join(output_base, f"{subject}_combined_channels.txt")

            # Process (windowing, wavelet decomposition, quantization) and save.
            process_and_save(prep_data, new_fs, coeffs_path, chans_path,
                             wavelet='db2', level=4, window_len_sec=1.8, plot_windows=True)
            print(f"Finished processing subject: {subject}")

    print("Done!")
