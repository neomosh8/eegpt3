#!/usr/bin/env python
"""
This script processes an EEG dataset recorded during an auditory attention experiment.
The dataset was acquired in an anechoic chamber where subjects listened to speech
presented from four loudspeakers. Each subject’s data is stored in a separate folder
inside the EEG_raw.zip archive, with each folder (e.g., "sub1", "sub2", … "sub16")
containing:
  - data.bdf         : the raw EEG recording.
  - evt.bdf          : event markers.
  - recordInformation.json

The processing pipeline is as follows:
  1. Download EEG_raw.zip from S3.
  2. Extract the archive.
  3. For each subject:
       a. Load the EEG data from data.bdf using MNE’s read_raw_bdf.
       b. Pick only EEG channels.
       c. Compute hemispheric averages:
            - Left hemisphere: average of channels whose names end in an odd digit.
            - Right hemisphere: average of channels whose names end in an even digit.
         (Midline channels—those whose names do not end in a digit—are ignored.)
       d. Preprocess the two‑channel data (using preprocess_data).
       e. Segment the data into non‑overlapping windows, perform wavelet decomposition and quantization.
       f. Save the quantized coefficients and channel labels to text files.
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
    Averages EEG channels into two hemispheric signals.

    Channels whose names end with an odd digit are considered left hemisphere,
    while channels whose names end with an even digit are considered right hemisphere.
    Channels that do not end with a digit (e.g., midline channels such as Fz, Cz, Pz)
    are ignored.

    Args:
        data: NumPy array of shape (n_channels, n_samples)
        ch_names: List of channel names (corresponding to the rows in data)

    Returns:
        A 2-channel NumPy array:
          - Channel 0: average of left hemisphere channels.
          - Channel 1: average of right hemisphere channels.
    """
    left_indices = []
    right_indices = []
    for i, ch in enumerate(ch_names):
        # Only process channels whose name ends with a digit.
        if ch and ch[-1].isdigit():
            # You can adjust these lists if your naming convention is different.
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
    Plots one window of the two-channel EEG data.

    Args:
        window_data: NumPy array of shape (2, n_samples) for the window.
        sps: Sampling rate (Hz).
        window_index: (Optional) window number (used in the title and filename).
    """
    n_samples = window_data.shape[1]
    time_axis = np.arange(n_samples) / sps
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, window_data[0, :], label='Left Hemisphere')
    plt.plot(time_axis, window_data[1, :], label='Right Hemisphere')
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
    Segments a two-channel signal into non-overlapping windows,
    performs wavelet decomposition and quantization on each window,
    and writes the results (quantized coefficients and channel labels) to text files.

    Args:
        data: NumPy array of shape (2, total_samples)
        sps: Sampling rate (Hz)
        coeffs_path: Path for the output coefficients text file.
        chans_path: Path for the output channel labels text file.
        wavelet: Wavelet type (default: 'db2').
        level: Decomposition level (default: 4).
        window_len_sec: Window length in seconds (e.g., 1.8 sec).
        plot_windows: If True, plots each window (or a random subset).
        plot_random_n: If set to an integer (less than total windows), randomly selects that many windows to plot.
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
    # --- S3 and Dataset Configuration for the Auditory Attention EEG Data ---
    s3_bucket = "dataframes--use1-az6--x-s3"
    s3_folder = "attention fintune/10803229"
    zip_filename = "EEG_raw.zip"
    s3_key = f"{s3_folder}/{zip_filename}"

    # Local output directory for processed files.
    output_base = "output-10803229"
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

        # Determine the dataset root.
        # The archive should contain a folder named "EEG_raw".
        dataset_root = os.path.join(extract_path, "EEG_raw")
        if not os.path.isdir(dataset_root):
            dataset_root = extract_path

        # List all subject directories in the dataset.
        subject_dirs = [d for d in os.listdir(dataset_root)
                        if os.path.isdir(os.path.join(dataset_root, d))]
        if not subject_dirs:
            print("No subject directories found in the dataset.")
            exit(1)

        for subj in sorted(subject_dirs):
            subject_path = os.path.join(dataset_root, subj)
            print(f"\nProcessing subject: {subj}")

            # Build the path to the data.bdf file in the subject folder.
            data_file = os.path.join(subject_path, "data.bdf")
            if not os.path.isfile(data_file):
                print(f"data.bdf not found in {subject_path}. Skipping.")
                continue

            try:
                raw = mne.io.read_raw_bdf(data_file, preload=True, verbose=False)
            except Exception as e:
                print(f"Error loading {data_file}: {e}")
                continue

            print("Channel names:", raw.info["ch_names"])
            print("Raw data shape:", raw.get_data().shape)

            # Select only EEG channels.
            raw.pick_types(eeg=True)
            print("After picking EEG channels, shape:", raw.get_data().shape)

            # Get the EEG data and sampling frequency.
            eeg_data = raw.get_data()
            fs = raw.info["sfreq"]

            # Compute hemispheric averages.
            try:
                twoch_data = average_hemispheric_channels(eeg_data, raw.info["ch_names"])
            except Exception as e:
                print(f"Error in hemispheric averaging for {subj}: {e}")
                continue
            print("Hemispheric averaged data shape:", twoch_data.shape)

            # Preprocess the two-channel data.
            prep_data, new_fs = preprocess_data(twoch_data, fs)

            # Define output file paths.
            coeffs_path = os.path.join(output_base, f"{subj}_combined_coeffs.txt")
            chans_path = os.path.join(output_base, f"{subj}_combined_channels.txt")

            # Process the preprocessed data (windowing, wavelet decomposition, quantization) and save.
            process_and_save(prep_data, new_fs, coeffs_path, chans_path,
                             wavelet='db2', level=4, window_len_sec=1.18, plot_windows=True)
            print(f"Finished processing subject: {subj}")

    print("Done!")
