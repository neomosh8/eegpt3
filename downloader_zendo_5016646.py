#!/usr/bin/env python
"""
Processing pipeline for a BCI2000 dataset for a 3-class BCI.
The dataset (from 12 healthy participants over up to 4 sessions) was recorded
using the BCI2000 file format. For each session (provided as a ZIP file on S3),
the EEG signal (channels 1–20) is loaded. Out of these, only the lateral channels
are used for hemispheric averaging. In this example we assume the following grouping:

  Left hemisphere:
      Fp1, F3, C3, P3, O1, F7, T7, P7
  Right hemisphere:
      Fp2, F4, C4, P4, O2, F8, T8, P8

After averaging, a two‐channel signal is obtained. This signal is preprocessed,
segmented into non-overlapping windows, decomposed with wavelets, and quantized.
The quantized coefficients and corresponding channel labels are saved to text files.

Note:
  - It is assumed that you have implemented a function `load_bcidat(filename)`
    (for example in a module named `bci_utils`) that loads BCI2000 files and returns:
         (signal, states, parameters)
    where `signal` is an array of shape (n_samples, n_channels),
    and `parameters` is a dictionary containing (at least) the sampling rate
    (e.g. parameters["SamplingRate"]) and channel names (e.g. parameters["ChannelNames"]).

  - The custom functions `preprocess_data`, `wavelet_decompose_window`, and
    `quantize_number` are imported from a local `utils` module.

  - AWS credentials must be properly configured for boto3 access.
"""

import os
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import zipfile
import tempfile
import boto3

# Import your custom utilities.
from utils import preprocess_data, wavelet_decompose_window, quantize_number

# Import your BCI2000 file loader.
from bci_utils import load_bcidat  # You must implement this function.

# Define the lateral channel lists for hemispheric averaging.
LEFT_CHANNELS = ["Fp1", "F3", "C3", "P3", "O1", "F7", "T7", "P7"]
RIGHT_CHANNELS = ["Fp2", "F4", "C4", "P4", "O2", "F8", "T8", "P8"]


def average_hemispheric_channels_bci(data, channel_names, left_list, right_list):
    """
    Averages EEG channels into left and right hemisphere signals.

    Args:
        data: NumPy array of shape (n_channels, n_samples)
        channel_names: List of channel names corresponding to the rows of data.
        left_list: List of channel names for the left hemisphere.
        right_list: List of channel names for the right hemisphere.

    Returns:
        A 2-channel NumPy array where:
          - Channel 0 is the average of the left hemisphere channels.
          - Channel 1 is the average of the right hemisphere channels.
    """
    left_indices = [i for i, ch in enumerate(channel_names) if ch in left_list]
    right_indices = [i for i, ch in enumerate(channel_names) if ch in right_list]

    if not left_indices:
        raise ValueError("No left hemisphere channels found.")
    if not right_indices:
        raise ValueError("No right hemisphere channels found.")

    left_avg = np.mean(data[left_indices, :], axis=0)
    right_avg = np.mean(data[right_indices, :], axis=0)

    return np.stack([left_avg, right_avg])


def plot_window(window_data, sps, window_index=None):
    """
    Plots a single window of a 2-channel EEG signal.

    Args:
        window_data: NumPy array of shape (2, n_samples) for the window.
        sps: Sampling rate in Hz.
        window_index: Optional window number (for title and filename).
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
    Segments a 2-channel signal into non-overlapping windows, applies wavelet
    decomposition and quantization, and writes the results to text files.

    Args:
        data: NumPy array of shape (2, total_samples)
        sps: Sampling rate (Hz)
        coeffs_path: File path for quantized coefficients output.
        chans_path: File path for channel labels output.
        wavelet: Type of wavelet to use (default 'db2').
        level: Decomposition level (default 4).
        window_len_sec: Length of each window in seconds (e.g., 1.8).
        plot_windows: If True, plot windows.
        plot_random_n: If an integer and less than total windows, randomly select that many windows to plot.
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
            window_data = data[:, start_idx:end_idx]  # (2, n_window_samples)

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
    # --- S3 and Dataset Configuration ---
    s3_bucket = "dataframes--use1-az6--x-s3"
    # For example, all session ZIP files are stored under this S3 folder.
    s3_folder = "attention fintune/5016646"

    # List of ZIP filenames to process.
    zip_filenames = [
        "P01Session01.zip", "P01Session02.zip", "P01Session03.zip", "P01Session04.zip",
        "P02Se01.zip", "P02Se02.zip", "P02Se03.zip", "P02Se04.zip",
        "P03Se01.zip", "P03Se02.zip", "P03Se03.zip", "P03Se04.zip",
        "P04Se01.zip", "P04Se02.zip", "P04Se03.zip", "P04Se04.zip",
        "P05Se01.zip", "P05Se02.zip", "P05Se03.zip", "P05Se04.zip",
        "P06Se01.zip", "P06Se02.zip", "P06Se03.zip", "P06Se04.zip",
        "P07Se01.zip", "P07Se02.zip", "P07Se03.zip", "P07Se04.zip",
        "P08Se01.zip", "P08Se02.zip", "P08Se03.zip", "P08Se04.zip",
        "P09Se01.zip", "P09Se02.zip", "P09Se03.zip", "P09Se04.zip",
        "P10Se01.zip", "P10Se02.zip", "P10Se03.zip", "P10Se04.zip",
        "P11Se01.zip", "P11Se02.zip", "P11Se03.zip", "P11Se04.zip",
        "P12Se01.zip", "P12Se02.zip", "P12Se03.zip", "P12Se04.zip",
    ]

    # Local output directory.
    output_base = "output-5016646"
    os.makedirs(output_base, exist_ok=True)

    # Initialize the boto3 S3 client.
    s3 = boto3.client("s3")

    # Create a temporary directory for downloads and extraction.
    with tempfile.TemporaryDirectory() as temp_dir:
        for zip_filename in zip_filenames:
            s3_key = f"{s3_folder}/{zip_filename}"
            local_zip_path = os.path.join(temp_dir, zip_filename)
            print(f"Downloading s3://{s3_bucket}/{s3_key} ...")
            try:
                s3.download_file(s3_bucket, s3_key, local_zip_path)
                print(f"Downloaded {zip_filename} to {local_zip_path}")
            except Exception as e:
                print(f"Error downloading {zip_filename}: {e}")
                continue

            # Create a folder for extraction.
            extract_path = os.path.join(temp_dir, os.path.splitext(zip_filename)[0])
            os.makedirs(extract_path, exist_ok=True)
            try:
                with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                print(f"Extracted {zip_filename} to {extract_path}")
            except Exception as e:
                print(f"Error extracting {zip_filename}: {e}")
                continue

            # Find the BCI2000 file (assumed to have a .dat extension).
            dat_files = [f for f in os.listdir(extract_path) if f.lower().endswith(".dat")]
            if not dat_files:
                print(f"No .dat file found in {zip_filename}. Skipping.")
                continue
            data_file = os.path.join(extract_path, dat_files[0])

            subject_id = os.path.splitext(zip_filename)[0]  # e.g., "P01Session01"
            print(f"\nProcessing {subject_id} ...")

            try:
                # Load the BCI2000 data. (Assumes signal shape is (n_samples, n_channels).)
                signal, states, parameters = load_bcidat(data_file)
            except Exception as e:
                print(f"Error loading {data_file}: {e}")
                continue

            # Transpose signal to shape (n_channels, n_samples) if needed.
            eeg_data = signal.T

            # Get sampling frequency.
            fs = parameters.get("SamplingRate", 250)  # Default to 250 Hz if not provided.

            # Get channel names from parameters or use a default list.
            channel_names = parameters.get("ChannelNames",
                                           ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
                                            "O1", "O2", "F7", "F8", "T7", "T8", "P7", "P8",
                                            "Fz", "NAS", "Fz", "Oz"])

            # For hemispheric averaging, select lateral channels only.
            # (Left hemisphere: LEFT_CHANNELS, Right hemisphere: RIGHT_CHANNELS)
            twoch_data = average_hemispheric_channels_bci(eeg_data, channel_names, LEFT_CHANNELS, RIGHT_CHANNELS)
            print("Hemispheric averaged data shape:", twoch_data.shape)

            # Preprocess the two-channel data.
            prep_data, new_fs = preprocess_data(twoch_data, fs)

            # Define output file paths.
            coeffs_path = os.path.join(output_base, f"{subject_id}_combined_coeffs.txt")
            chans_path = os.path.join(output_base, f"{subject_id}_combined_channels.txt")

            # Process: windowing, wavelet decomposition, quantization, and save.
            process_and_save(prep_data, new_fs, coeffs_path, chans_path,
                             wavelet='db2', level=4, window_len_sec=1.8, plot_windows=True)
            print(f"Finished processing {subject_id}")

    print("Done!")
