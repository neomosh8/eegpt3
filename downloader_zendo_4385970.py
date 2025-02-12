#!/usr/bin/env python
"""
This script processes a language-EEG dataset stored in BrainVision format on S3.
The files are located in the S3 directory:
    dataframes--use1-az6--x-s3/attention fintune/4385970/
Each subject’s recording is defined by three files with a common stem (e.g. “P10_6_12_2018”):
    - P10_6_12_2018.vhdr  (header)
    - P10_6_12_2018.eeg   (data)
    - P10_6_12_2018.vmrk  (markers)

For each subject, the script:
  1. Downloads the three files into a temporary folder.
  2. Loads the data using mne.io.read_raw_brainvision().
  3. Picks EEG channels.
  4. Computes a two-channel hemispheric average:
       - Left hemisphere: average of channels whose names end with an odd digit.
       - Right hemisphere: average of channels whose names end with an even digit.
       (Midline channels—those that do not end in a digit—are ignored.)
  5. Preprocesses the two-channel data using preprocess_data().
  6. Segments the data into windows, applies wavelet decomposition and quantization,
     and writes the quantized coefficients and channel labels to text files.
"""

import os
import boto3
import tempfile
import zipfile  # not used here since files are not zipped
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import pywt

# Import your custom processing functions.
from utils import preprocess_data, wavelet_decompose_window, quantize_number


def average_hemispheric_channels(data, ch_names):
    """
    Averages EEG channels into two hemispheric signals.

    Channels whose names end with an odd digit are assigned to the left hemisphere,
    while channels whose names end with an even digit are assigned to the right hemisphere.
    Channels that do not end with a digit (e.g. midline channels like Fz, Cz, Pz) are ignored.

    Args:
        data: NumPy array of shape (n_channels, n_samples)
        ch_names: List of channel names corresponding to rows in data.

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
        sps: Sampling rate (Hz).
        window_index: Optional window number for title/filename.
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
        coeffs_path: Path to the output coefficients text file.
        chans_path: Path to the output channel labels text file.
        wavelet: Wavelet type (default: 'db2').
        level: Decomposition level (default: 4).
        window_len_sec: Window length in seconds (e.g., 1.8 sec).
        plot_windows: If True, plots windows (or a random subset).
        plot_random_n: If an integer (and less than total number of windows),
                       randomly select that many windows to plot.
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


def download_subject_files(s3_client, bucket, subject_prefix, local_subject_dir):
    """
    Given a subject’s common file prefix (e.g. "attention fintune/4385970/P10_6_12_2018.vhdr"),
    download the associated BrainVision files (.vhdr, .eeg, .vmrk) into a local directory.

    Args:
        s3_client: An initialized boto3 S3 client.
        bucket: S3 bucket name.
        subject_prefix: Full S3 key of the subject's .vhdr file.
        local_subject_dir: Local directory where files will be saved.

    Returns:
        local_vhdr_path: Path to the downloaded .vhdr file.
    """
    # Extract the base filename (e.g., "P10_6_12_2018")
    base_filename = os.path.splitext(os.path.basename(subject_prefix))[0]
    extensions = ['.vhdr', '.eeg', '.vmrk']

    for ext in extensions:
        key = os.path.join(os.path.dirname(subject_prefix), base_filename + ext)
        local_path = os.path.join(local_subject_dir, base_filename + ext)
        try:
            s3_client.download_file(bucket, key, local_path)
            print(f"Downloaded {key} to {local_path}")
        except Exception as e:
            print(f"Error downloading {key}: {e}")
            raise e
    return os.path.join(local_subject_dir, base_filename + ".vhdr")


if __name__ == "__main__":
    # --- S3 and Dataset Configuration for the Language EEG Data ---
    s3_bucket = "dataframes--use1-az6--x-s3"
    s3_prefix = "attention fintune/4385970/"  # Note: files are directly under this prefix.

    # Output directory for processed files.
    output_base = "output-4385970"
    os.makedirs(output_base, exist_ok=True)

    # Initialize boto3 S3 client.
    s3 = boto3.client("s3")

    # List all objects under the specified prefix.
    response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=s3_prefix)
    if 'Contents' not in response:
        print("No files found in the specified S3 directory.")
        exit(1)

    # Identify all BrainVision header files (.vhdr).
    vhdr_keys = []
    for obj in response['Contents']:
        key = obj['Key']
        if key.lower().endswith(".vhdr"):
            vhdr_keys.append(key)
    if not vhdr_keys:
        print("No .vhdr files found in the S3 directory.")
        exit(1)

    # Process each subject.
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        for vhdr_key in sorted(vhdr_keys):
            subject_id = os.path.splitext(os.path.basename(vhdr_key))[0]
            print(f"\nProcessing subject: {subject_id}")

            # Create a local folder for this subject.
            local_subject_dir = os.path.join(temp_dir, subject_id)
            os.makedirs(local_subject_dir, exist_ok=True)

            try:
                local_vhdr_path = download_subject_files(s3, s3_bucket, vhdr_key, local_subject_dir)
            except Exception as e:
                print(f"Skipping subject {subject_id} due to download error.")
                continue

            # Load the subject's BrainVision data.
            try:
                raw = mne.io.read_raw_brainvision(local_vhdr_path, preload=True, verbose=False)
            except Exception as e:
                print(f"Error loading {local_vhdr_path}: {e}")
                continue

            print("Channel names:", raw.info["ch_names"])
            print("Raw data shape:", raw.get_data().shape)

            # Keep only EEG channels.
            raw.pick_types(eeg=True)
            print("After picking EEG channels, shape:", raw.get_data().shape)

            eeg_data = raw.get_data()  # shape: (n_channels, n_samples)
            fs = raw.info["sfreq"]  # sampling rate

            # Compute hemispheric averages.
            try:
                twoch_data = average_hemispheric_channels(eeg_data, raw.info["ch_names"])
            except Exception as e:
                print(f"Error computing hemispheric averages for {subject_id}: {e}")
                continue
            print("Hemispheric averaged data shape:", twoch_data.shape)

            # Preprocess the two-channel data.
            prep_data, new_fs = preprocess_data(twoch_data, fs)

            # Define output file paths.
            coeffs_path = os.path.join(output_base, f"{subject_id}_combined_coeffs.txt")
            chans_path = os.path.join(output_base, f"{subject_id}_combined_channels.txt")

            # Process the data (windowing, wavelet decomposition, quantization) and save.
            process_and_save(prep_data, new_fs, coeffs_path, chans_path,
                             wavelet='db2', level=4, window_len_sec=1.18, plot_windows=True)
            print(f"Finished processing subject: {subject_id}")

    print("Done!")
