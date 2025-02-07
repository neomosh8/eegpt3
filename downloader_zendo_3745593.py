#!/usr/bin/env python
"""
This script processes the mental_attention EEG dataset.

Dataset description:
  - 20 subjects performed an experiment in which they were instructed to
    mentally concentrate on a target for 15-second periods (interleaved with
    relax/get-ready phases).
  - Each subject’s data is stored in an MFF folder (e.g., "sub-001.mff") inside
    the mental_attention.zip archive.
  - The EEG is assumed to be recorded continuously; here we segment the data
    into non-overlapping 15-second windows.
  - For each subject, after picking only EEG channels, a hemispheric averaging is
    performed using the following logic:
       • Channels whose names end with an odd digit are averaged to produce a left-hemisphere signal.
       • Channels whose names end with an even digit are averaged to produce a right-hemisphere signal.
    (Channels without a trailing digit are ignored.)
  - The resulting 2-channel data is preprocessed, windowed, decomposed using wavelets,
    quantized, and then the quantized coefficients and their channel labels are saved to text files.

NOTE: This version patches each subject’s info.xml file to fix the recordTime field.
"""

# --- Preliminary Imports and Checks ---

# Check that defusedxml is installed, as it is required by MNE for reading EGI MFF data.
try:
    import defusedxml.ElementTree
except ImportError:
    raise ImportError(
        "For reading EGI MFF data, the module defusedxml is required. "
        "Please install it with 'pip install defusedxml' or 'conda install -c conda-forge defusedxml'."
    )

import re
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

# Import your custom processing functions.
from utils import preprocess_data, wavelet_decompose_window, quantize_number


# --- Helper Function to Patch info.xml ---

def patch_info_xml(subject_path):
    """
    Patches the info.xml file in the given subject folder by fixing the <recordTime> field.
    The problematic recordTime string (e.g., '2020-03-06T17:54:25.51953000:00')
    is truncated to include only six fractional digits (e.g., '2020-03-06T17:54:25.519530').

    Args:
        subject_path: Path to the subject folder (e.g., ".../sub-001.mff")
    """
    info_path = os.path.join(subject_path, "info.xml")
    if not os.path.exists(info_path):
        return
    with open(info_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Pattern to find the recordTime element.
    pattern = r"(<recordTime>)(.*?)(</recordTime>)"

    def repl(match):
        rt_str = match.group(2)
        # Use regex to capture the desired format: YYYY-MM-DDTHH:MM:SS.<6 digits>
        m = re.match(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6})", rt_str)
        if m:
            new_rt = m.group(1)
        else:
            new_rt = rt_str  # fallback (should not happen often)
        return f"{match.group(1)}{new_rt}{match.group(3)}"

    new_content = re.sub(pattern, repl, content)
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(new_content)


# --- Processing Functions ---

def average_hemispheric_channels(data, ch_names):
    """
    Averages EEG channels into two hemispheric signals.

    Channels whose names end with an odd digit are considered left-hemisphere,
    while those ending with an even digit are considered right-hemisphere.
    Channels that do not end with a digit (e.g. midline channels) are ignored.

    Args:
        data: NumPy array of shape (n_channels, n_samples)
        ch_names: List of channel names corresponding to rows in data.

    Returns:
        A 2-channel NumPy array:
          - Channel 0: Average of left hemisphere channels.
          - Channel 1: Average of right hemisphere channels.
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
    Plots one window of the two-channel EEG data.

    Args:
        window_data: NumPy array of shape (2, n_samples) for the window.
        sps: Sampling rate in Hz.
        window_index: (Optional) Window number (used in title and filename).
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
                     wavelet='db2', level=4, window_len_sec=15,
                     plot_windows=False, plot_random_n=1):
    """
    Segments a two-channel signal into non-overlapping windows,
    performs wavelet decomposition and quantization on each window,
    and writes the quantized coefficients and channel labels to text files.

    Args:
        data: NumPy array of shape (2, total_samples)
        sps: Sampling rate (Hz)
        coeffs_path: Output file path for quantized coefficients.
        chans_path: Output file path for channel labels.
        wavelet: Wavelet type (default 'db2').
        level: Decomposition level (default: 4).
        window_len_sec: Window length in seconds (here, 15 sec).
        plot_windows: If True, plots selected windows.
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


# --- Main Processing Pipeline ---

if __name__ == "__main__":
    # --- S3 and Dataset Configuration for the Mental Attention EEG Data ---
    s3_bucket = "dataframes--use1-az6--x-s3"
    s3_folder = "attention fintune/3745593"
    zip_filename = "mental_attention.zip"
    s3_key = f"{s3_folder}/{zip_filename}"

    # Local output directory for processed files.
    output_base = "output-3745593"
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

        # Determine the dataset root.
        if "mental_attention" in os.listdir(extract_path):
            dataset_root = os.path.join(extract_path, "mental_attention")
        else:
            dataset_root = extract_path

        # List subject entries ending with ".mff" (each is a directory).
        subject_entries = [d for d in os.listdir(dataset_root)
                           if d.endswith(".mff") and os.path.isdir(os.path.join(dataset_root, d))]
        if not subject_entries:
            print("No subject directories found in the dataset.")
            exit(1)

        for subj_entry in sorted(subject_entries):
            # Remove the ".mff" extension for a clean subject ID.
            subject_id = os.path.splitext(subj_entry)[0]
            subject_path = os.path.join(dataset_root, subj_entry)
            print(f"\nProcessing subject: {subject_id}")

            # Patch the info.xml file to fix recordTime format.
            patch_info_xml(subject_path)

            try:
                # Load the MFF data using MNE’s read_raw_egi.
                raw = mne.io.read_raw_egi(subject_path, preload=True, verbose=False)
            except Exception as e:
                print(f"Error loading {subject_path}: {e}")
                continue

            print("Channel names:", raw.info["ch_names"])
            print("Raw data shape:", raw.get_data().shape)

            # Retain only EEG channels.
            raw.pick_types(eeg=True)
            print("After picking EEG channels, shape:", raw.get_data().shape)

            # Get the EEG data and sampling frequency.
            eeg_data = raw.get_data()
            fs = raw.info["sfreq"]

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

            # Process the preprocessed data: segment into 15-sec windows, decompose and quantize.
            process_and_save(prep_data, new_fs, coeffs_path, chans_path,
                             wavelet='db2', level=4, window_len_sec=15, plot_windows=True)
            print(f"Finished processing subject: {subject_id}")

    print("Done!")
