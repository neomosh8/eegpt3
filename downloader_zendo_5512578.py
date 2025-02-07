#!/usr/bin/env python
"""
This script processes the EEG experiment data from the language/tactile dataset.
Each subject’s EEG data is provided as a ZIP file in the S3 directory:
    s3://dataframes--use1-az6--x-s3/attention fintune/5512578/
Subject ZIP files are named (e.g.) "al.zip", "yr.zip", etc. When extracted, the ZIP may
contain a subject folder (e.g. "chap") and inside that folder the session folders (e.g. "chap_EEG_1",
"chap_EEG_2") are located.

The processing pipeline is:
  1. Download the subject ZIP from S3 and extract it.
  2. Determine the subject folder and locate session folders.
  3. For each session, load the BrainVision files (.vhdr, .eeg, .vmrk) using MNE.
  4. Concatenate sessions (if >1), pick only EEG channels, and compute a two‐channel hemispheric average.
  5. Preprocess the two‑channel data.
  6. Segment the preprocessed data into windows, perform wavelet decomposition and quantization,
     and save the outputs.
  7. Optionally plot selected windows.
"""

import os
import boto3
import tempfile
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import pywt

# Import your custom functions (ensure these are available in your PYTHONPATH)
from utils import preprocess_data, wavelet_decompose_window, quantize_number

# List of subject IDs (18 subjects)
SUBJECT_IDS = ['al', 'yr', 'alio', 'chap', 'sep', 'phil', 'lad', 'calco',
               'hudi', 'nima', 'ogre', 'raqu', 'nikf', 'zartan', 'naga', 'miya',
               'elios', 'olio']

# S3 configuration
S3_BUCKET = "dataframes--use1-az6--x-s3"
S3_FOLDER = "attention fintune/5512578"  # Files are stored under this prefix

# Output directory for processed files (local)
OUTPUT_BASE = "output-5512578"
os.makedirs(OUTPUT_BASE, exist_ok=True)


# -----------------------------------------------------------------------------
# Helper functions

def get_session_folders(extracted_dir, subject_id):
    """
    Given the directory where a subject’s ZIP file has been extracted,
    attempt to find the subject folder and then list all session folders within it.
    Session folders are those whose names contain "EEG" (case-insensitive).

    Args:
        extracted_dir: Path to the extraction directory.
        subject_id: The subject ID string (e.g., "chap").

    Returns:
        A list of paths to session folders.
    """
    # First, check if there is a subfolder whose name matches the subject ID (case-insensitive)
    subject_folder = None
    for entry in os.listdir(extracted_dir):
        if os.path.isdir(os.path.join(extracted_dir, entry)) and entry.lower() == subject_id.lower():
            subject_folder = os.path.join(extracted_dir, entry)
            break
    base_dir = subject_folder if subject_folder is not None else extracted_dir

    session_folders = []
    for entry in os.listdir(base_dir):
        path = os.path.join(base_dir, entry)
        if os.path.isdir(path) and "EEG" in entry.upper():
            session_folders.append(path)
    return session_folders


def average_hemispheric_channels(data, ch_names):
    """
    Averages EEG channels into two hemispheric signals.

    Channels whose names end with an odd digit are assigned to the left hemisphere,
    while channels whose names end with an even digit are assigned to the right hemisphere.
    Channels that do not end with a digit (e.g., midline channels such as Fz, Cz, Pz) are ignored.

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


# -----------------------------------------------------------------------------
# Main processing function for a subject ZIP

def process_subject_zip(subject_id, s3_client):
    """
    Downloads and processes a subject's EEG experiment ZIP file.

    Args:
        subject_id: The subject's ID string (e.g., "chap", "hudi", etc.)
        s3_client: An initialized boto3 S3 client.

    Returns:
        None (results are saved to OUTPUT_BASE)
    """
    zip_filename = f"{subject_id}.zip"
    s3_key = f"{S3_FOLDER}/{zip_filename}"
    print(f"\nProcessing subject {subject_id} (S3 key: {s3_key})")

    # Create a temporary directory for this subject's files.
    with tempfile.TemporaryDirectory() as temp_dir:
        local_zip_path = os.path.join(temp_dir, zip_filename)
        try:
            s3_client.download_file(S3_BUCKET, s3_key, local_zip_path)
            print(f"Downloaded {zip_filename} to {local_zip_path}")
        except Exception as e:
            print(f"Error downloading {zip_filename}: {e}")
            return

        # Extract the ZIP file.
        try:
            import zipfile
            with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            print(f"Extracted {zip_filename} into {temp_dir}")
        except Exception as e:
            print(f"Error extracting {zip_filename}: {e}")
            return

        # Get the session folders.
        session_folders = get_session_folders(temp_dir, subject_id)
        if not session_folders:
            print(f"No session folders found for subject {subject_id}.")
            return

        raw_list = []
        # Process each session folder.
        for session_folder in sorted(session_folders):
            vhdr_files = [f for f in os.listdir(session_folder) if f.lower().endswith(".vhdr")]
            if not vhdr_files:
                print(f"No .vhdr file found in {session_folder}. Skipping this session.")
                continue
            vhdr_path = os.path.join(session_folder, vhdr_files[0])
            try:
                raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)
                print(f"Loaded session from {vhdr_path} with shape {raw.get_data().shape}")
                raw_list.append(raw)
            except Exception as e:
                print(f"Error loading {vhdr_path}: {e}")

        if not raw_list:
            print(f"No valid sessions loaded for subject {subject_id}.")
            return

        # Concatenate sessions if more than one exists.
        if len(raw_list) > 1:
            try:
                raw_combined = mne.concatenate_raws(raw_list)
                print(f"Concatenated {len(raw_list)} sessions; combined shape: {raw_combined.get_data().shape}")
            except Exception as e:
                print(f"Error concatenating sessions for subject {subject_id}: {e}")
                return
        else:
            raw_combined = raw_list[0]

        # Retain only EEG channels (exclude channels such as "Sound" and "Button").
        raw_combined.pick_types(eeg=True)
        print(f"After picking EEG channels, shape: {raw_combined.get_data().shape}")

        # Get the EEG data and sampling rate.
        eeg_data = raw_combined.get_data()  # shape: (n_channels, n_samples)
        fs = raw_combined.info["sfreq"]

        # Compute hemispheric averages.
        try:
            twoch_data = average_hemispheric_channels(eeg_data, raw_combined.info["ch_names"])
        except Exception as e:
            print(f"Error computing hemispheric averages for subject {subject_id}: {e}")
            return
        print(f"Hemispheric averaged data shape: {twoch_data.shape}")

        # Preprocess the two-channel data.
        prep_data, new_fs = preprocess_data(twoch_data, fs)

        # Define output file paths.
        coeffs_path = os.path.join(OUTPUT_BASE, f"{subject_id}_combined_coeffs.txt")
        chans_path = os.path.join(OUTPUT_BASE, f"{subject_id}_combined_channels.txt")

        # Process (windowing, wavelet decomposition, quantization) and save.
        process_and_save(prep_data, new_fs, coeffs_path, chans_path,
                         wavelet='db2', level=4, window_len_sec=1.8, plot_windows=True)
        print(f"Finished processing subject: {subject_id}")


# -----------------------------------------------------------------------------
# Main script

if __name__ == "__main__":
    # Initialize boto3 S3 client.
    s3_client = boto3.client("s3")

    # Process each subject.
    for subject_id in sorted(SUBJECT_IDS):
        process_subject_zip(subject_id, s3_client)

    print("Done!")
