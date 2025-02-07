#!/usr/bin/env python
"""
This script processes raw EEG data accompanying the paper
"Do syntactic and semantic similarity lead to interference effects? Evidence from self-paced reading and event-related potentials using German."

The raw EEG data are provided as several large ZIP files in S3 under:
    s3://dataframes--use1-az6--x-s3/attention fintune/10518106/

Each ZIP file (e.g., "EEG_rawdata_session1_subj01_50.zip") covers a range of subjects for one session.
It is assumed that each ZIP file, when extracted, contains one or more EEG data files in BrainVision format
(i.e. files ending with .vhdr, with corresponding .eeg and .vmrk files).

For each subject, the processing pipeline is:
  1. Load the subject’s EEG data using mne.io.read_raw_brainvision().
  2. Retain only EEG channels.
  3. Compute a two-channel hemispheric average:
       - Left hemisphere: average of lateral channels whose names end with an odd digit.
       - Right hemisphere: average of lateral channels whose names end with an even digit.
       (Midline channels are ignored.)
  4. Preprocess the resulting two-channel data using a custom function (preprocess_data).
  5. Segment the data into non-overlapping windows, perform wavelet decomposition and quantization,
     and save the quantized coefficients and channel labels to text files.
  6. Optionally plot selected windows.
"""

import os
import re
import boto3
import tempfile
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import pywt

# Import your custom functions – ensure these are available in your PYTHONPATH.
from utils import preprocess_data, wavelet_decompose_window, quantize_number

# --- S3 and Dataset Configuration ---
S3_BUCKET = "dataframes--use1-az6--x-s3"
S3_FOLDER = "attention fintune/10518106"  # folder containing the raw data ZIP files

# Output directory for processed files.
OUTPUT_BASE = "output-10518106"
os.makedirs(OUTPUT_BASE, exist_ok=True)


# --- Helper Functions ---

def list_zip_files(s3_client, bucket, prefix):
    """
    List all ZIP file keys in the given S3 prefix that start with "EEG_rawdata_session".
    """
    zip_keys = []
    continuation_token = None
    while True:
        list_kwargs = {"Bucket": bucket, "Prefix": prefix}
        if continuation_token:
            list_kwargs["ContinuationToken"] = continuation_token
        response = s3_client.list_objects_v2(**list_kwargs)
        for obj in response.get('Contents', []):
            key = obj['Key']
            if key.lower().endswith('.zip') and "EEG_rawdata_session" in key:
                zip_keys.append(key)
        if response.get('IsTruncated'):
            continuation_token = response.get('NextContinuationToken')
        else:
            break
    return zip_keys


def extract_session_from_zipname(zip_name):
    """
    Parse the ZIP file name to extract session information (e.g., "session1" or "session2").
    Assumes the ZIP file name contains 'session1' or 'session2'.
    """
    m = re.search(r'session(\d+)', zip_name, re.IGNORECASE)
    if m:
        return f"session{m.group(1)}"
    else:
        return "unknown_session"


def process_subject_file(vhdr_path, session_label, output_suffix=""):
    """
    Process a single subject’s EEG file in BrainVision format.
    Loads the data, picks EEG channels, computes hemispheric averages,
    preprocesses the data, segments into windows, performs wavelet decomposition/quantization,
    and writes the outputs.

    The subject ID is extracted from the vhdr file name (e.g., if the file name contains "subj01", that is used).

    Args:
        vhdr_path: Full local path to the .vhdr file.
        session_label: A string indicating the session (e.g., "session1").
        output_suffix: An optional extra suffix to append to output file names.
    """
    # Extract subject ID from the filename using a regex (e.g., "subj01" or similar).
    base = os.path.basename(vhdr_path)
    subj_match = re.search(r'(subj\d+)', base, re.IGNORECASE)
    subject_id = subj_match.group(1).lower() if subj_match else os.path.splitext(base)[0]
    # Append session label to subject id.
    full_id = f"{subject_id}_{session_label}{output_suffix}"
    print(f"Processing subject file: {base} (ID: {full_id})")

    try:
        raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)
    except Exception as e:
        print(f"Error loading {vhdr_path}: {e}")
        return
    print(f"Loaded data shape: {raw.get_data().shape}")

    # Retain only EEG channels.
    raw.pick_types(eeg=True)
    print(f"After picking EEG channels: {raw.get_data().shape}")

    eeg_data = raw.get_data()  # shape: (n_channels, n_samples)
    fs = raw.info["sfreq"]  # sampling rate (expected to be 1000 Hz for this dataset)

    try:
        twoch_data = average_hemispheric_channels(eeg_data, raw.info["ch_names"])
    except Exception as e:
        print(f"Error computing hemispheric averages for {full_id}: {e}")
        return
    print(f"Hemispheric averaged data shape: {twoch_data.shape}")

    # Preprocess the two-channel data.
    prep_data, new_fs = preprocess_data(twoch_data, fs)

    # Define output file paths.
    coeffs_path = os.path.join(OUTPUT_BASE, f"{full_id}_combined_coeffs.txt")
    chans_path = os.path.join(OUTPUT_BASE, f"{full_id}_combined_channels.txt")

    # Process (windowing, wavelet decomposition, quantization) and save.
    process_and_save(prep_data, new_fs, coeffs_path, chans_path,
                     wavelet='db2', level=4, window_len_sec=1.8, plot_windows=True)
    print(f"Finished processing subject: {full_id}")


def process_zip_file(zip_key, s3_client):
    """
    Downloads a ZIP file from S3, extracts it, and processes each subject file found inside.
    Assumes that the extracted contents contain one or more .vhdr files.

    Args:
        zip_key: The S3 key for the ZIP file.
        s3_client: An initialized boto3 S3 client.
    """
    zip_name = os.path.basename(zip_key)
    session_label = extract_session_from_zipname(zip_name)
    print(f"\nProcessing ZIP file: {zip_name} (Session: {session_label})")

    with tempfile.TemporaryDirectory() as temp_dir:
        local_zip_path = os.path.join(temp_dir, zip_name)
        try:
            s3_client.download_file(S3_BUCKET, zip_key, local_zip_path)
            print(f"Downloaded {zip_name} to {local_zip_path}")
        except Exception as e:
            print(f"Error downloading {zip_name}: {e}")
            return

        # Extract the ZIP file.
        try:
            import zipfile
            with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            print(f"Extracted {zip_name} into {temp_dir}")
        except Exception as e:
            print(f"Error extracting {zip_name}: {e}")
            return

        # Recursively find all .vhdr files in the extracted directory.
        vhdr_files = []
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.lower().endswith('.vhdr'):
                    vhdr_files.append(os.path.join(root, file))
        if not vhdr_files:
            print(f"No .vhdr files found in extracted {zip_name}.")
            return

        # Process each subject file.
        for vhdr_path in sorted(vhdr_files):
            process_subject_file(vhdr_path, session_label)


# --- Main Script ---
if __name__ == "__main__":
    s3_client = boto3.client("s3")
    zip_keys = list_zip_files(s3_client, S3_BUCKET, S3_FOLDER)

    if not zip_keys:
        print("No ZIP files found in the specified S3 folder.")
        exit(1)

    print(f"Found {len(zip_keys)} ZIP files. Processing them now...")
    for zip_key in sorted(zip_keys):
        process_zip_file(zip_key, s3_client)

    print("Done!")
