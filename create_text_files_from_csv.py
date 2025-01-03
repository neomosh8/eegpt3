import os
import glob
import random
import shutil
import tempfile
import time
import boto3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Wavelet parameters
wvlet = 'db2'
level = 2

# --------------------------------------------------------------------------------
# Import your custom wavelet/quantization utils (or define them here if needed)
# --------------------------------------------------------------------------------
from utils import (
    wavelet_decompose_window,
    wavelet_reconstruct_window,
    quantize_number,
    dequantize_number,
    pwelch_z, call_gpt_for_instructions,
    preprocess_data,
    calculate_sps,
    validate_round_trip, list_s3_folders, list_csv_files_in_folder
)
from concurrent.futures import ProcessPoolExecutor, as_completed

def parallel_process_csv_files(csv_files, max_workers=4):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_csv_file_s3, f): f for f in csv_files}
        for future in as_completed(futures):
            csvfile = futures[future]
            try:
                future.result()
                print(f"Done: {csvfile}")
            except Exception as e:
                print(f"Error: {csvfile} -> {e}")


def generate_quantized_files_local(
        csv_file: str,
        output_folder: str,
        window_length_sec: float = 2.0,
        wvlet: str = 'db2',
        level: int = 2
):
    """
    Iterate over a single local CSV file, wavelet-decompose+quantize
    each file's [Left, Right] average EEG signals, and write
    _quantized_coeffs.txt + _quantized_channels.txt to a local `output_folder`.
    """

    base_name = os.path.splitext(os.path.basename(csv_file))[0]

    # Create output file paths
    output_coeffs_file = os.path.join(output_folder, f"{base_name}_quantized_coeffs.txt")
    output_channels_file = os.path.join(output_folder, f"{base_name}_quantized_channels.txt")

    with open(output_coeffs_file, "w") as f_coeffs, open(output_channels_file, "w") as f_chans:
        df = pd.read_csv(csv_file)

        # Ask GPT if we skip or process => channels to drop
        all_columns = list(df.columns)
        instructions = call_gpt_for_instructions(
            channel_names=all_columns,
            dataset_id=base_name
        )
        if instructions["action"] == "skip":
            print(f"Skipping dataset '{base_name}' as instructed by GPT.")
            return

        channels_to_drop = instructions.get("channels_to_drop", [])
        print(f"Dropping channels: {str(channels_to_drop)}")
        filtered_columns = [col for col in all_columns if col not in channels_to_drop]

        # Identify left vs right channels
        left_chs_in_csv = []
        right_chs_in_csv = []
        for ch in filtered_columns:
            # skip if ends with 'z' or if last character isn't a digit, etc.
            if ch.endswith('z'):
                continue
            if not ch[-1].isdigit():
                continue
            # last character is digit => check odd/even
            if int(ch[-1]) % 2 == 0:
                right_chs_in_csv.append(ch)
            else:
                left_chs_in_csv.append(ch)

        if not left_chs_in_csv and not right_chs_in_csv:
            print(f"No valid left/right channels found in {csv_file}. Skipping.")
            return

        # Create the 2-channel data array: [Left, Right]
        if left_chs_in_csv:
            left_data = df[left_chs_in_csv].mean(axis=1).values
        else:
            left_data = np.zeros(len(df))

        if right_chs_in_csv:
            right_data = df[right_chs_in_csv].mean(axis=1).values
        else:
            right_data = np.zeros(len(df))

        data_2d = np.vstack([left_data, right_data])

        # Original sampling rate
        original_sps = calculate_sps(csv_file)
        # Preprocess (resample + bandpass, etc.)
        preprocessed_data, new_sps = preprocess_data(data_2d, original_sps)

        n_window_samples = int(window_length_sec * new_sps)
        num_channels, total_samples = preprocessed_data.shape

        if n_window_samples <= 0 or total_samples < n_window_samples:
            print(f"Skipping {csv_file} due to invalid window size or not enough samples.")
            return

        # Slide through the data in non-overlapping windows
        for window_start in range(0, total_samples - n_window_samples + 1, n_window_samples):
            window_end = window_start + n_window_samples

            all_channel_coeffs = []
            all_channel_names = []

            # For exactly 2 channels: 0 => left, 1 => right
            for ch_idx in range(2):
                ch_name_id = "1" if ch_idx == 0 else "2"
                channel_data = preprocessed_data[ch_idx, window_start:window_end]
                channel_data_2d = channel_data[np.newaxis, :]

                # Wavelet Decompose
                (decomposed_channels,
                 coeffs_lengths,
                 num_samples,
                 normalized_data) = wavelet_decompose_window(
                    channel_data_2d,
                    wavelet=wvlet,
                    level=level,
                    normalization=True
                )

                # Flatten for quantization
                coeffs_flat = decomposed_channels.flatten()
                q_ids = [str(quantize_number(c)) for c in coeffs_flat]  # << Convert to str here

                all_channel_coeffs.extend(q_ids)
                all_channel_names.extend([ch_name_id] * len(q_ids))

            # Write lines (for this single window)
            coeffs_line = " ".join(all_channel_coeffs) + " "  # << Added newline
            chans_line = " ".join(all_channel_names) + " "  # << Added newline

            f_coeffs.write(coeffs_line)
            f_chans.write(chans_line)

    validate_round_trip(
        csv_file_path=csv_file,  # Replace with your CSV path
        output_coeffs_file=output_coeffs_file,
        output_channels_file=output_channels_file,
        window_length_sec=2.0,
        show_plot=False,  # Set to False to hide plot
        mse_method="pwelch",  # Use "pwelch" to compute on pwelch
        plot_welch=False  # Set to True to plot pwelch next to the time series plot
    )
    print(f"Done generating quantized files for {csv_file}.")


def process_csv_file_s3(
    csv_key: str,
    bucket: str = "dataframes--use1-az6--x-s3",
    local_dir: str = "/tmp",
    output_prefix: str = "output"
):
    s3 = boto3.client("s3")
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    # Download CSV
    csv_name = os.path.basename(csv_key)
    local_csv = os.path.join(local_dir, csv_name)
    s3.download_file(bucket, csv_key, local_csv)

    # Generate local files
    generate_quantized_files_local(csv_file=local_csv, output_folder=local_dir)
    base = os.path.splitext(csv_name)[0]
    coeffs_file = os.path.join(local_dir, f"{base}_quantized_coeffs.txt")
    chans_file = os.path.join(local_dir, f"{base}_quantized_channels.txt")

    # Upload files
    s3.upload_file(coeffs_file, bucket, f"{output_prefix}/{os.path.basename(coeffs_file)}")
    s3.upload_file(chans_file, bucket, f"{output_prefix}/{os.path.basename(chans_file)}")

    # Cleanup
    os.remove(local_csv)
    os.remove(coeffs_file)
    os.remove(chans_file)

folders = list_s3_folders()
csv_files = []
for folder in folders[0:6]:
    print(f"looking into folder: {folder}")
    files = list_csv_files_in_folder(folder)
    csv_files.extend(files)
print(f"done with {len(csv_files)} files")

parallel_process_csv_files(csv_files, max_workers=4)
