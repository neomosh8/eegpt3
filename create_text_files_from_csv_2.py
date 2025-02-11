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
level = 4
epoch_len = 1.18

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

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed


def average_eeg_channels(df, channels_to_drop):
    """
    Given a DataFrame and a list of channels to drop,
    identify the remaining channels and then compute two averaged channels:
    one for "left" electrodes and one for "right" electrodes.

    The convention here is:
      - Channels whose name ends with a digit are candidates.
      - Among these, channels with an odd last digit are considered "left"
      - Those with an even last digit are considered "right"

    If a side has no valid channels, a zeros array is returned.
    """
    # Remove channels as per GPT instructions
    filtered_columns = [col for col in df.columns if col not in channels_to_drop]

    left_channels = []
    right_channels = []
    for ch in filtered_columns:
        # Skip channels that end with 'z' or do not end with a digit
        if ch.endswith('z'):
            continue
        if not ch[-1].isdigit():
            continue
        # Classify channel based on odd/even last digit
        if int(ch[-1]) % 2 == 0:
            right_channels.append(ch)
        else:
            left_channels.append(ch)

    # Compute the averaged signals
    left_data = df[left_channels].mean(axis=1).values if left_channels else np.zeros(len(df))
    right_data = df[right_channels].mean(axis=1).values if right_channels else np.zeros(len(df))

    return left_data, right_data


def generate_quantized_files_local(
        csv_file: str,
        output_folder: str,
        window_length_sec: float = epoch_len,
        wvlet: str = 'db2',
        level: int = level
):
    """
    Process a single CSV file:
      1. Load the CSV and drop channels as per GPT instructions.
      2. Compute two averaged channels (left/right).
      3. Plot the raw channels and the averaged channels.
      4. Preprocess the averaged channels (resample, bandpass, etc.).
      5. Plot the preprocessed averaged channels.
      6. Slide a window over the preprocessed data, perform wavelet decomposition,
         quantize the coefficients, interleave channel tokens, and write to file.
      7. Validate the round-trip reconstruction.
    """
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    # Define output file paths for quantized text files
    output_coeffs_file = os.path.join(output_folder, f"{base_name}_quantized_coeffs.txt")
    output_channels_file = os.path.join(output_folder, f"{base_name}_quantized_channels.txt")

    with open(output_coeffs_file, "w") as f_coeffs, open(output_channels_file, "w") as f_chans:
        df = pd.read_csv(csv_file)

        # Get GPT instructions to know which channels to drop
        all_columns = list(df.columns)
        instructions = call_gpt_for_instructions(
            channel_names=all_columns,
            dataset_id=base_name
        )
        if instructions.get("action") == "skip":
            print(f"Skipping dataset '{base_name}' as instructed by GPT.")
            return

        channels_to_drop = instructions.get("channels_to_drop", [])
        print(f"Dropping channels: {channels_to_drop}")

        # --- PLOT 1: All Channels (Raw) Before Preprocessing ---
        plt.figure(figsize=(15, 10))
        # Plot only the channels that are not dropped
        filtered_columns = [col for col in all_columns if col not in channels_to_drop]
        for col in filtered_columns:
            plt.plot(df[col], label=col)
        plt.title(f"All Channels (Raw) - {base_name}")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.legend(fontsize='small', loc='upper right')
        plt.tight_layout()
        raw_plot_file = os.path.join(output_folder, f"{base_name}_all_channels_before_preprocess.png")
        plt.savefig(raw_plot_file)
        plt.close()
        # --- END PLOT 1 ---

        # === STEP 1: Create Averaged Channels ===
        left_data, right_data = average_eeg_channels(df, channels_to_drop)

        # --- PLOT 2: Averaged Channels (Before Preprocessing) ---
        plt.figure(figsize=(15, 5))
        plt.plot(left_data, label="Left Averaged")
        plt.plot(right_data, label="Right Averaged")
        plt.title(f"Averaged Channels (Raw) - {base_name}")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
        avg_plot_file = os.path.join(output_folder, f"{base_name}_averaged_channels.png")
        plt.savefig(avg_plot_file)
        plt.close()
        # --- END PLOT 2 ---

        # === STEP 2: Preprocess the Averaged Channels ===
        # Calculate the original sampling rate (using your custom utility)
        original_sps = calculate_sps(csv_file)
        # Stack the two averaged channels into a 2D array for preprocessing
        data_2d = np.vstack([left_data, right_data])
        preprocessed_data, new_sps = preprocess_data(data_2d, original_sps)

        # --- PLOT 3: Preprocessed Averaged Channels ---
        plt.figure(figsize=(15, 5))
        plt.plot(preprocessed_data[0, :], label="Left Preprocessed")
        plt.plot(preprocessed_data[1, :], label="Right Preprocessed")
        plt.title(f"Preprocessed Averaged Channels - {base_name}")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
        prep_plot_file = os.path.join(output_folder, f"{base_name}_preprocessed_channels.png")
        plt.savefig(prep_plot_file)
        plt.close()
        # --- END PLOT 3 ---

        # --- Process the Preprocessed Data in Windows ---
        n_window_samples = int(window_length_sec * new_sps)
        num_channels, total_samples = preprocessed_data.shape

        if n_window_samples <= 0 or total_samples < n_window_samples:
            print(f"Skipping {csv_file} due to invalid window size or insufficient samples.")
            return

        # Slide through the data in non-overlapping windows
        for window_start in range(0, total_samples - n_window_samples + 1, n_window_samples):
            window_end = window_start + n_window_samples

            # Prepare lists for tokens from each channel
            channel0_tokens = []
            channel1_tokens = []

            for ch_idx in range(2):  # 0: left, 1: right
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

                # Flatten the coefficients and quantize
                coeffs_flat = decomposed_channels.flatten()
                q_ids = [str(quantize_number(c)) for c in coeffs_flat]

                if ch_idx == 0:
                    channel0_tokens = q_ids
                else:
                    channel1_tokens = q_ids

            # === INTERLEAVE THE CHANNEL TOKENS ===
            interleaved_ids = []
            interleaved_channels = []

            max_len = max(len(channel0_tokens), len(channel1_tokens))
            for i in range(max_len):
                if i < len(channel0_tokens):
                    interleaved_ids.append(channel0_tokens[i])
                    interleaved_channels.append("1")  # "1" corresponds to left channel
                if i < len(channel1_tokens):
                    interleaved_ids.append(channel1_tokens[i])
                    interleaved_channels.append("2")  # "2" corresponds to right channel

            # Write the interleaved tokens to the output files
            f_coeffs.write(" ".join(interleaved_ids) + " ")
            f_chans.write(" ".join(interleaved_channels) + " ")

    # Validate round-trip reconstruction
    print(2 * sum(coeffs_lengths[0]))
    validate_round_trip(
        csv_file_path=csv_file,
        coeff_lenght=sum(coeffs_lengths[0]),
        output_coeffs_file=output_coeffs_file,
        output_channels_file=output_channels_file,
        window_length_sec=epoch_len,
        show_plot=False,
        mse_method="timeseries",
        plot_welch=False
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

    # Download CSV from S3
    csv_name = os.path.basename(csv_key)
    local_csv = os.path.join(local_dir, csv_name)
    s3.download_file(bucket, csv_key, local_csv)

    # Process the file locally
    generate_quantized_files_local(csv_file=local_csv, output_folder=local_dir)
    base = os.path.splitext(csv_name)[0]
    coeffs_file = os.path.join(local_dir, f"{base}_quantized_coeffs.txt")
    chans_file = os.path.join(local_dir, f"{base}_quantized_channels.txt")

    # Upload the generated files back to S3
    s3.upload_file(coeffs_file, bucket, f"{output_prefix}/{os.path.basename(coeffs_file)}")
    s3.upload_file(chans_file, bucket, f"{output_prefix}/{os.path.basename(chans_file)}")

    # Cleanup local files
    os.remove(local_csv)
    os.remove(coeffs_file)
    os.remove(chans_file)


def parallel_process_csv_files(csv_files):
    # Set the number of workers (tweak as necessary)
    max_workers = multiprocessing.cpu_count() // 3
    total = len(csv_files)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_csv_file_s3, f): i for i, f in enumerate(csv_files, start=1)}
        for future in as_completed(futures):
            idx = futures[future]
            csvfile = csv_files[idx - 1]
            try:
                future.result()
                print(f"\033[94m[{idx}/{total}] Done: {csvfile}\033[0m")
            except Exception as e:
                print(f"\033[91mError: {csvfile} -> {e}\033[0m")


# --- MAIN EXECUTION ---

# List the S3 folders and CSV files (using your custom utility functions)
folders = list_s3_folders()
csv_files = []
i = 1
# Example: folders = ["ds004504", "ds004448", "ds004447", "ds004446", "ds004408"]
folders = ["ds004504"]
for folder in folders:
    print(f"{i}/{len(folders)}: looking into folder: {folder}")
    files = list_csv_files_in_folder(folder)
    csv_files.extend(files)
    i += 1
print(f"Done listing. Total files: {len(csv_files)}")

# Uncomment to process all CSV files in parallel:
# parallel_process_csv_files(csv_files)

# For testing, process a single CSV file from S3 locally:
process_csv_file_s3(csv_files[0])
