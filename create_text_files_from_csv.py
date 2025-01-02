import os
import glob
import random
import shutil
import tempfile
from queue import Queue

import boto3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

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
    validate_round_trip, list_s3_csv_files, list_s3_folders

)


# -------------------------------------------------------------------------------
# MAIN FUNCTION (Example: Original vs Reconstructed Plot for a Single Random Window)
# -------------------------------------------------------------------------------
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def main_s3_pipeline(
    bucket_name='dataframes--use1-az6--x-s3',
    prefix=''
):
    # 1) List top-level folders
    all_folders = list_s3_folders(bucket_name, prefix)  # e.g. ['ds004213', 'ds003144', ...]


    if not all_folders:
        print("No folders found in S3.")
        return

    # 2) Use multiprocessing to process each dataset folder
    futures = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        # create queue
        queue = Queue()
        for dataset_folder in all_folders:
            # Submit each dataset to a separate process, passing the queue
            fut = executor.submit(process_single_dataset_s3, dataset_folder, bucket_name, queue)
            futures.append(fut)
        # 3) Use tqdm to track progress
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Datasets processed"):
            # get and print from queue
            while not queue.empty():
                print(queue.get())

    print("All dataset folders processed.")




# --------------------------------------------------------------------------------
# NEW BLOCK:
# Iterate over all CSVs in "dataset" folder
# and create two large text files:
# 1) quantized_coeffs.txt  (space-separated coefficients, each line = one window)
# 2) quantized_channels.txt (space-separated channel names, repeated for each coeff)
#
# Default window_length_sec = 2 seconds
#
# Each line in both files covers BOTH HEMISPHERES in that window, concatenated:
#  - channel "1" for left, "2" for right.
# --------------------------------------------------------------------------------

import os
import glob
import json
import pandas as pd
import numpy as np

# ----------------------------------------------------------------------
# Assume these functions already exist somewhere in your code:
#
#   - calculate_sps(csv_file)
#   - preprocess_data(data_2d, original_sps)
#   - wavelet_decompose_window(channel_data_2d, wavelet, level, normalization)
#   - quantize_number(c)
#
# Also assume you have defined call_gpt_for_instructions(channel_names, dataset_id)
# which returns a dictionary with keys:
#   - "action": "process" or "skip"
#   - "channels_to_drop": [list of channel names to drop]
# ----------------------------------------------------------------------

# s3 client - can be shared or re-created in each process
s3 = boto3.client('s3')

def generate_quantized_files_local(
    dataset_folder: str,
    window_length_sec: float = 2.0,
    wvlet: str = 'db2',
    level: int = 2,
    output_folder: str = 'output',
    queue: Queue = None
):
    """
    Iterate over all CSV files in a *local* `dataset_folder`, wavelet-decompose+quantize
    each file's [Left, Right] average EEG signals, and write
    _quantized_coeffs.txt + _quantized_channels.txt to a local `output_folder`.
    """

    csv_files = sorted(glob.glob(os.path.join(dataset_folder, "*.csv")))
    if not csv_files:
        if queue:
          queue.put(f"No CSV files found in folder: {dataset_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)

    for csv_file in tqdm(csv_files, desc=f"Processing CSVs in {dataset_folder}"):
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
                if queue:
                  queue.put(f"Skipping dataset '{base_name}' as instructed by GPT.")
                continue

            channels_to_drop = instructions.get("channels_to_drop", [])
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
                if queue:
                    queue.put(f"No valid left/right channels found in {csv_file}. Skipping.")
                continue

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
                if queue:
                    queue.put(f"Skipping {csv_file} due to invalid window size or not enough samples.")
                continue

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
                    q_ids = [str(quantize_number(c)) for c in coeffs_flat] # << Convert to str here

                    all_channel_coeffs.extend(q_ids)
                    all_channel_names.extend([ch_name_id] * len(q_ids))

                # Write lines (for this single window)
                coeffs_line = " ".join(all_channel_coeffs) + "\n" # << Added newline
                chans_line  = " ".join(all_channel_names)  + "\n" # << Added newline

                f_coeffs.write(coeffs_line)
                f_chans.write(chans_line)
        if queue:
          queue.put(f"Validating: {csv_file}")
        validate_round_trip(
            csv_file_path=csv_file,  # Replace with your CSV path
            output_coeffs_file=output_coeffs_file,
            output_channels_file=output_channels_file,
            window_length_sec=2.0,
            show_plot=False,  # Set to False to hide plot
            mse_method="pwelch",  # Use "pwelch" to compute on pwelch
            plot_welch=False  # Set to True to plot pwelch next to the time series plot
        )
    if queue:
        queue.put(f"Done generating quantized files in {dataset_folder}.")




def process_single_dataset_s3(dataset_folder: str, bucket_name: str, queue: Queue):
    """
    Process one dataset (e.g., 'ds004213') by:
    - listing CSVs in S3 at `dataset_folder/`
    - downloading to local temp
    - running wavelet+quantization
    - uploading output txt files to S3
    - cleaning up
    """

    # 1) Make local temp dir
    temp_dir = tempfile.mkdtemp(prefix=f"{dataset_folder}_")
    local_dataset_dir = os.path.join(temp_dir, "data")
    os.makedirs(local_dataset_dir, exist_ok=True)

    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # 2) List CSV keys in S3
    dataset_folder_prefix = f"{dataset_folder}/"  # e.g. "ds004213/"
    csv_keys = list_s3_csv_files(bucket_name, dataset_folder_prefix)

    # If empty, skip
    if not csv_keys:
        if queue:
          queue.put(f"No CSVs found in {dataset_folder_prefix}")
        shutil.rmtree(temp_dir)  # cleanup
        return

    # 3) Download them locally
    for csv_key in csv_keys:
        filename = os.path.basename(csv_key)
        local_path = os.path.join(local_dataset_dir, filename)
        s3.download_file(bucket_name, csv_key, local_path)

    # 4) Generate quantized files in local `output_dir`
    generate_quantized_files_local(
        dataset_folder=local_dataset_dir,
        window_length_sec=2.0,
        wvlet='db2',
        level=2,
        output_folder=output_dir,
        queue = queue
    )

    # 5) Upload the generated outputs to S3 => "output/<dataset_folder>/..."
    for txt_file in glob.glob(os.path.join(output_dir, "*_quantized_*.txt")):
        base_txt_name = os.path.basename(txt_file)
        s3_upload_key = f"output/{dataset_folder}/{base_txt_name}"
        if queue:
            queue.put(f"Uploading {txt_file} => s3://{bucket_name}/{s3_upload_key}")
        s3.upload_file(txt_file, bucket_name, s3_upload_key)

    # 6) Cleanup local
    shutil.rmtree(temp_dir)
    if queue:
        queue.put(f"Completed processing for {dataset_folder}")


# ------------------------------------------------------------------------------
# Example usage of the new block
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main_s3_pipeline()