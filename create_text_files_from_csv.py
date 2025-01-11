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

def parallel_process_csv_files(csv_files):
    if True:
        max_workers = multiprocessing.cpu_count() //3
    else:
        max_workers = multiprocessing.cpu_count() -1
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


def generate_quantized_files_local(
        csv_file: str,
        output_folder: str,
        window_length_sec: float = epoch_len,
        wvlet: str = 'db2',
        level: int = level
):
    """
    Iterate over a single local CSV file, wavelet-decompose+quantize
    each file's [Left, Right] average EEG signals, and write
    _quantized_coeffs.txt + _quantized_channels.txt to a local `output_folder`,
    but with interleaved channel tokens.
    """

    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    # Create output file paths
    output_coeffs_file = os.path.join(output_folder, f"{base_name}_quantized_coeffs.txt")
    output_channels_file = os.path.join(output_folder, f"{base_name}_quantized_channels.txt")

    with open(output_coeffs_file, "w") as f_coeffs, open(output_channels_file, "w") as f_chans:
        # f_coeffs.write("trial ")
        # f_chans.write("trial ")
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

            # We will store channel-0 tokens in one list, channel-1 tokens in another
            channel0_tokens = []
            channel1_tokens = []

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
                q_ids = [str(quantize_number(c)) for c in coeffs_flat]  # Convert to str

                # Instead of extending a single list, we place them in the channel-specific list
                if ch_idx == 0:
                    channel0_tokens = q_ids
                else:
                    channel1_tokens = q_ids

            ### INTERLEAVING CHANGES HERE ###
            # Now we interleave channel0_tokens and channel1_tokens
            interleaved_ids = []
            interleaved_channels = []

            # In principle, wavelet decomposition for each channel
            # should yield the same length. But let's be safe:
            max_len = max(len(channel0_tokens), len(channel1_tokens))

            for i in range(max_len):
                if i < len(channel0_tokens):
                    interleaved_ids.append(channel0_tokens[i])
                    interleaved_channels.append("1")  # channel 0 => "1"
                if i < len(channel1_tokens):
                    interleaved_ids.append(channel1_tokens[i])
                    interleaved_channels.append("2")  # channel 1 => "2"
            ### END INTERLEAVING CHANGES ###

            # Now write the interleaved tokens to file
            coeffs_line = " ".join(interleaved_ids) + " "
            chans_line = " ".join(interleaved_channels) + " "

            f_coeffs.write(coeffs_line)
            f_chans.write(chans_line)

    # Validate round trip
    print(2*sum(coeffs_lengths[0]))
    validate_round_trip(
        csv_file_path=csv_file,
        coeff_lenght = sum(coeffs_lengths[0]),
        output_coeffs_file=output_coeffs_file,
        output_channels_file=output_channels_file,
        window_length_sec=epoch_len,
        show_plot=False,
        mse_method="timeseries",
        plot_welch=False

    )
    # print(f"Done generating quantized files for {csv_file}.")



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
i = 1
for folder in folders:
    print(f"{i}/{len(folders)}")
    print(f"looking into folder: {folder}")
    files = list_csv_files_in_folder(folder)
    csv_files.extend(files)
    i=i+1
print(f"done with {len(csv_files)} files")

for file in csv_files:
    if file.startswith("ds004408"):
        resume_index = csv_files.index(file)
        break
resume = csv_files[resume_index:]
print(f"resuming from: {resume_index}")
parallel_process_csv_files(resume)
# process_csv_file_s3(csv_files[0])




