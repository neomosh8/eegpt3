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
    pwelch_z,
    call_gpt_for_instructions,
    preprocess_data,
    calculate_sps,
    validate_round_trip,  # not used now
    list_s3_folders,
    list_csv_files_in_folder
)

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed


def average_eeg_channels(df, channels_to_drop):
    """
    From the remaining channels (after dropping channels_to_drop),
    compute overall left/right averages based solely on the ending digit:
      - Odd-ending channels are assumed to be left‐hemisphere.
      - Even-ending channels are right‐hemisphere.
    Returns (left_data, right_data)
    """
    filtered_columns = [col for col in df.columns if col not in channels_to_drop]

    left_channels = []
    right_channels = []
    for ch in filtered_columns:
        if ch.endswith('z'):
            continue
        if not ch[-1].isdigit():
            continue
        if int(ch[-1]) % 2 == 0:
            right_channels.append(ch)
        else:
            left_channels.append(ch)

    left_data = df[left_channels].mean(axis=1).values if left_channels else np.zeros(len(df))
    right_data = df[right_channels].mean(axis=1).values if right_channels else np.zeros(len(df))

    return left_data, right_data


def create_regional_bipolar_channels(df, channels_to_drop):
    """
    Groups channels into three regions based on the starting designator (case insensitive):
      - Frontal: channels starting with 'F'
      - Parietal-Occipital: channels starting with 'P' or 'O'
      - Motor-Temporal: channels starting with 'C' or 'T'

    For each region, channels are split into left/right using the ending digit
    (odd => left, even => right), averaged, then a bipolar signal is computed as:
         bipolar = (left average) - (right average)

    Returns a dictionary with keys:
       "frontal", "parietal_occipital", "motor_temporal"
    and values as the corresponding 1D bipolar signals.
    """
    valid_channels = [col for col in df.columns if col not in channels_to_drop]

    frontal = []
    parietal_occipital = []
    motor_temporal = []
    for ch in valid_channels:
        if not ch[-1].isdigit():
            continue
        first_letter = ch[0].upper()
        if first_letter == "F":
            frontal.append(ch)
        elif first_letter in ["P", "O"]:
            parietal_occipital.append(ch)
        elif first_letter in ["C", "T"]:
            motor_temporal.append(ch)

    def split_left_right(channel_list):
        left = []
        right = []
        for ch in channel_list:
            try:
                last_digit = int(ch[-1])
            except ValueError:
                continue
            if last_digit % 2 == 0:
                right.append(ch)
            else:
                left.append(ch)
        return left, right

    f_left, f_right = split_left_right(frontal)
    po_left, po_right = split_left_right(parietal_occipital)
    ct_left, ct_right = split_left_right(motor_temporal)

    def average_signals(channel_list):
        if channel_list:
            return df[channel_list].mean(axis=1).values
        else:
            return np.zeros(len(df))

    frontal_left_avg = average_signals(f_left)
    frontal_right_avg = average_signals(f_right)
    po_left_avg = average_signals(po_left)
    po_right_avg = average_signals(po_right)
    ct_left_avg = average_signals(ct_left)
    ct_right_avg = average_signals(ct_right)

    frontal_bipolar = frontal_left_avg - frontal_right_avg
    po_bipolar = po_left_avg - po_right_avg
    ct_bipolar = ct_left_avg - ct_right_avg

    return {
        "frontal": frontal_bipolar,
        "parietal_occipital": po_bipolar,
        "motor_temporal": ct_bipolar
    }


def generate_quantized_files_local(csv_file: str,
                                   output_folder: str,
                                   window_length_sec: float = epoch_len,
                                   wvlet: str = 'db2',
                                   level: int = level):
    """
    Process a single CSV file:
      1. Drop channels as per GPT instructions.
      2. Plot all raw channels and the (reference) overall bipolar channel.
      3. Create three regional bipolar channels (frontal, motor–temporal, parietal–occipital).
      4. Preprocess each regional bipolar signal and plot them.
      5. For each regional channel, slide over the preprocessed signal in non-overlapping windows,
         perform wavelet decomposition and quantization, and accumulate the quantized tokens.
      6. At the end, check that the three token sequences have the same length and then write
         each sequence to a separate text file.
    """
    base_name = os.path.splitext(os.path.basename(csv_file))[0]

    # --- Load CSV and get GPT instructions ---
    df = pd.read_csv(csv_file)
    all_columns = list(df.columns)
    instructions = call_gpt_for_instructions(channel_names=all_columns, dataset_id=base_name)
    if instructions.get("action") == "skip":
        print(f"Skipping dataset '{base_name}' as instructed by GPT.")
        return
    channels_to_drop = instructions.get("channels_to_drop", [])
    print(f"Dropping channels: {channels_to_drop}")

    # --- Plot all raw channels ---
    plt.figure(figsize=(15, 10))
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

    # --- (Optional) Create an overall bipolar channel for reference ---
    left_data, right_data = average_eeg_channels(df, channels_to_drop)
    overall_bipolar = left_data - right_data
    plt.figure(figsize=(15, 5))
    plt.plot(overall_bipolar, label="Overall Bipolar (Left-Right)", color='purple')
    plt.title(f"Overall Bipolar Channel (Raw) - {base_name}")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    overall_plot_file = os.path.join(output_folder, f"{base_name}_overall_bipolar_channel.png")
    plt.savefig(overall_plot_file)
    plt.close()

    # --- Create Regional Bipolar Channels ---
    regional_bipolar = create_regional_bipolar_channels(df, channels_to_drop)
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 1, 1)
    plt.plot(regional_bipolar["frontal"], label="Frontal Bipolar", color='blue')
    plt.title(f"Frontal Bipolar Channel - {base_name}")
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(regional_bipolar["motor_temporal"], label="Motor-Temporal Bipolar", color='red')
    plt.title(f"Motor-Temporal Bipolar Channel - {base_name}")
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(regional_bipolar["parietal_occipital"], label="Parietal-Occipital Bipolar", color='green')
    plt.title(f"Parietal-Occipital Bipolar Channel - {base_name}")
    plt.legend()
    plt.tight_layout()
    regional_plot_file = os.path.join(output_folder, f"{base_name}_regional_bipolar_channels.png")
    plt.savefig(regional_plot_file)
    plt.close()

    # --- Preprocess the signals ---
    original_sps = calculate_sps(csv_file)
    regional_preprocessed = {}
    for key, signal_array in regional_bipolar.items():
        signal_2d = signal_array[np.newaxis, :]
        preprocessed_signal, _ = preprocess_data(signal_2d, original_sps)
        regional_preprocessed[key] = preprocessed_signal[0, :]

    plt.figure(figsize=(15, 10))
    plt.subplot(3, 1, 1)
    plt.plot(regional_preprocessed["frontal"], label="Frontal Preprocessed", color='blue')
    plt.title(f"Preprocessed Frontal Bipolar Channel - {base_name}")
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(regional_preprocessed["motor_temporal"], label="Motor-Temporal Preprocessed", color='red')
    plt.title(f"Preprocessed Motor-Temporal Bipolar Channel - {base_name}")
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(regional_preprocessed["parietal_occipital"], label="Parietal-Occipital Preprocessed", color='green')
    plt.title(f"Preprocessed Parietal-Occipital Bipolar Channel - {base_name}")
    plt.legend()
    plt.tight_layout()
    regional_prep_plot_file = os.path.join(output_folder, f"{base_name}_regional_preprocessed_bipolar_channels.png")
    plt.savefig(regional_prep_plot_file)
    plt.close()

    # --- Wavelet Decomposition & Quantization for Each Regional Channel ---
    n_window_samples = int(
        window_length_sec * _)  # _ holds new_sps from preprocess_data; you can also recompute if needed
    # We'll accumulate tokens for each region in a dictionary.
    tokens_dict = {"frontal": [], "motor_temporal": [], "parietal_occipital": []}
    for region, preproc_signal in regional_preprocessed.items():
        total_samples_region = preproc_signal.shape[0]
        # Process in non-overlapping windows.
        for window_start in range(0, total_samples_region - n_window_samples + 1, n_window_samples):
            window_end = window_start + n_window_samples
            window_data = preproc_signal[window_start:window_end]
            window_data_2d = window_data[np.newaxis, :]
            (decomposed_channels,
             coeffs_lengths,
             num_samples,
             normalized_data) = wavelet_decompose_window(
                window_data_2d,
                wavelet=wvlet,
                level=level,
                normalization=True
            )
            coeffs_flat = decomposed_channels.flatten()
            q_ids = [str(quantize_number(c)) for c in coeffs_flat]
            tokens_dict[region].extend(q_ids)

    # --- Check that all token sequences have the same length ---
    lengths = {region: len(tokens) for region, tokens in tokens_dict.items()}
    print("Token counts per region:", lengths)
    if len(set(lengths.values())) != 1:
        print("Warning: The token sequences for the three regions are not the same length.")
    else:
        print("All regional token sequences have the same length.")

    # --- Write the tokens for each region to separate text files ---
    for region, tokens in tokens_dict.items():
        out_fname = os.path.join(output_folder, f"{base_name}_quantized_coeffs_{region}.txt")
        with open(out_fname, "w") as f_out:
            f_out.write(" ".join(tokens))
    print(f"Done generating quantized files for {csv_file}.")


def process_csv_file_s3(csv_key: str,
                        bucket: str = "dataframes--use1-az6--x-s3",
                        local_dir: str = "/tmp",
                        output_prefix: str = "output"):
    s3 = boto3.client("s3")
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    csv_name = os.path.basename(csv_key)
    local_csv = os.path.join(local_dir, csv_name)
    s3.download_file(bucket, csv_key, local_csv)
    generate_quantized_files_local(csv_file=local_csv, output_folder=local_dir)
    base = os.path.splitext(csv_name)[0]
    coeffs_files = [
        os.path.join(local_dir, f"{base}_quantized_coeffs_frontal.txt"),
        os.path.join(local_dir, f"{base}_quantized_coeffs_motor_temporal.txt"),
        os.path.join(local_dir, f"{base}_quantized_coeffs_parietal_occipital.txt")
    ]
    for fpath in coeffs_files:
        s3.upload_file(fpath, bucket, f"{output_prefix}/{os.path.basename(fpath)}")
        os.remove(fpath)
    os.remove(local_csv)


def parallel_process_csv_files(csv_files):
    max_workers = multiprocessing.cpu_count() // 3
    total = len(csv_files)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_csv_file_s3, f): i for i, f in enumerate(csv_files, start=1)}
        for future in as_completed(futures):
            idx = futures[future]
            csvfile = csv_files[idx - 1]
            try:
                future.result()
                print(f"[{idx}/{total}] Done: {csvfile}")
            except Exception as e:
                print(f"Error processing {csvfile}: {e}")


# --- MAIN EXECUTION ---
folders = list_s3_folders()
csv_files = []
i = 1
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
