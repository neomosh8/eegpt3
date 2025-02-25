import os
import glob
import random
import shutil
import tempfile
import time
import boto3
import joblib
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
epoch_len = 1.96

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
    list_s3_folders,
    list_csv_files_in_folder
)

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed


def average_eeg_channels(df, channels_to_drop):
    """
    From the remaining channels (after dropping channels_to_drop),
    compute overall left/right averages based solely on the ending digit.
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
    For each region, channels are split into left/right (odd => left, even => right),
    averaged, then a bipolar signal is computed.
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
                                   kmeans_model_paths: dict,
                                   window_length_sec: float = 2.0,
                                   wvlet: str = 'cmor1.5-1.0',
                                   scales: list = None,
                                   sampling_period: float = None):
    """
    Generate quantized token files from a CSV file using Morlet wavelet decomposition and pre-trained K-means models.

    Args:
        csv_file (str): Path to the input CSV file.
        output_folder (str): Directory to save output files.
        kmeans_model_paths (dict): Dictionary mapping regions to paths of pre-trained K-means models.
        window_length_sec (float): Length of the sliding window in seconds (default: 2.0).
        wvlet (str): Wavelet type (default: 'cmor1.5-1.0' for Morlet wavelet).
        scales (list): Scales for CWT (if None, computed for EEG bands: 0.5–40 Hz).
        sampling_period (float): Sampling period in seconds (if None, computed from data).

    Returns:
        tuple: (base_name, total_tokens, error_flag, message)
    """
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    try:
        # Load and preprocess CSV data
        df = pd.read_csv(csv_file)
        all_columns = list(df.columns)
        instructions = call_gpt_for_instructions(channel_names=all_columns, dataset_id=base_name)
        if instructions.get("action") == "skip":
            print(f"Skipping dataset '{base_name}' as instructed by GPT.")
            return base_name, 0, True, "Skipped by GPT"

        channels_to_drop = instructions.get("channels_to_drop", [])
        print(f"Processing dataset '{base_name}'. Dropping channels: {channels_to_drop}")

        # Create output directory
        dataset_dir = os.path.join(output_folder, base_name)
        os.makedirs(dataset_dir, exist_ok=True)

        # --- Step 1: Create Regional Bipolar Channels ---
        regional_bipolar = create_regional_bipolar_channels(df, channels_to_drop)

        # Plot regional bipolar channels
        plt.figure(figsize=(15, 10))
        for idx, (region, signal) in enumerate(regional_bipolar.items(), 1):
            plt.subplot(3, 1, idx)
            plt.plot(signal, label=f"{region.capitalize()} Bipolar", color=['blue', 'red', 'green'][idx-1])
            plt.title(f"{region.capitalize()} Bipolar Channel - {base_name}")
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(dataset_dir, f"{base_name}_regional_bipolar_channels.png"))
        plt.close()

        # --- Step 2: Preprocess Signals ---
        original_sps = calculate_sps(csv_file)
        regional_preprocessed = {}
        new_sps_val = None
        for key, signal_array in regional_bipolar.items():
            signal_2d = signal_array[np.newaxis, :]
            preprocessed_signal, new_sps = preprocess_data(signal_2d, original_sps)
            regional_preprocessed[key] = preprocessed_signal[0, :]
            if new_sps_val is None:
                new_sps_val = new_sps

        # Joint global normalization (consistent with training)
        all_data = np.concatenate([regional_preprocessed[region] for region in regional_preprocessed])
        global_mean = np.mean(all_data)
        global_std = np.std(all_data) if np.std(all_data) > 0 else 1e-8
        for key in regional_preprocessed:
            regional_preprocessed[key] = (regional_preprocessed[key] - global_mean) / global_std

        # Plot preprocessed signals
        plt.figure(figsize=(15, 10))
        for idx, (region, signal) in enumerate(regional_preprocessed.items(), 1):
            plt.subplot(3, 1, idx)
            plt.plot(signal, label=f"{region.capitalize()} Preprocessed", color=['blue', 'red', 'green'][idx-1])
            plt.title(f"Preprocessed {region.capitalize()} Bipolar Channel - {base_name}")
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(dataset_dir, f"{base_name}_regional_preprocessed_bipolar_channels.png"))
        plt.close()

        # --- Step 3: Load K-means Models ---
        kmeans_models = {region: joblib.load(path) for region, path in kmeans_model_paths.items()}

        # --- Step 4: Wavelet Decomposition and Quantization ---
        min_length = min(len(regional_preprocessed[region]) for region in regional_preprocessed)
        n_window_samples = int(window_length_sec * new_sps_val)
        num_windows = min_length // n_window_samples
        sampling_period = 1.0 / new_sps_val if sampling_period is None else sampling_period

        tokens_dict = {region: [] for region in ["frontal", "motor_temporal", "parietal_occipital"]}
        regions = list(tokens_dict.keys())

        for i in range(num_windows):
            window_start = i * n_window_samples
            window_end = window_start + n_window_samples
            window_data = np.vstack([
                regional_preprocessed[region][window_start:window_end] for region in regions
            ])

            # Perform CWT with Morlet wavelet
            decomposed_channels, scales, num_samples, normalized_data = wavelet_decompose_window(
                window_data,
                wavelet=wvlet,
                scales=scales,
                normalization=False,  # Normalization already done globally
                sampling_period=sampling_period
            )

            # Process complex coefficients
            for idx, region in enumerate(regions):
                coeffs_complex = decomposed_channels[idx].flatten()
                coeffs_real = coeffs_complex.real
                coeffs_imag = coeffs_complex.imag
                coeffs_combined = np.concatenate([coeffs_real, coeffs_imag])
                token = str(kmeans_models[region].predict(coeffs_combined.reshape(1, -1))[0])
                tokens_dict[region].append(token)

        # Verify token lengths
        lengths = {region: len(tokens) for region, tokens in tokens_dict.items()}
        print(f"Token counts per region for {base_name}: {lengths}")
        if len(set(lengths.values())) != 1:
            print("Warning: Token sequences for regions have different lengths.")

        # --- Step 5: Write Token Files ---
        token_file_paths = {}
        for region, tokens in tokens_dict.items():
            token_fname = f"{base_name}_quantized_coeffs_{region}.txt"
            out_fname = os.path.join(output_folder, token_fname)
            with open(out_fname, "w") as f_out:
                f_out.write(" ".join(tokens))
            token_file_paths[region] = out_fname
        print(f"Generated quantized files for {csv_file}.")

        total_tokens = sum(lengths.values())
        return base_name, total_tokens, False, "Processed successfully"

    except Exception as e:
        print(f"Error processing dataset '{base_name}': {e}")
        return base_name, 0, True, f"Error: {str(e)}"


def process_csv_file_s3(csv_key: str,
                        bucket: str = "dataframes--use1-az6--x-s3",
                        local_dir: str = "/tmp",
                        output_prefix: str = "output_emotiv"):
    """
    Downloads the CSV file from S3, processes it locally, and (if not skipped)
    uploads the resulting token files back to S3.
    Returns a tuple: (folder, base_name, total_tokens, skipped, reason)
    """
    s3 = boto3.client("s3")
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    csv_name = os.path.basename(csv_key)
    local_csv = os.path.join(local_dir, csv_name)
    s3.download_file(bucket, csv_key, local_csv)

    # Extract folder name from the S3 key.
    folder = os.path.dirname(csv_key)  # e.g., "ds004504"

    result = generate_quantized_files_local(csv_file=local_csv, output_folder=local_dir)
    base_name, token_count, skipped, reason = result

    if not skipped:
        for region in ["frontal", "motor_temporal", "parietal_occipital"]:
            token_fname = f"{base_name}_quantized_coeffs_{region}.txt"
            fpath = os.path.join(local_dir, token_fname)
            s3.upload_file(fpath, bucket, f"{output_prefix}/{token_fname}")
            os.remove(fpath)
    os.remove(local_csv)
    return folder, base_name, token_count, skipped, reason


def parallel_process_csv_files(csv_files):
    """
    Processes CSV files in parallel.
    Returns a list of tuples (folder, base_name, total_tokens, skipped, reason) for all files.
    """
    results = []
    max_workers = max(multiprocessing.cpu_count() // 3, 1)
    total = len(csv_files)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_csv_file_s3, f): i for i, f in enumerate(csv_files, start=1)}
        for future in as_completed(futures):
            idx = futures[future]
            csvfile = csv_files[idx - 1]
            try:
                res = future.result()
                results.append(res)
                folder, base, token_count, skipped, reason = res
                print(f"[{idx}/{total}] Done: {csvfile} -> Folder: {folder}, DB: {base}, tokens: {token_count}, skipped: {skipped}, reason: {reason}")
            except Exception as e:
                print(f"Error processing {csvfile}: {e}")
    return results


def write_final_report(results):
    """
    Writes a final report (final_report.txt) to the current working directory,
    listing skipped databases (with reason) and total token counts.
    """
    total_files = len(results)
    total_tokens = sum(token_count for (_, _, token_count, _, _) in results)
    skipped_dbs = [(folder, base, reason) for (folder, base, _, skipped, reason) in results if skipped]
    report_lines = []
    report_lines.append("Final Report:")
    report_lines.append(f"Total files processed: {total_files}")
    report_lines.append(f"Total tokens (all files, all channels): {total_tokens}")
    if skipped_dbs:
        report_lines.append("Skipped databases:")
        for folder, db, reason in skipped_dbs:
            report_lines.append(f"  - Folder: {folder}, Database: {db}, Reason: {reason}")
    else:
        report_lines.append("No databases were skipped.")
    report_lines.append("")
    report_lines.append("Details per database:")
    for folder, base, token_count, skipped, reason in results:
        report_lines.append(f"  - Folder: {folder}, Database: {base}: {token_count} tokens, skipped: {skipped}, Reason: {reason}")

    report_text = "\n".join(report_lines)
    report_file = os.path.join(os.getcwd(), "final_report.txt")
    with open(report_file, "w") as f:
        f.write(report_text)
    print(f"Final report written to {report_file}")



# --- MAIN EXECUTION ---
if __name__ == "__main__":
    folders = list_s3_folders()
    csv_files = []
    i = 1
    # folders = ["csv_emotiv"]

    # folders_to_delete = ["ds002338", "ds002336","ds001849","ds001971","ds002718","ds002814","ds003380","ds003474"]

    # new_folders = [folder for folder in folders if folder not in folders_to_delete]

    # folders = new_folders  # If you want to update the original 'folders' variable
    # Example: processing folder "ds004504"


    for folder in folders:
        print(f"{i}/{len(folders)}: Looking into folder: {folder}")
        files = list_csv_files_in_folder(folder)
        csv_files.extend(files)
        i += 1
    print(f"Done listing. Total files: {len(csv_files)}")

    # Choose one of the following processing methods:

    # Option 1: Process all CSV files in parallel.
    results = parallel_process_csv_files(csv_files)

    # Option 2: For testing, process a single CSV file from S3 locally.
    # Uncomment the following lines to test with a single file.
    # result = process_csv_file_s3(csv_files[0])
    # results = [result]

    # Write final report with overall statistics including folder names.
    # write_final_report(results)
