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
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the VQCAETokenizer class (definition from previous code)
from VQCAE_TOKENIZER import (VQCAETokenizer)

# --------------------------------------------------------------------------------
# Import your custom wavelet/quantization utils (or define them here if needed)
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

# Global variable to store tokenizer instance per process
_GLOBAL_TOKENIZER = None


# Function to initialize the tokenizer in each process
def init_worker(tokenizer_path):
    """Initialize the tokenizer once for each worker process"""
    global _GLOBAL_TOKENIZER
    print(f"Initializing tokenizer in process {os.getpid()}")
    _GLOBAL_TOKENIZER = VQCAETokenizer(model_path=tokenizer_path)


# --------------------------------------------------------------------------------
# Reuse the existing create_regional_bipolar_channels function
def create_regional_bipolar_channels(df, channels_to_drop):
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


# --------------------------------------------------------------------------------
# Modified function to process a single CSV file, using global tokenizer
def process_csv_file_with_global_tokenizer(csv_key,
                                         bucket="dataframes--use1-az6--x-s3",
                                         local_dir="/tmp",
                                         output_dir="training_data_shards",
                                         overlap_percent=50,
                                         window_length_sec=2.0):
    """
    Downloads a CSV file from S3, processes it using the global VQCAE tokenizer,
    and saves encoded tokens as a single tensor file.

    Args:
        csv_key (str): S3 key for the CSV file
        bucket (str): S3 bucket name
        local_dir (str): Local directory for temporary files
        output_dir (str): Output directory for tokenized tensors
        overlap_percent (float): Percentage of overlap between consecutive windows (default: 50)

    Returns:
        tuple: (folder, base_name, total_tokens, skipped, reason)
    """
    # Access the global tokenizer initialized for this process
    global _GLOBAL_TOKENIZER
    if _GLOBAL_TOKENIZER is None:
        return os.path.dirname(csv_key), os.path.splitext(os.path.basename(csv_key))[
            0], 0, True, "Tokenizer not initialized"

    # Create required directories
    s3 = boto3.client("s3")
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    csv_name = os.path.basename(csv_key)
    base_name = os.path.splitext(csv_name)[0]
    local_csv = os.path.join(local_dir, csv_name)

    try:
        # Download the CSV file
        print(f"Downloading {csv_key} to {local_csv}")
        s3.download_file(bucket, csv_key, local_csv)

        # Extract folder name from the S3 key
        folder = os.path.dirname(csv_key)

        # Read CSV into a DataFrame
        df = pd.read_csv(local_csv)
        all_columns = list(df.columns)
        instructions = call_gpt_for_instructions(channel_names=all_columns, dataset_id=base_name)
        original_sps = calculate_sps(local_csv)

        if instructions.get("action") == "skip":
            print(f"Skipping dataset '{base_name}'.")
            if os.path.exists(local_csv):
                os.remove(local_csv)
            return folder, base_name, 0, True, "Skipped by GPT"

        channels_to_drop = instructions.get("channels_to_drop", [])
        print(f"Processing '{base_name}'. Dropping: {channels_to_drop}")

        # Create regional bipolar channels
        regional_bipolar = create_regional_bipolar_channels(df, channels_to_drop)
        channels = list(regional_bipolar.keys())
        data_2d = np.vstack([regional_bipolar[ch] for ch in channels])

        # Preprocess data
        preprocessed_data, new_sps = preprocess_data(data_2d, original_sps)
        regional_preprocessed = {ch: preprocessed_data[i, :] for i, ch in enumerate(channels)}
        new_sps_val = new_sps

        # Standardize the signals
        for key in regional_preprocessed:
            signal = regional_preprocessed[key]
            mean_signal = np.mean(signal)
            std_signal = np.std(signal) if np.std(signal) > 0 else 1e-8
            regional_preprocessed[key] = (signal - mean_signal) / std_signal

        # Calculate window parameters
        min_length = min(len(regional_preprocessed[region]) for region in regional_preprocessed)
        if min_length == 0:
            if os.path.exists(local_csv):
                os.remove(local_csv)
            return folder, base_name, 0, True, "No valid data in regional channels"

        n_window_samples = int(window_length_sec * new_sps_val)

        # Calculate step size based on overlap percentage
        step_size = int(n_window_samples * (1 - overlap_percent / 100))

        # Calculate number of windows with overlap
        num_windows = (min_length - n_window_samples) // step_size + 1 if step_size > 0 else 1

        # Compute window statistics for artifact rejection
        window_stats = []
        for i in range(num_windows):
            window_start = i * step_size
            window_end = window_start + n_window_samples
            window_data = np.vstack([regional_preprocessed[region][window_start:window_end]
                                     for region in regional_preprocessed])
            window_mean = np.mean(window_data)
            window_stats.append(window_mean)

        window_stats = np.array(window_stats)
        window_mu = np.mean(window_stats)
        window_sigma = np.std(window_stats) if np.std(window_stats) > 0 else 1e-8
        z_scores = (window_stats - window_mu) / window_sigma

        keep_indices = np.where(np.abs(z_scores) <= 2.0)[0]  # Z threshold of 2.0
        rejected_indices = np.where(np.abs(z_scores) > 2.0)[0]
        discarded_count = len(rejected_indices)
        print(f"Discarded {discarded_count} windows out of {num_windows} due to artifact rejection (|Z| > 2.0).")

        # Initialize a single token list
        token_list = []

        # Process all windows sequentially
        for i in range(num_windows):
            # Skip windows that didn't pass artifact rejection
            if i not in keep_indices:
                continue

            window_start = i * step_size
            window_end = window_start + n_window_samples
            window_data = np.vstack([
                regional_preprocessed[region][window_start:window_end]
                for region in regional_preprocessed
            ])

            # Perform wavelet decomposition
            decomposed_channels, _, _, _ = wavelet_decompose_window(
                window_data,
                wavelet='cmor1.5-1.0',
                scales=None,
                normalization=True,
                sampling_period=1.0 / new_sps_val,
                verbose=False
            )

            # Check if we have the expected number of channels
            if len(decomposed_channels) < len(regional_preprocessed):
                continue

            # Combine the decomposed channels into a single 3-channel image
            combined_image = np.stack([
                decomposed_channels[idx] for idx, region in enumerate(regional_preprocessed.keys())
            ], axis=0)

            # Encode the combined image with the global tokenizer
            token_indices = _GLOBAL_TOKENIZER.encode(combined_image)

            # Add pad token to the end of this window's tokens
            pad_token = torch.tensor([_GLOBAL_TOKENIZER.get_pad_token()], device=token_indices.device)
            window_tokens_with_pad = torch.cat([token_indices.flatten(), pad_token])

            # Store the flattened token indices
            token_list.append(window_tokens_with_pad.flatten())
            print(np.shape(token_indices))
        # Check if we have any tokens
        if not token_list:
            if os.path.exists(local_csv):
                os.remove(local_csv)
            return folder, base_name, 0, True, "No valid windows after artifact rejection"

        # Concatenate all tokens
        all_tokens = torch.cat(token_list)

        # Add EOS token
        eos_token = torch.tensor([_GLOBAL_TOKENIZER.get_eos_token()], device=all_tokens.device)
        all_tokens_with_eos = torch.cat([all_tokens, eos_token])

        # Create output filename
        output_file = f"{base_name}_tokens.pt"
        output_path = os.path.join(output_dir, output_file)

        # Save the tensor
        torch.save(all_tokens_with_eos, output_path)
        total_tokens = len(all_tokens_with_eos)
        print(f"Saved {total_tokens} tokens to {output_path}")

        # Clean up local CSV file
        if os.path.exists(local_csv):
            os.remove(local_csv)

        return folder, base_name, total_tokens, False, f"Processed {len(keep_indices)}/{num_windows} windows"

    except Exception as e:
        if os.path.exists(local_csv):
            os.remove(local_csv)
        print(f"Error processing {csv_key}: {e}")
        return os.path.dirname(csv_key), base_name, 0, True, f"Error: {str(e)}"


# --------------------------------------------------------------------------------
# Function to process multiple CSV files in parallel using the global tokenizer
def parallel_process_csv_files_with_tokenizer(csv_files, tokenizer_model_path,
                                            bucket="dataframes--use1-az6--x-s3",
                                            local_dir="/tmp",
                                            output_dir="training_data_shards",
                                            overlap_percent=50,
                                            window_length_sec=2.0):
    """
    Processes CSV files in parallel using the VQCAE tokenizer.
    Uses a global tokenizer instance per worker process.

    Args:
        csv_files (list): List of CSV file S3 keys
        tokenizer_model_path (str): Path to the trained VQCAE model
        bucket (str): S3 bucket name
        local_dir (str): Local directory for temporary files
        output_dir (str): Output directory for tokenized tensors
        overlap_percent (float): Percentage of overlap between consecutive windows (default: 50)

    Returns:
        list: List of results tuples (folder, base_name, total_tokens, skipped, reason)
    """
    results = []
    max_workers = max(multiprocessing.cpu_count() // 3, 1)
    total = len(csv_files)

    print(f"Starting processing pool with {max_workers} workers")
    with ProcessPoolExecutor(max_workers=max_workers,
                             initializer=init_worker,
                             initargs=(tokenizer_model_path,)) as executor:
        futures = {}
        for i, csv_file in enumerate(csv_files, start=1):
            future = executor.submit(
                process_csv_file_with_global_tokenizer,
                csv_file,
                bucket,
                local_dir,
                output_dir,
                overlap_percent,
                window_length_sec
            )
            futures[future] = i

        for future in as_completed(futures):
            idx = futures[future]
            csvfile = csv_files[idx - 1]
            try:
                res = future.result()
                results.append(res)
                folder, base, token_count, skipped, reason = res
                print(
                    f"[{idx}/{total}] Done: {csvfile} -> Folder: {folder}, DB: {base}, tokens: {token_count}, skipped: {skipped}, reason: {reason}")
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
        report_lines.append(
            f"  - Folder: {folder}, Database: {base}: {token_count} tokens, skipped: {skipped}, Reason: {reason}")

    report_text = "\n".join(report_lines)
    report_file = os.path.join(os.getcwd(), "final_report.txt")
    with open(report_file, "w") as f:
        f.write(report_text)
    print(f"Final report written to {report_file}")


# --------------------------------------------------------------------------------
# Main execution
if __name__ == "__main__":
    # Configuration parameters
    bucket = "dataframes--use1-az6--x-s3"
    local_dir = "/tmp/eeg_processing"
    output_dir = "training_data_shards"
    tokenizer_model_path = "output/vqcae_final.pt"  # Update with your model path
    overlap_percent = 50  # Default overlap percentage
    window_length_sec = 0.5  # Default window length in seconds
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List CSV files from S3
    print("Listing S3 folders...")
    folders = list_s3_folders()  # Process first 40 folders

    csv_files = []
    for i, folder in enumerate(folders, 1):
        print(f"{i}/{len(folders)}: Looking into folder: {folder}")
        files = list_csv_files_in_folder(folder)
        csv_files.extend(files)

    print(f"Done listing. Found {len(csv_files)} CSV files to process.")

    # Process all CSV files in parallel using the VQCAE tokenizer
    results = parallel_process_csv_files_with_tokenizer(
        csv_files,
        tokenizer_model_path,
        bucket=bucket,
        local_dir=local_dir,
        output_dir=output_dir,
        overlap_percent=overlap_percent,
        window_length_sec=window_length_sec
    )

    # Write the final report
    write_final_report(results)

    print(f"Processing complete. Token files saved to: {output_dir}")