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
    # filtered_columns = [col for col in df.columns if col not in channels_to_drop]
    filtered_columns = [col for col in df.columns if col.lower() not in [c.lower() for c in channels_to_drop]]# for emotiv

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
                                   window_length_sec: float = epoch_len,
                                   wvlet: str = 'db2',
                                   level: int = level):
    """
    Process a single CSV file:
      1. Drop channels as per GPT instructions.
      2. Create three regional bipolar channels.
      3. Preprocess and plot signals.
      4. Perform joint (global) normalization.
      5. Do wavelet decomposition (with normalization turned off) and quantize.
      6. Write token sequences to text files.
      7. Return a tuple: (base_name, total_tokens, skipped, reason)
         - If an error occurs, mark as skipped with an error reason.
         - If skipped by GPT, mark as such.
    """
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    try:
        df = pd.read_csv(csv_file)
        all_columns = list(df.columns)
        instructions = call_gpt_for_instructions(channel_names=all_columns, dataset_id=base_name)
        if instructions.get("action") == "skip":
            print(f"Skipping dataset '{base_name}' as instructed by GPT.")
            return base_name, 0, True, "Skipped by GPT: instructions indicated skip"

        channels_to_drop = instructions.get("channels_to_drop", [])
        print(f"Processing dataset '{base_name}'. Dropping channels: {channels_to_drop}")

        # Create a folder for this dataset's plots.
        dataset_dir = os.path.join(output_folder, base_name)
        os.makedirs(dataset_dir, exist_ok=True)

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
        regional_plot_file = os.path.join(dataset_dir, f"{base_name}_regional_bipolar_channels.png")
        plt.savefig(regional_plot_file)
        plt.close()

        # --- Preprocess the Regional Signals ---
        original_sps = calculate_sps(csv_file)
        regional_preprocessed = {}
        new_sps_val = None
        for key, signal_array in regional_bipolar.items():
            signal_2d = signal_array[np.newaxis, :]
            preprocessed_signal, new_sps = preprocess_data(signal_2d, original_sps)
            regional_preprocessed[key] = preprocessed_signal[0, :]
            if new_sps_val is None:
                new_sps_val = new_sps

        # --- Joint Global Normalization ---
        all_data = np.concatenate([
            regional_preprocessed["frontal"],
            regional_preprocessed["motor_temporal"],
            regional_preprocessed["parietal_occipital"]
        ])
        global_mean = np.mean(all_data)
        global_std = np.std(all_data)
        if np.isclose(global_std, 0):
            print(f"Warning: Global standard deviation is zero for dataset '{base_name}'.")
            global_std = 1e-8
        for key in regional_preprocessed:
            regional_preprocessed[key] = (regional_preprocessed[key] - global_mean) / global_std

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
        regional_prep_plot_file = os.path.join(dataset_dir, f"{base_name}_regional_preprocessed_bipolar_channels.png")
        plt.savefig(regional_prep_plot_file)
        plt.close()

        # --- Joint Wavelet Decomposition & Quantization ---
        min_length = min(
            len(regional_preprocessed["frontal"]),
            len(regional_preprocessed["motor_temporal"]),
            len(regional_preprocessed["parietal_occipital"])
        )
        n_window_samples = int(window_length_sec * new_sps_val)
        num_windows = min_length // n_window_samples

        tokens_dict = {"frontal": [], "motor_temporal": [], "parietal_occipital": []}
        regions = ["frontal", "motor_temporal", "parietal_occipital"]

        for i in range(num_windows):
            window_start = i * n_window_samples
            window_end = window_start + n_window_samples
            window_data = np.vstack([
                regional_preprocessed["frontal"][window_start:window_end],
                regional_preprocessed["motor_temporal"][window_start:window_end],
                regional_preprocessed["parietal_occipital"][window_start:window_end]
            ])
            (decomposed_channels,
             coeffs_lengths,
             num_samples,
             normalized_data) = wavelet_decompose_window(
                window_data,
                wavelet=wvlet,
                level=level,
                normalization=False  # Data is already normalized globally
            )
            for idx, region in enumerate(regions):
                coeffs_for_channel = decomposed_channels[idx].flatten()
                q_ids = [str(quantize_number(c)) for c in coeffs_for_channel]
                tokens_dict[region].extend(q_ids)
                print(f"Window {i + 1}/{num_windows} - {region}: {len(q_ids)} tokens")

        lengths = {region: len(tokens) for region, tokens in tokens_dict.items()}
        print("Total token counts per region for", base_name, ":", lengths)
        if len(set(lengths.values())) != 1:
            print("Warning: The token sequences for the three regions are not the same length.")
        else:
            print("All regional token sequences have the same length.")

        # --- Write the token files (saved in output_folder with dataset prefix) ---
        token_file_paths = {}
        for region, tokens in tokens_dict.items():
            token_fname = f"{base_name}_quantized_coeffs_{region}.txt"
            out_fname = os.path.join(output_folder, token_fname)
            with open(out_fname, "w") as f_out:
                f_out.write(" ".join(tokens))
            token_file_paths[region] = out_fname
        print(f"Done generating quantized files for {csv_file}.")

        total_tokens = sum(len(tokens) for tokens in tokens_dict.values())
        return base_name, total_tokens, False, "Processed successfully"
    except Exception as e:
        print(f"Error processing dataset '{base_name}': {e}")
        return base_name, 0, True, f"Error: {str(e)}"


def process_csv_file_s3(csv_key: str,
                        bucket: str = "dataframes--use1-az6--x-s3",
                        local_dir: str = "/tmp",
                        output_prefix: str = "output"):
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
    # folders = list_s3_folders()
    csv_files = []
    i = 1
    folders = ["csv_emotiv"]

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
