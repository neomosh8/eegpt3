import os
import glob
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

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
    dequantize_number
)


# --------------------------------------------------------------------------------
# Calculate sampling rate from CSV
# --------------------------------------------------------------------------------
def calculate_sps(csv_file_path):
    # Read the CSV
    df = pd.read_csv(csv_file_path)

    # Extract the 'timestamp' column into a NumPy array
    timestamps = df['timestamp'].values

    # Option A: Compute the mean difference between consecutive timestamps
    avg_dt = np.mean(np.diff(timestamps))  # average time-step (seconds)
    sps = 1.0 / avg_dt

    return sps


# --------------- Preprocessing Functions (resample, filter) -----------
def resample_windows(ndarray, original_sps, new_rate=128):
    """
    Resample the data from original_sps to new_rate.
    :param ndarray: 2D numpy array (channels x samples).
    :param original_sps: Original sampling rate.
    :param new_rate: New sampling rate to resample to (default 128).
    :return: 2D numpy array with resampled data.
    """
    num_channels = ndarray.shape[0]
    resampled_ndarray = []
    for channel_data in ndarray:
        number_of_samples = round(len(channel_data) * float(new_rate) / original_sps)
        resampled_data = signal.resample(channel_data, number_of_samples)
        resampled_ndarray.append(resampled_data)
    return np.array(resampled_ndarray)


def filter_band_pass_windows(ndarray, sps):
    """
    Apply a bandpass filter to the data.
    :param ndarray: 2D numpy array (channels x samples).
    :param sps: Sampling rate.
    :return: 2D numpy array with bandpass filtered data.
    """
    # Example bandpass between 0.1 - 48 Hz
    f_b, f_a = signal.butter(N=5, Wn=[0.1, 48], btype='bandpass', fs=sps)
    filtered_data = signal.filtfilt(f_b, f_a, ndarray, axis=1)
    return filtered_data


def preprocess_data(data, original_sps):
    """
    Preprocess the data by resampling and filtering.
    :param data: 2D numpy array (channels x samples).
    :param original_sps: Original sampling rate.
    :return: Preprocessed 2D numpy array, and the new sampling rate (after resampling).
    """
    # Resample the data to 128 Hz (example)
    resampled_data = resample_windows(data, original_sps, new_rate=128)
    new_sps = 128

    # Filter (Band-pass)
    filtered_data = filter_band_pass_windows(resampled_data, new_sps)
    return filtered_data, new_sps


# -------------------------------------------------------------------------------
# MAIN FUNCTION (Example: Original vs Reconstructed Plot for a Single Random Window)
# -------------------------------------------------------------------------------
def main():
    # === 1. Load one CSV data ===
    df = pd.read_csv('dataset.csv')

    # Identify possible EEG columns (exclude non-EEG columns).
    all_columns = list(df.columns)
    exclude_cols = ['timestamp', 'VEOG', 'X', 'Y', 'Z']  # adjust as needed
    eeg_channels = [col for col in all_columns if col not in exclude_cols]

    # -----------------------------------------------------------------------
    # CHANGED HERE:
    # Automatically determine left vs. right hemisphere based on last character
    # -----------------------------------------------------------------------
    left_chs_in_csv = []
    right_chs_in_csv = []

    for ch in eeg_channels:
        # If it ends with 'z', ignore it
        if ch.endswith('z'):
            continue
        # If it doesn't end with a digit, ignore
        if not ch[-1].isdigit():
            continue

        # We have a digit => check odd/even
        if int(ch[-1]) % 2 == 0:
            right_chs_in_csv.append(ch)  # even => right
        else:
            left_chs_in_csv.append(ch)  # odd => left

    # If both lists are empty => no valid channels
    if not left_chs_in_csv and not right_chs_in_csv:
        raise ValueError("No left or right hemisphere channels found in this dataset.")

    # Average left channels
    if left_chs_in_csv:
        left_data = df[left_chs_in_csv].mean(axis=1).values
    else:
        left_data = np.zeros(len(df))

    # Average right channels
    if right_chs_in_csv:
        right_data = df[right_chs_in_csv].mean(axis=1).values
    else:
        right_data = np.zeros(len(df))

    # Now we have 2 channels: [Left, Right]
    # shape => (2, samples)
    data_2d = np.vstack([left_data, right_data])

    # Original sampling rate from the CSV
    original_sps = calculate_sps('dataset.csv')
    print(f"Calculated sampling rate: {original_sps} Hz")

    # Preprocess the data (2 channels only)
    preprocessed_data, new_sps = preprocess_data(data_2d, original_sps)
    # preprocessed_data shape => (2, total_samples)

    # Now pick ONE channel (either 0 = left, or 1 = right) for demonstration
    chosen_channel_idx = random.randint(0, 1)
    if chosen_channel_idx == 0:
        chosen_channel_name = "Left"
    else:
        chosen_channel_name = "Right"

    num_channels, total_samples = preprocessed_data.shape

    # === 2. Pick one 4-second window ===
    num_samples_4sec = int(4 * new_sps)
    if total_samples < num_samples_4sec:
        raise ValueError("Not enough data to extract 4 seconds from this dataset.")

    max_start_index = total_samples - num_samples_4sec
    start_idx = random.randint(0, max_start_index)
    end_idx = start_idx + num_samples_4sec

    # Extract the channel data
    original_signal = preprocessed_data[chosen_channel_idx, start_idx:end_idx]
    window = np.expand_dims(original_signal, axis=0)  # shape (1, N)

    # === 3. Wavelet Decomposition (with internal z-score) ===
    (decomposed_channels,
     coeffs_lengths,
     num_samples,
     normalized_data) = wavelet_decompose_window(
        window,
        wavelet=wvlet,
        level=level,
        normalization=True
    )

    # === 4. Quantize & Dequantize the wavelet coefficients ===
    quantized_coeffs_list = []
    for coeff_val in decomposed_channels.flatten():
        q_id = quantize_number(coeff_val)
        quantized_coeffs_list.append(q_id)

    # Convert quantized string identifiers back to float
    dequantized_coeffs = [dequantize_number(qid) for qid in quantized_coeffs_list]
    dequantized_coeffs = np.array(dequantized_coeffs).reshape(decomposed_channels.shape)

    # === 5. Wavelet Reconstruction (still in "z-scored" domain) ===
    reconstructed_window = wavelet_reconstruct_window(
        dequantized_coeffs,
        coeffs_lengths,
        num_samples,
        wavelet=wvlet
    )
    reconstructed_signal_z = reconstructed_window[0, :]

    # === 6. (DEMO) Rescale using stats from the PREVIOUS 4s window, if available
    next_start_idx = start_idx - num_samples_4sec
    next_end_idx = start_idx
    if next_start_idx >= 0:
        next_window_data = preprocessed_data[chosen_channel_idx, next_start_idx:next_end_idx]
        next_mean = np.mean(next_window_data)
        next_std = np.std(next_window_data)
    else:
        next_mean = 0.0
        next_std = 1.0
    if next_std == 0:
        next_std = 1.0

    reconstructed_signal_scaled = reconstructed_signal_z * next_std + next_mean

    # === 7. Plot everything ===
    time_axis = np.arange(num_samples_4sec) / float(new_sps)

    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, original_signal, label='Original (this 4s window)', alpha=0.7)
    plt.plot(time_axis, reconstructed_signal_scaled,
             label='Reconstructed (rescaled w/ prev 4s mean/std)',
             alpha=0.7)
    plt.title(f'Original vs Reconstructed - Channel: {chosen_channel_name}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (Preprocessed)')
    plt.legend()
    plt.tight_layout()
    plt.show()


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

def generate_quantized_files(
        dataset_folder="dataset",
        output_coeffs_file="quantized_coeffs.txt",
        output_channels_file="quantized_channels.txt",
        window_length_sec=2.0
):
    """
    Iterate over all CSV files in `dataset_folder`, preprocess them,
    and for each consecutive window of length `window_length_sec`,
    wavelet-decompose+quantize the *averaged left channel* and
    *averaged right channel*. Append the resulting coefficients +
    channel identifiers ("1" for left, "2" for right) to text files.

    :param dataset_folder: Folder path containing .csv files
    :param output_coeffs_file: Name of the output text file for quantized coefficients
    :param output_channels_file: Name of the output text file for repeated channel names
    :param window_length_sec: Window length in seconds (default 2.0)
    """

    # Open in append mode ("a"). If you want to overwrite each time, use "w".
    f_coeffs = open(output_coeffs_file, "a")
    f_chans = open(output_channels_file, "a")

    # Find all CSV files in the folder
    csv_files = sorted(glob.glob(os.path.join(dataset_folder, "*.csv")))
    if not csv_files:
        print(f"No CSV files found in folder: {dataset_folder}")
        f_coeffs.close()
        f_chans.close()
        return

    for csv_file in csv_files:
        print(f"Processing {csv_file} ...")

        # 1) Load CSV
        df = pd.read_csv(csv_file)

        # 2) Identify EEG channels
        all_columns = list(df.columns)
        exclude_cols = ['timestamp', 'VEOG', 'X', 'Y', 'Z', "EXG1", "EXG2", "EXG7", "EXG8"]
        eeg_channels = [col for col in all_columns if col not in exclude_cols]

        # --------------------------------------------------------------------
        # CHANGED HERE:
        # Determine left vs. right sets by last-digit parity
        # --------------------------------------------------------------------
        left_chs_in_csv = []
        right_chs_in_csv = []

        for ch in eeg_channels:
            # skip if ends with 'z'
            if ch.endswith('z'):
                continue
            # skip if last char is not a digit
            if not ch[-1].isdigit():
                continue

            # last character is digit => check odd/even
            if int(ch[-1]) % 2 == 0:
                right_chs_in_csv.append(ch)
            else:
                left_chs_in_csv.append(ch)

        if not left_chs_in_csv and not right_chs_in_csv:
            print(f"No valid left/right channels found in {csv_file}. Skipping.")
            continue

        # 3) Create 2-channel data: [Left, Right]
        if left_chs_in_csv:
            left_data = df[left_chs_in_csv].mean(axis=1).values
        else:
            left_data = np.zeros(len(df))

        if right_chs_in_csv:
            right_data = df[right_chs_in_csv].mean(axis=1).values
        else:
            right_data = np.zeros(len(df))

        # shape => (2, samples)
        data_2d = np.vstack([left_data, right_data])

        # 4) Calculate original sampling rate
        original_sps = calculate_sps(csv_file)

        # 5) Preprocess (resample + bandpass)
        preprocessed_data, new_sps = preprocess_data(data_2d, original_sps)
        # shape => (2, total_samples)

        # 6) Break into consecutive windows of length `window_length_sec`
        n_window_samples = int(window_length_sec * new_sps)
        num_channels, total_samples = preprocessed_data.shape

        if n_window_samples <= 0:
            print(f"Invalid window_length_sec: {window_length_sec}")
            continue
        if total_samples < n_window_samples:
            print(f"Not enough samples to form even one window in {csv_file}. Skipping.")
            continue

        # 7) Slide through the data in blocks of `n_window_samples`
        for window_start in range(0, total_samples - n_window_samples + 1, n_window_samples):
            window_end = window_start + n_window_samples

            # We'll build up a list of strings for the coefficients & channel-names
            all_channel_coeffs = []
            all_channel_names = []

            # Process exactly 2 channels: index 0 => left, 1 => right
            for ch_idx in range(2):
                if ch_idx == 0:
                    ch_name_id = "1"  # Left hemisphere
                else:
                    ch_name_id = "2"  # Right hemisphere

                channel_data = preprocessed_data[ch_idx, window_start:window_end]
                channel_data_2d = channel_data[np.newaxis, :]

                # Wavelet Decompose (with internal z-score)
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

                # Quantize
                q_ids = [quantize_number(c) for c in coeffs_flat]

                # Append these quantized values to our "one window" list
                all_channel_coeffs.extend(q_ids)

                # Repeat the channel name ID ("1" or "2") for each coefficient
                all_channel_names.extend([ch_name_id] * len(q_ids))

            # Now we have a single line representing BOTH channels for this window
            coeffs_line = " ".join(all_channel_coeffs)
            chans_line = " ".join(all_channel_names)

            # Write them to the respective files
            f_coeffs.write(coeffs_line + "\n")
            f_chans.write(chans_line + "\n")

    # Close the files
    f_coeffs.close()
    f_chans.close()
    print("Done generating quantized files.")


# ------------------------------------------------------------------------------
# Example usage of the new block
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # You can run the normal "main" for a single-file plot demonstration
    # main()

    # Or run our new function to process an entire folder of CSVs
    generate_quantized_files(
        dataset_folder="dataset",  # Folder containing your CSV files
        output_coeffs_file="quantized_coeffs.txt",
        output_channels_file="quantized_channels.txt",
        window_length_sec=2.0
    )
