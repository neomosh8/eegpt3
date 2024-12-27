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

# --------------- Preprocessing Functions (resample, remontage, filter) -----------
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

def remontage_windows(ndarray, montage='AR'):
    """
    Apply remontage to the data (e.g., average reference).
    :param ndarray: 2D numpy array (channels x samples).
    :param montage: Type of montage. Currently, only 'AR' is implemented.
    :return: 2D numpy array with remontaged data.
    """
    num_channels = ndarray.shape[0]
    if montage == 'AR':
        i_ = np.eye(num_channels)
        r_ave = np.ones((num_channels, num_channels)) * (1.0 / num_channels)
        x_ = i_ - r_ave
        remontaged_data = np.dot(x_, ndarray)
        return remontaged_data
    else:
        # Implement other montage methods if needed
        return ndarray

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
    Preprocess the data by resampling, remontaging, and filtering.
    :param data: 2D numpy array (channels x samples).
    :param original_sps: Original sampling rate.
    :return: Preprocessed 2D numpy array, and the new sampling rate (after resampling).
    """
    # Resample the data to 128 Hz (example)
    resampled_data = resample_windows(data, original_sps, new_rate=128)
    new_sps = 128
    
    # Remontage (Average Reference)
    remontaged_data = remontage_windows(resampled_data, montage='AR')
    
    # Filter (Band-pass)
    filtered_data = filter_band_pass_windows(remontaged_data, new_sps)
    return filtered_data, new_sps

# -------------------------------------------------------------------------------
# MAIN FUNCTION (Example: Original vs Reconstructed Plot for a Single Random Window)
# -------------------------------------------------------------------------------
def main():
    # === 1. Load one CSV data ===
    df = pd.read_csv('dataset.csv')

    # Identify possible EEG channels (exclude non-EEG columns)
    all_columns = list(df.columns)
    exclude_cols = ['timestamp', 'VEOG', 'X', 'Y', 'Z']  # adjust as needed
    eeg_channels = [col for col in all_columns if col not in exclude_cols]

    # --- Convert to 2D array [channels x samples] ---
    data_2d = df[eeg_channels].T.values

    # Original sampling rate from the CSV
    original_sps = calculate_sps('dataset.csv')
    print(f"Calculated sampling rate: {original_sps} Hz")

    # Preprocess the data
    preprocessed_data, new_sps = preprocess_data(data_2d, original_sps)

    # Now we have: preprocessed_data.shape = (num_channels, new_num_samples)
    num_channels, total_samples = preprocessed_data.shape

    # === 2. Randomly pick ONE channel and one 4-second window ===
    chosen_channel_idx = random.randint(0, num_channels - 1)
    chosen_channel_name = eeg_channels[chosen_channel_idx]
    
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
# Each line in both files covers ALL channels in that window, concatenated.
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
    wavelet-decompose+quantize all channels. 
    Append the resulting coefficients + channel names to text files.

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
        exclude_cols = ['timestamp', 'VEOG', 'X', 'Y', 'Z']  # adjust as needed
        eeg_channels = [col for col in all_columns if col not in exclude_cols]

        if len(eeg_channels) == 0:
            print(f"No EEG channels found in {csv_file}. Skipping.")
            continue

        # 3) Convert to 2D array: shape (num_channels, num_samples)
        data_2d = df[eeg_channels].T.values

        # 4) Calculate original sampling rate
        original_sps = calculate_sps(csv_file)

        # 5) Preprocess (resample, montage, bandpass)
        preprocessed_data, new_sps = preprocess_data(data_2d, original_sps)

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
        #    For each block => wavelet-decompose each channel => quantize => store
        for window_start in range(0, total_samples - n_window_samples + 1, n_window_samples):
            window_end = window_start + n_window_samples

            # We'll build up a list of strings for the coefficients & channel-names
            all_channel_coeffs = []
            all_channel_names = []

            # For each channel, wavelet-decompose this window
            for ch_idx, ch_name in enumerate(eeg_channels):
                channel_data = preprocessed_data[ch_idx, window_start:window_end]

                # Reshape to (1, samples)
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
                # decomposed_channels shape -> (1, total_coeffs) (flatten of subbands)
                # Flatten for quantization
                coeffs_flat = decomposed_channels.flatten()

                # Quantize
                q_ids = [quantize_number(c) for c in coeffs_flat]

                # Append these quantized values to our "one window" list
                all_channel_coeffs.extend(q_ids)

                # Repeat the channel name the same number of times
                all_channel_names.extend([ch_name] * len(q_ids))

            # Now we have a single line representing ALL channels for this window
            #  - in quantized form
            #  - and repeated channel names
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
    # You can run the normal "main" if you want the single-file plot demonstration
    # main()

    # Or run our new function to process an entire folder of CSVs
    generate_quantized_files(
        dataset_folder="dataset",                  # Folder containing your CSV files
        output_coeffs_file="quantized_coeffs.txt", # Space-separated quantized numbers
        output_channels_file="quantized_channels.txt", # Parallel list of channel names
        window_length_sec=2.0                      # Default 2-second windows
    )
