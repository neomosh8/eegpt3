import os
import glob
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.animation import FuncAnimation

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
    pwelch_z
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
        window_length_sec=2.0
):
    """
    Iterate over all CSV files in `dataset_folder`, preprocess them,
    and for each consecutive window of length `window_length_sec`,
    wavelet-decompose+quantize the *averaged left channel* and
    *averaged right channel*. Append the resulting coefficients +
    channel identifiers ("1" for left, "2" for right) to text files.

    :param dataset_folder: Folder path containing .csv files
    :param window_length_sec: Window length in seconds (default 2.0)
    """

    # Find all CSV files in the folder
    csv_files = sorted(glob.glob(os.path.join(dataset_folder, "*.csv")))
    if not csv_files:
        print(f"No CSV files found in folder: {dataset_folder}")
        return

    os.makedirs('output', exist_ok=True)

    for csv_file in csv_files:
        print(f"Processing {csv_file} ...")

        # --- Output files per CSV ---
        base_name = os.path.splitext(os.path.basename(csv_file))[0]

        # Create full paths for the output files inside the "output" folder
        output_coeffs_file = os.path.join('output', f"{base_name}_quantized_coeffs.txt")
        output_channels_file = os.path.join('output', f"{base_name}_quantized_channels.txt")

        # Open in write mode ("w"). Overwriting previous files each time
        f_coeffs = open(output_coeffs_file, "w")
        f_chans = open(output_channels_file, "w")

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
            f_coeffs.close()
            f_chans.close()
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
            f_coeffs.close()
            f_chans.close()
            continue
        if total_samples < n_window_samples:
            print(f"Not enough samples to form even one window in {csv_file}. Skipping.")
            f_coeffs.close()
            f_chans.close()
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
# NEW BLOCK:
# Validation Function
# ------------------------------------------------------------------------------
def validate_round_trip(
        csv_file_path,
        window_length_sec=2.0,
        show_plot=True,
        mse_method="timeseries",  # options: "timeseries" or "pwelch"
        plot_welch=False
):
    """
    Loads data from original CSV, quantized coeffs, and channel txt files.
    It then performs the round trip (decompose->quantize->dequantize->reconstruct)
    and optionally generates an animation of the original vs reconstructed signal.
    It also calculates and prints the average MSE between original and reconstructed data.
    :param csv_file_path: path of the input csv
    :param quantized_coeffs_file_path: path to the quantized_coeffs.txt file
    :param quantized_channels_file_path: path to the quantized_channels.txt file
    :param window_length_sec: Window length for plotting
    :param show_plot: Boolean, whether to show animation plot (default: True).
    :param mse_method: String, method for MSE calculation ("timeseries" or "pwelch").
    :param plot_welch: Boolean, whether to plot the Welch spectrogram next to the signal,
    """

    base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    quantized_coeffs_file_path = f'{base_name}_quantized_coeffs.txt'
    quantized_channels_file_path = f'{base_name}_quantized_channels.txt'

    def calculate_mse(original, reconstructed):
        return np.mean((np.array(original) - np.array(reconstructed)) ** 2)

    # 1) Load CSV and Preprocess Data
    df = pd.read_csv(csv_file_path)

    # Identify EEG channels
    all_columns = list(df.columns)
    exclude_cols = ['timestamp', 'VEOG', 'X', 'Y', 'Z', "EXG1", "EXG2", "EXG7", "EXG8"]
    eeg_channels = [col for col in all_columns if col not in exclude_cols]

    # Determine left vs. right sets by last-digit parity
    left_chs_in_csv = []
    right_chs_in_csv = []

    for ch in eeg_channels:
        if ch.endswith('z'):
            continue
        if not ch[-1].isdigit():
            continue
        if int(ch[-1]) % 2 == 0:
            right_chs_in_csv.append(ch)
        else:
            left_chs_in_csv.append(ch)

    if not left_chs_in_csv and not right_chs_in_csv:
        raise ValueError(f"No valid left/right channels found in {csv_file_path}.")

    # Create 2-channel data: [Left, Right]
    if left_chs_in_csv:
        left_data = df[left_chs_in_csv].mean(axis=1).values
    else:
        left_data = np.zeros(len(df))

    if right_chs_in_csv:
        right_data = df[right_chs_in_csv].mean(axis=1).values
    else:
        right_data = np.zeros(len(df))

    data_2d = np.vstack([left_data, right_data])

    # Calculate sampling rate
    original_sps = calculate_sps(csv_file_path)

    # Preprocess (resample + bandpass)
    preprocessed_data, new_sps = preprocess_data(data_2d, original_sps)

    # Break into consecutive windows
    n_window_samples = int(window_length_sec * new_sps)
    num_channels, total_samples = preprocessed_data.shape

    if n_window_samples <= 0:
        raise ValueError(f"Invalid window_length_sec: {window_length_sec}")
    if total_samples < n_window_samples:
        raise ValueError(f"Not enough samples to form even one window in {csv_file_path}.")

    # Load quantized coeffs and channels
    with open(quantized_coeffs_file_path, 'r') as f_coeffs, open(quantized_channels_file_path, 'r') as f_chans:
        coeffs_lines = f_coeffs.readlines()
        chans_lines = f_chans.readlines()

    # Basic checks
    if len(coeffs_lines) != len(chans_lines):
        raise ValueError(
            "Quantized coeffs and channels files have a different number of lines."
        )

    num_windows = len(coeffs_lines)
    if num_windows * n_window_samples > total_samples:
        raise ValueError(
            "There are more windows in the coeff files then in the data"
        )

    # initialize variables to store MSEs
    channel_mses = {
        "Left": [],
        "Right": []
    }

    if show_plot:
        # --- Setup for Animation ---
        if plot_welch:
            fig, axes = plt.subplots(2, 2, figsize=(15, 8))  # Create 2 subplots for each channel, plus welch
            ax_time_left = axes[0, 0]
            ax_welch_left = axes[0, 1]
            ax_time_right = axes[1, 0]
            ax_welch_right = axes[1, 1]
        else:
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))  # Create 2 subplots for each channel
            ax_time_left = axes[0]
            ax_time_right = axes[1]

        line1, = ax_time_left.plot([], [], label='Original Left', alpha=0.7)
        line2, = ax_time_left.plot([], [], label='Reconstructed Left', alpha=0.7)
        line3, = ax_time_right.plot([], [], label='Original Right', alpha=0.7)
        line4, = ax_time_right.plot([], [], label='Reconstructed Right', alpha=0.7)

        ax_time_left.set_xlabel('Time (s)')
        ax_time_left.set_ylabel('Amplitude')
        ax_time_left.legend()
        ax_time_left.set_title('Left Channel')

        ax_time_right.set_xlabel('Time (s)')
        ax_time_right.set_ylabel('Amplitude')
        ax_time_right.legend()
        ax_time_right.set_title('Right Channel')

        if plot_welch:
            ax_welch_left.set_xlabel('Frequency (Hz)')
            ax_welch_left.set_ylabel('Power Spectral Density')
            ax_welch_left.set_title('Welch PSD Left')
            ax_welch_right.set_xlabel('Frequency (Hz)')
            ax_welch_right.set_ylabel('Power Spectral Density')
            ax_welch_right.set_title('Welch PSD Right')
            line5, = ax_welch_left.plot([], [], label='Original Welch Left', alpha=0.7)
            line6, = ax_welch_left.plot([], [], label='Reconstructed Welch Left', alpha=0.7)
            line7, = ax_welch_right.plot([], [], label='Original Welch Right', alpha=0.7)
            line8, = ax_welch_right.plot([], [], label='Reconstructed Welch Right', alpha=0.7)
            ax_welch_left.legend()
            ax_welch_right.legend()

        time_axis = np.arange(n_window_samples) / float(new_sps)

        # --- Manual Y-Axis Limits ---
        # Calculate range of original signal (across all windows)
        min_val = np.min(preprocessed_data)
        max_val = np.max(preprocessed_data)

        ax_time_left.set_ylim(min_val, max_val)
        ax_time_right.set_ylim(min_val, max_val)

        def update(frame):
            """
            Updates the plot for the given animation frame (one window).
            This version auto-scales the Welch PSD subplots each frame
            while keeping the time-domain signal subplots at a fixed manual y-range.
            """
            window_start = frame * n_window_samples
            window_end = window_start + n_window_samples

            # Read one line of quantized coeffs and channels
            coeffs_line = coeffs_lines[frame].strip()
            chans_line = chans_lines[frame].strip()

            coeffs_list = coeffs_line.split()
            chans_list = chans_line.split()

            if len(coeffs_list) != len(chans_list):
                raise ValueError(f"Coeff and channel length mismatch at frame {frame}")

            # Prepare storage for original/reconstructed signals
            original_signals = []
            reconstructed_signals = []

            # Prepare storage for Welch PSD (if requested)
            welch_originals = []
            welch_reconstructed = []

            # Process exactly 2 channels: [Left, Right]
            for ch_idx in range(2):
                # Channel name for labeling
                if ch_idx == 0:
                    ch_name_id = "Left"
                else:
                    ch_name_id = "Right"

                # Extract the original data for this window
                channel_data = preprocessed_data[ch_idx, window_start:window_end]
                original_signals.append(channel_data)

                # Wavelet Decompose (with z-score)
                channel_data_2d = channel_data[np.newaxis, :]
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

                # Determine which slice of coeffs_list belongs to this channel
                # (Because we have exactly 2 channels per window, each channel
                #  gets a contiguous chunk of coeffs in the text file.)
                # --- For 2 channels, a simple way is to split half for left, half for right ---
                # But if you want to do it precisely by channel IDs in chans_list,
                # you'd parse which are '1' vs. '2'. We'll keep it simple here.
                total_coeffs_for_one_channel = coeffs_flat.shape[0]
                # For example: first half belongs to ch_idx=0, second half to ch_idx=1
                start_idx = ch_idx * total_coeffs_for_one_channel
                end_idx = start_idx + total_coeffs_for_one_channel
                coeffs_flat_from_file = coeffs_list[start_idx:end_idx]

                # Dequantize
                dequantized_coeffs = [dequantize_number(qid) for qid in coeffs_flat_from_file]
                dequantized_coeffs = np.array(dequantized_coeffs).reshape(decomposed_channels.shape)

                # Wavelet Reconstruct
                reconstructed_window = wavelet_reconstruct_window(
                    dequantized_coeffs,
                    coeffs_lengths,
                    num_samples,
                    wavelet=wvlet
                )
                reconstructed_signal_z = reconstructed_window[0, :]

                # Rescale with previous window stats (demo style)
                prev_window_start = window_start - n_window_samples
                prev_window_end = window_start
                if prev_window_start >= 0:
                    prev_window_data = preprocessed_data[ch_idx, prev_window_start:prev_window_end]
                    prev_mean = np.mean(prev_window_data)
                    prev_std = np.std(prev_window_data)
                else:
                    prev_mean = 0.0
                    prev_std = 1.0

                if prev_std == 0:
                    prev_std = 1.0

                reconstructed_signal_scaled = reconstructed_signal_z * prev_std + prev_mean
                reconstructed_signals.append(reconstructed_signal_scaled)

                # If using Welch for MSE or plotting, compute PSD
                if mse_method == "pwelch" or plot_welch:
                    welch_orig = pwelch_z(channel_data, new_sps)
                    welch_rec = pwelch_z(reconstructed_signal_scaled, new_sps)
                    welch_originals.append(welch_orig)
                    welch_reconstructed.append(welch_rec)

                # Calculate MSE (if desired)
                if mse_method == "timeseries":
                    mse_val = calculate_mse(channel_data, reconstructed_signal_scaled)
                elif mse_method == "pwelch":
                    mse_val = calculate_mse(welch_orig.flatten(), welch_rec.flatten())
                else:
                    raise ValueError("Invalid MSE method")
                channel_mses[ch_name_id].append(mse_val)

            # Update time-domain lines
            line1.set_data(time_axis, original_signals[0])  # Left Original
            line2.set_data(time_axis, reconstructed_signals[0])  # Left Reconstructed
            line3.set_data(time_axis, original_signals[1])  # Right Original
            line4.set_data(time_axis, reconstructed_signals[1])  # Right Reconstructed

            # Update Welch lines (if plot_welch = True)
            if plot_welch and len(welch_originals) == 2:
                # Left Channel PSD
                line5.set_data(np.arange(welch_originals[0].shape[-1]),
                               welch_originals[0].flatten())
                line6.set_data(np.arange(welch_reconstructed[0].shape[-1]),
                               welch_reconstructed[0].flatten())

                # Right Channel PSD
                line7.set_data(np.arange(welch_originals[1].shape[-1]),
                               welch_originals[1].flatten())
                line8.set_data(np.arange(welch_reconstructed[1].shape[-1]),
                               welch_reconstructed[1].flatten())

                # ------------------------------------------------------------
                #  AUTO-SCALE Welch PSD subplots each frame
                # ------------------------------------------------------------
                ax_welch_left.relim()
                ax_welch_left.autoscale_view()
                ax_welch_right.relim()
                ax_welch_right.autoscale_view()

            # If you want to RE-FIT the time-domain axes each frame, uncomment these:
            # ax_time_left.relim()
            # ax_time_left.autoscale_view()
            # ax_time_right.relim()
            # ax_time_right.autoscale_view()

            # Otherwise, they stay at the fixed limits you set before (e.g. min_val, max_val).

            # --- Return all lines to be updated (include welch lines if plotting PSD) ---
            if plot_welch:
                return line1, line2, line3, line4, line5, line6, line7, line8
            else:
                return line1, line2, line3, line4

        ani = FuncAnimation(fig, update, frames=num_windows, blit=True)
        plt.tight_layout()
        plt.show()
    else:
        for frame in range(num_windows):
            window_start = frame * n_window_samples
            window_end = window_start + n_window_samples

            coeffs_line = coeffs_lines[frame].strip()
            chans_line = chans_lines[frame].strip()

            coeffs_list = coeffs_line.split()
            chans_list = chans_line.split()

            if len(coeffs_list) != len(chans_list):
                raise ValueError(f"Coeff and channel length mismatch at frame {frame}")

            all_channel_coeffs = []
            all_channel_names = []
            original_signals = []
            reconstructed_signals = []

            means = []
            stds = []

            welch_originals = []
            welch_reconstructed = []

            # Process exactly 2 channels: index 0 => left, 1 => right
            for ch_idx in range(2):
                if ch_idx == 0:
                    ch_name_id = "Left"  # Left hemisphere
                else:
                    ch_name_id = "Right"  # Right hemisphere

                channel_data = preprocessed_data[ch_idx, window_start:window_end]
                original_signals.append(channel_data)
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

                # Get correct coeffs from file
                coeffs_flat_from_file = []
                start_idx = 0
                for i in range(ch_idx):
                    start_idx += sum([len(c.flatten()) for c in decomposed_channels])
                end_idx = start_idx + sum([len(c.flatten()) for c in decomposed_channels])
                for i in range(start_idx, end_idx):
                    coeffs_flat_from_file.append(coeffs_list[i])

                # Convert back to float
                dequantized_coeffs = [dequantize_number(qid) for qid in coeffs_flat_from_file]
                dequantized_coeffs = np.array(dequantized_coeffs).reshape(decomposed_channels.shape)

                # Reconstruct
                reconstructed_window = wavelet_reconstruct_window(
                    dequantized_coeffs,
                    coeffs_lengths,
                    num_samples,
                    wavelet=wvlet
                )
                reconstructed_signal_z = reconstructed_window[0, :]

                # === 6. (DEMO) Rescale using stats from the PREVIOUS 4s window, if available
                prev_window_start = window_start - n_window_samples
                prev_window_end = window_start

                if prev_window_start >= 0:
                    prev_window_data = preprocessed_data[ch_idx, prev_window_start:prev_window_end]
                    prev_mean = np.mean(prev_window_data)
                    prev_std = np.std(prev_window_data)
                else:
                    prev_mean = 0.0
                    prev_std = 1.0

                if prev_std == 0:
                    prev_std = 1.0

                reconstructed_signal_scaled = reconstructed_signal_z * prev_std + prev_mean
                reconstructed_signals.append(reconstructed_signal_scaled)

                if mse_method == "pwelch" or plot_welch:
                    welch_orig = pwelch_z(original_signals[ch_idx], new_sps)
                    welch_rec = pwelch_z(reconstructed_signals[ch_idx], new_sps)
                    welch_originals.append(welch_orig)
                    welch_reconstructed.append(welch_rec)

                if mse_method == "timeseries":
                    mse = calculate_mse(original_signals[ch_idx], reconstructed_signals[ch_idx])
                elif mse_method == "pwelch":
                    if welch_originals and welch_reconstructed:
                        mse = calculate_mse(welch_originals[-1].flatten(), welch_reconstructed[-1].flatten())
                    else:
                        mse = 0
                else:
                    raise ValueError("Invalid MSE method")

                channel_mses[ch_name_id].append(mse)

    avg_mse_left = np.mean(channel_mses["Left"])
    avg_mse_right = np.mean(channel_mses["Right"])

    print(f"Average MSE - Left Channel: {avg_mse_left:.6f}")
    print(f"Average MSE - Right Channel: {avg_mse_right:.6f}")

# ------------------------------------------------------------------------------
# Example usage of the new block
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # You can run the normal "main" for a single-file plot demonstration
    # main()

    # # Or run our new function to process an entire folder of CSVs
    generate_quantized_files(
        dataset_folder="dataset",  # Folder containing your CSV files
        window_length_sec=2.0
    )



    # # Example usage with different options:
    # validate_round_trip(
    #     csv_file_path='dataset/sub-000_task-proposer_run-1_eeg.csv',  # Replace with your CSV path
    #     window_length_sec=2.0,
    #     show_plot=False,  # Set to False to hide plot
    #     mse_method="timeseries",  # Use "pwelch" to compute on pwelch
    #     plot_welch=True  # Set to True to plot pwelch next to the time series plot
    # )