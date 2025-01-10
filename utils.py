#utils.py
import json
import os
import math

import boto3
import pandas as pd
from dotenv import load_dotenv
from matplotlib.animation import FuncAnimation

from scipy import signal
import numpy as np
import pywt
import matplotlib.pyplot as plt  # <-- needed for plotting histogram
from  openai import OpenAI

load_dotenv()
wvlet = 'db2'
level = 2
client = OpenAI(
    api_key=os.getenv('api_key')
)

def calculate_stats(data):
    """Calculate mean and order of magnitude."""
    mean_val = np.mean(data)
    oom = np.floor(np.log10(np.abs(mean_val))) if mean_val != 0 else 0
    return mean_val, oom

def plot_amplitude_histogram(data, channel_name='Unknown', bins=50):
    """
    Plot a histogram of the amplitude distribution.

    :param data: 1D or 2D numpy array of signal data
    :param channel_name: Name of the channel for labeling
    :param bins: Number of bins for the histogram
    """
    plt.figure(figsize=(7, 5))
    # If data is 2D (channels x samples), flatten it:
    flattened_data = data.flatten()
    plt.hist(flattened_data, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
    plt.title(f'Amplitude Distribution - {channel_name}')
    plt.xlabel('Amplitude')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def wavelet_decompose_window(window, wavelet='db2', level=4, normalization=True):
    """
    Apply wavelet decomposition to each channel in the window.

    :param window: 2D numpy array (channels x samples)
    :param wavelet: Wavelet type
    :param level: Decomposition level
    :param normalization: If True, use z-score normalization. If False, use magnitude-based scaling
    :return: (decomposed_channels, coeffs_lengths, original_signal_length, normalized_data)
    """
    num_channels, num_samples = window.shape
    decomposed_channels = []
    coeffs_lengths = []
    normalized_data = []

    # ---- NORMALIZATION / SCALING ----
    if normalization:
        # Z-score normalization for the entire dataset (all channels combined).
        mean = np.mean(window)
        std = np.std(window)
        if std == 0:
            std = 1  # Prevent division by zero
        window_normalized = (window - mean) / std
    else:
        # Magnitude-based scaling for the entire dataset
        mean_val, oom = calculate_stats(window)
        # Example logic: if order of magnitude is >= -4, multiply by 10^6 (adjust to your preference)
        if oom >= -4:
            window_normalized = window * (1000000)
        else:
            window_normalized = window.copy()  # Keep original values

    # ---- PLOT HISTOGRAM: amplitude distribution before wavelet decomposition ----
    # You can call the plotting function here to see how your data looks
    # (either channel-by-channel or the entire window at once)
    # plot_amplitude_histogram(window_normalized, channel_name='All Channels', bins=50)

    # ---- PER-CHANNEL WAVELET DECOMPOSITION ----
    for channel_index in range(num_channels):
        # Extract single channel data
        channel_data_normalized = window_normalized[channel_index, :]
        normalized_data.append(channel_data_normalized)

        # Perform wavelet decomposition on normalized data
        coeffs = pywt.wavedec(channel_data_normalized, wavelet, level=level)

        # Flatten the coefficients
        flattened_coeffs = np.hstack([comp.flatten() for comp in coeffs])

        # Store the lengths of each coefficient array (needed for reconstruction)
        lengths = np.array([len(comp) for comp in coeffs])

        decomposed_channels.append(flattened_coeffs)
        coeffs_lengths.append(lengths)

    decomposed_channels = np.array(decomposed_channels)
    coeffs_lengths = np.array(coeffs_lengths)  # shape: (channels x (num_levels + 1))
    normalized_data = np.array(normalized_data)

    return decomposed_channels, coeffs_lengths, num_samples, normalized_data

def wavelet_reconstruct_window(decomposed_channels, coeffs_lengths, num_samples, wavelet='db2'):
    """
    Reconstruct the normalized signal from decomposed channels.

    :param decomposed_channels: 2D numpy array (channels x flattened_coefficients)
    :param coeffs_lengths: 2D numpy array (channels x num_levels+1)
    :param num_samples: Original signal length
    :param wavelet: Wavelet type
    :return: Reconstructed window (channels x samples)
    """
    num_channels = decomposed_channels.shape[0]
    reconstructed_window = []

    for channel_index in range(num_channels):
        flattened_coeffs = decomposed_channels[channel_index]
        lengths = coeffs_lengths[channel_index]
        # Split the flattened coefficients back into list of arrays
        coeffs = []
        idx = 0
        for length in lengths:
            coeff = flattened_coeffs[idx:idx+length]
            coeffs.append(coeff)
            idx += length
        # Reconstruct the signal using waverec
        channel_data_normalized = pywt.waverec(coeffs, wavelet)[:num_samples]
        reconstructed_window.append(channel_data_normalized)

    reconstructed_window = np.array(reconstructed_window)
    return reconstructed_window


def get_sorted_file_list(file_list):
    """
    Groups files by their base pattern (everything except the window number)
    and sorts windows consecutively for each file.
    """
    # Dictionary to store files grouped by their base pattern
    file_groups = {}

    for filepath in file_list:
        base_filename = os.path.basename(filepath)
        # Split the window number from rest of filename
        window_num = int(base_filename.split('_')[0])
        # Get the rest of filename (base pattern)
        base_pattern = '_'.join(base_filename.split('_')[1:])

        # Group files by their base pattern
        if base_pattern not in file_groups:
            file_groups[base_pattern] = []
        file_groups[base_pattern].append((window_num, filepath))

    # Sort each group by window number
    sorted_files = []
    for base_pattern in file_groups:
        files_for_pattern = file_groups[base_pattern]
        files_for_pattern.sort(key=lambda x: x[0])  # Sort by window number
        sorted_files.extend([f[1] for f in files_for_pattern])

    return sorted_files


import math


def quantize_number(z_value, resolution=77):
    """
    Quantize a z-scored value into a token: e.g. 'C23', 'D150', etc.
    with finer granularity near -1 < z < 1, and coarser outside.

    Parameters
    ----------
    z_value : float
        The input z-scored value to quantize.
    resolution : int
        The total number of quantization levels across all ranges.

    Returns
    -------
    token : str
        The quantization token, e.g., 'D230'.
    """
    # 1) Clamp to [-5, 5] (modify if needed)
    z_clamped = max(min(z_value, 5), -5)

    # 2) Define the ranges (in ascending order)
    #    Each range dict has:
    #    'id' (symbol), 'start' (inclusive), 'end' (exclusive), 'proportion'
    ranges = [
        {'id': 'A', 'start': -5, 'end': -3, 'proportion': 0.05},  # z < -3
        {'id': 'B', 'start': -3, 'end': -2, 'proportion': 0.10},  # -3 <= z < -2
        {'id': 'C', 'start': -2, 'end': -1, 'proportion': 0.15},  # -2 <= z < -1
        {'id': 'D', 'start': -1, 'end': 1, 'proportion': 0.40},  # -1 <= z < 1
        {'id': 'E', 'start': 1, 'end': 2, 'proportion': 0.15},  # 1 <= z < 2
        {'id': 'F', 'start': 2, 'end': 3, 'proportion': 0.10},  # 2 <= z < 3
        {'id': 'G', 'start': 3, 'end': 5, 'proportion': 0.05},  # z >= 3
    ]

    # 3) Allocate number of tokens for each range
    for r in ranges:
        r['tokens'] = int(round(r['proportion'] * resolution))

    # 4) Assign a starting token index for each range
    cumulative = 0
    for r in ranges:
        r['token_start'] = cumulative
        cumulative += r['tokens']

    # 5) Find which range the clamped z-value belongs to
    for r in ranges:
        if r['start'] <= z_clamped < r['end']:
            range_id = r['id']
            start = r['start']
            end = r['end']
            tokens_in_range = r['tokens']
            token_offset = r['token_start']
            break
    else:
        # Handle edge case if z_clamped == 5 exactly
        # (by definition above, 'G' range is up to 5, so let's just re-use that)
        range_id = 'G'
        start = 3
        end = 5
        tokens_in_range = ranges[-1]['tokens']
        token_offset = ranges[-1]['token_start']

    # 6) Linear quantization within that range
    #    (We assume a direct linear scale from start to end)
    if tokens_in_range <= 1:
        # Edge case: if tokens_in_range is 0 or 1, fallback or just yield 0
        quant_level = 0
    else:
        q = (z_clamped - start) / (end - start)  # 0..1
        quant_level = int(round(q * (tokens_in_range - 1)))

    # 7) Combine with offset (if you want a single global numbering system).
    #    But if you only want *per-range* quantization level, you can skip offset.
    global_quant_index = token_offset + quant_level

    # 8) Form the token.
    #    If you prefer per-range indexing, use f"{range_id}{quant_level}".
    #    If you want a single 0..499 scale, you might do something else.
    token = f"{range_id}{global_quant_index}"
    return token


def dequantize_number(token, resolution=77):
    """
    Convert a token like 'C23' or 'D144' back to a z-score approximation.
    """
    # 1) Separate the range ID (the first character) and the index (rest)
    range_id = token[0]
    idx_str = token[1:]
    if not idx_str.isdigit():
        raise ValueError(f"Invalid token format: {token}")
    global_quant_index = int(idx_str)

    # 2) Same range definitions
    ranges = [
        {'id': 'A', 'start': -5, 'end': -3, 'proportion': 0.05},
        {'id': 'B', 'start': -3, 'end': -2, 'proportion': 0.10},
        {'id': 'C', 'start': -2, 'end': -1, 'proportion': 0.15},
        {'id': 'D', 'start': -1, 'end':  1, 'proportion': 0.40},
        {'id': 'E', 'start':  1, 'end':  2, 'proportion': 0.15},
        {'id': 'F', 'start':  2, 'end':  3, 'proportion': 0.10},
        {'id': 'G', 'start':  3, 'end':  5, 'proportion': 0.05},
    ]

    # 3) Recompute #tokens and token_start
    cumulative = 0
    for r in ranges:
        r['tokens'] = int(round(r['proportion'] * resolution))
    for r in ranges:
        r['token_start'] = cumulative
        cumulative += r['tokens']

    # 4) Find the matching range
    r_found = None
    for r in ranges:
        if r['id'] == range_id:
            r_found = r
            break
    if not r_found:
        raise ValueError(f"Unknown range ID {range_id} in token {token}")

    # 5) Derive the per-range index
    #    If you used 'global_quant_index' in quantize,
    #    we must subtract r['token_start'] to get the local index in that range.
    local_index = global_quant_index - r_found['token_start']

    tokens_in_range = r_found['tokens']
    if tokens_in_range <= 1:
        # Edge case
        return (r_found['start'] + r_found['end']) / 2.0  # middle of that bin

    # 6) The ratio in that range
    q = local_index / (tokens_in_range - 1)  # 0..1

    # 7) Re-map from [0..1] to [start..end]
    z_approx = r_found['start'] + q * (r_found['end'] - r_found['start'])

    return z_approx

def pwelch_z(data, sps):
    if len(data.shape) < 2:
        data = np.expand_dims(data, axis=0)
    win_len = min(2 * sps, len(data[0]))
    f_, pxx = signal.welch(data, fs=sps, window=signal.windows.tukey(win_len, sym=False, alpha=.17),
                           nperseg=win_len, noverlap=int(win_len * 0.75), nfft=2 * sps, return_onesided=True,
                           scaling='spectrum', axis=-1, average='mean')

    return pxx[:, 2:82] * 0.84



def call_gpt_for_instructions(channel_names, dataset_id):
    """
    Calls the GPT model with channel names to decide whether to skip the dataset,
    and which channels to drop if processing.

    :param channel_names: List of channel names.
    :param dataset_id: ID of the dataset being processed.
    :return: A dictionary with instructions.
    """
    prompt = f'''
You are an assistant that helps in processing EEG datasets.
Based on the following channel names, specify which channels to drop.
If there are channels that are auxiliary channels or information (like timestamp, temp, EOG, ECG, GSR, Trigger, EMG, eye signals, etc.), those channels should be dropped.


Provide the response in the following JSON format:


    "action": "process" 
    "channels_to_drop": ["channelname1", "channelname2", ...]

Channel Names for Dataset {dataset_id}:
{', '.join(channel_names)}
'''

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                },
            ],
            temperature=0.0,
            response_format={
                "type": "json_object"
            }
        )

        # Extract the assistant's reply
        gpt_reply = json.loads(str(response.choices[0].message.content.strip()))

        # Attempt to parse the JSON from GPT's response
        instructions = gpt_reply
        return instructions
    except Exception as e:
        print(f"Error communicating with GPT: {e}")
        # In case of error, default to processing without changes
        return {
            "action": "process",
            "channels_to_drop": []
        }


# Initialize the S3 client
s3 = boto3.client('s3')
def list_s3_folders(bucket_name='dataframes--use1-az6--x-s3', prefix=''):
    # List all objects in the bucket that start with the given prefix and use '/' as a delimiter to identify folders
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/')

    # Extract folder names, strip the trailing '/'
    folders = [prefix.get('Prefix').rstrip('/') for prefix in response.get('CommonPrefixes', [])]

    return folders


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
    resampled_data = resample_windows(data, original_sps, new_rate=123)
    new_sps = 123

    # Filter (Band-pass)
    filtered_data = filter_band_pass_windows(resampled_data, new_sps)
    return filtered_data, new_sps

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



# ------------------------------------------------------------------------------
# NEW BLOCK:
# Validation Function
# ------------------------------------------------------------------------------
def validate_round_trip(
        csv_file_path,
        output_coeffs_file,
        output_channels_file,
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

    quantized_coeffs_file_path = output_coeffs_file
    quantized_channels_file_path = output_channels_file


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


def list_csv_files_in_folder(folder_name , bucket_name='dataframes--use1-az6--x-s3', ):
    """
    List all CSV files in a specific folder within an S3 bucket.

    :param bucket_name: Name of the S3 bucket.
    :param folder_name: Folder name (prefix) inside the bucket.
    :return: List of CSV file keys.
    """
    s3 = boto3.client('s3')
    csv_files = []

    try:
        paginator = s3.get_paginator('list_objects_v2')
        response_iterator = paginator.paginate(Bucket=bucket_name, Prefix=f"{folder_name}/")

        for page in response_iterator:
            for content in page.get('Contents', []):
                key = content.get('Key', '')
                if key.endswith('.csv'):
                    csv_files.append(key)
    except Exception as e:
        print(f"Error listing CSV files: {e}")

    return csv_files


### TEMP UTILS
import os
import boto3

def aggregate_coeffs_files_s3(
    bucket='dataframes--use1-az6--x-s3',
    output_prefix="output/",
    local_dir="/tmp",
    aggregated_name="all_coeffs.txt"
):
    s3 = boto3.client('s3')
    local_agg_path = os.path.join(local_dir, aggregated_name)

    print(f"\033[94mStarting aggregation of _coeffs.txt files from s3://{bucket}/{output_prefix} -> {aggregated_name}\033[0m")

    if os.path.exists(local_agg_path):
        print(f"\033[93mRemoving existing local file: {local_agg_path}\033[0m")
        os.remove(local_agg_path)

    # Gather all relevant files
    paginator = s3.get_paginator('list_objects_v2')
    all_keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=output_prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('_coeffs.txt'):
                all_keys.append(key)

    total = len(all_keys)
    if total == 0:
        print("\033[93mNo _coeffs.txt files found.\033[0m")
        return

    print(f"\033[94mFound {total} files.\033[0m")

    for i, key in enumerate(all_keys, start=1):
        percentage = 100 * i / total
        print(f"\033[94m[{i}/{total} | {percentage:.1f}%] {key}\033[0m")

        local_temp = os.path.join(local_dir, os.path.basename(key))
        print(f"\033[92mDownloading {key} -> {local_temp}\033[0m")
        s3.download_file(bucket, key, local_temp)

        with open(local_temp, 'r') as src, open(local_agg_path, 'a') as dst:
            content = src.read()
            dst.write(content)
            print(f"\033[92mAppended {len(content)} chars from {key}\033[0m")

        print(f"\033[91mRemoving temp {local_temp}\033[0m")
        os.remove(local_temp)

    print(f"\033[93mUploading aggregated file to s3://{bucket}/{aggregated_name}\033[0m")
    s3.upload_file(local_agg_path, bucket, aggregated_name)

    print(f"\033[91mRemoving local aggregated file {local_agg_path}\033[0m")
    os.remove(local_agg_path)

    print("\033[94mAll done.\033[0m")