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
        output_channels_file,   # you can ignore or remove this if not needed
        window_length_sec=2.0,
        show_plot=True,
        mse_method="timeseries",  # "timeseries" or "pwelch"
        plot_welch=False
):
    """
    Loads data from original CSV and from a single (or few) lines of quantized wavelet coeffs
    that contain all windows back-to-back. Each window = 256 tokens (128 for channel-0, 128 for channel-1).
    Reconstructs signals and optionally plots or calculates MSE.
    """
    def calculate_mse(original, reconstructed):
        return np.mean((np.array(original) - np.array(reconstructed)) ** 2)

    # 1) Load CSV and Preprocess Data
    df = pd.read_csv(csv_file_path)
    all_columns = list(df.columns)

    # Ask GPT if we skip or process => channels to drop
    all_columns = list(df.columns)
    instructions = call_gpt_for_instructions(
        channel_names=all_columns,
        dataset_id=csv_file_path
    )
    if instructions["action"] == "skip":
        print(f"Skipping dataset '{csv_file_path}' as instructed by GPT.")
        return

    channels_to_drop = instructions.get("channels_to_drop", [])
    print(f"Dropping channels: {str(channels_to_drop)}")
    filtered_columns = [col for col in all_columns if col not in channels_to_drop]

    left_chs_in_csv = []
    right_chs_in_csv = []
    for ch in filtered_columns:
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
    # Preprocess (resample + bandpass) => shape = [2, total_samples]
    preprocessed_data, new_sps = preprocess_data(data_2d, original_sps)

    # Determine how many windows from the raw data
    n_window_samples = int(window_length_sec * new_sps)
    num_channels, total_samples = preprocessed_data.shape
    if n_window_samples <= 0:
        raise ValueError(f"Invalid window_length_sec: {window_length_sec}")
    if total_samples < n_window_samples:
        raise ValueError("Not enough samples for even one window.")

    # The number of windows in the raw signal
    num_windows = total_samples // n_window_samples

    # 2) Load all quantized tokens from the file (single line or few lines)
    with open(output_coeffs_file, 'r') as f_coeffs:
        all_text = f_coeffs.read().strip()
    # Split into tokens
    all_tokens = all_text.split()

    # We expect exactly `num_windows * 256` tokens (128 for channel0 + 128 for channel1 per window)
    TOKENS_PER_WINDOW = 128 * 2  # 256
    expected_total_tokens = num_windows * TOKENS_PER_WINDOW
    if len(all_tokens) < expected_total_tokens:
        raise ValueError(
            f"Not enough tokens in {output_coeffs_file}. "
            f"Expected at least {expected_total_tokens}, got {len(all_tokens)}."
        )
    if len(all_tokens) > expected_total_tokens:
        print(
            f"WARNING: More tokens ({len(all_tokens)}) than expected "
            f"({expected_total_tokens}). We'll only parse the first {expected_total_tokens} tokens."
        )

    # We'll parse exactly num_windows windows from the token list
    all_tokens = all_tokens[:expected_total_tokens]

    # We'll store the MSE for each channel across windows
    channel_mses = {"Left": [], "Right": []}

    # -----------------------------------------
    #  OPTIONAL: PLOTTING / ANIMATION SETUP
    # -----------------------------------------
    if show_plot:
        # For animation
        if plot_welch:
            fig, axes = plt.subplots(2, 2, figsize=(15, 8))
            ax_time_left = axes[0, 0]
            ax_welch_left = axes[0, 1]
            ax_time_right = axes[1, 0]
            ax_welch_right = axes[1, 1]
        else:
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
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
            line5, = ax_welch_left.plot([], [], label='Original Welch Left', alpha=0.7)
            line6, = ax_welch_left.plot([], [], label='Reconstructed Welch Left', alpha=0.7)
            line7, = ax_welch_right.plot([], [], label='Original Welch Right', alpha=0.7)
            line8, = ax_welch_right.plot([], [], label='Reconstructed Welch Right', alpha=0.7)
            ax_welch_left.set_xlabel('Frequency (Hz)')
            ax_welch_left.set_ylabel('PSD')
            ax_welch_right.set_xlabel('Frequency (Hz)')
            ax_welch_right.set_ylabel('PSD')
            ax_welch_left.set_title('Welch PSD Left')
            ax_welch_right.set_title('Welch PSD Right')
            ax_welch_left.legend()
            ax_welch_right.legend()

        time_axis = np.arange(n_window_samples) / float(new_sps)

        # Manual Y-limits for time plots
        min_val = np.min(preprocessed_data)
        max_val = np.max(preprocessed_data)
        ax_time_left.set_ylim(min_val, max_val)
        ax_time_right.set_ylim(min_val, max_val)

        def update(frame):
            """
            frame => which window from 0 to num_windows-1
            We'll read 256 tokens from all_tokens for the given window, parse channel0 vs channel1.
            """
            window_start = frame * n_window_samples
            window_end = window_start + n_window_samples

            # Extract chunk of tokens for THIS window
            start_tok = frame * TOKENS_PER_WINDOW
            end_tok = start_tok + TOKENS_PER_WINDOW
            chunk = all_tokens[start_tok:end_tok]
            if len(chunk) != TOKENS_PER_WINDOW:
                raise ValueError(f"Token chunk size mismatch. Got {len(chunk)}, expected {TOKENS_PER_WINDOW}")

            # Split into ch0 vs ch1
            ch0_coefs_str = chunk[:128]
            ch1_coefs_str = chunk[128:]

            # Reconstruct Left
            ch0_data = preprocessed_data[0, window_start:window_end]
            left_orig, left_recon = wavelet_reconstruct_one_window(
                ch0_data, ch0_coefs_str, new_sps, mse_method
            )
            channel_mses["Left"].append(calculate_mse(left_orig, left_recon))

            # Reconstruct Right
            ch1_data = preprocessed_data[1, window_start:window_end]
            right_orig, right_recon = wavelet_reconstruct_one_window(
                ch1_data, ch1_coefs_str, new_sps, mse_method
            )
            channel_mses["Right"].append(calculate_mse(right_orig, right_recon))

            # Update lines
            line1.set_data(time_axis, left_orig)
            line2.set_data(time_axis, left_recon)
            line3.set_data(time_axis, right_orig)
            line4.set_data(time_axis, right_recon)

            if plot_welch:
                # compute PSD
                welch_orig_left = pwelch_z(left_orig, new_sps)
                welch_rec_left = pwelch_z(left_recon, new_sps)
                line5.set_data(np.arange(len(welch_orig_left[0])), welch_orig_left[0])
                line6.set_data(np.arange(len(welch_rec_left[0])), welch_rec_left[0])

                welch_orig_right = pwelch_z(right_orig, new_sps)
                welch_rec_right = pwelch_z(right_recon, new_sps)
                line7.set_data(np.arange(len(welch_orig_right[0])), welch_orig_right[0])
                line8.set_data(np.arange(len(welch_rec_right[0])), welch_rec_right[0])

                # Auto-scale PSD
                ax_welch_left.relim()
                ax_welch_left.autoscale_view()
                ax_welch_right.relim()
                ax_welch_right.autoscale_view()

                return line1, line2, line3, line4, line5, line6, line7, line8

            return line1, line2, line3, line4

        ani = FuncAnimation(fig, update, frames=num_windows, blit=True)
        plt.tight_layout()
        plt.show()

    else:
        # Non-animated version: just loop over windows
        for frame in range(num_windows):
            window_start = frame * n_window_samples
            window_end = window_start + n_window_samples

            start_tok = frame * TOKENS_PER_WINDOW
            end_tok = start_tok + TOKENS_PER_WINDOW
            chunk = all_tokens[start_tok:end_tok]
            if len(chunk) != TOKENS_PER_WINDOW:
                raise ValueError(f"Token chunk size mismatch. Got {len(chunk)}, expected {TOKENS_PER_WINDOW}")

            # ch0 vs ch1
            ch0_coefs_str = chunk[:128]
            ch1_coefs_str = chunk[128:]

            # Left
            ch0_data = preprocessed_data[0, window_start:window_end]
            left_orig, left_recon = wavelet_reconstruct_one_window(
                ch0_data, ch0_coefs_str, new_sps, mse_method
            )
            channel_mses["Left"].append(calculate_mse(left_orig, left_recon))

            # Right
            ch1_data = preprocessed_data[1, window_start:window_end]
            right_orig, right_recon = wavelet_reconstruct_one_window(
                ch1_data, ch1_coefs_str, new_sps, mse_method
            )
            channel_mses["Right"].append(calculate_mse(right_orig, right_recon))

    # Print final average MSE
    avg_mse_left = np.mean(channel_mses["Left"])
    avg_mse_right = np.mean(channel_mses["Right"])
    print(f"Average MSE - Left Channel: {avg_mse_left:.6f}")
    print(f"Average MSE - Right Channel: {avg_mse_right:.6f}")


def wavelet_reconstruct_one_window(original_data, coeffs_str_list, new_sps, mse_method):
    """
    Helper function: takes the original_data for one window and the *stringified* wavelet coeffs.
    Dequantizes, wavelet-reconstructs, and returns (original_data, reconstructed_data).
    If needed, you can incorporate the "previous window's mean/std" logic here as well.
    """
    # Flatten length must match the wavelet decomposition length
    # You likely know the wavelet level in advance, or we can deduce it
    # from wavelet_decompose_window. For simplicity, let's do the same steps:

    channel_data_2d = original_data[np.newaxis, :]
    (decomposed_channels,
     coeffs_lengths,
     num_samples,
     normalized_data) = wavelet_decompose_window(
        channel_data_2d,
        wavelet='db2',   # or your wavelet
        level=2,         # or your wavelet level
        normalization=True
    )

    # Check size
    flatten_len = decomposed_channels.flatten().shape[0]
    if len(coeffs_str_list) != flatten_len:
        raise ValueError(f"Got {len(coeffs_str_list)} tokens, expected {flatten_len} for wavelet reconst.")

    # Dequantize
    dequantized_coeffs = [dequantize_number(x) for x in coeffs_str_list]
    dequantized_coeffs = np.array(dequantized_coeffs).reshape(decomposed_channels.shape)

    # Reconstruct
    reconstructed_window = wavelet_reconstruct_window(
        dequantized_coeffs,
        coeffs_lengths,
        num_samples,
        wavelet='db2'
    )
    reconstructed_signal_z = reconstructed_window[0, :]

    # If you want to incorporate "previous window" stats, you can do it here
    # For now, we do a naive approach => assume we keep the same normalization as wavelet_decompose_window
    # or do some final rescaling if needed.

    # Example of no extra scaling:
    reconstructed_signal = reconstructed_signal_z

    return original_data, reconstructed_signal




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