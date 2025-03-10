#utils.py
import json
import os
import math

import boto3
import pandas as pd
from dotenv import load_dotenv
# from dotenv import load_dotenv
from matplotlib.animation import FuncAnimation

from scipy import signal
import numpy as np
import pywt
import matplotlib.pyplot as plt  # <-- needed for plotting histogram
from  openai import OpenAI

load_dotenv()
wvlet = 'db2'
level = 4
client = OpenAI(
    api_key=os.getenv('api_key'))
    #  api_key='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')

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


import numpy as np
import pywt
import matplotlib.pyplot as plt


def wavelet_decompose_window(window, wavelet='cmor1.5-1.0', scales=None, normalization=True, sampling_period=1.0,
                             verbose=True):
    """
    Apply continuous wavelet transform (CWT) with emphasis on frequency bands related to
    higher cognitive functions, emotions, attention, and decision-making.

    Args:
        window (numpy.ndarray): 2D array (channels x samples).
        wavelet (str): Wavelet type (default: 'cmor1.5-1.0').
        scales (list or None): Scales for CWT. If None, computed based on cognitive bands.
        normalization (bool): If True, apply per-scale z-score normalization to coefficients.
        sampling_period (float): Sampling period in seconds.
        verbose (bool): If True, plot the wavelet decomposition result for each channel.

    Returns:
        tuple: (decomposed_channels, scales, original_signal_length, normalized_data)
    """
    num_channels, num_samples = window.shape
    decomposed_channels = []
    normalized_data = []

    # Make a copy of the input window data
    window_normalized = window.copy()

    # Define scales with emphasis on cognitive function bands
    if scales is None:
        fs = 1.0 / sampling_period  # For example, if sampling_period = 1/256, then fs = 256 Hz.

        # Define EEG bands with their frequency ranges and cognitive associations
        eeg_bands = {
            "delta": (0.1, 4),  # Sleep, unconscious processes
            "theta": (4, 8),  # Memory, emotional processing, meditation
            "alpha": (8, 12),  # Attention, inhibition, relaxation, awareness
            "low_beta": (12, 16),  # Active thinking, focus, problem-solving
            "mid_beta": (16, 20),  # Active thinking, anxiety
            "high_beta": (20, 30),  # Alert, anxiety, excitement
            "low_gamma": (30, 50),  # Cognitive processing, feature binding
            "high_gamma": (50, 80)  # Advanced cognitive processing
        }

        # Allocate more scales to bands associated with cognitive functions
        scales_per_band = {
            "delta": 15,  # Fewer scales for delta
            "theta": 60,  # More for memory, emotional processing
            "alpha": 70,  # Most for attention, awareness
            "low_beta": 50,  # High for problem-solving, decision-making
            "mid_beta": 40,  # Good amount for active thinking
            "high_beta": 30,  # Moderate for alertness
            "low_gamma": 20,  # Some for cognitive processing
            "high_gamma": 10  # Fewer for highest frequencies
        }

        scales = []
        # Generate scales for each band, with more detail in cognitively important frequencies
        for band_name, (f_min, f_max) in eeg_bands.items():
            scale_min = fs / f_max  # Smaller scale for higher frequency components
            scale_max = fs / f_min  # Larger scale for lower frequency components
            band_scales = np.logspace(
                np.log10(scale_min),
                np.log10(scale_max),
                scales_per_band[band_name]
            )
            scales.extend(band_scales)

        scales = np.unique(np.sort(scales))  # Ensure unique and sorted scales

    # Process each channel
    for channel_idx in range(num_channels):
        # Perform continuous wavelet transform on the channel data
        coeffs, _ = pywt.cwt(window_normalized[channel_idx, :], scales, wavelet, sampling_period=sampling_period)
        # Take the absolute value to get the magnitude
        magnitude = np.abs(coeffs)

        # If normalization is enabled, standardize each scale separately
        if normalization:
            for j in range(magnitude.shape[0]):
                mean_scale = np.mean(magnitude[j, :])
                std_scale = np.std(magnitude[j, :]) if np.std(magnitude[j, :]) > 0 else 1e-8
                magnitude[j, :] = (magnitude[j, :] - mean_scale) / std_scale

        decomposed_channels.append(magnitude)
        normalized_data.append(window_normalized[channel_idx, :])

        # If verbose, plot the wavelet coefficients with cognitive band emphasis
        if verbose:
            plt.figure(figsize=(12, 8))

            # Use a perceptually uniform colormap for better feature visibility
            plt.imshow(magnitude, aspect='auto', cmap='plasma',
                       extent=[0, num_samples, 0, len(scales)], origin='lower')

            # Set custom y-ticks to show frequency instead of scale
            freq_ticks = []
            tick_positions = []
            cognitive_functions = {
                "delta": "Sleep, unconscious",
                "theta": "Memory, emotion",
                "alpha": "Attention, awareness",
                "low_beta": "Problem-solving",
                "mid_beta": "Active thinking",
                "high_beta": "Alertness",
                "low_gamma": "Cognitive processing",
                "high_gamma": "Advanced cognition"
            }

            # Add markers for EEG band boundaries
            for band, (f_min, f_max) in eeg_bands.items():
                # Find scales closest to band boundaries
                s_min_idx = np.argmin(np.abs(scales - fs / f_max))
                s_max_idx = np.argmin(np.abs(scales - fs / f_min))

                # Add frequency ticks
                freq_ticks.extend([f"{f_max} Hz", f"{f_min} Hz"])
                tick_positions.extend([s_min_idx, s_max_idx])

                # Add band label with cognitive function
                mid_pos = (s_min_idx + s_max_idx) // 2
                plt.text(num_samples + 10, mid_pos, f"{band}: {cognitive_functions[band]}",
                         verticalalignment='center', fontsize=8)

            plt.yticks(tick_positions, freq_ticks)

            plt.colorbar(label='Coefficient (normalized)' if normalization else 'Coefficient magnitude')
            plt.title(f'Wavelet Decomposition (Ch{channel_idx}) - Enhanced for Cognitive Functions')
            plt.xlabel('Time')
            plt.ylabel('Frequency (Hz)')
            plt.tight_layout()
            plt.show()
    return (decomposed_channels, scales, num_samples, np.array(normalized_data))


def wavelet_decompose_window_(window, wavelet='cmor1.5-1.0', scales=None, normalization=True, sampling_period=1.0,
                             verbose=True):
    """
    Apply continuous wavelet transform (CWT) with Morlet wavelet to each channel in the window.
    Optionally, perform per-scale z-score normalization and plot the result.

    Args:
        window (numpy.ndarray): 2D array (channels x samples).
        wavelet (str): Wavelet type (default: 'cmor1.5-1.0').
        scales (list or None): Scales for CWT. If None, computed based on EEG bands.
        normalization (bool): If True, apply per-scale z-score normalization to coefficients.
        sampling_period (float): Sampling period in seconds.
        verbose (bool): If True, plot the wavelet decomposition result for each channel.

    Returns:
        tuple: (decomposed_channels, scales, original_signal_length, normalized_data)
               - decomposed_channels: List of 2D arrays, each with shape (scales, time_points),
                                      containing the (optionally normalized) coefficient magnitudes.
               - scales: Array of scales used in CWT.
               - original_signal_length: Number of samples in the input signal.
               - normalized_data: Array of the input signals after any pre-normalization (here simply copied).
    """
    num_channels, num_samples = window.shape
    decomposed_channels = []
    normalized_data = []

    # Make a copy of the input window data (if any pre-normalization is needed)
    window_normalized = window.copy()

    # Define scales if not provided, based on typical EEG bands
    if scales is None:
        fs = 1.0 / sampling_period  # For example, if sampling_period = 1/256, then fs = 256 Hz.
        eeg_bands = {"delta": (0.1, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 30), "gamma": (30, 80)}
        scales_per_band = 28  # Number of scales per EEG band
        scales = []
        for f_min, f_max in eeg_bands.values():
            scale_min = fs / f_max  # Smaller scale for higher frequency components
            scale_max = fs / f_min  # Larger scale for lower frequency components
            band_scales = np.logspace(np.log10(scale_min), np.log10(scale_max), scales_per_band)
            scales.extend(band_scales)
        scales = np.unique(np.sort(scales))  # Ensure unique and sorted scales

    # Process each channel
    for channel_idx in range(num_channels):
        # Perform continuous wavelet transform on the channel data
        coeffs, _ = pywt.cwt(window_normalized[channel_idx, :], scales, wavelet, sampling_period=sampling_period)
        # Take the absolute value to get the magnitude (resulting shape: [num_scales x num_samples])
        magnitude = np.abs(coeffs)

        # If normalization is enabled, standardize each scale (row) separately
        if normalization:
            for j in range(magnitude.shape[0]):
                mean_scale = np.mean(magnitude[j, :])
                std_scale = np.std(magnitude[j, :]) if np.std(magnitude[j, :]) > 0 else 1e-8
                magnitude[j, :] = (magnitude[j, :] - mean_scale) / std_scale

        decomposed_channels.append(magnitude)
        normalized_data.append(window_normalized[channel_idx, :])

        # If verbose, plot the wavelet coefficients for the current channel
        if verbose:
            plt.figure(figsize=(10, 6))
            plt.imshow(magnitude, aspect='auto', extent=[0, num_samples, scales[0], scales[-1]], origin='lower')
            plt.colorbar(label='Coefficient (normalized)' if normalization else 'Coefficient magnitude')
            plt.title(f'Wavelet Decomposition (Channel {channel_idx})')
            plt.xlabel('Time')
            plt.ylabel('Scale')
            plt.show()

    return (decomposed_channels, scales, num_samples, np.array(normalized_data))


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


def quantize_number(z_value, resolution=80):
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


def dequantize_number(token, resolution=80):
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
If channel names are not conventional (eg E1, E2 , A28, B24, etc) , you need to skip databse with     "action": "skip" 

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


def filter_band_pass_windows(ndarray, sps,plot=False):
    """
    Apply a bandpass filter to the data and plot the PSD before and after filtering.
    :param ndarray: 2D numpy array (channels x samples).
    :param sps: Sampling rate.
    :return: 2D numpy array with bandpass filtered data.
    """
    # Compute PSD before filtering for the first channel
    freqs_before, psd_before = signal.welch(ndarray[0, :], fs=sps, nperseg=1024)

    # Existing filter implementation
    f_b, f_a = signal.butter(N=30, Wn=( sps / 4), btype='low', fs=sps)
    filtered_data = signal.filtfilt(f_b, f_a, ndarray, axis=1)
    # Define notch filter parameters
    quality_factor = 30  # Adjust this Q-factor for a "strong" (narrow) notch
    # notch_freqs = [50, 60]  # Frequencies to notch out (in Hz)
    # notch_freqs = [50]  #for bci val
    # # Apply each notch filter in series
    # for f0 in notch_freqs:
    #     b_notch, a_notch = signal.iirnotch(w0=f0, Q=quality_factor, fs=sps)
    #     filtered_data = signal.filtfilt(b_notch, a_notch, filtered_data, axis=1)

    if plot:
        # Compute PSD after filtering for the first channel
        freqs_after, psd_after = signal.welch(filtered_data[0, :], fs=sps, nperseg=1024)

        # Create inline plot
        plt.figure()
        plt.semilogy(freqs_before, psd_before, label='Before Filtering')
        plt.semilogy(freqs_after, psd_after, label='After Filtering')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density (dB/Hz)')
        plt.title('PSD Before and After Filtering (First Channel)')
        plt.legend()
        plt.grid(True)
        plt.show()

    return filtered_data

def preprocess_data(data, original_sps):
    """
    Preprocess the data by first applying a bandpass filter at the original sampling rate
    and then resampling to a new sampling rate.

    :param data: 2D numpy array (channels x samples).
    :param original_sps: Original sampling rate.
    :return: Preprocessed 2D numpy array (channels x new_samples), and the new sampling rate.
    """
    # First, filter the data with the original sampling rate
    filtered_data = filter_band_pass_windows(data, original_sps)

    # Then, resample the filtered data to the new rate (e.g., 100 Hz)
    new_rate = 256
    resampled_data = resample_windows(filtered_data, original_sps, new_rate=new_rate)

    return resampled_data, new_rate


# --------------------------------------------------------------------------------
# Calculate sampling rate from CSV
# --------------------------------------------------------------------------------
def calculate_sps(csv_file_path):
    # Read the CSV
    df = pd.read_csv(csv_file_path)

    # Extract the 'timestamp' column into a NumPy array
    try:
        timestamps = df['timestamp'].values
    except:
        timestamps = df['TimeStamp'].values


    # Option A: Compute the mean difference between consecutive timestamps
    avg_dt = np.mean(np.diff(timestamps))  # average time-step (seconds)
    sps = 1.0 / avg_dt

    return sps



# ------------------------------------------------------------------------------
# NEW BLOCK:
# Validation Function
# ------------------------------------------------------------------------------

def list_csv_files_in_folder(folder_name , bucket_name='dataframes--use1-az6--x-s3', ):
    """
    List all CSV files in a specific folder within an S3 bucket.

    :param bucket_name: Name of the S3 bucket
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