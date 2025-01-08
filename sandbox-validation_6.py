import numpy as np
import pandas as pd
import scipy.io
import pywt
import matplotlib.pyplot as plt
import os
from utils import preprocess_data,wavelet_decompose_window,quantize_number

def average_alternate_channels(data):
    """
    data shape: (channels, samples).
    Return a 2-channel array, where:
      ch0 = average of even-indexed channels (0,2,4,...)
      ch1 = average of odd-indexed channels (1,3,5,...)
    """
    even_channels = data[0::2, :]  # 0-based indexing
    odd_channels = data[1::2, :]  # 1-based in 0-based indexing
    even_avg = np.mean(even_channels, axis=0)
    odd_avg = np.mean(odd_channels, axis=0)
    return np.stack([even_avg, odd_avg])


def plot_eeg_channels(df, fs=512, title="EEG Channels"):
    """
    Plots each column in df as a separate subplot.
    df shape: (samples, channels).
    """
    n_samples = len(df)
    time_axis = np.arange(n_samples) / fs
    n_channels = df.shape[1]

    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 2.5 * n_channels), sharex=True)
    if n_channels == 1:
        axes = [axes]

    for i, col_name in enumerate(df.columns):
        axes[i].plot(time_axis, df[col_name], label=col_name)
        axes[i].set_ylabel(col_name)
        axes[i].legend(loc='upper right')

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


###############################################################################
# 2) Loading data & extracting segments using event codes
###############################################################################

def load_mat_data(mat_file_path):
    """
    Loads the .mat file containing:
       - signal  ( shape: (n_channels, n_samples) )
       - events  ( struct with fields 'codes' and 'positions' )
       - header  ( struct with at least 'channels_eeg', 'channels_eog', etc. )

    Returns:
      data_dict with:
        'fs': int, sample rate
        'EEG': (n_eeg_channels, n_samples)
        'events_codes': (1, num_events)
        'events_positions': (1, num_events)
    """
    mat_data = scipy.io.loadmat(mat_file_path, squeeze_me=True, struct_as_record=False)

    signal = mat_data['signal']  # shape: (n_channels, n_samples)
    events = mat_data['events']  # has fields 'codes', 'positions'
    header = mat_data['header']  # has fields 'channels_eeg', 'channels_eog', etc.

    # Only keep EEG channels (exclude EOG)
    eeg_ch_idx = header.channels_eeg  # e.g. [0..57] or [1..58], watch out for Python vs MATLAB indexing
    # In MATLAB, channels might be 1-based. If your data is 0-based in Python, you might need to subtract 1.
    # But from your example, it looks like they might already be 1-based in the .mat
    # so we correct it for Python 0-based indexing:
    if np.min(eeg_ch_idx) == 1:
        eeg_ch_idx = eeg_ch_idx - 1  # shift to 0-based

    eeg_signal = signal[eeg_ch_idx, :]  # shape: (n_eeg_channels, n_samples)

    fs = header.sample_rate

    events_codes = events.codes  # shape: (num_events,)
    events_positions = events.positions  # shape: (num_events,)

    data_dict = {
        'fs': fs,
        'EEG': eeg_signal,
        'events_codes': events_codes,
        'events_positions': events_positions
    }
    return data_dict


def extract_segments_by_event(eeg_data, events_codes, events_positions,
                              onset_code, offset_code):
    """
    Extracts segments from 'onset_code' to 'offset_code'.
    For each pair of matching onset->offset in time order, we slice the EEG.

    Args:
      eeg_data: (channels, samples)
      events_codes: array of shape (num_events,)
      events_positions: array of shape (num_events,)
      onset_code: int, e.g. 503587 for "palmar movement onset"
      offset_code: int, e.g. 534562 for "palmar grasp offset"

    Returns:
      A single pd.DataFrame (samples x channels) with all extracted segments
      concatenated, or empty DataFrame if no pairs found.
    """
    # Find indices of 'onset_code' events
    onset_idx = np.where(events_codes == onset_code)[0]
    # Find indices of 'offset_code' events
    offset_idx = np.where(events_codes == offset_code)[0]

    # We'll match them in chronological order, but make sure
    # each onset is followed by an offset that occurs later in time.
    segments = []
    j = 0  # pointer for offset events

    for i in range(len(onset_idx)):
        onset_event_i = onset_idx[i]
        onset_pos = events_positions[onset_event_i]  # sample index

        # Move j forward until we find an offset that is after onset
        while j < len(offset_idx) and (events_positions[offset_idx[j]] <= onset_pos):
            j += 1
        if j >= len(offset_idx):
            break  # no more matching offsets

        # Now offset_idx[j] is presumably the first offset after onset_pos
        offset_event_j = offset_idx[j]
        offset_pos = events_positions[offset_event_j]

        if offset_pos > onset_pos:
            # Valid pair => slice
            segment = eeg_data[:, onset_pos:offset_pos]
            # Convert to DataFrame (samples x channels)
            # Name columns "Ch1", "Ch2", etc.
            channel_count = segment.shape[0]
            ch_names = [f"Ch{i + 1}" for i in range(channel_count)]
            df_segment = pd.DataFrame(segment.T, columns=ch_names)
            segments.append(df_segment)

        # Increment j for next round
        j += 1

    # Concatenate all segments
    if len(segments) > 0:
        all_data = pd.concat(segments, ignore_index=True)
    else:
        all_data = pd.DataFrame()
    return all_data


###############################################################################
# 3) The main processing pipeline
###############################################################################

def process_and_save(data, sps, coeffs_path, chans_path,
                     wavelet='db2', level=2, window_len_sec=1.0):
    """
    data shape: (2, total_samples)
    We break it into non-overlapping windows of length window_len_sec * sps,
    wavelet-decompose, quantize, and save to text files.
    """
    n_window_samples = int(window_len_sec * sps)
    total_samples = data.shape[1]

    # Ensure output directories exist
    os.makedirs(os.path.dirname(coeffs_path), exist_ok=True)

    with open(coeffs_path, 'w') as f_coeffs, open(chans_path, 'w') as f_chans:
        # Slide in non-overlapping windows
        for start_idx in range(0, total_samples - n_window_samples + 1, n_window_samples):
            end_idx = start_idx + n_window_samples

            all_channel_coeffs = []
            all_channel_names = []

            for ch_idx in range(2):
                ch_name_id = str(ch_idx)
                channel_data = data[ch_idx, start_idx:end_idx]
                channel_data_2d = channel_data[np.newaxis, :]

                (decomposed_channels,
                 coeffs_lengths,
                 num_samples,
                 normalized_data) = wavelet_decompose_window(
                    channel_data_2d,
                    wavelet=wavelet,
                    level=level,
                    normalization=True
                )

                coeffs_flat = decomposed_channels.flatten()
                q_ids = [str(quantize_number(c)) for c in coeffs_flat]

                all_channel_coeffs.extend(q_ids)
                all_channel_names.extend([ch_name_id] * len(q_ids))

            coeffs_line = " ".join(all_channel_coeffs) + " "
            chans_line = " ".join(all_channel_names) + " "

            f_coeffs.write(coeffs_line)
            f_chans.write(chans_line)


###############################################################################
# 4) Example usage: segment by event codes instead of runs
###############################################################################
if __name__ == "__main__":
    # ---------------------------------------------------------------------------
    # 1) Load data from .mat
    # ---------------------------------------------------------------------------
    mat_file_path = "G01.mat"  # Adjust filename
    data_dict = load_mat_data(mat_file_path)
    eeg_data = data_dict['EEG']
    fs = data_dict['fs']
    event_codes = data_dict['events_codes']
    event_positions = data_dict['events_positions']

    # ---------------------------------------------------------------------------
    # 2) Define which tasks (onset->offset) we want to extract
    #    Example: Palmar grasp from code 503587 (movement onset) to 534562 (grasp offset)
    #             Lateral grasp from code 503588 to 534563
    #             Rest from 768 to 769
    #    Adjust codes to your needs.
    # ---------------------------------------------------------------------------
    # Mapping of "task name" -> (onset_code, offset_code)
    task_code_pairs = {
        "palmar": (503587, 534562),
        "lateral": (503588, 534563),
        "rest": (768, 769),
        # Possibly define eye tasks, etc.:
        # "eye_vertical": (10, 11),
        # "eye_horizontal": (12, 13),
        # "blinking": (14, 15),
    }

    # ---------------------------------------------------------------------------
    # 3) For each task, extract segments, preprocess, average channels, etc.
    # ---------------------------------------------------------------------------
    for task_name, (onset_c, offset_c) in task_code_pairs.items():
        print(f"Processing task: {task_name}")

        # Extract all segments for this task
        task_df = extract_segments_by_event(
            eeg_data, event_codes, event_positions,
            onset_code=onset_c,
            offset_code=offset_c
        )
        if task_df.empty:
            print(f"No segments found for {task_name}. Skipping.")
            continue

        # Convert to numpy (channels x samples)
        data_np = task_df.to_numpy().T

        # Preprocess
        prep_data, new_sps = preprocess_data(data_np, fs)

        # Optional: reduce to 2 channels by averaging
        twoch_data = average_alternate_channels(prep_data)

        # Optional: plot
        # df_plot = pd.DataFrame(twoch_data.T, columns=["EvenAvg", "OddAvg"])
        # plot_eeg_channels(df_plot, fs=new_sps, title=f"Task: {task_name}")

        # -----------------------------------------------------------------------
        # 4) Save wavelet decomposition results
        # -----------------------------------------------------------------------
        wavelet_name = 'db2'
        level = 2
        window_len_sec = 1.0  # e.g. 1-second windows

        # Example output paths
        coeffs_path = f"output/{task_name}_coeffs.txt"
        chans_path = f"output/{task_name}_channels.txt"

        process_and_save(
            data=twoch_data,
            sps=new_sps,
            coeffs_path=coeffs_path,
            chans_path=chans_path,
            wavelet=wavelet_name,
            level=level,
            window_len_sec=window_len_sec
        )

    print("Done!")
