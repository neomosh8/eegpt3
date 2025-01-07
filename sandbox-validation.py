import numpy as np
import pandas as pd
import scipy.io

from utils import preprocess_data, wavelet_decompose_window, quantize_number


def load_structured_data(mat_file_path):
    """
    Loads a .mat file assumed to have a variable named 'data' in the structure
    described in your question. Returns a Python list of dictionaries (structured_data).
    Each dict has keys: X, trial, y, fs, classes.
    """
    mat_data = scipy.io.loadmat(mat_file_path)
    raw_data = mat_data['data']  # shape (1, N) - each column is one block/run

    structured_data = []
    n_items = raw_data.shape[1]

    for i in range(n_items):
        entry = raw_data[0, i]  # shape (1,) with fields
        # Extract fields from the structured array
        X = entry['X'][0, 0]  # EEG signals (samples x channels)
        trial = entry['trial'][0, 0]  # array of trial start indices
        y = entry['y'][0, 0]  # array of labels (1 or 2)
        fs = entry['fs'][0, 0]  # sampling frequency
        classes = entry['classes'][0, 0]  # class names

        structured_data.append({
            'X': X,
            'trial': trial,
            'y': y,
            'fs': fs,
            'classes': classes
        })
    return structured_data


def extract_task_data(structured_data, task_label=1, channel_names=None):
    """
    Extracts EEG samples from all blocks (runs) for a given task (label).

    Args:
        structured_data (list): The list of dictionaries returned by load_structured_data().
        task_label (int): The label identifying the task of interest
                          (e.g., 1 for 'right hand', 2 for 'feet').
        channel_names (list): List of channel names. If None, defaults to generic names.

    Returns:
        Pandas DataFrame of the concatenated samples for the specified task,
        with each channel as a column.
    """
    if channel_names is None:
        # If you know the exact electrode names, replace these with real ones.
        # Example: ['C3', 'C3a', 'C3b', 'Cz', ... ] up to 15 channels
        channel_count = structured_data[0]['X'].shape[1]
        channel_names = [f"Ch{i + 1}" for i in range(channel_count)]

    all_task_segments = []

    for block in structured_data:
        X = block['X']  # shape (num_samples, num_channels)
        trials = block['trial'][0]  # shape (20,) => 20 trials
        labels = block['y'][0]  # shape (20,) => each is 1 or 2
        # fs = block['fs'][0, 0]       # sampling frequency (512), if needed
        # classes = block['classes']   # 'right hand', 'feet', if needed

        for i in range(len(trials)):
            # Check if this trial matches the requested task label
            if labels[i] == task_label:
                start_idx = trials[i]
                # If it's not the last trial, go until the next trial start_idx
                if i < len(trials) - 1:
                    end_idx = trials[i + 1]
                else:
                    end_idx = X.shape[0]  # or end of the recording

                # Extract the EEG samples for this trial
                segment = X[start_idx:end_idx, :]  # shape (samples_in_trial, channels)

                # Convert to DataFrame
                df_segment = pd.DataFrame(segment, columns=channel_names)
                # Append to the list
                all_task_segments.append(df_segment)

    # Concatenate all segments for this task from all blocks
    if all_task_segments:
        task_data = pd.concat(all_task_segments, ignore_index=True)
    else:
        # If, for some reason, no trials matched the label (unlikely), return empty
        task_data = pd.DataFrame(columns=channel_names)

    return task_data


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_eeg_channels(df, fs=512, title="EEG Channels"):
    """
    Plots all channels in a given DataFrame,
    each channel in its own subplot (stacked vertically).

    Args:
        df (pd.DataFrame): DataFrame with shape (n_samples, n_channels).
                           Columns are channel names, rows are samples.
        fs (int): Sampling frequency (default 512).
        title (str): Figure title.
    """
    # Create a time axis in seconds based on the number of samples and the sampling frequency
    n_samples = len(df)
    time_axis = np.arange(n_samples) / fs

    # Number of channels
    n_channels = df.shape[1]

    # Create subplots: one row per channel, sharing the same x-axis
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 2.5 * n_channels), sharex=True)
    # Handle the case where there's only one channel (axes won't be an array)
    if n_channels == 1:
        axes = [axes]

    # Plot each channel in its respective subplot
    for i, col_name in enumerate(df.columns):
        axes[i].plot(time_axis, df[col_name], label=col_name)
        axes[i].set_ylabel(col_name)
        axes[i].legend(loc='upper right')

    # Set x-label for the last subplot and add a global title
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def average_alternate_channels(data):
    # Assuming data is ch x sample array
    # Get odd and even indexed channels
    odd_channels = data[1::2]  # Start from index 1, step by 2
    even_channels = data[0::2]  # Start from index 0, step by 2

    # Average along the channel dimension (axis 0)
    odd_avg = np.mean(odd_channels, axis=0)
    even_avg = np.mean(even_channels, axis=0)


    # Stack the averages to get 2 x sample array
    return np.stack([even_avg, odd_avg])


import numpy as np

# Assuming all necessary functions are already imported:
# load_structured_data, extract_task_data, preprocess_data, average_alternate_channels,
# wavelet_decompose_window, quantize_number

# Parameters for processing
window_len = 1  # window length in seconds (adjust as needed)
# Assuming 'new_sps' is the sampling rate returned from preprocess_data
# If 'new_sps' is not defined yet, ensure it's set appropriately
n_window_samples = window_len * 512  # using the original sampling rate; adjust if 'new_sps' is different

# Wavelet parameters
wvlet = 'db2'  # example wavelet; change as needed
level = 2     # decomposition level; change as needed

# File paths for saving coefficients and channel information
# Right Hand
coeffs_right_hand_path = "validation_datasets/right_hand_coeffs_1.txt"
chans_right_hand_path = "validation_datasets/right_hand_channels_1.txt"

# Feet
coeffs_feet_path = "validation_datasets/feet_coeffs_1.txt"
chans_feet_path = "validation_datasets/feet_channels_1.txt"

# Function to process and save data
def process_and_save(data, sps, coeffs_path, chans_path, wavelet, level, window_len):
    n_window_samples = window_len * sps
    total_samples = data.shape[1]

    with open(coeffs_path, 'w') as f_coeffs, open(chans_path, 'w') as f_chans:
        # Slide through the data in non-overlapping windows
        for window_start in range(0, total_samples - n_window_samples + 1, n_window_samples):
            window_end = window_start + n_window_samples

            all_channel_coeffs = []
            all_channel_names = []

            # For exactly 2 channels: 0 => first channel, 1 => second channel
            for ch_idx in range(2):
                ch_name_id = "0" if ch_idx == 0 else "1"  # 0 for first channel, 1 for second
                channel_data = data[ch_idx, window_start:window_end]
                channel_data_2d = channel_data[np.newaxis, :]  # Reshape for processing

                # Wavelet Decompose
                (decomposed_channels,
                 coeffs_lengths,
                 num_samples,
                 normalized_data) = wavelet_decompose_window(
                    channel_data_2d,
                    wavelet=wavelet,
                    level=level,
                    normalization=True
                )

                # Flatten for quantization
                coeffs_flat = decomposed_channels.flatten()
                q_ids = [str(quantize_number(c)) for c in coeffs_flat]  # Convert to strings

                all_channel_coeffs.extend(q_ids)
                all_channel_names.extend([ch_name_id] * len(q_ids))

            # Write lines for this window
            coeffs_line = " ".join(all_channel_coeffs) + " "
            chans_line = " ".join(all_channel_names) + " "

            f_coeffs.write(coeffs_line)
            f_chans.write(chans_line)


if __name__ == "__main__":
    # 1) Load the structured data from your .mat
    mat_file_path = "S02T.mat"
    structured_data = load_structured_data(mat_file_path)

    # 2) Extract all samples corresponding to "right hand" trials.
    #    By your description, label=1 means "right hand."
    right_hand_df = extract_task_data(structured_data, task_label=1)
    prep,new_sps = preprocess_data(np.array(right_hand_df).transpose(),512)
    twoch = average_alternate_channels(prep)
    # plot_eeg_channels(pd.DataFrame(twoch.transpose()), fs=new_sps, title="Right Hand EEG")
    # 3) Do the same for "feet" if you want:
    feet_df = extract_task_data(structured_data, task_label=2)
    prep_f,new_sps = preprocess_data(np.array(feet_df).transpose(),512)
    twoch_f = average_alternate_channels(prep_f)

    # Process and save Right Hand data
    process_and_save(
        data=twoch,
        sps=new_sps,
        coeffs_path=coeffs_right_hand_path,
        chans_path=chans_right_hand_path,
        wavelet=wvlet,
        level=level,
        window_len=window_len
    )

    # Process and save Feet data
    process_and_save(
        data=twoch_f,
        sps=new_sps,
        coeffs_path=coeffs_feet_path,
        chans_path=chans_feet_path,
        wavelet=wvlet,
        level=level,
        window_len=window_len
    )

    # # 5) (Optional) Save the DataFrame(s) to CSV or pickle
    # right_hand_df.to_csv("right_hand_data.csv", index=False)
    # feet_df.to_csv("feet_data.csv", index=False)
