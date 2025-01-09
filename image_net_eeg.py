import numpy as np
import torch
import pandas as pd
import os
from utils import preprocess_data
import matplotlib.pyplot as plt
from utils import preprocess_data, wavelet_decompose_window, quantize_number

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
wvlet = 'db2'  # example wavelet; change as needed
level = 2     # decomposition level; change as needed

def load_pth_file(file_path):
    """
    Loads the .pth file and returns the Python object (usually a dict).
    """
    try:
        data = torch.load(file_path, map_location='cpu')
        print(f"Successfully loaded '{file_path}'")
        return data
    except Exception as e:
        print(f"Error loading '{file_path}': {e}")
        return None


def get_unique_groups(dataset):
    """
    Returns a list of unique (subject, image, granularity) tuples found in 'dataset'.
    """
    unique_keys = set()
    for entry in dataset:
        subj = entry.get('subject')
        img = entry.get('label')
        gran = entry.get('granularity')
        eeg = entry.get('eeg_data')
        if subj is not None and img is not None and gran is not None and eeg is not None:
            unique_keys.add((subj, img, gran))
    return list(unique_keys)


def build_df_for_group(eeg_list, subject, image, granularity):
    """
    Given:
    - eeg_list: List of Tensors, each (62 x 501)
    - subject: int
    - image: str
    - granularity: str

    1) For each Tensor: transpose to (501 x 62), so rows = time-samples, columns = channels.
    2) Concatenate all Tensors row-wise => final shape = (501*n) x 62.
    3) Create a DataFrame, add metadata columns, return it.
    """
    if not eeg_list:
        return None

    # Stack all trials into one tensor of shape (n, 62, 501)
    stacked = torch.stack(eeg_list)  # (n, 62, 501)
    n_trials = stacked.shape[0]

    # Permute to (n, 501, 62) so that dimension 1 = time, dimension 2 = channels
    stacked = stacked.permute(0, 2, 1)  # shape: (n, 501, 62)

    # Reshape => (n * 501, 62)
    # i.e., all time-samples from all trials stacked row-wise
    stacked_2d = stacked.reshape(-1, 62)  # (501*n, 62)

    # Convert to numpy for DataFrame
    eeg_array = stacked_2d.numpy()

    # Channel column names
    channel_cols = [f"Channel_{i + 1}" for i in range(62)]

    # Build DataFrame: rows = time-samples, columns = channels
    df = pd.DataFrame(eeg_array, columns=channel_cols)

    # Insert metadata columns at the front, repeating for each row
    df.insert(0, "Subject", subject)
    df.insert(1, "Image", image)
    df.insert(2, "Granularity", granularity)

    return df


def main():
    file_path = "dataset/EEG-ImageNet_1.pth"  # <-- Change this
    data = load_pth_file(file_path)
    if data is None:
        print("No data loaded.")
        return

    dataset = data.get('dataset')
    if not dataset:
        print("dataset is missing or empty in the loaded file.")
        return

    # 1) Identify unique (subject, image, granularity) combos
    groups = get_unique_groups(dataset)
    print(f"Found {len(groups)} unique groups in dataset.")

    # 2) Prepare an output folder
    output_dir = "validation_datasets_imageNet"
    os.makedirs(output_dir, exist_ok=True)

    # 3) Process each group in series to avoid big memory usage
    for idx, (subj, img, gran) in enumerate(groups, start=1):
        # Collect all trials for this group
        eeg_list = []
        for entry in dataset:
            if (
                    entry.get('subject') == subj
                    and entry.get('label') == img
                    and entry.get('granularity') == gran
                    and entry.get('eeg_data') is not None
            ):
                eeg_list.append(entry['eeg_data'])

        if not eeg_list:
            continue

        # 4) Build the DataFrame
        df = build_df_for_group(eeg_list, subj, img, gran)
        if df is None:
            continue
        df.drop(df.columns[:3], axis=1, inplace=True)
        prep,new_sps = preprocess_data(np.array(df).transpose(), 1000)
        twoch = average_alternate_channels(prep)
        # plot_eeg_channels(pd.DataFrame(twoch.transpose()), fs=new_sps, title="EEG")
        # 5) Save to CSV
        # Clean up the image string to avoid weird characters in filename
        img_basename = os.path.splitext(os.path.basename(img))[0]
        out_name = f"subject_{subj}_image_{img_basename}_gran_{gran}"
        coeffs_path = f"validation_datasets_imageNet/{out_name}_coeffs.txt"
        chans_path= f"validation_datasets_imageNet/{out_name}_channels.txt"
        # Process and save Right Hand data
        process_and_save(
            data=twoch,
            sps=new_sps,
            coeffs_path=coeffs_path,
            chans_path=chans_path,
            wavelet=wvlet,
            level=level,
            window_len=2
        )

        # Free memory
        del df
        del eeg_list

    print("Done! All groups processed and saved.")


if __name__ == "__main__":
    main()
