import numpy as np
import pandas as pd
import scipy.io
import pywt
import matplotlib.pyplot as plt
import os
from utils import preprocess_data, wavelet_decompose_window, quantize_number


def average_alternate_channels(data):
    """
    data shape: (channels, samples).
    Return a 2-channel array, where:
      ch0 = average of even-indexed channels (0,2,4,...)
      ch1 = average of odd-indexed channels (1,3,5,...)
    """
    even_channels = data[0::2, :]
    odd_channels = data[1::2, :]
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


def load_mat_data(mat_file_path):
    """
    Loads the .mat file containing:
       - 'data': shape (n_epochs, epoch_length, n_channels)
       - 'label': shape (1, n_epochs) or (n_epochs,)
         which indicates the class of each epoch.

    Returns:
      data (np.ndarray): shape (n_epochs, epoch_length, n_channels)
      labels (np.ndarray): shape (n_epochs,)
      fs: int, sample rate
    """
    mat_data = scipy.io.loadmat(mat_file_path)

    data = mat_data['data']  # shape: (n_epochs, epoch_length, n_channels)
    labels = mat_data['label']  # shape: (1, n_epochs) or (n_epochs,)
    labels = np.squeeze(labels)  # make it (n_epochs,)

    # According to your description, each epoch is 1 second at 500 Hz
    fs = 500

    return data, labels, fs


def process_and_save(data, sps, coeffs_path, chans_path,
                     wavelet='db2', level=2, window_len_sec=1.8):
    """
    data shape: (2, total_samples)
    We break it into non-overlapping windows of length window_len_sec * sps,
    wavelet-decompose, quantize, and save to text files.

    Args:
      data: shape (2, total_samples)
      sps: sampling rate
      coeffs_path, chans_path: output text file paths
      wavelet, level: wavelet parameters
      window_len_sec: window length in seconds (e.g., 1.8 means 900 samples at 500 Hz)
    """
    n_window_samples = int(window_len_sec * sps)  # 1.8 * 500 = 900
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


if __name__ == "__main__":
    # Load your data/labels
    mat_file_path = 'dataset/Subject2_attention_4.mat'

    data, labels, fs = load_mat_data(mat_file_path)
    # data: shape (n_epochs, epoch_length=500, n_channels=32)
    # labels: shape (n_epochs,)

    unique_labels = np.unique(labels)
    print(f"Unique labels found: {unique_labels}")

    # Wavelet parameters
    wavelet_name = 'db2'
    level = 4
    window_len_sec = 1.8  # <--- your desired 1.8-second window

    for label_id in unique_labels:
        print(f"\nProcessing label: {label_id}")
        idx_label = np.where(labels == label_id)[0]
        if idx_label.size == 0:
            print(f"No epochs for label {label_id}. Skipping.")
            continue

        # data_label => shape (num_epochs_for_label, 500, 32)
        data_label = data[idx_label, :, :]

        # Concatenate these epochs along the time axis (after transposing to (epochs, channels, samples)):
        # final shape will be (32, num_epochs_for_label * 500)
        combined_data = data_label.transpose(0, 2, 1).reshape(32, -1)

        # Preprocess the combined data
        prep_data, new_sps = preprocess_data(combined_data, fs)

        # Reduce from 32 channels to 2 channels
        twoch_data = average_alternate_channels(prep_data)

        # (Optional) plot
        # df_plot = pd.DataFrame(twoch_data.T, columns=["EvenAvg", "OddAvg"])
        # plot_eeg_channels(df_plot, fs=new_sps, title=f"Label: {label_id}")

        # Save wavelet decomposition results with your 1.8-second window
        coeffs_path = f"output/label_{label_id}_coeffs.txt"
        chans_path = f"output/label_{label_id}_channels.txt"
        process_and_save(
            data=twoch_data,
            sps=new_sps,
            coeffs_path=coeffs_path,
            chans_path=chans_path,
            wavelet=wavelet_name,
            level=level,
            window_len_sec=window_len_sec
        )

    print("\nDone!")
