
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
    Returns a 2-channel array, where:
      ch0 = average of even-indexed channels (0,2,4,...)
      ch1 = average of odd-indexed channels (1,3,5,...)
    """
    even_channels = data[0::2, :]  # even-indexed channels
    odd_channels = data[1::2, :]  # odd-indexed channels
    even_avg = np.mean(even_channels, axis=0)
    odd_avg = np.mean(odd_channels, axis=0)
    return np.stack([even_avg, odd_avg])


def plot_eeg_channels(df, fs=500, title="EEG Channels"):
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

def plot_window(window_data, sps, window_index=None):
    """
    Plots one window of two-channel EEG data.

    Args:
        window_data: NumPy array of shape (2, n_samples) for the window.
        sps: Sampling rate in Hz.
        window_index: Optional window number (used in title and filename).
    """
    n_samples = window_data.shape[1]
    time_axis = np.arange(n_samples) / sps
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, window_data[0, :], label="Left Hemisphere")
    plt.plot(time_axis, window_data[1, :], label="Right Hemisphere")
    if window_index is not None:
        plt.title(f"Window {window_index}")
    else:
        plt.title("EEG Window")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"window_{window_index}.png")
    plt.close()

def process_and_save(data, sps, coeffs_path, chans_path,
                     wavelet='db2', level=4, window_len_sec=1.8,
                     plot_windows=False, plot_random_n=1):
    """
    Segments a two-channel EEG signal into non-overlapping windows,
    performs wavelet decomposition and quantization on each window,
    and writes the resulting quantized coefficients and channel labels to text files.

    Args:
        data: NumPy array of shape (2, total_samples)
        sps: Sampling rate in Hz.
        coeffs_path: Output file path for quantized coefficients.
        chans_path: Output file path for channel labels.
        wavelet: Wavelet type (default 'db2').
        level: Decomposition level (default 4).
        window_len_sec: Window length in seconds (e.g., 1.8 sec).
        plot_windows: If True, plots windows (or a random subset).
        plot_random_n: If an integer (and less than total windows), randomly selects that many windows to plot.
    """
    n_window_samples = int(window_len_sec * sps)
    total_samples = data.shape[1]
    total_windows = len(range(0, total_samples - n_window_samples + 1, n_window_samples))

    if plot_random_n is not None and plot_random_n < total_windows:
        selected_windows = np.random.choice(range(1, total_windows + 1), size=plot_random_n, replace=False)
    else:
        selected_windows = None

    os.makedirs(os.path.dirname(coeffs_path), exist_ok=True)
    window_counter = 0
    with open(coeffs_path, 'w') as f_coeffs, open(chans_path, 'w') as f_chans:
        for start_idx in range(0, total_samples - n_window_samples + 1, n_window_samples):
            window_counter += 1
            end_idx = start_idx + n_window_samples
            window_data = data[:, start_idx:end_idx]

            if selected_windows is not None:
                if window_counter in selected_windows:
                    plot_window(window_data, sps, window_index=window_counter)
            elif plot_windows:
                plot_window(window_data, sps, window_index=window_counter)

            (decomposed_channels,
             coeffs_lengths,
             num_samples,
             normalized_data) = wavelet_decompose_window(
                window_data,
                wavelet=wavelet,
                level=level,
                normalization=True
            )

            all_channel_coeffs = []
            all_channel_names = []
            for ch_idx in range(decomposed_channels.shape[0]):
                ch_name = str(ch_idx)
                coeffs_flat = decomposed_channels[ch_idx].flatten()
                q_ids = [str(quantize_number(c)) for c in coeffs_flat]
                all_channel_coeffs.extend(q_ids)
                all_channel_names.extend([ch_name] * len(q_ids))

            f_coeffs.write(" ".join(all_channel_coeffs) + " ")
            f_chans.write(" ".join(all_channel_names) + " ")


def load_mat_epoch_data(mat_file_path):
    """
    Loads a .mat file containing:
      - 'data': ndarray of shape (n_epochs, n_samples, n_channels)
      - 'label': ndarray of shape (1, n_epochs) or (n_epochs,)

    Returns:
      epochs: ndarray of shape (n_epochs, n_samples, n_channels)
      labels: ndarray of shape (n_epochs,)
    """
    mat_data = scipy.io.loadmat(mat_file_path, squeeze_me=True)
    data = mat_data['data']  # e.g., shape: (653, 500, 32)
    label = mat_data['label']  # e.g., shape: (1, 653) or (653,)
    if label.ndim > 1:
        label = label.flatten()
    return data, label


###############################################################################
#  Main processing pipeline for epoch data grouped by label
###############################################################################

if __name__ == "__main__":
    # ---------------------------
    # 1) Load epoch data and labels
    # ---------------------------
    mat_file_path = "dataset/Subject2_dom.mat"  # Change to your file name/path
    epochs, labels = load_mat_epoch_data(mat_file_path)
    fs = 500  # Sample rate (given that each epoch has 500 samples corresponding to 1 second)

    # ---------------------------
    # 2) Process epochs grouped by label
    # ---------------------------
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        print(f"Processing class (label): {lab}")

        # Select all epochs belonging to the current label.
        # epochs has shape (n_epochs, n_samples, n_channels)
        epochs_class = epochs[labels == lab]

        processed_epochs = []
        for epoch in epochs_class:
            # Each epoch is of shape (n_samples, n_channels) i.e. (500, 32).
            # Transpose so that the shape becomes (n_channels, n_samples).
            epoch_T = epoch.T

            # Preprocess (for example: filtering, detrending, etc.)
            prep_data, new_sps = preprocess_data(epoch_T, fs)

            # Average alternate channels to reduce the 32-channel data to 2 channels.
            twoch_data = average_alternate_channels(prep_data)
            # # (Optional) plot
            # if lab==2:
            #     df_plot = pd.DataFrame(twoch_data.T, columns=["EvenAvg", "OddAvg"])
            #     plot_eeg_channels(df_plot, fs=new_sps, title=f"Label: {lab}")

            processed_epochs.append(twoch_data)

        if len(processed_epochs) == 0:
            print(f"No epochs found for label {lab}.")
            continue

        # Concatenate the processed epochs along the time axis.
        # Each processed epoch has shape (2, n_samples) so the combined_data has shape (2, total_samples)
        combined_data = np.concatenate(processed_epochs, axis=1)

        # ---------------------------
        # 3) Wavelet Decomposition & Saving
        # ---------------------------
        # Since each epoch is 1 second, we use a window length of 1 second.
        wavelet_name = 'db2'
        level = 4
        window_len_sec = 1.18

        # Define output paths (one set of files per class label)
        coeffs_path = f"output_MEMA/class_{lab}_coeffs.txt"
        chans_path = f"output_MEMA/class_{lab}_channels.txt"

        process_and_save(
            data=combined_data,
            sps=new_sps,
            coeffs_path=coeffs_path,
            chans_path=chans_path,
            wavelet=wavelet_name,
            level=level,
            window_len_sec=window_len_sec
        )

    print("Done!")
