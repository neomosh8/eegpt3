#https://zenodo.org/records/4518754
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import matplotlib.pyplot as plt
import mne  # MNE-Python
import pywt
from utils import preprocess_data, wavelet_decompose_window, quantize_number


def average_alternate_channels(data):
    """
    Averages alternate channels.

    Args:
        data: NumPy array of shape (channels, samples)

    Returns:
        A 2-channel NumPy array where:
          - Channel 0 is the average of even-indexed channels (0, 2, 4, ...)
          - Channel 1 is the average of odd-indexed channels (1, 3, 5, ...)
    """
    even_channels = data[0::2, :]
    odd_channels = data[1::2, :]
    even_avg = np.mean(even_channels, axis=0)
    odd_avg = np.mean(odd_channels, axis=0)
    return np.stack([even_avg, odd_avg])


def plot_window(window_data, sps, window_index=None):
    """
    Plots a single window of EEG data (2 channels).

    Args:
        window_data: NumPy array of shape (2, n_samples) for the window.
        sps: Sampling rate in Hz.
        window_index: (Optional) window number for the title.
    """
    n_samples = window_data.shape[1]
    time_axis = np.arange(n_samples) / sps
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, window_data[0, :], label='Even channels average')
    plt.plot(time_axis, window_data[1, :], label='Odd channels average')
    if window_index is not None:
        plt.title(f"Window {window_index}")
    else:
        plt.title("EEG Window")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


def process_and_save(data, sps, coeffs_path, chans_path,
                     wavelet='db2', level=4, window_len_sec=1.8,
                     plot_windows=False, plot_random_n=None):
    """
    Segments a 2-channel signal into non-overlapping windows,
    performs wavelet decomposition and quantization on each window,
    and writes the results to text files.

    Optionally, plots a random subset of windows.

    Args:
        data: NumPy array of shape (2, total_samples)
        sps: Sampling rate (Hz)
        coeffs_path: Output file path for quantized coefficients
        chans_path: Output file path for channel labels
        wavelet: Wavelet type (default 'db2')
        level: Decomposition level (default 4)
        window_len_sec: Window length in seconds (here, 1.8 sec)
        plot_windows: If True, plots every window.
        plot_random_n: If set to an integer, randomly selects that many windows to plot.
    """
    n_window_samples = int(window_len_sec * sps)
    total_samples = data.shape[1]

    # Determine total number of windows.
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

            all_channel_coeffs = []
            all_channel_names = []
            for ch_idx in range(2):
                ch_name = str(ch_idx)
                channel_data = window_data[ch_idx, :]
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
                all_channel_names.extend([ch_name] * len(q_ids))
            f_coeffs.write(" ".join(all_channel_coeffs) + " ")
            f_chans.write(" ".join(all_channel_names) + " ")


def load_subject_raw_data(raw_folder, subject_id):
    """
    Loads raw EEG data for a given subject using MNE’s read_raw_curry.

    Assumes that for each subject there are four files with names:
        {subject_id}_AAD_1L.dat, {subject_id}_AAD_1R.dat,
        {subject_id}_AAD_2L.dat, {subject_id}_AAD_2R.dat

    Uses mne.io.read_raw_curry to read each .dat file. The function then concatenates
    all available recordings into one Raw object.

    Returns:
        raw_combined: An MNE Raw object with concatenated data.
    """
    file_variants = ["1L", "1R", "2L", "2R"]
    raw_list = []
    for variant in file_variants:
        fname = os.path.join(raw_folder, f"{subject_id}_AAD_{variant}.dat")
        if os.path.exists(fname):
            try:
                # Preload the data for faster indexing
                raw = mne.io.read_raw_curry(fname, preload=True, verbose=False)
                raw_list.append(raw)
            except Exception as e:
                print(f"Error reading {fname}: {e}")
        else:
            print(f"File {fname} not found.")
    if len(raw_list) == 0:
        raise ValueError(f"No CURRY files found for {subject_id} in {raw_folder}.")
    # Concatenate the raw recordings along the time axis.
    raw_combined = mne.concatenate_raws(raw_list)
    return raw_combined


if __name__ == "__main__":
    # Set the folder where the raw Curry files (extracted from Sx.tar.gzip) are located.
    raw_eeg_folder = "C:/Users/rabie/Desktop/gg/S0"  # Adjust this path accordingly
    subject_ids = [f"S{n}" for n in range(2)]  # S0, S1, ..., S29
    output_base = "output"

    for subject_id in subject_ids:
        print(f"Processing subject: {subject_id}")
        try:
            raw = load_subject_raw_data(raw_eeg_folder, subject_id)
        except Exception as e:
            print(f"Error loading subject {subject_id}: {e}")
            continue

        # Get the EEG data and sampling rate from the Raw object.
        # raw.get_data() returns an array of shape (n_channels, n_samples)
        eeg_data = raw.get_data()
        fs = raw.info["sfreq"]

        # (Optional) Select a subset of channels if needed, for example the first 64 channels:
        # eeg_data = eeg_data[:64, :]

        # Preprocess each segment of data as if it were a trial.
        # In this example we split the continuous data into trials based on events
        # (if available) or simply treat the entire recording as one long trial.
        # Here we simply treat the entire data as one "trial" for simplicity.
        trial_T = eeg_data  # Already in (channels, samples)
        # Preprocess (e.g., filtering, downsampling, etc.)
        prep_data, new_fs = preprocess_data(trial_T, fs)
        # Reduce to 2 channels by averaging alternate channels.
        twoch_data = average_alternate_channels(prep_data)
        # In this example we treat the whole processed data as one combined recording.
        combined_data = twoch_data  # Shape: (2, total_samples)

        # Define output file paths.
        coeffs_path = os.path.join(output_base, f"{subject_id}_combined_coeffs.txt")
        chans_path = os.path.join(output_base, f"{subject_id}_combined_channels.txt")

        # Process (window, wavelet decompose, quantize) the combined data.
        # For example, we can plot 5 random windows:
        process_and_save(combined_data, new_fs, coeffs_path, chans_path,
                         wavelet='db2', level=4, window_len_sec=1.8,
                         plot_windows=True, plot_random_n=5)
    print("Done!")
