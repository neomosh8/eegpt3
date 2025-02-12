import os
import numpy as np
import scipy.io
from utils import preprocess_data, wavelet_decompose_window, quantize_number


def average_left_right_channels(data):
    """
    Given data of shape (channels, samples) with 14 EEG channels,
    average the first 7 channels for the left hemisphere and the last 7 channels
    for the right hemisphere, returning a 2-channel array.
    """
    n_channels = data.shape[0]
    half = n_channels // 2  # Expected to be 7 for 14 channels
    left_avg = np.mean(data[:half, :], axis=0)
    right_avg = np.mean(data[half:, :], axis=0)
    return np.stack([left_avg, right_avg])


def load_mat_experiment_data(mat_file_path):
    """
    Loads a MATLAB file containing an experiment object with a field 'data'.

    The MATLAB file is assumed to have a variable 'o' with field 'data' (or a key 'data' directly)
    that is originally of shape (n_samples, 25). This function extracts only the EEG channels,
    which correspond to MATLAB columns 4–17 (i.e. Python indices 3 to 16).

    Returns:
        ndarray: EEG data of shape (n_samples, 14)
    """
    mat_data = scipy.io.loadmat(mat_file_path, squeeze_me=True)

    if 'o' in mat_data:
        o = mat_data['o']
        try:
            data = o.data
        except Exception:
            data = o['data']
    elif 'data' in mat_data:
        data = mat_data['data']
    else:
        raise ValueError("No valid data found in the MAT file.")

    # If data comes in as a memoryview, convert it.
    if isinstance(data, memoryview):
        data = np.asarray(data)

    # Unwrap 0-dimensional arrays.
    if isinstance(data, np.ndarray) and data.ndim == 0:
        data = data.item()

    # If the unwrapped data is a tuple, search for a 2D array with 25 columns.
    if isinstance(data, tuple):
        candidate = None
        for element in data:
            if isinstance(element, np.ndarray) and element.ndim == 2 and element.shape[1] == 25:
                candidate = element
                break
        if candidate is None:
            raise ValueError("Could not find the raw data element (2D array with 25 columns) in the tuple.")
        data = candidate

    # Extract the EEG channels (MATLAB channels 4–17 correspond to Python indices 3 to 16)
    eeg_data = data[:, 3:17]
    return eeg_data


def process_segment(data, sps, wavelet='db2', level=4, window_len_sec=1.18, plot_windows=False, plot_random_n=1):
    """
    Processes a two-channel EEG segment by splitting it into non-overlapping windows,
    performing wavelet decomposition and quantization on each window.

    Args:
        data (ndarray): Two-channel EEG data of shape (2, total_samples)
        sps (int): Sampling rate (Hz)
        wavelet (str): Wavelet name
        level (int): Decomposition level
        window_len_sec (float): Window length in seconds

    Returns:
        tuple: Two strings (coeffs_str, chans_str) containing space‐separated quantized coefficients
               and corresponding channel labels, respectively.
    """
    n_window_samples = int(window_len_sec * sps)
    total_samples = data.shape[1]
    coeffs_list = []
    chans_list = []

    # Loop over non-overlapping windows.
    for start_idx in range(0, total_samples - n_window_samples + 1, n_window_samples):
        window_data = data[:, start_idx:start_idx + n_window_samples]
        (decomposed_channels,
         coeffs_lengths,
         num_samples,
         normalized_data) = wavelet_decompose_window(window_data, wavelet=wavelet, level=level, normalization=True)

        for ch_idx in range(decomposed_channels.shape[0]):
            coeffs_flat = decomposed_channels[ch_idx].flatten()
            q_ids = [str(quantize_number(c)) for c in coeffs_flat]
            coeffs_list.append(" ".join(q_ids))
            chans_list.append(" ".join([str(ch_idx)] * len(q_ids)))

    return " ".join(coeffs_list), " ".join(chans_list)


def main():
    # Set input and output folders.
    dataset_folder = "dataset/EEG_Data"  # Adjust this to your data folder
    output_folder = "output_EMOTIV"
    os.makedirs(output_folder, exist_ok=True)

    # Prepare aggregated output file handles (one pair per attention label).
    attention_labels = ["Focused", "Unfocused", "Sleep"]
    agg_files = {}
    for lab in attention_labels:
        coeffs_path = os.path.join(output_folder, f"all_{lab}_coeffs.txt")
        chans_path = os.path.join(output_folder, f"all_{lab}_channels.txt")
        agg_files[lab] = (open(coeffs_path, "w"), open(chans_path, "w"))

    # Get list of all .mat files.
    mat_files = [f for f in os.listdir(dataset_folder) if f.endswith('.mat')]

    # Parameters
    fs = 128
    wavelet_name = 'db2'
    level = 4
    window_len_sec = 1.8
    focused_duration_sec = 10 * 60  # 10 minutes for Focused
    unfocused_duration_sec = 10 * 60  # next 10 minutes for Unfocused

    # Loop over each subject file.
    for mat_file in mat_files:
        mat_file_path = os.path.join(dataset_folder, mat_file)
        print(f"Processing file: {mat_file}")

        try:
            # Load the raw EEG data; expected shape: (n_samples, 14)
            raw_data = load_mat_experiment_data(mat_file_path)
        except Exception as e:
            print(f"Error loading {mat_file}: {e}")
            continue

        # Transpose to shape (channels, samples) for processing.
        raw_data_T = raw_data.T
        try:
            preprocessed_data, new_fs = preprocess_data(raw_data_T, fs)
        except Exception as e:
            print(f"Error preprocessing {mat_file}: {e}")
            continue

        # Reduce from 14 channels to 2 channels by averaging left and right hemispheres.
        twoch_data = average_left_right_channels(preprocessed_data)
        total_samples = twoch_data.shape[1]
        total_time_sec = total_samples / new_fs
        print(f"  Total duration: {total_time_sec:.2f} sec")

        # Segment the data based on time.
        segments = {}
        samples_focused = int(focused_duration_sec * new_fs)
        samples_unfocused = int((focused_duration_sec + unfocused_duration_sec) * new_fs)

        if total_samples >= samples_focused:
            segments["Focused"] = twoch_data[:, :samples_focused]
        else:
            print(f"  {mat_file}: too short for Focused segment.")

        if total_samples >= samples_unfocused:
            segments["Unfocused"] = twoch_data[:, samples_focused:samples_unfocused]
        else:
            print(f"  {mat_file}: too short for Unfocused segment.")

        if total_samples > samples_unfocused:
            segments["Sleep"] = twoch_data[:, samples_unfocused:]
        else:
            print(f"  {mat_file}: too short for Sleep segment.")

        # Process each attention segment and append results to the aggregated files.
        for lab, seg_data in segments.items():
            coeffs_str, chans_str = process_segment(seg_data, new_fs,
                                                    wavelet=wavelet_name,
                                                    level=level,
                                                    window_len_sec=window_len_sec)
            coeffs_fh, chans_fh = agg_files[lab]
            coeffs_fh.write(coeffs_str + " ")
            chans_fh.write(chans_str + " ")

    # Close all aggregated output files.
    for lab in agg_files:
        agg_files[lab][0].close()
        agg_files[lab][1].close()

    print("Aggregated outputs for all subjects written successfully.")


if __name__ == "__main__":
    main()
