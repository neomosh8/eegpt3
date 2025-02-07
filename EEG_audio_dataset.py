#https://zenodo.org/records/1199011
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


import pywt
from utils import preprocess_data, wavelet_decompose_window, quantize_number
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
    plt.savefig(f"window_{window_index}.png")
    plt.close()

def average_alternate_channels(data):
    """
    Averages alternate channels.

    Args:
        data: NumPy array of shape (channels, samples)

    Returns:
        A 2-channel NumPy array where:
          - Channel 0 is the average of even-indexed channels (0,2,4,...)
          - Channel 1 is the average of odd-indexed channels (1,3,5,...)
    """
    even_channels = data[0::2, :]
    odd_channels = data[1::2, :]
    even_avg = np.mean(even_channels, axis=0)
    odd_avg = np.mean(odd_channels, axis=0)
    return np.stack([even_avg, odd_avg])


def plot_eeg_channels(df, fs, title="EEG Channels"):
    """
    Plots each column in the given DataFrame as a separate subplot.

    Args:
        df: DataFrame of shape (samples, channels)
        fs: Sampling rate in Hz
        title: Plot title
    """
    n_samples = len(df)
    time_axis = np.arange(n_samples) / fs
    n_channels = df.shape[1]
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 2.5 * n_channels), sharex=True)
    if n_channels == 1:
        axes = [axes]
    for i, col in enumerate(df.columns):
        axes[i].plot(time_axis, df[col], label=col)
        axes[i].set_ylabel(col)
        axes[i].legend(loc='upper right')
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def process_and_save(data, sps, coeffs_path, chans_path,
                     wavelet='db2', level=4, window_len_sec=1.8, plot_windows=False, plot_random_n=5):
    """
    Segments a 2-channel signal into non-overlapping windows,
    performs wavelet decomposition and quantization on each window,
    and writes the results to text files.

    Optionally, it plots a subset of the windows.

    Args:
        data: NumPy array of shape (2, total_samples)
        sps: Sampling rate (Hz)
        coeffs_path: Output file path for quantized coefficients
        chans_path: Output file path for channel labels
        wavelet: Wavelet type (default 'db2')
        level: Decomposition level (default 4)
        window_len_sec: Window length in seconds (here, 1.8 sec)
        plot_windows: If True, plots every window.
        plot_random_n: If set to an integer, randomly select that many windows to plot.
    """
    n_window_samples = int(window_len_sec * sps)
    total_samples = data.shape[1]

    # Calculate the total number of windows.
    total_windows = len(range(0, total_samples - n_window_samples + 1, n_window_samples))

    # If you want to plot a random subset of windows, select them now.
    if plot_random_n is not None and plot_random_n < total_windows:
        selected_windows = np.random.choice(range(1, total_windows + 1), size=plot_random_n, replace=False)
    else:
        selected_windows = None  # This means plot all windows if plot_windows is True

    os.makedirs(os.path.dirname(coeffs_path), exist_ok=True)
    window_counter = 0

    with open(coeffs_path, 'w') as f_coeffs, open(chans_path, 'w') as f_chans:
        # Process each non-overlapping window.
        for start_idx in range(0, total_samples - n_window_samples + 1, n_window_samples):
            window_counter += 1
            end_idx = start_idx + n_window_samples
            window_data = data[:, start_idx:end_idx]

            # Determine whether to plot this window.
            if selected_windows is not None:
                if window_counter in selected_windows:
                    plot_window(window_data, sps, window_index=window_counter)
            elif plot_windows:
                plot_window(window_data, sps, window_index=window_counter)

            # Now do the wavelet decomposition and quantization.
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

def load_subject_data(mat_file_path):
    """
    Loads an original EEG MAT file from the /eeg folder.

    The MAT file is expected to contain:
      - A variable 'data' (with fields: eeg, fsample.eeg, and event.eeg.sample)
      - Experimental information stored in a MATLAB table. If the top‑level key
        'expinfo' is missing, we attempt to extract it from a MatlabOpaque object
        stored under the key 'None'.

    Returns:
        trials: a list of NumPy arrays; each array is one trial (shape: [n_samples_trial, n_channels])
                (the first trial is assumed to be pre-stim and is ignored)
        fs: the sampling rate (int)
        labels: trial labels extracted from the experimental info if available;
                if not found, dummy labels (default 1) are assigned.
    """
    # Load the file with options to access struct attributes.
    mat = scipy.io.loadmat(mat_file_path, squeeze_me=True, struct_as_record=False)

    # --- Extract experimental info ---
    if 'expinfo' in mat:
        expinfo = mat['expinfo']
    elif 'None' in mat:
        mopaque = mat['None']
        expinfo = None
        try:
            if isinstance(mopaque, (list, np.ndarray)):
                for item in mopaque:
                    if item[0] == b'expinfo':
                        table_struct = item[3]
                        if isinstance(table_struct, np.ndarray) and table_struct.dtype.names is not None:
                            if 'arr' in table_struct.dtype.names:
                                expinfo = table_struct['arr']
                            else:
                                expinfo = table_struct
                        else:
                            expinfo = table_struct
                        break
            if expinfo is None:
                raise KeyError("expinfo not found inside the MatlabOpaque structure.")
        except Exception as e:
            raise KeyError("Could not extract expinfo from MAT file.") from e
    else:
        raise KeyError("MAT file does not contain 'expinfo' or an equivalent structure.")

    # --- Extract the data structure ---
    if 'data' not in mat:
        raise KeyError("MAT file does not contain 'data'")
    data_struct = mat['data']
    if isinstance(data_struct, np.ndarray):
        data_struct = data_struct.item()

    # Access continuous EEG data, sampling rate, and event sample indices.
    try:
        eeg_continuous = data_struct.eeg
        fs = data_struct.fsample.eeg
        splits = np.array(data_struct.event.eeg.sample, dtype=float) - 1  # convert from 1-indexed to 0-indexed
        splits = np.sort(splits)
    except AttributeError as e:
        raise ValueError("The 'data' structure does not have the expected fields.") from e

    # --- Split the continuous EEG into trials ---
    trials = []
    for i in range(1, len(splits)):
        start = int(splits[i])
        end = int(splits[i + 1]) if i < len(splits) - 1 else eeg_continuous.shape[0]
        trial = eeg_continuous[start:end, :]
        trials.append(trial)

    # --- Extract trial labels from the experimental info ---
    # Since you don't care about separating by label, we simply assign a dummy label (1) to each trial.
    labels = np.ones(len(trials))

    return trials, fs, labels


if __name__ == "__main__":
    import boto3
    import zipfile
    import tempfile

    # Download EEG.zip from S3 and extract it
    s3_bucket = "dataframes--use1-az6--x-s3"
    s3_key = "attention fintune/1199011/EEG.zip"
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "EEG.zip")

    s3 = boto3.client('s3')
    print("Downloading EEG.zip from S3...")
    s3.download_file(s3_bucket, s3_key, zip_path)
    print(f"Downloaded EEG.zip to {zip_path}")

    extract_dir = os.path.join(temp_dir, "EEG")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Extracted EEG.zip to {extract_dir}")

    # Set eeg_folder to the extracted directory
    eeg_folder = extract_dir
    subject_files = [f"S{ss}.mat" for ss in range(1, 19)]  # e.g., S1.mat to S18.mat
    output_base = "output_1199011"

    for subj_file in subject_files:
        subj_path = os.path.join(eeg_folder, subj_file)
        print(f"Processing subject file: {subj_path}")
        try:
            trials, fs, trial_labels = load_subject_data(subj_path)
        except Exception as e:
            print(f"Error loading {subj_file}: {e}")
            continue

        processed_trials = []  # Processed (2-channel) trials

        for trial in trials:
            # Select only the first 64 channels (scalp EEG)
            trial = trial[:, :64]
            # Transpose so that the data shape is (channels, samples)
            trial_T = trial.T
            # Preprocess the trial (e.g., filtering, downsampling)
            prep_data, new_fs = preprocess_data(trial_T, fs)
            # Reduce to 2 channels by averaging alternate channels
            twoch_data = average_alternate_channels(prep_data)
            processed_trials.append(twoch_data)

        if processed_trials:
            # Combine all processed trials (all attention EEG) into one continuous signal.
            combined_data = np.concatenate(processed_trials, axis=1)
            base_name = os.path.splitext(subj_file)[0]
            coeffs_path = os.path.join(output_base, f"{base_name}_combined_coeffs.txt")
            chans_path = os.path.join(output_base, f"{base_name}_combined_channels.txt")
            process_and_save(combined_data, new_fs, coeffs_path, chans_path,
                             wavelet='db2', level=4, window_len_sec=1.8, plot_windows=True)
    print("Done!")
