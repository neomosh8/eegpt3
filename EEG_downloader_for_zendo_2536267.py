import mne
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pywt
from utils import preprocess_data, wavelet_decompose_window, quantize_number


def average_left_right(raw):
    """
    Average EEG channels separately for left and right hemispheres.

    This function first selects only EEG channels (thereby excluding non‐EEG channels
    such as EXG, GSR, etc.). It then ensures a montage is set (using the standard
    BioSemi64 montage) so that the channel locations are available. Channels with an
    x-coordinate (first element in the location vector) < 0 are considered left-hemisphere
    and those with x > 0 as right-hemisphere.

    Args:
        raw : mne.io.Raw object with EEG data.

    Returns:
        combined_data : A 2-channel NumPy array of shape (2, total_samples) where
                        combined_data[0, :] is the average left hemisphere signal and
                        combined_data[1, :] is the average right hemisphere signal.
    """
    # Pick only EEG channels (excludes non-EEG channels like EXG, GSR, etc.)
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')

    # Ensure the montage is set. If not, assign the standard BioSemi64 montage.
    if raw.get_montage() is None:
        montage = mne.channels.make_standard_montage("biosemi64")  # :contentReference[oaicite:0]{index=0}
        raw.set_montage(montage)

    data = raw.get_data(picks=picks)

    left_signals = []
    right_signals = []

    # Loop over the picked EEG channels.
    for i, idx in enumerate(picks):
        ch_info = raw.info["chs"][idx]
        # The first three entries in 'loc' are the x, y, z coordinates (in meters)
        loc = ch_info["loc"][:3]
        if loc[0] < 0:
            left_signals.append(data[i])
        elif loc[0] > 0:
            right_signals.append(data[i])
        else:
            # In case x == 0, you might choose to assign it to one side or ignore.
            pass

    # Compute the average for each hemisphere.
    left_avg = np.mean(left_signals, axis=0) if left_signals else np.zeros(data.shape[1])
    right_avg = np.mean(right_signals, axis=0) if right_signals else np.zeros(data.shape[1])

    return np.stack([left_avg, right_avg])


def plot_window(window_data, sps, window_index=None):
    """
    Plots a single window of EEG data (2 channels).

    Args:
        window_data: NumPy array of shape (2, n_samples)
        sps: Sampling rate in Hz.
        window_index: (Optional) Window number for the title.
    """
    n_samples = window_data.shape[1]
    time_axis = np.arange(n_samples) / sps
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, window_data[0, :], label='Left hemisphere average')
    plt.plot(time_axis, window_data[1, :], label='Right hemisphere average')
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
                     wavelet='db2', level=4, window_len_sec=1.8, plot_windows=False, plot_random_n=1):
    """
    Segments a 2-channel signal into non-overlapping windows,
    performs wavelet decomposition and quantization on each window,
    and writes the results to text files.

    Args:
        data: NumPy array of shape (2, total_samples)
        sps: Sampling rate (Hz)
        coeffs_path: Output file path for quantized coefficients
        chans_path: Output file path for channel labels
        wavelet: Wavelet type (default 'db2')
        level: Decomposition level (default 4)
        window_len_sec: Window length in seconds (default 1.8 sec)
        plot_windows: If True, plots each window.
        plot_random_n: If an integer, randomly selects that many windows to plot.
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


def load_subject_raw_data_bids(subject_folder):
    """
    Loads raw EEG data for a given subject from a BIDS-formatted dataset.
    Iterates over session folders, loads BDF files from the 'eeg' subfolder,
    and concatenates them.

    Args:
        subject_folder: Path to the subject folder (e.g. "path/to/sub-001")

    Returns:
        raw_combined: An MNE Raw object with concatenated EEG data.
    """
    raw_list = []
    for ses in sorted(os.listdir(subject_folder)):
        ses_path = os.path.join(subject_folder, ses)
        if os.path.isdir(ses_path) and ses.startswith("ses-"):
            eeg_path = os.path.join(ses_path, "eeg")
            if os.path.isdir(eeg_path):
                bdf_files = [f for f in os.listdir(eeg_path) if f.endswith("_eeg.bdf")]
                if not bdf_files:
                    print(f"No BDF files found in {eeg_path}.")
                for bdf_file in bdf_files:
                    full_path = os.path.join(eeg_path, bdf_file)
                    try:
                        raw = mne.io.read_raw_bdf(full_path, preload=True, verbose=False)
                        raw_list.append(raw)
                    except Exception as e:
                        print(f"Error reading {full_path}: {e}")
            else:
                print(f"EEG folder not found in session folder: {ses_path}")
    if not raw_list:
        raise ValueError(f"No BDF files found in subject folder: {subject_folder}")
    if len(raw_list) > 1:
        raw_combined = mne.concatenate_raws(raw_list)
    else:
        raw_combined = raw_list[0]
    return raw_combined


if __name__ == "__main__":
    import boto3
    import zipfile
    import tempfile

    # --- S3 and Dataset Configuration for BIDS Data ---
    s3_bucket = "dataframes--use1-az6--x-s3"
    s3_folder = "attention fintune/2536267"
    zip_filename = "BIDS_EEG_meditation_experiment_v2.zip"
    s3_key = f"{s3_folder}/{zip_filename}"

    output_base = "output-2536267"
    os.makedirs(output_base, exist_ok=True)

    s3 = boto3.client("s3")

    with tempfile.TemporaryDirectory() as temp_dir:
        local_zip_path = os.path.join(temp_dir, zip_filename)
        print(f"Downloading s3://{s3_bucket}/{s3_key} ...")
        try:
            s3.download_file(s3_bucket, s3_key, local_zip_path)
            print(f"Downloaded {zip_filename} to {local_zip_path}")
        except Exception as e:
            print(f"Error downloading {zip_filename}: {e}")
            exit(1)

        extract_path = os.path.join(temp_dir, "extracted")
        os.makedirs(extract_path, exist_ok=True)
        try:
            with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print(f"Extracted {zip_filename} to {extract_path}")
        except Exception as e:
            print(f"Error extracting {zip_filename}: {e}")
            exit(1)

        extracted_contents = os.listdir(extract_path)
        if len(extracted_contents) == 1 and os.path.isdir(os.path.join(extract_path, extracted_contents[0])):
            dataset_root = os.path.join(extract_path, extracted_contents[0])
        else:
            dataset_root = extract_path

        subject_dirs = sorted([d for d in os.listdir(dataset_root)
                               if d.startswith("sub-") and os.path.isdir(os.path.join(dataset_root, d))])

        for subject in subject_dirs:
            subject_folder = os.path.join(dataset_root, subject)
            print(f"\nProcessing subject: {subject}")
            try:
                raw = load_subject_raw_data_bids(subject_folder)
            except Exception as e:
                print(f"Error loading subject {subject}: {e}")
                continue

            # Retrieve EEG data and sampling rate
            fs = raw.info["sfreq"]

            # Preprocess data using the provided function (e.g., filtering, downsampling, etc.)
            prep_data, new_fs = preprocess_data(raw.get_data(), fs)
            # Now average only EEG channels into left and right hemispheric signals.
            combined_data = average_left_right(raw)

            coeffs_path = os.path.join(output_base, f"{subject}_combined_coeffs.txt")
            chans_path = os.path.join(output_base, f"{subject}_combined_channels.txt")

            process_and_save(combined_data, new_fs, coeffs_path, chans_path,
                             wavelet='db2', level=4, window_len_sec=1.8, plot_windows=True)
            print(f"Finished processing subject: {subject}")

    print("Done!")
