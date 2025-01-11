import numpy as np
import torch
import pandas as pd
import os
import matplotlib.pyplot as plt

from utils import (
    preprocess_data,
    wavelet_decompose_window,
    quantize_number
)

def average_alternate_channels(data):
    """
    data is shape [ch, samples].
    We interpret channel 0,2,4,... as "even" channels,
    and channel 1,3,5,... as "odd" channels.
    We then average each group => returns 2xSamples array.
    """
    # 'even' channels: 0, 2, 4
    even_channels = data[0::2]  # start=0 step=2
    # 'odd' channels: 1, 3, 5
    odd_channels  = data[1::2]  # start=1 step=2

    even_avg = np.mean(even_channels, axis=0) if even_channels.size > 0 else np.zeros(data.shape[1])
    odd_avg  = np.mean(odd_channels, axis=0)  if odd_channels.size  > 0 else np.zeros(data.shape[1])

    return np.stack([even_avg, odd_avg], axis=0)  # shape: [2, samples]


def process_and_save(
        data,         # shape [2, total_samples]
        sps,          # sampling rate
        coeffs_path,  # path to write the wavelet-coeff tokens
        chans_path,   # path to write the channel-IDs
        wavelet,
        level,
        window_len
):
    """
    - data: shape [2, total_samples] (the two averaged channels).
    - sps: samples per second
    - wavelet, level: wavelet decomposition parameters
    - window_len: length of each window in seconds
    Writes out a single line per window containing interleaved tokens:
      ch0_token_1, ch1_token_1, ch0_token_2, ch1_token_2, ...
    and a second line with interleaved channel IDs:
      0, 1, 0, 1, ...
    """

    n_window_samples = int(window_len * sps)
    total_samples = data.shape[1]

    with open(coeffs_path, 'w') as f_coeffs, open(chans_path, 'w') as f_chans:
        # Slide through data in non-overlapping windows of size n_window_samples
        for window_start in range(0, total_samples - n_window_samples + 1, n_window_samples):
            window_end = window_start + n_window_samples

            # ---------------------------
            # Decompose Channel 0
            # ---------------------------
            ch0_data = data[0, window_start:window_end]
            ch0_data_2d = ch0_data[np.newaxis, :]  # shape (1, n_window_samples)

            (ch0_decomp, ch0_lengths, ch0_num_samps, ch0_normed_data) = wavelet_decompose_window(
                ch0_data_2d,
                wavelet=wavelet,
                level=level,
                normalization=True
            )

            ch0_flat = ch0_decomp.flatten()  # 1D array
            ch0_quant = [str(quantize_number(val)) for val in ch0_flat]

            # ---------------------------
            # Decompose Channel 1
            # ---------------------------
            ch1_data = data[1, window_start:window_end]
            ch1_data_2d = ch1_data[np.newaxis, :]

            (ch1_decomp, ch1_lengths, ch1_num_samps, ch1_normed_data) = wavelet_decompose_window(
                ch1_data_2d,
                wavelet=wavelet,
                level=level,
                normalization=True
            )

            ch1_flat = ch1_decomp.flatten()
            ch1_quant = [str(quantize_number(val)) for val in ch1_flat]

            # Quick sanity check: both channels should produce the SAME length
            if len(ch0_quant) != len(ch1_quant):
                raise ValueError(
                    f"Wavelet mismatch: ch0 has {len(ch0_quant)} coeffs, "
                    f"ch1 has {len(ch1_quant)}. They must match!"
                )

            # ---------------------------
            # Interleave them
            # ---------------------------
            interleaved_coeffs = []
            interleaved_chans  = []
            for i in range(len(ch0_quant)):
                # Channel 0 token
                interleaved_coeffs.append(ch0_quant[i])
                interleaved_chans.append("0")

                # Channel 1 token
                interleaved_coeffs.append(ch1_quant[i])
                interleaved_chans.append("1")

            # Convert to strings, add trailing space or newline
            coeffs_line = " ".join(interleaved_coeffs) + "\n"
            chans_line  = " ".join(interleaved_chans)  + "\n"

            # Write out
            f_coeffs.write(coeffs_line)
            f_chans.write(chans_line)


# --------------- EVERYTHING BELOW HERE is your main script logic ---------------

wvlet = 'db2'  # example wavelet; change as needed
level = 4      # wavelet decomposition level; change as needed

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

    1) Stack the EEG data so that dimension 0 = trials, dimension 1=channels, dimension 2=time
    2) Reshape so rows=all time from all trials, columns=62 EEG channels
    3) Return as a DataFrame
    """
    if not eeg_list:
        return None

    # (n, 62, 501)
    stacked = torch.stack(eeg_list)
    n_trials = stacked.shape[0]

    # Swap time <-> channel to get shape (n, time, channels) => (n, 501, 62)
    stacked = stacked.permute(0, 2, 1)

    # Reshape => (n*501, 62)
    stacked_2d = stacked.reshape(-1, 62)

    # Convert to numpy for DataFrame
    eeg_array = stacked_2d.numpy()

    # Channel column names
    channel_cols = [f"Channel_{i + 1}" for i in range(62)]
    df = pd.DataFrame(eeg_array, columns=channel_cols)

    # (Optionally) insert metadata columns
    df.insert(0, "Subject", subject)
    df.insert(1, "Image", image)
    df.insert(2, "Granularity", granularity)

    return df


def main():
    file_path = "dataset/EEG-ImageNet_1.pth"  # <-- Your input .pth file
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

    # 3) Process each group
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

        # 4) Build DataFrame
        df = build_df_for_group(eeg_list, subj, img, gran)
        if df is None:
            continue

        # Just keep the EEG columns (drop the first 3 metadata columns)
        df.drop(df.columns[:3], axis=1, inplace=True)

        # shape => (channels, samples)
        # but df is (samples, channels). So do transpose => shape [channels, samples]
        data_2d = df.values.transpose()

        original_sps = 1000  # your nominal sampling rate
        prep_data, new_sps = preprocess_data(data_2d, original_sps)

        # Now reduce from many channels to exactly 2 by averaging "odd" vs. "even" channels
        twoch = average_alternate_channels(prep_data)  # => shape [2, samples]

        # Build output filenames
        img_basename = os.path.splitext(os.path.basename(img))[0]
        out_name = f"subject_{subj}_image_{img_basename}_gran_{gran}"
        coeffs_path = os.path.join(output_dir, f"{out_name}_coeffs.txt")
        chans_path  = os.path.join(output_dir, f"{out_name}_channels.txt")

        # 5) Actually run wavelet decomposition in windows & interleave
        process_and_save(
            data=twoch,
            sps=new_sps,
            coeffs_path=coeffs_path,
            chans_path=chans_path,
            wavelet=wvlet,
            level=level,
            window_len=1.18  # e.g. 2-second windows
        )

        # free memory
        del df, eeg_list

    print("Done! All groups processed and saved.")


if __name__ == "__main__":
    main()
