import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from sandbox2 import calculate_sps

wvlet = 'db2'
level = 2
# Import your utility functions
from utils import (
    wavelet_decompose_window,
    wavelet_reconstruct_window,
    quantize_number,
    dequantize_number
)


def main():
    # === 1. Load the CSV data ===
    path = 'dataset/sub-000_task-proposer_run-1_eeg.csv'
    df = pd.read_csv(path)
    fs = int(calculate_sps(path))  # sampling rate (Hz)
    print(fs)

    num_samples_4sec = 1 * fs
    # Identify possible EEG channels
    all_columns = list(df.columns)
    exclude_cols = ['timestamp','VEOG','X','Y','Z',"EXG1","EXG2","EXG7","EXG8"]  # adjust as needed
    eeg_channels = [col for col in all_columns if col not in exclude_cols]



    # === 2. Randomly pick ONE channel and one 4-second window ===
    chosen_channel = random.choice(eeg_channels)
    total_samples = len(df)

    max_start_index = total_samples - num_samples_4sec
    if max_start_index <= 0:
        raise ValueError("Not enough data to extract 4 seconds from this dataset.")

    start_idx = random.randint(0, max_start_index)
    end_idx = int(start_idx + num_samples_4sec)

    original_signal = df[chosen_channel].values[start_idx:end_idx]
    window = np.expand_dims(original_signal, axis=0)  # shape: (1, num_samples_4sec)

    # === 3. Wavelet Decomposition (with internal z-score) ===
    (decomposed_channels,
     coeffs_lengths,
     num_samples,
     normalized_data) = wavelet_decompose_window(
         window, wavelet=wvlet, level=level, normalization=True
     )

    # === 4. Quantize & Dequantize the wavelet coefficients ===
    quantized_coeffs_list = []
    for coeff_val in decomposed_channels.flatten():
        q_id = quantize_number(coeff_val)  # quantize -> string identifier
        quantized_coeffs_list.append(q_id)
    print(quantized_coeffs_list)
    print(len(quantized_coeffs_list))
    # Convert quantized string identifiers back to float
    dequantized_coeffs = [dequantize_number(qid) for qid in quantized_coeffs_list]
    dequantized_coeffs = np.array(dequantized_coeffs).reshape(decomposed_channels.shape)

    # === 5. Wavelet Reconstruction (still in "z-scored" domain) ===
    reconstructed_window = wavelet_reconstruct_window(
        dequantized_coeffs,
        coeffs_lengths,
        num_samples,
        wavelet=wvlet
    )
    reconstructed_signal_z = reconstructed_window[0, :]  # shape: (num_samples_4sec,)

    # --------------------------------------------------------------------------------
    # NOTE: At this point, 'reconstructed_signal_z' is in the SAME z-score space that
    # was used *inside* wavelet_decompose_window. Typically, you'd unscale using the
    # SAME mean/std from that window, but here we want to demonstrate using the
    # "next 4-second window" to do it.
    # --------------------------------------------------------------------------------

    # === 6. (NEW) Get the NEXT 4-second window to compute new z-score parameters ===
    # next_start_idx = end_idx
    # next_end_idx = next_start_idx + num_samples_4sec
    next_start_idx = start_idx - num_samples_4sec
    next_end_idx = start_idx

    # Check if we have enough data for the next 4-second window
    if next_end_idx <= total_samples:
        # Extract that next window
        next_window_data = df[chosen_channel].values[next_start_idx:next_end_idx]

        # Calculate mean and std of the NEXT window
        next_mean = np.mean(next_window_data)
        next_std = np.std(next_window_data)
    else:
        # If there's no enough data, just do something arbitrary
        # (e.g., re-use the old mean/std or skip)
        next_mean = 0.0
        next_std = 1.0

    if next_std == 0:
        next_std = 1.0  # avoid division by zero

    # === 7. Rescale the reconstructed signal using the NEXT window's mean/std ===
    reconstructed_signal_scaled = reconstructed_signal_z * next_std + next_mean

    # === 8. Plot everything ===
    time_axis = np.arange(num_samples_4sec) / fs  # time in seconds

    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, original_signal, label='Original (this 4s window)', alpha=0.7)
    plt.plot(time_axis, reconstructed_signal_scaled,
             label='Reconstructed (rescaled w/ NEXT 4s mean/std)',
             alpha=0.7)
    plt.title(f'Original vs Reconstructed (Next-Window Z-score) - Channel: {chosen_channel}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
