import os
import random
import numpy as np
import matplotlib.pyplot as plt
import mne

def main():
    # Path to the directory where your files reside
    data_path = "original_database"

    # Name of your EEGLAB .set file (must be accompanied by the .fdt)
    set_filename = "sub-002_ses-02_task-ReinforcementLearning_eeg.set"

    # -------------------------------------------------------------------------
    # 1. Load the EEG dataset
    # -------------------------------------------------------------------------
    raw = mne.io.read_raw_eeglab(os.path.join(data_path, set_filename), preload=True)
    print("Data loaded successfully.")

    # Retrieve sampling frequency and available channel names
    sfreq = raw.info['sfreq']
    ch_names = raw.ch_names
    print(f"Sampling Frequency (Hz): {sfreq}")
    print(f"Available Channels: {ch_names}")

    # -------------------------------------------------------------------------
    # 2. Randomly pick a channel and 4 seconds of data
    # -------------------------------------------------------------------------
    chosen_channel = random.choice(ch_names)
    channel_idx = ch_names.index(chosen_channel)
    print(f"\nChosen Channel: {chosen_channel}")

    # Extract data for the chosen channel
    # 'data' shape: (1, n_samples), 'times' shape: (n_samples,)
    data, times = raw[channel_idx, :]
    total_time = times[-1]  # total duration in seconds

    # Ensure we have at least 4 seconds of data
    if total_time < 4:
        raise ValueError("The dataset does not contain at least 4 seconds of data.")

    # Random start time (between 0 and total_time-4)
    start_sec = random.uniform(0, total_time - 4)
    end_sec = start_sec + 200

    # Convert times to sample indices
    start_samp = int(start_sec * sfreq)
    end_samp = int(end_sec * sfreq)

    # Snippet of data
    snippet = data[0, start_samp:end_samp]  # shape: (end_samp - start_samp,)
    snippet_times = times[start_samp:end_samp]

    # -------------------------------------------------------------------------
    # 3. Print amplitude statistics
    # -------------------------------------------------------------------------
    mean_amp = np.mean(snippet)
    std_amp = np.std(snippet)
    min_amp = np.min(snippet)
    max_amp = np.max(snippet)

    print("\nAmplitude Statistics for the 4s snippet:")
    print(f" - Mean: {mean_amp:.5f}")
    print(f" - Std:  {std_amp:.5f}")
    print(f" - Min:  {min_amp:.5f}")
    print(f" - Max:  {max_amp:.5f}")

    # -------------------------------------------------------------------------
    # 4. Plot the 4-second snippet
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(snippet_times, snippet, label=f"{chosen_channel} (4s snippet)")
    plt.title(f"Random 4-second snippet from channel '{chosen_channel}'")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (?)")  # or your unit of measurement
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
