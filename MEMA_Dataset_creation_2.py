import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import os


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


def plot_eeg_channels(df, fs=500, title="EEG Channels"):
    """
    Plots each column in df as a separate subplot.
    Args:
        df: Pandas DataFrame where each column is one EEG channel.
        fs: Sampling rate in Hz.
        title: Title of the plot.
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


if __name__ == "__main__":
    # ---------------------------
    # 1) Load raw epoch data and labels from the .mat file
    # ---------------------------
    mat_file_path = "dataset/Subject2_dom.mat"  # Change this path as needed
    epochs, labels = load_mat_epoch_data(mat_file_path)
    fs = 500  # Sampling frequency (Hz)

    # ---------------------------
    # 2) For each label (task), concatenate raw epochs and save as CSV
    # ---------------------------
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        print(f"Processing task (label): {lab}")

        # Select all epochs belonging to the current label.
        # epochs has shape (n_epochs, n_samples, n_channels)
        epochs_class = epochs[labels == lab]
        # Ensure the data has three dimensions (in case there is only one epoch)
        if epochs_class.ndim == 2:
            epochs_class = np.expand_dims(epochs_class, axis=0)

        # Concatenate the epochs along the time (sample) axis.
        # Each epoch is of shape (n_samples, n_channels), so the combined data
        # will have shape (total_samples, n_channels).
        combined_data = np.concatenate(epochs_class, axis=0)
        total_samples = combined_data.shape[0]

        # Create a time stamp column (in seconds).
        time_stamps = np.arange(total_samples) / fs

        # Create a DataFrame with columns: TimeStamp, ch_1, ch_2, ...
        n_channels = combined_data.shape[1]
        channel_names = [f"ch_{i + 1}" for i in range(n_channels)]
        df_data = pd.DataFrame(combined_data, columns=channel_names)
        df_data.insert(0, "TimeStamp", time_stamps)

        # Save CSV file for this task/label
        output_folder = "csv_tasks"
        os.makedirs(output_folder, exist_ok=True)
        csv_filename = os.path.join(output_folder, f"task_{lab}.csv")
        df_data.to_csv(csv_filename, index=False)
        print(f"Saved CSV for task {lab} to {csv_filename}")

        # ---------------------------
        # 3) Plot all channels for this task
        # ---------------------------
        # (TimeStamp is not plotted; only the channel data is plotted.)
        plot_eeg_channels(df_data[channel_names], fs=fs, title=f"Task (Label): {lab}")
