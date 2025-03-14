import os
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt

# Full list of channel names as per EMOTIV documentation
EMOTIV_CHANNEL_NAMES = [
    'ED_COUNTER',      # 1
    'ED_INTERPOLATED', # 2
    'ED_RAW_CQ',       # 3
    'ED_AF3',          # 4
    'ED_F7',           # 5
    'ED_F3',           # 6
    'ED_FC5',          # 7
    'ED_T7',           # 8
    'ED_P7',           # 9
    'ED_O1',           # 10
    'ED_O2',           # 11
    'ED_P8',           # 12
    'ED_T8',           # 13
    'ED_FC6',          # 14
    'ED_F4',           # 15
    'ED_F8',           # 16
    'ED_AF4',          # 17
    'ED_GYROX',        # 18
    'ED_GYROY',        # 19
    'ED_TIMESTAMP',    # 20
    'ED_ES_TIMESTAMP', # 21
    'ED_FUNC_ID',      # 22
    'ED_FUNC_VALUE',   # 23
    'ED_MARKER',       # 24
    'ED_SYNC_SIGNAL'   # 25
]

# For EEG, we only need channels 4 to 17 (MATLAB indexing).
# In Python (0-indexed), that corresponds to indices 3 to 16.
EEG_CHANNEL_NAMES = EMOTIV_CHANNEL_NAMES[3:17]

# Convert to conventional EEG channel names by dropping the "ED_" prefix.
CONVENTIONAL_CHANNEL_NAMES = [name.replace("ED_", "") for name in EEG_CHANNEL_NAMES]

def process_emotiv_file(mat_file_path):
    """
    Loads an Emotiv .mat file and extracts the EEG data (channels 4–17).

    Args:
        mat_file_path: Path to the .mat file.

    Returns:
        df: Pandas DataFrame with a "TimeStamp" column and one column per EEG channel,
            using conventional channel names.
    """
    # Load the MATLAB file.
    mat_data = scipy.io.loadmat(mat_file_path, squeeze_me=True, struct_as_record=False)

    # The EMOTIV data is stored in an object "o". Its field "data" holds the raw data.
    o = mat_data['o']
    raw_data = o.data  # Shape: (n_samples, 25)

    # Extract the EEG channels: MATLAB channels 4–17 are Python indices 3:17.
    eeg_data = raw_data[:, 3:17]
    n_samples = eeg_data.shape[0]
    fs = 128  # Sampling frequency for EMOTIV data (Hz)

    # Create a time stamp array in seconds.
    time_stamps = np.arange(n_samples) / fs

    # Create the DataFrame with a TimeStamp column and one column per EEG channel,
    # using conventional channel names.
    df = pd.DataFrame(eeg_data, columns=CONVENTIONAL_CHANNEL_NAMES)
    df.insert(0, "TimeStamp", time_stamps)

    return df

def plot_emotiv_eeg(df, fs=128, title="Emotiv EEG Data"):
    """
    Plots all EEG channels from the DataFrame in separate subplots.

    Args:
        df: Pandas DataFrame with a "TimeStamp" column followed by EEG channel columns.
        fs: Sampling frequency.
        title: Plot title.
    """
    # Exclude the "TimeStamp" and "Label" columns for plotting.
    channel_columns = [col for col in df.columns if col not in ["TimeStamp", "Label"]]
    n_channels = len(channel_columns)
    n_samples = len(df)
    time_axis = np.arange(n_samples) / fs

    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 2.5 * n_channels), sharex=True)
    if n_channels == 1:
        axes = [axes]

    for i, col in enumerate(channel_columns):
        axes[i].plot(time_axis, df[col], label=col)
        axes[i].set_ylabel(col)
        axes[i].legend(loc='upper right')

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Folder containing the Emotiv experiment .mat files.
    input_folder = "dataset/emotiv_experiments"  # <-- Update this path as needed.

    # Output folder to save aggregated CSV files.
    output_folder = "csv_emotiv"
    os.makedirs(output_folder, exist_ok=True)

    # Containers for aggregating data across all subjects by label.
    focused_list = []
    unfocused_list = []
    sleep_list = []

    # List all .mat files in the input folder.
    mat_files = [f for f in os.listdir(input_folder) if f.endswith('.mat')]

    for mat_file in mat_files:
        mat_file_path = os.path.join(input_folder, mat_file)
        print(f"Processing file: {mat_file_path}")

        # Process the file to get a DataFrame.
        df = process_emotiv_file(mat_file_path)

        # Create a "Label" column based on TimeStamp (in seconds).
        # 0 - 10 minutes: focused; 10 - 20 minutes: unfocused; 20+ minutes: sleep.
        conditions = [
            (df["TimeStamp"] < 600),
            (df["TimeStamp"] >= 600) & (df["TimeStamp"] < 1200),
            (df["TimeStamp"] >= 1200)
        ]
        choices = ["focused", "unfocused", "sleep"]
        df["Label"] = np.select(conditions, choices, default="unknown")

        # Append rows to the aggregated lists for each label.
        focused_list.append(df[df["Label"] == "focused"])
        unfocused_list.append(df[df["Label"] == "unfocused"])
        sleep_list.append(df[df["Label"] == "sleep"])

        # Optionally, plot the EEG channels for this file.
        # plot_emotiv_eeg(df, fs=128, title=f"Emotiv EEG Data: {mat_file}")

    # After processing all files, aggregate data across subjects for each label.
    focused_df = pd.concat(focused_list, ignore_index=True) if focused_list else pd.DataFrame()
    unfocused_df = pd.concat(unfocused_list, ignore_index=True) if unfocused_list else pd.DataFrame()
    sleep_df = pd.concat(sleep_list, ignore_index=True) if sleep_list else pd.DataFrame()

    # Remove the "Label" column before saving the aggregated CSV files.
    focused_csv = os.path.join(output_folder, "focused.csv")
    unfocused_csv = os.path.join(output_folder, "unfocused.csv")
    sleep_csv = os.path.join(output_folder, "sleep.csv")

    focused_df.drop(columns=["Label"]).to_csv(focused_csv, index=False)
    unfocused_df.drop(columns=["Label"]).to_csv(unfocused_csv, index=False)
    sleep_df.drop(columns=["Label"]).to_csv(sleep_csv, index=False)

    print("Aggregated CSVs saved (without the 'Label' column):")
    print(f" - Focused: {focused_csv}")
    print(f" - Unfocused: {unfocused_csv}")
    print(f" - Sleep: {sleep_csv}")
