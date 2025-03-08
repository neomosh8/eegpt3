import numpy as np
import pandas as pd
import os
import glob
import re
from tqdm import tqdm


def load_bci_dataset(base_dir):
    """
    Load BCI Competition IV Dataset 1 files with improved error handling
    """
    # Find all dataset files
    cnt_files = sorted(glob.glob(os.path.join(base_dir, "*_cnt.txt")))
    mrk_files = sorted(glob.glob(os.path.join(base_dir, "*_mrk.txt")))
    nfo_files = sorted(glob.glob(os.path.join(base_dir, "*_nfo.txt")))

    if not cnt_files:
        print(f"No *_cnt.txt files found in {base_dir}. Check the directory path.")
        return []

    datasets = []

    for i, cnt_file in enumerate(cnt_files):
        # Extract dataset identifier from filename
        dataset_id = os.path.basename(cnt_file).split('_cnt.txt')[0]
        print(f"Loading dataset {dataset_id}...")

        # Load continuous EEG data
        cnt_data = np.loadtxt(cnt_file)
        print(f"  Loaded EEG data shape: {cnt_data.shape}")

        # Find corresponding NFO file
        matching_nfo_files = [f for f in nfo_files if dataset_id in f]
        if not matching_nfo_files:
            print(f"  Warning: No NFO file found for {dataset_id}, skipping.")
            continue

        nfo_file = matching_nfo_files[0]

        # Load the entire NFO file content for debugging
        with open(nfo_file, 'r') as f:
            nfo_content = f.read()

        # Print a sample of the NFO content to debug
        print(f"  NFO file sample content (first 200 chars):\n{nfo_content[:200]}...")

        # Try different patterns to extract channel information
        channel_names = None

        # Try to find a line with 'clab' that might contain channel info
        for line in nfo_content.split('\n'):
            if 'clab' in line.lower():
                print(f"  Found clab line: {line}")

                # Try various patterns
                patterns = [
                    r'clab\s*:\s*\[(.*?)\]',  # Format: clab : [ch1 ch2 ch3]
                    r'clab\s*:\s*(.*)',  # Format: clab : ch1 ch2 ch3
                    r'clab\s*=\s*\{(.*?)\}',  # Format: clab = {ch1 ch2 ch3}
                    r'clab\s*=\s*(.*)'  # Format: clab = ch1 ch2 ch3
                ]

                for pattern in patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        channel_names = match.group(1).split()
                        break

                if channel_names:
                    break

        # If we still don't have channel names, count the columns and use generic names
        if not channel_names:
            print("  Could not extract channel names using regex patterns.")
            print("  Creating generic channel names based on data dimensions.")
            num_channels = cnt_data.shape[1]
            channel_names = [f"CH{i + 1}" for i in range(num_channels)]

        print(f"  Found {len(channel_names)} channels: {channel_names[:5]}...")

        # Extract sampling rate
        fs = None
        for line in nfo_content.split('\n'):
            if 'fs' in line.lower():
                print(f"  Found fs line: {line}")
                match = re.search(r'fs\s*[:=]\s*(\d+)', line)
                if match:
                    fs = float(match.group(1))
                    break

        if fs is None:
            print("  Could not extract sampling rate. Using default of 100 Hz.")
            fs = 100.0

        print(f"  Sampling rate: {fs} Hz")

        # Extract class information
        classes = []
        for line in nfo_content.split('\n'):
            if 'classes' in line.lower():
                print(f"  Found classes line: {line}")
                # Try different patterns
                patterns = [
                    r'classes\s*:\s*\[(.*?)\]',  # Format: classes : [class1 class2]
                    r'classes\s*:\s*(.*)',  # Format: classes : class1 class2
                    r'classes\s*=\s*\{(.*?)\}',  # Format: classes = {class1 class2}
                    r'classes\s*=\s*(.*)'  # Format: classes = class1 class2
                ]

                for pattern in patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        classes = match.group(1).split()
                        break

                if classes:
                    break

        if not classes:
            print("  Could not extract class information. Using default classes.")
            classes = ['left', 'right']  # Default motor imagery classes

        print(f"  Classes: {classes}")

        # Determine if this is calibration or evaluation data
        is_calibration = False
        markers = None
        corresponding_mrk_file = [f for f in mrk_files if dataset_id in f]

        if corresponding_mrk_file:
            is_calibration = True
            # Load marker information
            try:
                markers = np.loadtxt(corresponding_mrk_file[0])
                if markers.ndim == 1:  # Handle case with only one marker
                    markers = markers.reshape(1, -1)
                print(f"  Loaded {len(markers)} markers")
            except Exception as e:
                print(f"  Error loading markers: {e}")
                markers = None

        datasets.append({
            'id': dataset_id,
            'cnt_data': cnt_data,
            'channel_names': channel_names,
            'fs': fs,
            'classes': classes,
            'is_calibration': is_calibration,
            'markers': markers
        })

    return datasets


def process_calibration_data(dataset, output_dir, window_size=4.0, offset=0.0):
    """
    Process calibration data into CSV files by class

    Args:
        dataset: Dictionary containing dataset information
        output_dir: Directory to save CSV files
        window_size: Duration of each trial in seconds (default is 4.0 as per description)
        offset: Time offset from cue in seconds (to skip initial transient)
    """
    os.makedirs(output_dir, exist_ok=True)

    cnt_data = dataset['cnt_data']
    channel_names = dataset['channel_names']
    fs = dataset['fs']
    markers = dataset['markers']
    classes = dataset['classes']

    print(f"Processing dataset {dataset['id']}")
    print(f"  Data shape: {cnt_data.shape}")
    print(f"  Number of channels: {len(channel_names)}")
    print(f"  Classes: {classes}")

    if markers is None or len(markers) == 0:
        print("  No markers found, cannot process calibration data.")
        return

    # Create dataframe with timestamps
    timestamps = np.arange(cnt_data.shape[0]) / fs

    # Convert EEG values to microvolts
    cnt_data = 0.1 * cnt_data  # As per the instructions in the dataset description

    # Create full dataframe with all channels
    full_df = pd.DataFrame(cnt_data, columns=channel_names)
    full_df.insert(0, 'timestamp', timestamps)

    # Check if we have at least 2 unique class codes in markers
    unique_class_codes = np.unique(markers[:, 1])
    if len(unique_class_codes) < 2:
        print(f"  Warning: Only found {len(unique_class_codes)} unique class codes: {unique_class_codes}")
        print("  Will try to process anyway.")

    # Extract trials for each class
    for i, class_code in enumerate(unique_class_codes):
        # If we have more class codes than class names, use generic names
        if i < len(classes):
            class_name = classes[i]
        else:
            class_name = f"class{int(class_code)}"

        print(f"  Processing class: {class_name} (code {class_code})")

        # Get all trials for this class
        class_trials = markers[markers[:, 1] == class_code]
        print(f"  Found {len(class_trials)} trials for class {class_name}")

        class_data = []

        for trial_idx, (pos, _) in enumerate(class_trials):
            # Calculate start and end sample
            start_sample = int(pos + offset * fs)
            end_sample = int(start_sample + window_size * fs)

            # Check if within data bounds
            if end_sample <= cnt_data.shape[0]:
                # Extract trial data
                trial_data = full_df.iloc[start_sample:end_sample].copy()
                trial_data['trial'] = trial_idx + 1
                class_data.append(trial_data)
            else:
                print(f"  Warning: Trial {trial_idx} exceeds data bounds, skipping.")

        if class_data:
            # Combine all trials for this class
            class_df = pd.concat(class_data, ignore_index=True)

            # Save to CSV
            csv_filename = f"{dataset['id']}_{class_name}.csv"
            csv_path = os.path.join(output_dir, csv_filename)
            class_df.to_csv(csv_path, index=False)
            print(f"  Saved {csv_path} with {len(class_data)} trials, {class_df.shape[0]} rows")
        else:
            print(f"  No valid trials for class {class_name}, skipping CSV creation.")

    # Create a dataframe for rest/no control periods (intermitting periods)
    # Intermitting periods are between trials
    print("  Processing rest/no control periods")
    rest_data = []

    # Identify rest periods - we'll consider time between trials as rest
    all_trial_pos = markers[:, 0].astype(int)
    all_trial_pos.sort()

    # Add rest periods between trials
    rest_periods_found = 0
    for i in range(len(all_trial_pos) - 1):
        start_sample = int(all_trial_pos[i] + window_size * fs)  # End of current trial
        end_sample = int(all_trial_pos[i + 1])  # Start of next trial

        # Check if there's a valid rest period
        if end_sample - start_sample > fs:  # At least 1 second
            rest_period = full_df.iloc[start_sample:end_sample].copy()
            rest_period['trial'] = i + 1
            rest_data.append(rest_period)
            rest_periods_found += 1

    print(f"  Found {rest_periods_found} rest periods between trials")

    if rest_data:
        # Combine all rest periods
        rest_df = pd.concat(rest_data, ignore_index=True)

        # Save to CSV
        csv_filename = f"{dataset['id']}_rest.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        rest_df.to_csv(csv_path, index=False)
        print(f"  Saved {csv_path} with {len(rest_data)} periods, {rest_df.shape[0]} rows")
    else:
        print("  No valid rest periods found.")

    return f"Processed {dataset['id']} into CSV files by class"


def main():
    # Directory where the BCI Competition IV dataset files are located
    base_dir = "/Volumes/Untitled/BCICIV_1_asc"
    output_dir = "/Volumes/Untitled/processed_data"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load all datasets
    print("Loading BCI datasets...")
    datasets = load_bci_dataset(base_dir)

    if not datasets:
        print("No datasets were successfully loaded. Check the input files and directory.")
        return

    print(f"Successfully loaded {len(datasets)} datasets.")

    # Process each calibration dataset
    for dataset in datasets:
        if dataset['is_calibration']:
            print(f"\nProcessing calibration dataset {dataset['id']}...")
            process_calibration_data(dataset, output_dir)
        else:
            print(f"\nSkipping evaluation dataset {dataset['id']} (no markers)")

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()