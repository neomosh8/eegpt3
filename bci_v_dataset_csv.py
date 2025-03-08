import numpy as np
import pandas as pd
import os
import glob
import re


def load_bci_dataset(base_dir):
    """
    Load BCI Competition IV Dataset files
    """
    # Find all dataset files
    cnt_files = sorted(glob.glob(os.path.join(base_dir, "BCICIV_calib_*_cnt.txt")))
    mrk_files = sorted(glob.glob(os.path.join(base_dir, "BCICIV_calib_*_mrk.txt")))
    nfo_files = sorted(glob.glob(os.path.join(base_dir, "BCICIV_calib_*_nfo.txt")))

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

        # Load NFO file content
        with open(nfo_file, 'r') as f:
            nfo_content = f.read()

        # Extract sampling rate
        fs = 100  # Default
        fs_match = re.search(r'fs:\s*(\d+)', nfo_content)
        if fs_match:
            fs = float(fs_match.group(1))
        print(f"  Sampling rate: {fs} Hz")

        # Extract channel names - make sure to strip any whitespace or commas
        clab_match = re.search(r'clab:\s*(.*)', nfo_content)
        if clab_match:
            # Split by comma and strip whitespace from each channel name
            channel_names = [ch.strip() for ch in clab_match.group(1).split(',')]
        else:
            # Create generic channel names
            channel_names = [f"CH{i + 1}" for i in range(cnt_data.shape[1])]
        print(f"  Found {len(channel_names)} channels")

        # Extract class information - make sure to strip any whitespace or commas
        classes = ['left', 'right']  # Default
        classes_match = re.search(r'classes:\s*(.*)', nfo_content)
        if classes_match:
            # Split by comma and strip whitespace from each class name
            classes = [cls.strip() for cls in classes_match.group(1).split(',')]
        print(f"  Classes: {classes}")

        # Load marker information
        markers = None
        corresponding_mrk_file = [f for f in mrk_files if dataset_id in f]
        if corresponding_mrk_file:
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
            'markers': markers
        })

    return datasets


def process_calibration_data(dataset, output_dir, window_size=4.0, offset=0.0):
    """
    Process calibration data into CSV files by class
    """
    os.makedirs(output_dir, exist_ok=True)

    cnt_data = dataset['cnt_data']
    channel_names = dataset['channel_names']
    fs = dataset['fs']
    markers = dataset['markers']
    classes = dataset['classes']

    print(f"Processing dataset {dataset['id']}")
    print(f"  Data shape: {cnt_data.shape}")
    print(f"  Classes: {classes}")

    if markers is None or len(markers) == 0:
        print("  No markers found, cannot process calibration data.")
        return

    # Create dataframe with timestamps
    timestamps = np.arange(cnt_data.shape[0]) / fs

    # Convert EEG values to microvolts (if needed)
    cnt_data = 0.1 * cnt_data  # Adjust scaling factor if needed

    # Create full dataframe with all channels
    full_df = pd.DataFrame(cnt_data, columns=channel_names)
    full_df.insert(0, 'timestamp', timestamps)

    # Check marker class codes
    unique_class_codes = np.unique(markers[:, 1])
    print(f"  Found {len(unique_class_codes)} unique class codes: {unique_class_codes}")

    # Map class codes to class names
    class_mapping = {}
    for i, code in enumerate(unique_class_codes):
        if i < len(classes):
            class_mapping[code] = classes[i]
        else:
            class_mapping[code] = f"class{int(code)}"

    # Extract trials for each class
    for class_code, class_name in class_mapping.items():
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

            # Save to CSV - ensure clean filename with no commas
            csv_filename = f"{dataset['id']}_{class_name}.csv"
            csv_path = os.path.join(output_dir, csv_filename)
            class_df.to_csv(csv_path, index=False)
            print(f"  Saved {csv_path} with {len(class_data)} trials, {class_df.shape[0]} rows")
        else:
            print(f"  No valid trials for class {class_name}, skipping CSV creation.")

    # Create a dataframe for rest periods
    print("  Processing rest periods")
    rest_data = []

    # Identify rest periods between trials
    all_trial_pos = markers[:, 0].astype(int)
    all_trial_pos.sort()

    # Add rest periods between trials
    rest_periods_found = 0
    for i in range(len(all_trial_pos) - 1):
        start_sample = int(all_trial_pos[i] + window_size * fs)  # End of current trial
        end_sample = int(all_trial_pos[i + 1])  # Start of next trial

        # Check if there's a valid rest period (at least 1 second)
        if end_sample - start_sample > fs:
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
        if dataset['markers'] is not None:
            print(f"\nProcessing calibration dataset {dataset['id']}...")
            process_calibration_data(dataset, output_dir)
        else:
            print(f"\nSkipping dataset {dataset['id']} (no markers)")

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
