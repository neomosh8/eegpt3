import os
import glob
import multiprocessing
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from VQCAE_TOKENIZER import VQCAETokenizer
from create_text_files_from_csv_3_optimized import create_regional_bipolar_channels

# Global variable for tokenizer instance
_GLOBAL_TOKENIZER = None


def init_worker(tokenizer_path):
    """Initialize the tokenizer once for each worker process"""
    global _GLOBAL_TOKENIZER
    print(f"Initializing tokenizer in process {os.getpid()}")
    _GLOBAL_TOKENIZER = VQCAETokenizer(model_path=tokenizer_path)


def process_csv_file_for_bci(csv_file, output_dir, overlap_percent=50, window_length_sec=2.0):
    """Process a local CSV file using the global tokenizer with overlapping windows"""
    global _GLOBAL_TOKENIZER
    if _GLOBAL_TOKENIZER is None:
        return os.path.basename(csv_file), 0, True, "Tokenizer not initialized"

    # Extract base name from file path
    base_name = os.path.splitext(os.path.basename(csv_file))[0]

    try:
        # Import the VQCAETokenizer class (definition from previous code)
        from VQCAE_TOKENIZER import (VQCAETokenizer)

        # --------------------------------------------------------------------------------
        # Import your custom wavelet/quantization utils (or define them here if needed)
        from utils import (
            wavelet_decompose_window,
            wavelet_reconstruct_window,
            quantize_number,
            dequantize_number,
            pwelch_z,
            call_gpt_for_instructions,
            preprocess_data,
            calculate_sps,
            list_s3_folders,
            list_csv_files_in_folder
        )

        # Read CSV into a DataFrame
        import pandas as pd
        import numpy as np

        print(f"Processing '{base_name}'")
        df = pd.read_csv(csv_file)

        # For BCI data, we don't need GPT to analyze it
        # We'll use predefined channel handling
        all_columns = list(df.columns)

        # Remove 'timestamp' and 'trial' columns if they exist
        channels_to_drop = ['timestamp', 'trial']
        valid_channels = [col for col in df.columns if col not in channels_to_drop]

        # Calculate original sampling rate
        original_sps = calculate_sps(csv_file)

        # Create regional bipolar channels
        regional_bipolar = create_regional_bipolar_channels(df, channels_to_drop)
        channels = list(regional_bipolar.keys())
        data_2d = np.vstack([regional_bipolar[ch] for ch in channels])

        # Preprocess data
        preprocessed_data, new_sps = preprocess_data(data_2d, original_sps)
        regional_preprocessed = {ch: preprocessed_data[i, :] for i, ch in enumerate(channels)}
        new_sps_val = new_sps

        # Standardize the signals
        for key in regional_preprocessed:
            signal = regional_preprocessed[key]
            mean_signal = np.mean(signal)
            std_signal = np.std(signal) if np.std(signal) > 0 else 1e-8
            regional_preprocessed[key] = (signal - mean_signal) / std_signal

        # Calculate window parameters
        min_length = min(len(regional_preprocessed[region]) for region in regional_preprocessed)
        if min_length == 0:
            return base_name, 0, True, "No valid data in regional channels"

        # Use the windowing logic from paste-2.txt
        n_window_samples = int(window_length_sec * new_sps_val)

        # Calculate step size based on overlap percentage
        step_size = int(n_window_samples * (1 - overlap_percent / 100))

        # Calculate number of windows with overlap
        num_windows = (min_length - n_window_samples) // step_size + 1 if step_size > 0 else 1

        # Compute window statistics for artifact rejection
        window_stats = []
        for i in range(num_windows):
            window_start = i * step_size
            window_end = window_start + n_window_samples
            window_data = np.vstack([regional_preprocessed[region][window_start:window_end]
                                     for region in regional_preprocessed])
            window_mean = np.mean(window_data)
            window_stats.append(window_mean)

        window_stats = np.array(window_stats)
        window_mu = np.mean(window_stats)
        window_sigma = np.std(window_stats) if np.std(window_stats) > 0 else 1e-8
        z_scores = (window_stats - window_mu) / window_sigma

        keep_indices = np.where(np.abs(z_scores) <= 2.0)[0]  # Z threshold of 2.0
        rejected_indices = np.where(np.abs(z_scores) > 2.0)[0]
        discarded_count = len(rejected_indices)
        print(f"Discarded {discarded_count} windows out of {num_windows} due to artifact rejection (|Z| > 2.0).")

        # Initialize a single token list
        token_list = []

        # Process all windows with overlap
        for i in range(num_windows):
            # Skip windows that didn't pass artifact rejection
            if i not in keep_indices:
                continue

            window_start = i * step_size
            window_end = window_start + n_window_samples
            window_data = np.vstack([
                regional_preprocessed[region][window_start:window_end]
                for region in regional_preprocessed
            ])

            # Perform wavelet decomposition
            decomposed_channels, _, _, _ = wavelet_decompose_window(
                window_data,
                wavelet='cmor1.5-1.0',
                scales=None,
                normalization=True,
                sampling_period=1.0 / new_sps_val,
                verbose=False
            )

            # Check if we have the expected number of channels
            if len(decomposed_channels) < len(regional_preprocessed):
                continue

            # Combine the decomposed channels into a single 3-channel image
            combined_image = np.stack([
                decomposed_channels[idx] for idx, region in enumerate(regional_preprocessed.keys())
            ], axis=0)

            # Encode the combined image with the global tokenizer
            token_indices = _GLOBAL_TOKENIZER.encode(combined_image)

            # Store the flattened token indices
            token_list.append(token_indices.flatten())

        # Check if we have any tokens
        if not token_list:
            return base_name, 0, True, "No valid windows after artifact rejection"

        # Concatenate all tokens
        all_tokens = torch.cat(token_list)

        # Add EOS token
        eos_token = torch.tensor([_GLOBAL_TOKENIZER.get_eos_token()], device=all_tokens.device)
        all_tokens_with_eos = torch.cat([all_tokens, eos_token])

        # Create output filename
        output_file = f"{base_name}_tokens.pt"
        output_path = os.path.join(output_dir, output_file)

        # Save the tensor
        torch.save(all_tokens_with_eos, output_path)
        total_tokens = len(all_tokens_with_eos)
        print(f"Saved {total_tokens} tokens to {output_path}")

        return base_name, total_tokens, False, f"Processed {len(keep_indices)}/{num_windows} windows"

    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        return base_name, 0, True, f"Error: {str(e)}"


def tokenize_bci_dataset(csv_dir="processed_data", output_dir="tokenized_bci_data",
                         tokenizer_model_path="output/vqcae_final.pt",
                         overlap_percent=50, window_length_sec=2.0):
    """
    Tokenize all CSV files in the specified directory using the VQCAE tokenizer

    Args:
        csv_dir: Directory containing processed BCI CSV files
        output_dir: Directory to save tokenized data
        tokenizer_model_path: Path to the trained VQCAE model
        overlap_percent: Percentage of overlap between consecutive windows (default: 50)
        window_length_sec: Length of each window in seconds (default: 2.0)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all CSV files
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
        return []

    print(f"Found {len(csv_files)} CSV files to process")
    print(f"Using window length: {window_length_sec}s with {overlap_percent}% overlap")

    # Process files in parallel
    results = []
    max_workers = max(multiprocessing.cpu_count() // 2, 1)

    print(f"Starting processing pool with {max_workers} workers")
    with ProcessPoolExecutor(max_workers=max_workers,
                             initializer=init_worker,
                             initargs=(tokenizer_model_path,)) as executor:
        futures = {}
        for i, csv_file in enumerate(csv_files, start=1):
            future = executor.submit(
                process_csv_file_for_bci,
                csv_file,
                output_dir,
                overlap_percent,
                window_length_sec
            )
            futures[future] = i

        for future in tqdm(as_completed(futures), total=len(futures), desc="Tokenizing files"):
            idx = futures[future]
            csvfile = csv_files[idx - 1]
            try:
                res = future.result()
                results.append(res)
                base, token_count, skipped, reason = res
                print(
                    f"[{idx}/{len(csv_files)}] Done: {csvfile} -> {base}, tokens: {token_count}, skipped: {skipped}, reason: {reason}")
            except Exception as e:
                print(f"Error processing {csvfile}: {e}")

    # Create label mapping file for classification
    create_label_mapping(csv_files, output_dir)

    return results


def create_label_mapping(csv_files, output_dir):
    """Create a label mapping file for classification tasks"""
    import json
    import re

    label_mapping = {}
    class_counts = {}

    # Extract class names from filenames
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        match = re.search(r'(.+)_(\w+)\.csv', filename)

        if match:
            dataset_id = match.group(1)
            class_name = match.group(2)

            # Skip rest class
            if class_name == "rest":
                continue

            # Add to label mapping if it's a new class
            if class_name not in label_mapping:
                label_mapping[class_name] = len(label_mapping)

            # Count instances per class
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1

    # Save label mapping
    mapping_file = os.path.join(output_dir, "label_mapping.json")
    with open(mapping_file, "w") as f:
        json.dump({
            "label_mapping": label_mapping,
            "class_counts": class_counts
        }, f, indent=4)

    print(f"Created label mapping with {len(label_mapping)} classes: {label_mapping}")
    print(f"Class counts: {class_counts}")


if __name__ == "__main__":
    # Set paths
    csv_dir = "processed_data"
    output_dir = "tokenized_bci_data"
    tokenizer_model_path = "output/vqcae_final.pt"  # Update this with your actual model path

    # Window parameters
    overlap_percent = 50  # 50% overlap between windows
    window_length_sec = 0.5  # 2 second windows

    # Run tokenization
    results = tokenize_bci_dataset(
        csv_dir,
        output_dir,
        tokenizer_model_path,
        overlap_percent,
        window_length_sec
    )

    # Summarize results
    total_files = len(results)
    total_tokens = sum(token_count for (_, token_count, _, _) in results)
    skipped_files = [(base, reason) for (base, _, skipped, reason) in results if skipped]

    print("\nTokenization Summary:")
    print(f"Total files processed: {total_files}")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Files skipped: {len(skipped_files)}")

    if skipped_files:
        print("\nSkipped files:")
        for base, reason in skipped_files:
            print(f"  - {base}: {reason}")