import os
import random

import torch
from tokenizer2 import BPE_RLE_Tokenizer as Tokenizer, apply_alignment_to_channels


def VAL_preprocess_multi(
        source_dirs: list,
        target_data_dir: str,
        shard_prefix: str = "shard",
        limit_files: int = None
):
    """
    - Lists *_coeffs.txt files in each directory in `source_dirs`.
    - Pairs each with its corresponding *_channels.txt file.
    - Shuffles the collected file pairs.
    - Tokenizes and aligns the data.
    - Saves the processed data as .pt shards in `target_data_dir`.

    Parameters:
      source_dirs (list): List of directories containing the *_coeffs.txt and *_channels.txt files.
      target_data_dir (str): Directory where the processed .pt shards will be saved.
      shard_prefix (str): Prefix for the shard filenames.
      limit_files (int, optional): Maximum number of file pairs to process. If None, process all.
    """
    # Ensure the target directory exists
    os.makedirs(target_data_dir, exist_ok=True)

    # Collect all file pairs from the given source directories
    file_pairs = []
    for source_dir in source_dirs:
        print(f"Processing source directory: {source_dir}")
        if not os.path.isdir(source_dir):
            print(f"Warning: {source_dir} is not a valid directory. Skipping.")
            continue

        # List all *_coeffs.txt files in the current directory
        all_coeffs_files = [f for f in os.listdir(source_dir) if f.endswith('_coeffs.txt')]
        print(f"  Found {len(all_coeffs_files)} coeffs files in {source_dir}.")

        for coeffs_file in all_coeffs_files:
            channels_file = coeffs_file.replace('_coeffs.txt', '_channels.txt')
            coeffs_path = os.path.join(source_dir, coeffs_file)
            channels_path = os.path.join(source_dir, channels_file)

            if os.path.isfile(channels_path):
                file_pairs.append((coeffs_path, channels_path))
            else:
                print(f"  Warning: Channels file {channels_file} not found for {coeffs_file} in {source_dir}.")

    total_pairs_found = len(file_pairs)
    print(f"\nTotal file pairs collected from all source directories: {total_pairs_found}")

    # Shuffle the file pairs so the ordering is random
    random.shuffle(file_pairs)

    # If a limit is specified, keep only that many pairs (after shuffling)
    if limit_files is not None:
        file_pairs = file_pairs[:limit_files]
        print(f"Limiting processing to {limit_files} file pairs.")

    if not file_pairs:
        raise ValueError("No valid coeffs/channels file pairs found in the provided source directories.")

    # Load the tokenizer
    tokenizer = Tokenizer()
    tokenizer.load_merges("neo_tokenizer/merges.json")
    tokenizer.load_vocab("neo_tokenizer/vocab.json")
    print("Tokenizer loaded successfully.\n")

    # Process each file pair and save them as shards
    shard_id = 0
    total_pairs = len(file_pairs)
    for i, (coeffs_path, channels_path) in enumerate(file_pairs, start=1):
        print(f"[PROCESSING] Pair {i}/{total_pairs}:")
        print(f"  - Coeffs: {coeffs_path}")
        print(f"  - Channels: {channels_path}")

        # Read and tokenize the coefficients file
        with open(coeffs_path, 'r', encoding='utf-8') as f:
            text = f.read()
        raw_tokens = text.strip().split()
        raw_tokens.insert(0, "|trial|")
        encoded, pos = tokenizer.encode_with_alignment(raw_tokens, as_ids=True)
        tokens_tensor = torch.tensor(encoded, dtype=torch.long)

        # Read and process the channels file
        with open(channels_path, 'r', encoding='utf-8') as f:
            chan_text = f.read().strip().split()
        chan_text.insert(0, "1")
        final_channels = apply_alignment_to_channels(chan_text, pos)
        channels_tensor = torch.tensor([int(x) for x in final_channels], dtype=torch.long)

        # Ensure tokens and channels are aligned
        if len(tokens_tensor) != len(channels_tensor):
            raise ValueError(f"Token/channel length mismatch for files {coeffs_path} and {channels_path}.")

        # Save the processed data as a shard (with original file pair info)
        shard_filename = f"{shard_prefix}_{shard_id}.pt"
        shard_path = os.path.join(target_data_dir, shard_filename)
        torch.save({
            'tokens': tokens_tensor,
            'channels': channels_tensor,
            'original_pair': (os.path.basename(coeffs_path), os.path.basename(channels_path))
        }, shard_path)
        print(f"  - Shard saved: {shard_path}\n")
        shard_id += 1

    print(f"Saved {shard_id} shards in '{target_data_dir}'.")
    print("Finished preprocessing.")


# Example usage:
source_directories = [
    "output-2536267",
    "output-3745593",
    "output-4385970",
    "output-4518754",
    "output-5016646",
    "output-515278",
    "output-7650679",
    "output-10518106",
    "output-10803229",
    "output-12734987",
    "output-1199011",
]

VAL_preprocess_multi(
    source_dirs=source_directories,
    target_data_dir="output_finetue/shards",
    shard_prefix="shard",
    limit_files=None  # Set to None to process all file pairs
)
