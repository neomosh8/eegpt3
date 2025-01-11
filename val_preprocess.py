import os
import random

import torch
from tokenizer2 import BPE_RLE_Tokenizer as Tokenizer, apply_alignment_to_channels


def VAL_preprocess(
        source_dir: str,
        target_data_dir: str,
        shard_prefix: str = "shard",
        limit_files: int = None
):
    """
    - Lists *_coeffs.txt files in `source_dir`.
    - Pairs them up with corresponding *_channels.txt files.
    - Shuffles the file pairs.
    - Tokenizes and aligns the data.
    - Saves the processed data as .pt shards in `target_data_dir`.

    Parameters:
    - source_dir (str): Directory containing the *_coeffs.txt and *_channels.txt files.
    - target_data_dir (str): Directory where the processed .pt shards will be saved.
    - shard_prefix (str): Prefix for the shard filenames.
    - limit_files (int, optional): Maximum number of file pairs to process. Processes all if None.
    """

    # Ensure the target directory exists
    os.makedirs(target_data_dir, exist_ok=True)

    # List all *_coeffs.txt files in the source directory
    all_coeffs_files = [
        f for f in os.listdir(source_dir)
        if f.endswith('_coeffs.txt')
    ]
    print(f"Found {len(all_coeffs_files)} coeffs files in {source_dir}.")

    # Apply file limit if specified
    if limit_files is not None:
        all_coeffs_files = all_coeffs_files[:limit_files]
        print(f"Limiting to the first {limit_files} files.")

    # Pair each *_coeffs.txt with its corresponding *_channels.txt
    file_pairs = []
    for coeffs_file in all_coeffs_files:
        channels_file = coeffs_file.replace('_coeffs.txt', '_channels.txt')
        coeffs_path = os.path.join(source_dir, coeffs_file)
        channels_path = os.path.join(source_dir, channels_file)

        if os.path.isfile(channels_path):
            file_pairs.append((coeffs_path, channels_path))
        else:
            print(f"Warning: Channels file {channels_file} not found for {coeffs_file}.")

    if not file_pairs:
        raise ValueError("No valid coeffs/channels file pairs found.")

    print(f"Total valid pairs: {len(file_pairs)}")

    # Shuffle the file pairs
    random.shuffle(file_pairs)
    print("Shuffled the file pairs.")

    # Load the tokenizer
    tokenizer = Tokenizer()
    tokenizer.load_merges("neo_tokenizer/merges.json")
    tokenizer.load_vocab("neo_tokenizer/vocab.json")
    print("Tokenizer loaded successfully.")

    # Initialize shard ID
    shard_id = 0
    total_pairs = len(file_pairs)

    # Process each file pair
    for i, (coeffs_path, channels_path) in enumerate(file_pairs, start=1):
        print(f"[TRAIN] Processing pair {i}/{total_pairs}:")
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
            raise ValueError(f"Token / channel length mismatch in pair {coeffs_path} and {channels_path}.")

        # Save the processed data as a shard (added original pair info below)
        shard_filename = f"{shard_prefix}_train_{shard_id}.pt"
        shard_path = os.path.join(target_data_dir, shard_filename)
        torch.save({
            'tokens': tokens_tensor,
            'channels': channels_tensor,
            'original_pair': (os.path.basename(coeffs_path), os.path.basename(channels_path))
        }, shard_path)
        print(f"  - Shard saved: {shard_path}")
        shard_id += 1

    print(f"Saved {shard_id} train shards in '{target_data_dir}'.")
    print("Finished preprocessing.")

# Example usage:
VAL_preprocess(
    source_dir="validation_datasets_imageNet",
    target_data_dir="validation_datasets_imageNet/shards",
    shard_prefix="shard",
    limit_files=200000  # or None to process all files
)
