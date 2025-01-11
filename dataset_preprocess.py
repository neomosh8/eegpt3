# dataset_preprocess.py
import os
import random
import boto3
import json
import torch
import concurrent.futures
from pathlib import Path
from typing import List

from tokenizer2 import BPE_RLE_Tokenizer as Tokenizer, apply_alignment_to_channels


# We'll define a global variable so that each worker can lazy-load
# (and re-use) the tokenizer rather than trying to pickle a single instance.
_GLOBAL_TOKENIZER = None

def _get_tokenizer():
    """
    Lazy-load the tokenizer in each process.
    This avoids trying to pickle a tokenizer object,
    and ensures each process has its own instance.
    """
    global _GLOBAL_TOKENIZER
    if _GLOBAL_TOKENIZER is None:
        tok = Tokenizer()
        tok.load_merges("neo_tokenizer/merges.json")
        tok.load_vocab("neo_tokenizer/vocab.json")
        _GLOBAL_TOKENIZER = tok
    return _GLOBAL_TOKENIZER


def _process_single_pair(args):
    """
    Worker function to handle a single (coeffs_key, channels_key) pair.
    Creates its own S3 client, downloads files, tokenizes, saves shard.
    """
    (
        coeffs_key,
        channels_key,
        shard_prefix,
        local_data_dir,
        bucket_name,
        split_name,
        pair_index,     # 1-based index
        total_pairs     # total number of pairs for that split
    ) = args

    # Create a local S3 client in each process:
    s3 = boto3.client('s3')
    tokenizer = _get_tokenizer()  # get (or load) our tokenizer

    print(f"[{split_name.upper()}] Processing pair {pair_index}/{total_pairs}:")
    print(f"  - Coeffs: {coeffs_key}")
    print(f"  - Channels: {channels_key}")

    # Download local copies
    coeffs_local = os.path.join(local_data_dir, os.path.basename(coeffs_key))
    channels_local = os.path.join(local_data_dir, os.path.basename(channels_key))

    print("  - Downloading coeffs file...")
    s3.download_file(bucket_name, coeffs_key, coeffs_local)
    print("  - Downloading channels file...")
    s3.download_file(bucket_name, channels_key, channels_local)

    # Read & tokenize
    with open(coeffs_local, 'r', encoding='utf-8') as f:
        text = f.read()
    raw_tokens = text.strip().split()
    # Insert a marker token at the start
    raw_tokens.insert(0, "|trial|")
    encoded, pos = tokenizer.encode_with_alignment(raw_tokens, as_ids=True)
    tokens_tensor = torch.tensor(encoded, dtype=torch.long)

    # Channels
    with open(channels_local, 'r', encoding='utf-8') as f:
        chan_text = f.read().strip().split()
    # Insert at the start as well
    chan_text.insert(0, "1")
    # NEW PART
    if len(raw_tokens) != len(chan_text):
        print("########################## Length mismatch in tokens/channels ###################")
        last_chan = int(chan_text[-1])
        # If the last channel is 1, keep alternating [2,1,2,1,...] until lengths match.
        # If the last channel is 2, keep alternating [1,2,1,2,...] until lengths match.
        while len(chan_text) < len(raw_tokens):
            if last_chan == 1:
                next_val = 2
            else:
                next_val = 1
            chan_text.append(str(next_val))
            last_chan = next_val

    final_channels = apply_alignment_to_channels(chan_text, pos)
    channels_tensor = torch.tensor([int(x) - 1 for x in final_channels], dtype=torch.long)

    if len(tokens_tensor) != len(channels_tensor):
        raise ValueError("Token / channel length mismatch.")

    # We can derive a shard_id from pair_index-1 to ensure unique file naming
    shard_id = pair_index - 1
    shard_path = os.path.join(local_data_dir, f"{shard_prefix}_{split_name}_{shard_id}.pt")
    torch.save({'tokens': tokens_tensor, 'channels': channels_tensor}, shard_path)

    print(f"  - Shard saved: {shard_path}")

    # Clean up temp files
    try:
        os.remove(coeffs_local)
        os.remove(channels_local)
        print("  - Temporary files removed.")
    except OSError as e:
        print(f"  - Error removing temporary files: {e}")

    return shard_path  # not strictly necessary, but can be useful


def download_and_preprocess_s3(
    bucket_name: str,
    s3_prefix: str,
    local_data_dir: str,
    shard_prefix: str = "shard",
    limit_files: int = None,
    val_ratio: float = 0.1
):
    """
    - Lists *_coeffs.txt files in S3 under `bucket_name` and `s3_prefix`.
    - Pairs them up with *_channels.txt`.
    - Shuffles them.
    - Splits them into train/val by `val_ratio`.
    - Tokenizes, aligns, and saves them as .pt files in `local_data_dir`, in parallel.
    """

    os.makedirs(local_data_dir, exist_ok=True)

    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)

    all_files = []
    for page_index, page in enumerate(page_iterator, start=1):
        contents = page.get('Contents', [])
        print(f"[Page {page_index}] Found {len(contents)} items in this S3 page.")
        for obj in contents:
            key = obj['Key']
            if key.endswith('_coeffs.txt'):
                all_files.append(key)

    if limit_files is not None:
        all_files = all_files[:limit_files]

    # Pair up coeffs with channels
    file_pairs = []
    for coeffs_key in all_files:
        channels_key = coeffs_key.replace('_coeffs.txt', '_channels.txt')
        try:
            s3.head_object(Bucket=bucket_name, Key=channels_key)
            file_pairs.append((coeffs_key, channels_key))
        except:
            pass

    print(f"Found {len(file_pairs)} Pairs")
    if not file_pairs:
        raise ValueError("No valid coeffs/channels file pairs found.")

    # Shuffle the pairs
    random.shuffle(file_pairs)
    print("shuffled")

    # Train/val split
    val_count = int(len(file_pairs) * val_ratio)
    val_pairs = file_pairs[:val_count]
    train_pairs = file_pairs[val_count:]
    print(f"Total pairs: {len(file_pairs)} | Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    # Parallel processing function for a whole split
    def process_split(pairs, split_name):
        """
        Process all pairs for this split in parallel.
        """
        if not pairs:
            print(f"No pairs to process for {split_name}.")
            return

        tasks = []
        total_pairs = len(pairs)

        for i, (coeffs_key, channels_key) in enumerate(pairs, start=1):
            tasks.append((
                coeffs_key,
                channels_key,
                shard_prefix,
                local_data_dir,
                bucket_name,
                split_name,
                i,           # 1-based index
                total_pairs
            ))

        # Use all available CPUs for max parallelism
        max_workers = os.cpu_count() or 1
        print(f"Processing {len(pairs)} items for '{split_name}' split with {max_workers} workers...")

        # Map over the tasks in parallel
        shard_paths = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for shard_path in executor.map(_process_single_pair, tasks):
                shard_paths.append(shard_path)

        print(f"Finished saving {len(shard_paths)} shards for split '{split_name}'.")

    # Process TRAIN (parallel)
    process_split(train_pairs, "train")

    # Process VAL (parallel)
    process_split(val_pairs, "val")

    print("Finished preprocessing.")


if __name__ == "__main__":
    BUCKET_NAME = "dataframes--use1-az6--x-s3"
    S3_PREFIX = "output/"
    LOCAL_DIR = "./local_shards"

    download_and_preprocess_s3(
        bucket_name=BUCKET_NAME,
        s3_prefix=S3_PREFIX,
        local_data_dir=LOCAL_DIR,
        shard_prefix="mydata",
        limit_files=None,
        val_ratio=0.1  # 10% for validation
    )
