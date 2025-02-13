# dataset_preprocess.py
import os
import random
import boto3
import torch
import concurrent.futures
from pathlib import Path
from typing import List

from tokenizer2 import BPE_RLE_Tokenizer as Tokenizer

# Global tokenizer instance for each worker.
_GLOBAL_TOKENIZER = None


def _get_tokenizer():
    global _GLOBAL_TOKENIZER
    if _GLOBAL_TOKENIZER is None:
        tok = Tokenizer()
        tok.load_merges("neo_tokenizer/merges.json")
        tok.load_vocab("neo_tokenizer/vocab.json")
        _GLOBAL_TOKENIZER = tok
    return _GLOBAL_TOKENIZER


# Define the regions (which now represent channels)
REGIONS = ["frontal", "motor_temporal", "parietal_occipital"]


def _process_single_group(args):
    """
    Process one group of files (one base name) corresponding to all regions.
    For each region file:
      - Download the file,
      - Tokenize its contents,
      - Store the resulting tokens.
    Then, trim all region token sequences to the same length,
    concatenate them (in the order given by REGIONS),
    and create a channels tensor labeling tokens with region indices.
    This produces a shard whose length is divisible by num_channels.
    """
    (base_name, files, shard_prefix, local_data_dir, bucket_name,
     split_name, group_index, total_groups) = args

    s3 = boto3.client('s3')
    tokenizer = _get_tokenizer()
    print(f"[{split_name.upper()}] Processing group {group_index}/{total_groups} for base '{base_name}'")

    region_tokens = []
    for region in REGIONS:
        key = None
        # Look for the file that ends with _{region}.txt
        for f in files:
            if f.endswith(f"_{region}.txt"):
                key = f
                break
        if key is None:
            print(f"Warning: Missing file for region {region} in group {base_name}")
            return None
        local_path = os.path.join(local_data_dir, os.path.basename(key))
        print(f"  - Downloading file for region {region}: {key}")
        s3.download_file(bucket_name, key, local_path)
        with open(local_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        raw_tokens = text.split()
        # (Optionally, insert a marker token if desired)
        encoded, pos = tokenizer.encode_with_alignment(raw_tokens, as_ids=True)
        tokens_tensor = torch.tensor(encoded, dtype=torch.long)
        region_tokens.append(tokens_tensor)
        try:
            os.remove(local_path)
        except OSError as e:
            print(f"  - Error removing {local_path}: {e}")

    # Ensure all regions have the same length; trim if necessary.
    lengths = [t.size(0) for t in region_tokens]
    min_length = min(lengths)
    if any(l != min_length for l in lengths):
        print(f"  - Length mismatch in group {base_name}, trimming to {min_length} tokens per region.")
        region_tokens = [t[:min_length] for t in region_tokens]

    # Concatenate tokens from all regions.
    final_tokens = torch.cat(region_tokens, dim=0)
    # Build a channels tensor: first min_length tokens are channel 0, next min_length tokens channel 1, etc.
    channel_list = [torch.full((min_length,), i, dtype=torch.long) for i in range(len(REGIONS))]
    final_channels = torch.cat(channel_list, dim=0)

    # Ensure the total length is divisible by the number of channels.
    total_length = final_tokens.size(0)
    num_channels = len(REGIONS)
    if total_length % num_channels != 0:
        new_length = (total_length // num_channels) * num_channels
        final_tokens = final_tokens[:new_length]
        final_channels = final_channels[:new_length]

    shard_id = group_index - 1
    shard_path = os.path.join(local_data_dir, f"{shard_prefix}_{split_name}_{shard_id}.pt")
    torch.save({'tokens': final_tokens, 'channels': final_channels}, shard_path)
    print(f"  - Shard saved: {shard_path}")
    return shard_path


def download_and_preprocess_s3(bucket_name: str, s3_prefix: str, local_data_dir: str,
                               shard_prefix: str = "shard", limit_files: int = None,
                               val_ratio: float = 0.1):
    """
    - Lists all *_quantized_coeffs_*.txt files in S3 under the given prefix.
    - Groups files by base name (i.e. common part before '_quantized_coeffs_').
    - Each valid group must contain files for all regions (channels).
    - Splits the groups into train and validation splits based on val_ratio.
    - Processes each group in parallel and saves shards.
    """
    os.makedirs(local_data_dir, exist_ok=True)
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)

    all_files = []
    for page in page_iterator:
        contents = page.get('Contents', [])
        for obj in contents:
            key = obj['Key']
            if key.endswith('.txt') and '_quantized_coeffs_' in key:
                all_files.append(key)

    if limit_files is not None:
        all_files = all_files[:limit_files]

    # Group files by base_name (everything before '_quantized_coeffs_')
    groups = {}
    for key in all_files:
        parts = key.split('_quantized_coeffs_')
        if len(parts) != 2:
            continue
        base_name = parts[0]
        groups.setdefault(base_name, []).append(key)

    # Keep only groups that have exactly the expected number of regions.
    valid_groups = {base: files for base, files in groups.items() if len(files) == len(REGIONS)}
    print(f"Found {len(valid_groups)} valid groups (each with {len(REGIONS)} regions).")
    if not valid_groups:
        raise ValueError("No valid groups found.")

    group_items = list(valid_groups.items())
    random.shuffle(group_items)

    total_groups = len(group_items)
    val_count = int(total_groups * val_ratio)
    val_groups = group_items[:val_count]
    train_groups = group_items[val_count:]
    print(f"Total groups: {total_groups} | Train: {len(train_groups)} | Val: {len(val_groups)}")

    def process_split(groups_list, split_name):
        if not groups_list:
            print(f"No groups to process for {split_name}.")
            return
        tasks = []
        total = len(groups_list)
        for i, (base_name, files) in enumerate(groups_list, start=1):
            tasks.append((base_name, files, shard_prefix, local_data_dir,
                          bucket_name, split_name, i, total))
        max_workers = os.cpu_count() or 1
        print(f"Processing {total} items for '{split_name}' split with {max_workers} workers...")
        shard_paths = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for shard_path in executor.map(_process_single_group, tasks):
                if shard_path is not None:
                    shard_paths.append(shard_path)
        print(f"Finished saving {len(shard_paths)} shards for split '{split_name}'.")

    process_split(train_groups, "train")
    process_split(val_groups, "val")
    print("Finished preprocessing.")


if __name__ == "__main__":
    BUCKET_NAME = "dataframes--use1-az6--x-s3"
    S3_PREFIX = "output/"
    LOCAL_DIR = "./local_shards"
    download_and_preprocess_s3(bucket_name=BUCKET_NAME,
                               s3_prefix=S3_PREFIX,
                               local_data_dir=LOCAL_DIR,
                               shard_prefix="mydata",
                               limit_files=None,
                               val_ratio=0.1)
