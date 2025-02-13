# dataset_preprocess.py
import os
import random
import boto3
import torch
import concurrent.futures
from pathlib import Path
from typing import List

# Use the new tokenizer.
from Tokenizer_new_arch import StreamingPassBasedWordLevelBPETokenizer

# Global tokenizer instance for each worker.
_GLOBAL_TOKENIZER = None
def _get_tokenizer():
    global _GLOBAL_TOKENIZER
    if _GLOBAL_TOKENIZER is None:
        tok = StreamingPassBasedWordLevelBPETokenizer()
        # Load the pretrained tokenizer model (merges and vocab) from a single JSON file.
        tok.load("neotokenizer_2/tokenizer_model.json")
        _GLOBAL_TOKENIZER = tok
    return _GLOBAL_TOKENIZER

# Define the regions (channels)
REGIONS = ["frontal", "motor_temporal", "parietal_occipital"]

def _process_single_group(args):
    """
    Process one group of files (one base name) corresponding to all regions.
    For each region file:
      - Download the file,
      - Prepend the token '|trial|' and a space,
      - Tokenize its contents using the new tokenizer.
    Then, trim all region token sequences to the same length and concatenate them.
    This produces a shard whose total token length is divisible by the number of regions.
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
            # Prepend the token '|trial|' plus a space to the file content.
            text = '|trial| ' + f.read().strip()
        # Use the new tokenizer's encode method (no alignment here).
        encoded = tokenizer.encode(text)
        tokens_tensor = torch.tensor(encoded, dtype=torch.long)
        region_tokens.append(tokens_tensor)
        try:
            os.remove(local_path)
        except OSError as e:
            print(f"  - Error removing {local_path}: {e}")

    # Ensure all regions have the same token length; trim if necessary.
    lengths = [t.size(0) for t in region_tokens]
    min_length = min(lengths)
    if any(l != min_length for l in lengths):
        print(f"  - Length mismatch in group {base_name}, trimming to {min_length} tokens per region.")
        region_tokens = [t[:min_length] for t in region_tokens]

    # Concatenate tokens from all regions.
    final_tokens = torch.cat(region_tokens, dim=0)
    # Ensure the total length is divisible by the number of regions.
    num_regions = len(REGIONS)
    total_length = final_tokens.size(0)
    if total_length % num_regions != 0:
        new_length = (total_length // num_regions) * num_regions
        final_tokens = final_tokens[:new_length]

    shard_id = group_index - 1
    shard_path = os.path.join(local_data_dir, f"{shard_prefix}_{split_name}_{shard_id}.pt")
    # Save only the tokens; the model will infer channel structure from token order.
    torch.save({'tokens': final_tokens}, shard_path)
    print(f"  - Shard saved: {shard_path}")
    return shard_path

def download_and_preprocess_s3(bucket_name: str, s3_prefix: str, local_data_dir: str,
                               shard_prefix: str = "shard", limit_files: int = None,
                               val_ratio: float = 0.1):
    """
    This function performs the following steps:
      - Lists all *_quantized_coeffs_*.txt files in S3 under the specified prefix.
      - Groups files by their base name (common part before '_quantized_coeffs_'),
        so that each group represents one subject/instance.
      - Filters groups so that only those with files for all defined regions (channels) are retained.
      - Splits the groups into training and validation splits based on val_ratio.
      - Processes each group in parallel:
          For each group, it downloads the corresponding files from S3, prepends the token '|trial|'
          to the contents of each file, tokenizes the text using the new StreamingPassBasedWordLevelBPETokenizer,
          trims the token sequences to the same length, concatenates them (in the defined order of regions),
          and saves the result as a shard.
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

    # Retain only groups that have exactly the expected number of regions.
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
