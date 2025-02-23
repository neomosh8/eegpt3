import os
import random
import boto3
import torch
import concurrent.futures
from pathlib import Path

from very_simple_tokenizer import SimpleQuantizerTokenizer

_GLOBAL_TOKENIZER = None
def _get_tokenizer(resolution=80):
    global _GLOBAL_TOKENIZER
    if _GLOBAL_TOKENIZER is None:
        tok = SimpleQuantizerTokenizer(resolution=resolution)
        # Optionally save vocab for inspection or reuse
        vocab_path = "quantizer_vocab.json"
        tok.save_vocab(vocab_path)
        print(f"Tokenizer vocab built and saved to {vocab_path} with {tok.vocab_size()} tokens.")
        _GLOBAL_TOKENIZER = tok
    return _GLOBAL_TOKENIZER

REGIONS = ["frontal", "motor_temporal", "parietal_occipital"]

def _process_single_group(args):
    (base_name, files, shard_prefix, local_data_dir, bucket_name,
     split_name, group_index, total_groups) = args

    s3 = boto3.client('s3')
    tokenizer = _get_tokenizer()
    print(f"[{split_name.upper()}] Processing group {group_index}/{total_groups} for base '{base_name}'")

    region_tokens = {}
    for region in REGIONS:
        key = None
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
            text = '|trial| ' + f.read().strip()
        encoded = tokenizer.encode(text)
        tokens_tensor = torch.tensor(encoded, dtype=torch.long)
        region_tokens[region] = tokens_tensor
        try:
            os.remove(local_path)
        except OSError as e:
            print(f"  - Error removing {local_path}: {e}")

    min_length = min(t.size(0) for t in region_tokens.values())
    if any(t.size(0) != min_length for t in region_tokens.values()):
        print(f"  - Length mismatch in group {base_name}, trimming all channels to {min_length} tokens.")
    for region in region_tokens:
        region_tokens[region] = region_tokens[region][:min_length]

    shard_id = group_index - 1
    shard_path = os.path.join(local_data_dir, f"{shard_prefix}_{split_name}_{shard_id}.pt")
    torch.save(region_tokens, shard_path)
    print(f"  - Shard saved: {shard_path}")
    return shard_path

def download_and_preprocess_s3(bucket_name: str, s3_prefix: str, local_data_dir: str,
                               shard_prefix: str = "shard", limit_files: int = None,
                               val_ratio: float = 0.1, resolution: int = 80):
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

    groups = {}
    for key in all_files:
        parts = key.split('_quantized_coeffs_')
        if len(parts) != 2:
            continue
        base_name = parts[0]
        groups.setdefault(base_name, []).append(key)

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
    S3_PREFIX = "output_emotiv/"
    LOCAL_DIR = "./local_shards_val"
    download_and_preprocess_s3(bucket_name=BUCKET_NAME,
                              s3_prefix=S3_PREFIX,
                              local_data_dir=LOCAL_DIR,
                              shard_prefix="mydata",
                              limit_files=None,
                              val_ratio=0.0,
                              resolution=80)