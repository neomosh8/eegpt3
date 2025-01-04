# dataset_preprocess.py
import os
import boto3
import json
import torch
from pathlib import Path
from typing import List

from tokenizer2 import BPE_RLE_Tokenizer as Tokenizer,apply_alignment_to_channels



def download_and_preprocess_s3(
        bucket_name: str,
        s3_prefix: str,
        local_data_dir: str,
        shard_prefix: str = "shard",
        limit_files: int = None,
):
    """
    - Lists *_coeffs.txt files in S3 under `bucket_name` and `s3_prefix`.
    - For each valid pair of coeffs/channels, downloads them locally.
    - Tokenizes, aligns, and saves them as a .pt file (each file can be considered a 'shard').
    - Optionally limit the number of files for testing via `limit_files`.
    """

    # Ensure local data directory exists
    os.makedirs(local_data_dir, exist_ok=True)

    # Create S3 client
    s3 = boto3.client('s3')

    # 1) Find all *_coeffs.txt
    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)

    all_files = []
    for page in page_iterator:
        contents = page.get('Contents', [])
        for obj in contents:
            key = obj['Key']
            if key.endswith('_coeffs.txt'):
                all_files.append(key)

    if limit_files is not None:
        all_files = all_files[:limit_files]

    # 2) Build file pairs
    file_pairs = []
    for coeffs_key in all_files:
        channels_key = coeffs_key.replace('_coeffs.txt', '_channels.txt')
        try:
            s3.head_object(Bucket=bucket_name, Key=channels_key)
            file_pairs.append((coeffs_key, channels_key))
        except:
            pass

    if not file_pairs:
        raise ValueError("No valid coeffs/channels file pairs found in S3 prefix.")

    # Load tokenizer
    tokenizer = Tokenizer()
    tokenizer.load_merges("neo_tokenizer/merges.json")
    tokenizer.load_vocab("neo_tokenizer/vocab.json")

    shard_id = 0
    for coeffs_key, channels_key in file_pairs:
        # Local paths
        coeffs_local = os.path.join(local_data_dir, os.path.basename(coeffs_key))
        channels_local = os.path.join(local_data_dir, os.path.basename(channels_key))

        # Download
        s3.download_file(bucket_name, coeffs_key, coeffs_local)
        s3.download_file(bucket_name, channels_key, channels_local)

        # Read & tokenize coeffs
        with open(coeffs_local, 'r', encoding='utf-8') as f:
            text = f.read()
        raw_tokens = text.strip().split()
        raw_tokens.insert(0, "|trial|")  # Add special token
        encoded, pos = tokenizer.encode_with_alignment(raw_tokens, as_ids=True)
        tokens_tensor = torch.tensor(encoded, dtype=torch.long)

        # Read channels
        with open(channels_local, 'r', encoding='utf-8') as f:
            chan_text = f.read().strip().split()
        chan_text.insert(0, "1")  # or "0"
        final_channels = apply_alignment_to_channels(chan_text, pos, combine_mode="first")
        channels_tensor = torch.tensor([int(x) - 1 for x in final_channels], dtype=torch.long)

        if len(tokens_tensor) != len(channels_tensor):
            raise ValueError("Token / channel length mismatch.")

        # Save shard
        shard_path = os.path.join(local_data_dir, f"{shard_prefix}_{shard_id}.pt")
        torch.save({
            'tokens': tokens_tensor,
            'channels': channels_tensor
        }, shard_path)

        shard_id += 1

        # Cleanup local copies if you want
        try:
            os.remove(coeffs_local)
            os.remove(channels_local)
        except OSError:
            pass

    print(f"Finished preprocessing. Created {shard_id} shards in {local_data_dir}.")


if __name__ == "__main__":
    # Example usage:
    BUCKET_NAME = "dataframes--use1-az6--x-s3"
    S3_PREFIX = "output/"
    LOCAL_DIR = "./local_shards"

    download_and_preprocess_s3(
        bucket_name=BUCKET_NAME,
        s3_prefix=S3_PREFIX,
        local_data_dir=LOCAL_DIR,
        shard_prefix="mydata",
        limit_files=10  # or some small number for testing
    )
