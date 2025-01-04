import os
import boto3
import torch
from mne.fixes import rng_uniform

from tokenizer import apply_alignment_to_channels, save_channels_as_text
from tokenizer2 import BPE_RLE_Tokenizer as Tokenizer

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes,
                 bucket_name='dataframes--use1-az6--x-s3', s3_prefix='output/'):
        """
        Args:
            B: Batch size
            T: Sequence length
            process_rank: (DDP) process rank
            num_processes: total number of processes (DDP)
            bucket_name: S3 bucket name
            s3_prefix: the directory/prefix inside the bucket where data files live
        """
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.total_num_tokens = 0
        # Create S3 client; adapt as needed.
        self.s3 = boto3.client('s3')

        # 1) Find all *_coeffs.txt files in s3_prefix using a paginator to handle large listings
        paginator = self.s3.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)

        all_files = []
        page_num = 0
        for page in page_iterator:
            page_num += 1
            contents = page.get('Contents', [])
            print(f"Processing page {page_num} with {len(contents)} items ...")
            for obj in contents:
                key = obj['Key']
                if key.endswith('_coeffs.txt'):
                    all_files.append(key)
            print(f"  -> Accumulated so far: {len(all_files)} files.")

        print(f"Total number of *_coeffs.txt files found: {len(all_files)}")

        # Optionally sort or shuffle.
        # For demonstration, let's keep them all but limit to 10 in code for testing:
        all_files = all_files

        # 2) Prepare a list of file pairs: (coeffs_file, channels_file)
        self.file_pairs = []
        for coeffs_key in all_files:
            channels_key = coeffs_key.replace('_coeffs.txt', '_channels.txt')
            try:
                self.s3.head_object(Bucket=bucket_name, Key=channels_key)
                self.file_pairs.append((coeffs_key, channels_key))
            except:
                # If the channels file doesn't exist, skip it
                pass

        if len(self.file_pairs) == 0:
            raise ValueError("No valid coeffs/channels file pairs found in S3 prefix.")

        # Load up a tokenizer once (or pass it in)
        self.tokenizer = Tokenizer()
        self.tokenizer.load_merges("neo_tokenizer/merges.json")
        self.tokenizer.load_vocab("neo_tokenizer/vocab.json")

        self.bucket_name = bucket_name
        self.s3_prefix = s3_prefix

        # Initialize state
        self.current_file_idx = 0  # which pair of files we are on
        self.tokens = None
        self.channels = None
        self.current_position = 0

        # Load the first file
        self._load_current_file()

    def _download_s3_file(self, key, local_path):
        """Download a single key from S3 to a local path."""
        self.s3.download_file(self.bucket_name, key, local_path)

    def _load_current_file(self):
        """
        Download the current file pair from S3, tokenize and store in self.tokens / self.channels.
        Reset self.current_position for the new file. Then remove local files.
        """
        coeffs_key, channels_key = self.file_pairs[self.current_file_idx]

        # In production, you might want to store these in /tmp
        coeffs_local = os.path.basename(coeffs_key)
        channels_local = os.path.basename(channels_key)

        # Download both files from S3
        self._download_s3_file(coeffs_key, coeffs_local)
        self._download_s3_file(channels_key, channels_local)

        # Read & tokenize coeffs
        with open(coeffs_local, 'r', encoding='utf-8') as f:
            text = f.read()
        # Convert text to a list of tokens (words) and prepend the special token
        raw_tokens = text.strip().split()
        raw_tokens.insert(0, "|trial|")  # <-- add the special token here

        # Now encode with alignment
        encoded, pos = self.tokenizer.encode_with_alignment(raw_tokens, as_ids=True)
        self.total_num_tokens += len(encoded)
        self.tokens = torch.tensor(encoded, dtype=torch.long)

        # Read channels
        with open(channels_local, 'r', encoding='utf-8') as f:
            chan_text = f.read().strip().split()
            chan_text.insert(0, "1")  # or "0", or whichever makes sense
            final_channels = apply_alignment_to_channels(chan_text, pos, combine_mode="first")
            chan_text = final_channels

        # Convert e.g. '1'->0, '2'->1, ...
        self.channels = torch.tensor([int(x) - 1 for x in chan_text], dtype=torch.long)

        # Make sure length matches
        if len(self.tokens) != len(self.channels):
            raise ValueError("tokens and channels length mismatch!")

        # Cleanup local files now that we've read them
        try:
            os.remove(coeffs_local)
        except OSError:
            pass
        try:
            os.remove(channels_local)
        except OSError:
            pass

        # Reset position for the fresh file
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        """
        Fetch the next batch of data: (x, c, y)
          - x, c: the input tokens and channel IDs
          - y: the target tokens for cross-entropy
        If the current file is exhausted, move to the next file.
        """
        B, T = self.B, self.T

        start = self.current_position
        end = self.current_position + B * T + 1

        buf_tokens = self.tokens[start:end]
        buf_channels = self.channels[start:end]

        # If we don't have enough tokens to form a full batch, move to the next file and try again.
        if len(buf_tokens) < B * T + 1:
            self.current_file_idx = (self.current_file_idx + 1) % len(self.file_pairs)
            self._load_current_file()
            return self.next_batch()

        # x, y, c
        x = buf_tokens[:-1].view(B, T)
        y = buf_tokens[1:].view(B, T)
        c = buf_channels[:-1].view(B, T)

        # Advance the position
        self.current_position += B * T * self.num_processes

        # If next batch goes out of range, switch to the next file.
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_file_idx = (self.current_file_idx + 1) % len(self.file_pairs)
            self._load_current_file()
        print(self.total_num_tokens)
        print(self.current_file_idx,"/",len(self.file_pairs))
        return x, c, y

train_loader = DataLoaderLite(B=32, T=1024 , process_rank=0, num_processes=1)
for i in range(1000000):
    print(i,"\n")
    (train_loader.next_batch())
b=6

"""
771479260 tokens
44386 / 44389 files
13098 steps
"""