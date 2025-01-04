import boto3


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, bucket_name='my-bucket', s3_prefix='output'):
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

        # Create S3 client; adapt as needed (credentials, region, etc.)
        self.s3 = boto3.client('s3')

        # 1) Find all *_coeffs.txt files in s3_prefix
        #    Typically you'd list objects in the bucket and filter them. Something like:
        response = self.s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_prefix)
        # Make a list of all keys
        all_files = []
        if 'Contents' in response:
            all_files = [obj['Key'] for obj in response['Contents']
                         if obj['Key'].endswith('_coeffs.txt')]
        # Sort or shuffle as you see fit
        all_files.sort()

        # 2) Prepare a list of file pairs: (coeffs_file, channels_file)
        #    For each 'something_coeffs.txt', we also expect 'something_channels.txt'
        self.file_pairs = []
        for coeffs_key in all_files:
            channels_key = coeffs_key.replace('_coeffs.txt', '_channels.txt')
            # We only add pairs if the channels file also exists.
            # (If you're sure both exist always, you could skip this check.)
            # We'll do a quick object head check:
            try:
                self.s3.head_object(Bucket=bucket_name, Key=channels_key)
                self.file_pairs.append((coeffs_key, channels_key))
            except:
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
        Reset self.current_position for the new file.
        """
        coeffs_key, channels_key = self.file_pairs[self.current_file_idx]

        # Create local temp paths; in real code you might want to store them in /tmp or somewhere
        coeffs_local = os.path.basename(coeffs_key)
        channels_local = os.path.basename(channels_key)

        # Download both files from S3
        self._download_s3_file(coeffs_key, coeffs_local)
        self._download_s3_file(channels_key, channels_local)

        # Read & tokenize coeffs
        with open(coeffs_local, 'r', encoding='utf-8') as f:
            text = f.read()
        encoded = self.tokenizer.encode_wi`th_alignment(text.strip().split(), as_ids=True,alignment_filepath=f'temp_{coeffs_key}.txt')
        self.tokens = torch.tensor(encoded, dtype=torch.long)

        # Read channels
        with open(channels_local, 'r', encoding='utf-8') as f:
            chan_text = f.read().strip().split()
            final_channels = apply_alignment_to_channels(chan_text, final_rle_positions, combine_mode="first")
            save_channels_as_text(final_channels, 'final_channels.txt')
        # Convert e.g. '1'->0, '2'->1, ...
        self.channels = torch.tensor([int(x) - 1 for x in chan_text], dtype=torch.long)

        # Make sure length matches
        if len(self.tokens) != len(self.channels):
            raise ValueError("tokens and channels length mismatch!")

        # Reset position to reflect a fresh file
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        """
        Fetch the next batch of data: (x, c, y)
          - x, c: the input tokens and channel IDs
          - y: the target tokens for cross-entropy
        If the current file is exhausted, move to the next file.
        """
        B, T = self.B, self.T

        # Take enough tokens from current position
        start = self.current_position
        end = self.current_position + B * T + 1

        buf_tokens = self.tokens[start:end]
        buf_channels = self.channels[start:end]

        # If we don't have enough tokens to form a full batch,
        # move to the next file and try again.
        if len(buf_tokens) < B * T + 1:
            self.current_file_idx = (self.current_file_idx + 1) % len(self.file_pairs)
            self._load_current_file()
            return self.next_batch()

        # Construct x (inputs), y (targets), c (channels)
        x = buf_tokens[:-1].view(B, T)
        y = buf_tokens[1:].view(B, T)
        c = buf_channels[:-1].view(B, T)
        # We do not shift c by 1 since the channel "belongs" to the same token in x.

        # Advance the position
        self.current_position += B * T * self.num_processes

        # If the next batch would go out of range, we'll switch to the next file next time.
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_file_idx = (self.current_file_idx + 1) % len(self.file_pairs)
            self._load_current_file()

        return x, c, y




train_loader = DataLoaderLite(B=4, T=1024 , process_rank=0, num_processes=1)
train_loader.next_batch()
b=6