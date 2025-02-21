import os
import glob
import random

import torch


# Define regions (channels) as in your original code.
REGIONS = ["frontal", "motor_temporal", "parietal_occipital"]

class DataLoaderLiteAllInMemory:
    def __init__(self, B: int, T: int, process_rank: int, num_processes: int,
                 local_data_dir: str = "./local_shards", shard_prefix: str = "mydata",
                 split: str = "train", shuffle_shards: bool = False, pad_token: int = 0):
        self.B = B
        self.per_channel_length = T  # Sequence length per channel
        self.num_channels = len(REGIONS)
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.pad_token = pad_token  # Token to use for padding

        # Locate shard files
        pattern = os.path.join(local_data_dir, f"{shard_prefix}_{split}_*.pt")
        self.shard_files = sorted(glob.glob(pattern))[0:10]
        if not self.shard_files:
            raise ValueError(f"No {split} shards found in {local_data_dir} with prefix {shard_prefix}_{split}_")
        if shuffle_shards:
            random.shuffle(self.shard_files)

        # Load and concatenate tokens
        self.tokens = {region: [] for region in REGIONS}
        for shard_path in self.shard_files:
            loaded = torch.load(shard_path, map_location="cpu", weights_only=False)
            for region in REGIONS:
                if region not in loaded:
                    available_regions = list(loaded.keys())
                    if available_regions:
                        loaded[region] = loaded[available_regions[0]]
                        print(f"Warning: Shard {shard_path} missing {region}, using {available_regions[0]}.")
                    else:
                        raise ValueError(f"Shard {shard_path} has no channels for {region}.")
                self.tokens[region].append(loaded[region])
        for region in REGIONS:
            self.tokens[region] = torch.cat(self.tokens[region], dim=0)

        # Check minimum length
        min_length = min(t.size(0) for t in self.tokens.values())
        required_length = self.B * self.per_channel_length * self.num_processes
        if min_length < required_length:
            print(f"Warning: Shortest channel has {min_length} tokens, less than required {required_length}. Padding will be used.")

        self.start_ptr = self.B * self.per_channel_length * self.process_rank
        self.ptr = self.start_ptr

    def _get_slice(self, token_tensor: torch.Tensor, start: int, length: int, pad_value: int) -> torch.Tensor:
        total_length = token_tensor.size(0)
        start = start % total_length  # Wrap around if exceeding length
        end = start + length
        if end <= total_length:
            return token_tensor[start:end]
        else:
            first_part = token_tensor[start:]
            remaining = length - first_part.size(0)
            second_part = token_tensor[:remaining]
            return torch.cat((first_part, second_part), dim=0)
    def next_batch(self):
        inputs_list = []
        targets_list = []

        for region in REGIONS:
            token_tensor = self.tokens[region]
            channel_inputs = []
            channel_targets = []
            for b in range(self.B):
                start = self.ptr + b * self.per_channel_length
                # Input: T tokens, pad with pad_token
                seq = self._get_slice(token_tensor, start, self.per_channel_length, pad_value=self.pad_token)  # [T]
                channel_inputs.append(seq.unsqueeze(0))  # [1, T]
                # Target: Next T tokens, pad with -100
                target = self._get_slice(token_tensor, start + 1, self.per_channel_length, pad_value=-100)  # [T]
                channel_targets.append(target.unsqueeze(0))  # [1, T]
            channel_inputs = torch.cat(channel_inputs, dim=0)  # [B, T]
            channel_targets = torch.cat(channel_targets, dim=0)  # [B, T]
            inputs_list.append(channel_inputs.unsqueeze(1))  # [B, 1, T]
            targets_list.append(channel_targets.unsqueeze(1))  # [B, 1, T]

        inputs = torch.cat(inputs_list, dim=1)  # [B, num_channels, T]
        targets = torch.cat(targets_list, dim=1)  # [B, num_channels, T]

        self.ptr += self.B * self.per_channel_length * self.num_processes
        return inputs, targets

    def reset(self):
        self.ptr = self.start_ptr

    @property
    def total_len(self):
        return self.tokens[REGIONS[0]].size(0)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch()
#############################################
# Create Dummy Shard Files for Testing
#############################################
dummy_dir = "./dummy_shards"
if not os.path.exists(dummy_dir):
    os.makedirs(dummy_dir)

def create_dummy_shard(filepath, length=50):
    """
    Create a dummy shard with a small sequence of tokens for each channel.
    For clarity, each region has a different range of tokens.
    """
    data = {}
    data["frontal"] = torch.arange(1, length + 1)                   # [1, 2, ..., 50]
    data["motor_temporal"] = torch.arange(1001, 1001 + length)        # [1001, 1002, ..., 1050]
    data["parietal_occipital"] = torch.arange(2001, 2001 + length)      # [2001, 2002, ..., 2050]
    torch.save(data, filepath)

# Create dummy shards if none exist.
shard_pattern = os.path.join(dummy_dir, "mydata_train_*.pt")
if not glob.glob(shard_pattern):
    for i in range(2):  # Create 2 shards for testing.
        shard_path = os.path.join(dummy_dir, f"mydata_train_{i}.pt")
        create_dummy_shard(shard_path, length=50)
print(f"Dummy shards created in {dummy_dir}: {glob.glob(shard_pattern)}")
#############################################
# Emulate DDP: Instantiate two dataloaders (ranks 0 and 1)
#############################################
ddp_world_size = 2  # Emulated number of processes.
B = 2       # Batch size (samples per process)
T = 10      # Sequence length per channel

# Create two dataloader instances for two DDP ranks.
loader_rank0 = DataLoaderLiteAllInMemory(
    B=B, T=T, process_rank=0, num_processes=ddp_world_size,
    local_data_dir=dummy_dir, shard_prefix="mydata", split="train", shuffle_shards=False
)
loader_rank1 = DataLoaderLiteAllInMemory(
    B=B, T=T, process_rank=1, num_processes=ddp_world_size,
    local_data_dir=dummy_dir, shard_prefix="mydata", split="train", shuffle_shards=False
)

#############################################
# Emulate Multiple Steps/Passes
#############################################
num_steps = 3  # How many batches (or passes) to fetch.
for step in range(num_steps):
    print(f"\n=== Step {step} ===")
    # Each "rank" fetches its own batch.
    inputs0, targets0 = loader_rank0.next_batch()
    inputs1, targets1 = loader_rank1.next_batch()

    print("\n-- Rank 0 --")
    print("Pointer:", loader_rank0.ptr)
    for b in range(B):
        print(f"Sample {b}:")
        for c, region in enumerate(REGIONS):
            inp = inputs0[b, c].tolist()
            tgt = targets0[b, c].tolist()
            print(f"  {region} - Input:  {inp}")
            print(f"             Target: {tgt}")

    print("\n-- Rank 1 --")
    print("Pointer:", loader_rank1.ptr)
    for b in range(B):
        print(f"Sample {b}:")
        for c, region in enumerate(REGIONS):
            inp = inputs1[b, c].tolist()
            tgt = targets1[b, c].tolist()
            print(f"  {region} - Input:  {inp}")
            print(f"             Target: {tgt}")

#############################################
# Reset and Fetch Again to Verify Consistency
#############################################
print("\n=== Resetting loaders and fetching the first batch again ===")
loader_rank0.reset()
loader_rank1.reset()

inputs0, targets0 = loader_rank0.next_batch()
inputs1, targets1 = loader_rank1.next_batch()

print("\nAfter reset - Rank 0, first sample:")
for c, region in enumerate(REGIONS):
    print(f"  {region} - Input:  {inputs0[0, c].tolist()}")
    print(f"             Target: {targets0[0, c].tolist()}")

print("\nAfter reset - Rank 1, first sample:")
for c, region in enumerate(REGIONS):
    print(f"  {region} - Input:  {inputs1[0, c].tolist()}")
    print(f"             Target: {targets1[0, c].tolist()}")