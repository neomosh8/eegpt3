import os
import torch
from torch import distributed as dist

# Match the REGIONS from your original code
REGIONS = ["frontal", "motor_temporal", "parietal_occipital"]

# Use your DataLoaderLiteAllInMemory class (assuming it's defined or imported)
# If not defined here, ensure it's available by copying the class definition from your original code
class DataLoaderLiteAllInMemory:
    def __init__(self, B: int, T: int, process_rank: int, num_processes: int,
                 local_data_dir: str = "./local_shards", shard_prefix: str = "mydata",
                 split: str = "train", shuffle_shards: bool = False, pad_token: int = 0):
        self.B = B
        self.per_channel_length = T
        self.num_channels = len(REGIONS)
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.pad_token = pad_token

        # Locate shard files (simplified for testing; adjust path as needed)
        pattern = os.path.join(local_data_dir, f"{shard_prefix}_{split}_*.pt")
        self.shard_files = sorted(glob.glob(pattern))
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

    def _get_slice(self, token_tensor: torch.Tensor, start: int, length: int) -> torch.Tensor:
        total_length = token_tensor.size(0)
        end = start + length
        if end <= total_length:
            return token_tensor[start:end]
        else:
            available = token_tensor[start:] if start < total_length else torch.tensor([], dtype=token_tensor.dtype)
            padding_needed = length - available.size(0)
            padding = torch.full((padding_needed,), self.pad_token, dtype=token_tensor.dtype)
            return torch.cat((available, padding), dim=0)

    def next_batch(self):
        inputs_list = []
        targets_list = []

        for region in REGIONS:
            token_tensor = self.tokens[region]
            channel_inputs = []
            channel_targets = []
            for b in range(self.B):
                start = self.ptr + b * self.per_channel_length
                seq = self._get_slice(token_tensor, start, self.per_channel_length)
                channel_inputs.append(seq.unsqueeze(0))
                target = self._get_slice(token_tensor, start + self.per_channel_length, 1)
                channel_targets.append(target)
            channel_inputs = torch.cat(channel_inputs, dim=0)
            channel_targets = torch.cat(channel_targets, dim=0)
            inputs_list.append(channel_inputs.unsqueeze(1))
            targets_list.append(channel_targets.unsqueeze(1))

        inputs = torch.cat(inputs_list, dim=1)
        targets = torch.cat(targets_list, dim=1)

        self.ptr += self.B * self.per_channel_length * self.num_processes
        return inputs, targets

    def reset(self):
        self.ptr = self.start_ptr

    @property
    def total_len(self):
        return self.tokens[REGIONS[0]].size(0)

# DDP Setup (matching your original code)
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "CUDA is required for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

# Training hyperparameters (matching your original code)
B = 4  # micro-batch size
T = 1024  # sequence length
desired_B_eff = 32 * 4  # effective batch size
grad_accum_steps = desired_B_eff // B

# Create the dataloader
train_loader = DataLoaderLiteAllInMemory(
    B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size,
    local_data_dir="./local_shards", shard_prefix="mydata", split='train', shuffle_shards=True
)

# Calculate max_steps (matching your original code)
num_passes = 5
tokens_per_optim = B * T * grad_accum_steps * ddp_world_size
steps_per_pass = (train_loader.total_len - 1) // tokens_per_optim
max_steps = num_passes * steps_per_pass

if master_process:
    print(f"Total tokens in training set: {train_loader.total_len}")
    print(f"Tokens per optimization step: {tokens_per_optim}")
    print(f"Steps per pass: {steps_per_pass}")
    print(f"Total steps: {max_steps}")

# Test the dataloader
def test_dataloader(loader, max_steps, pad_token=0):
    total_padding = 0
    loader.reset()  # Start from the beginning

    for step in range(max_steps):
        inputs, targets = loader.next_batch()
        batch_start = loader.ptr - (B * T * ddp_world_size)  # Pointer before the batch
        batch_end = loader.ptr - 1  # Pointer after the batch (last token accessed)

        # Count padding in this batch
        padding_mask = (inputs == pad_token)
        batch_padding = padding_mask.sum().item()
        total_padding += batch_padding

        if master_process and (step < 5 or step >= max_steps - 5 or step % 100 == 0):
            print(f"\nStep {step}:")
            print(f"  Batch Start Index: {batch_start}")
            print(f"  Batch End Index: {batch_end}")
            print(f"  Inputs Shape: {inputs.shape} [B, num_channels, T]")
            print(f"  Targets Shape: {targets.shape} [B, num_channels]")
            print(f"  Padding in this batch: {batch_padding}")
            print(f"  Sample input (first sequence, first channel): {inputs[0, 0, :10]}")
            print(f"  Sample target (first sequence, first channel): {targets[0, 0]}")

    if master_process:
        print(f"\nTest completed over {max_steps} steps.")
        print(f"Total padding tokens added: {total_padding}")
        print(f"Percentage of tokens that are padding: {100 * total_padding / (max_steps * B * T * len(REGIONS)):.2f}%")

# Run the test
test_dataloader(train_loader, max_steps)

# Clean up DDP (matching your original code)
if ddp:
    dist.destroy_process_group()