import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import random
import argparse


def plot_token_histogram(tokens, region, filename, output_dir):
    """Plot histogram of token frequencies"""

    # Remove EOS token for frequency analysis
    eos_token = tokens[-1].item()
    tokens_without_eos = tokens[:-1]

    # Count token frequencies
    token_counts = Counter(tokens_without_eos.numpy())

    # Plot settings
    plt.figure(figsize=(12, 6))

    # Sort tokens by index
    sorted_tokens = sorted(token_counts.items())
    indices = [t[0] for t in sorted_tokens]
    freqs = [t[1] for t in sorted_tokens]

    # Create bar plot
    plt.bar(indices, freqs, alpha=0.7)
    plt.title(f"Token Distribution - {region} - {os.path.basename(filename)}")
    plt.xlabel("Token Index")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Annotate EOS token information
    plt.figtext(0.02, 0.02, f"EOS Token: {eos_token}", fontsize=10)

    # Add percentage of unused tokens
    unique_tokens = len(token_counts)
    max_token = max(indices)
    unused = max_token - unique_tokens + 1  # +1 because tokens are zero-indexed
    plt.figtext(0.02, 0.05,
                f"Unique tokens: {unique_tokens}/{max_token + 1} ({unique_tokens / (max_token + 1) * 100:.1f}%)",
                fontsize=10)

    # Save the plot
    save_path = os.path.join(output_dir, f"{os.path.basename(filename)}_{region}_histogram.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return save_path


def plot_token_sequence(tokens, region, filename, output_dir, segment_size=None):
    """Plot token sequence as a heatmap"""

    # Default segment size (take entire sequence or limit to 2000 tokens if too long)
    if segment_size is None:
        segment_size = min(len(tokens), 2000)

    # If sequence is very long, take the first segment_size tokens
    if len(tokens) > segment_size:
        tokens_slice = tokens[:segment_size]
    else:
        tokens_slice = tokens

    # Convert to numpy array
    tokens_np = tokens_slice.numpy()

    # Create figure
    plt.figure(figsize=(20, 6))

    # Line plot for sequence
    plt.plot(tokens_np, marker='.', markersize=2, linestyle='-', linewidth=0.5)
    plt.title(f"Token Sequence - {region} - {os.path.basename(filename)}")
    plt.xlabel("Position")
    plt.ylabel("Token Index")
    plt.grid(True, linestyle='--', alpha=0.7)

    # Annotate EOS token
    if tokens[-1] != tokens_slice[-1]:
        plt.figtext(0.02, 0.02,
                    f"Showing first {segment_size} of {len(tokens)} tokens (EOS token at position {len(tokens) - 1})",
                    fontsize=10)

    # Save the plot
    save_path = os.path.join(output_dir, f"{os.path.basename(filename)}_{region}_sequence.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return save_path


def plot_token_heatmap(tokens, region, filename, output_dir, max_cols=100):
    """Plot tokens as a 2D heatmap to visualize patterns"""

    # Remove EOS token for the heatmap
    tokens_without_eos = tokens[:-1]

    # Calculate grid dimensions for the heatmap
    total_tokens = len(tokens_without_eos)
    n_cols = min(max_cols, total_tokens)
    n_rows = (total_tokens + n_cols - 1) // n_cols  # Ceiling division

    # Create a 2D grid filled with the token values
    grid = np.full((n_rows, n_cols), -1)  # Fill with -1 to indicate empty cells
    for i, token in enumerate(tokens_without_eos):
        row = i // n_cols
        col = i % n_cols
        grid[row, col] = token.item()

    # Plot settings
    plt.figure(figsize=(min(20, n_cols / 5), min(10, n_rows / 5)))

    # Create heatmap
    sns.heatmap(grid, cmap='viridis', cbar=True,
                xticklabels=50, yticklabels=50)
    plt.title(f"Token Pattern Heatmap - {region} - {os.path.basename(filename)}")
    plt.xlabel("Position (column)")
    plt.ylabel("Position (row)")

    # Save the plot
    save_path = os.path.join(output_dir, f"{os.path.basename(filename)}_{region}_heatmap.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return save_path


def compare_regions(tokens_dict, filename, output_dir):
    """Create a plot comparing token patterns across regions"""

    # Get region names and corresponding tokens
    regions = list(tokens_dict.keys())
    if len(regions) < 2:
        return None  # Not enough regions to compare

    # Create figure with subplots for each region
    fig, axes = plt.subplots(len(regions), 1, figsize=(15, 3 * len(regions)), sharex=True)

    # Plot each region's tokens
    for i, region in enumerate(regions):
        tokens = tokens_dict[region]
        tokens_without_eos = tokens[:-1]

        # Plot tokens (select first 1000 tokens if longer)
        plot_length = min(len(tokens_without_eos), 1000)
        tokens_to_plot = tokens_without_eos[:plot_length].numpy()

        # Plot as line
        axes[i].plot(tokens_to_plot, marker='.', markersize=2, linestyle='-', linewidth=0.5)
        axes[i].set_title(f"{region} Tokens")
        axes[i].set_ylabel("Token Index")
        axes[i].grid(True, linestyle='--', alpha=0.5)

    # Add overall title
    plt.suptitle(f"Region Comparison - {os.path.basename(filename)}")
    axes[-1].set_xlabel("Position")

    # Save the plot
    save_path = os.path.join(output_dir, f"{os.path.basename(filename)}_region_comparison.png")
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for suptitle
    plt.savefig(save_path)
    plt.close()

    return save_path


def visualize_token_tensors(input_dir, output_dir, num_files=5):
    """
    Load token tensors from input_dir, create visualizations, and save to output_dir

    Args:
        input_dir: Directory containing .pt token tensor files
        output_dir: Directory to save visualizations to
        num_files: Number of files to randomly select for visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all .pt files in the input directory
    pt_files = glob.glob(os.path.join(input_dir, "*.pt"))

    # Get unique base filenames (without region suffix)
    base_files = set()
    for file in pt_files:
        basename = os.path.basename(file)
        base_name = '_'.join(basename.split('_')[:-2])  # Remove _region_tokens.pt
        base_files.add(base_name)

    # Select random samples if we have more than requested
    if len(base_files) > num_files:
        base_files = random.sample(list(base_files), num_files)

    print(f"Processing {len(base_files)} token tensor sets...")

    # Process each selected file
    for base_name in base_files:
        # Find all region files for this base name
        region_files = [f for f in pt_files if os.path.basename(f).startswith(base_name)]

        # Skip if no files found (shouldn't happen)
        if not region_files:
            continue

        print(f"Processing {base_name} with {len(region_files)} regions")

        # Load token tensors for each region
        tokens_dict = {}
        for file in region_files:
            # Extract region name from filename
            filename = os.path.basename(file)
            region = filename.split('_')[-2]  # Format: base_name_region_tokens.pt

            # Load tensor
            tokens = torch.load(file)
            tokens_dict[region] = tokens

            # Create visualizations for this region
            plot_token_histogram(tokens, region, filename, output_dir)
            plot_token_sequence(tokens, region, filename, output_dir)
            plot_token_heatmap(tokens, region, filename, output_dir)

        # Create region comparison if we have multiple regions
        if len(tokens_dict) > 1:
            compare_regions(tokens_dict, base_name, output_dir)

    print(f"Visualization complete. Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize EEG token tensors")
    parser.add_argument("--input_dir", type=str, default="training_data_shards",
                        help="Directory containing token tensor files")
    parser.add_argument("--output_dir", type=str, default="token_visualizations",
                        help="Directory to save visualizations")
    parser.add_argument("--num_files", type=int, default=5,
                        help="Number of files to visualize")

    args = parser.parse_args()

    # Run the visualization
    visualize_token_tensors(args.input_dir, args.output_dir, args.num_files)