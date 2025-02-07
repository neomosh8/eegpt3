#!/usr/bin/env python
"""
This script processes the neural responses dataset accompanying the paper
"Do syntactic and semantic similarity lead to interference effects? Evidence from self-paced reading and event-related potentials using German."

The dataset file is:
    neural_responses.mat (≈3.6 GB)
stored in S3 at:
    s3://dataframes--use1-az6--x-s3/attention fintune/8173495/neural_responses.mat

This script:
  1. Downloads the MATLAB file from S3.
  2. Loads the file using scipy.io.loadmat.
  3. Extracts neural spiking data, time, condition labels, and layer labels from a structure named "neural_responses".
  4. Computes the average spiking response for target and distractor trials in each cortical layer.
  5. Plots the time course of the average responses for each layer.
  6. Saves the output plot to a local directory.

Adjust the field names and processing as needed for your dataset.
"""

import os
import boto3
import tempfile
import scipy.io as sio
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# S3 configuration
S3_BUCKET = "dataframes--use1-az6--x-s3"
S3_KEY = "attention fintune/8173495/neural_responses.mat"

# Local output directory
OUTPUT_DIR = "output_8173495"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_mat_file(s3_bucket, s3_key, local_path):
    s3 = boto3.client("s3")
    try:
        s3.download_file(s3_bucket, s3_key, local_path)
        print(f"Downloaded {s3_key} to {local_path}")
    except Exception as e:
        print(f"Error downloading {s3_key}: {e}")
        raise e


def load_neural_data(mat_path):
    print(f"Loading MATLAB file: {mat_path}")
    mat_data = sio.loadmat(mat_path)
    print("MATLAB file keys:", mat_data.keys())
    return mat_data


def process_neural_data(mat_data):
    """
    Process the neural data contained in the .mat file.

    This function assumes that the MATLAB file contains a structure variable
    named "neural_responses" with at least the following fields:
      - spiking: a 3D array (trials × time × layers)
      - time: a 1D array of time points
      - condition: an array of strings (one per trial) with entries such as 'target' or 'distractor'
      - layer_labels: a cell array of strings naming each cortical layer

    Adjust the indexing as needed.
    """
    if "neural_responses" not in mat_data:
        raise KeyError("Variable 'neural_responses' not found in the MATLAB file.")

    # Depending on how MATLAB structures are imported, fields may be nested.
    # Here we assume a common pattern where the struct is stored as a NumPy structured array.
    nr = mat_data["neural_responses"]

    # The following indexing may need to be adjusted for your file.
    spiking = nr["spiking"][0, 0]  # e.g., shape: (num_trials, num_timepoints, num_layers)
    time = nr["time"][0, 0].squeeze()  # 1D array
    condition = nr["condition"][0, 0].squeeze()  # array of strings or numeric codes
    layer_labels = nr["layer_labels"][0, 0]  # cell array of strings

    # Convert layer_labels to a Python list of strings.
    # This assumes layer_labels is a 2D cell array with one row.
    layer_labels = [str(x[0]) if isinstance(x, np.ndarray) else str(x) for x in layer_labels[0]]

    print("Extracted neural responses:")
    print(f"  spiking shape: {spiking.shape}")  # e.g., (trials, time, layers)
    print(f"  time shape: {time.shape}, first 10 values: {time[:10]}")
    print(f"  Number of trials: {spiking.shape[0]}, Number of layers: {spiking.shape[2]}")
    print("  Layer labels:", layer_labels)

    # For demonstration, compute the average spiking response separately for target and distractor conditions.
    # Here we assume that condition is an array of strings.
    condition = np.array([str(c[0]) if isinstance(c, np.ndarray) else str(c) for c in condition])
    target_trials = np.where(condition == "target")[0]
    distractor_trials = np.where(condition == "distractor")[0]

    if target_trials.size == 0 or distractor_trials.size == 0:
        print("Warning: One or both conditions have zero trials. Check the 'condition' field in the .mat file.")

    avg_target = np.mean(spiking[target_trials, :, :], axis=0)  # shape: (time, layers)
    avg_distractor = np.mean(spiking[distractor_trials, :, :], axis=0)  # shape: (time, layers)

    return time, layer_labels, avg_target, avg_distractor


def plot_layer_responses(time, layer_labels, avg_target, avg_distractor, output_dir):
    """
    Plot the average spiking responses for target and distractor conditions for each cortical layer.
    """
    num_layers = avg_target.shape[1]
    plt.figure(figsize=(12, 3 * num_layers))
    for i in range(num_layers):
        plt.subplot(num_layers, 1, i + 1)
        plt.plot(time, avg_target[:, i], label="Target", color="blue")
        plt.plot(time, avg_distractor[:, i], label="Distractor", color="red")
        plt.title(f"Layer {layer_labels[i]}")
        if i == 0:
            plt.legend()
        plt.ylabel("Spiking Activity (a.u.)")
    plt.xlabel("Time")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "layer_responses.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved layer response plot to {plot_path}")


if __name__ == "__main__":
    # Create a temporary directory for the download.
    with tempfile.TemporaryDirectory() as temp_dir:
        local_mat_path = os.path.join(temp_dir, "neural_responses.mat")
        download_mat_file(S3_BUCKET, S3_KEY, local_mat_path)

        # Load the MATLAB file.
        mat_data = load_neural_data(local_mat_path)

        # Process the neural data.
        try:
            time, layer_labels, avg_target, avg_distractor = process_neural_data(mat_data)
        except Exception as e:
            print("Error processing neural data:", e)
            exit(1)

        # Plot the average responses per layer.
        plot_layer_responses(time, layer_labels, avg_target, avg_distractor, OUTPUT_DIR)

    print("Done!")
