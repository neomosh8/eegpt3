import numpy as np
from sklearn.cluster import KMeans
import joblib
import boto3
import pandas as pd
import asyncio
import aioboto3
from concurrent.futures import ProcessPoolExecutor
import os
import io

from create_text_files_from_csv_2 import create_regional_bipolar_channels
from utils import list_csv_files_in_folder, list_s3_folders, call_gpt_for_instructions, calculate_sps, preprocess_data, wavelet_decompose_window

# Updated calculate_sps (example implementation - adjust based on your utils.py)
def calculate_sps_from_df(df):
    # Assuming timestamp is in a column named 'timestamp' or the first column
    time_col = df.get('timestamp', df.columns[0])
    dt = np.diff(time_col).mean()
    return 1.0 / dt if dt != 0 else 1.0

def process_csv_for_coeffs(csv_key, bucket, local_dir=None, window_length_sec=1.96, wvlet='db2', level=4, num_samples_per_file=10):
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=csv_key)
    df = pd.read_csv(response['Body'])

    base_name = os.path.splitext(os.path.basename(csv_key))[0]
    all_columns = list(df.columns)
    instructions = call_gpt_for_instructions(channel_names=all_columns, dataset_id=base_name)

    if instructions.get("action") == "skip":
        print(f"Skipping dataset '{base_name}' as instructed by GPT.")
        return {"frontal": [], "motor_temporal": [], "parietal_occipital": []}

    channels_to_drop = instructions.get("channels_to_drop", [])
    print(f"Processing dataset '{base_name}'. Dropping channels: {channels_to_drop}")

    regional_bipolar = create_regional_bipolar_channels(df, channels_to_drop)
    original_sps = calculate_sps_from_df(df)  # Pass DataFrame instead of csv_key
    regional_preprocessed = {}
    new_sps_val = None
    for key, signal_array in regional_bipolar.items():
        signal_2d = signal_array[np.newaxis, :]
        preprocessed_signal, new_sps = preprocess_data(signal_2d, original_sps)
        regional_preprocessed[key] = preprocessed_signal[0, :]
        if new_sps_val is None:
            new_sps_val = new_sps

    all_data = np.concatenate([regional_preprocessed[region] for region in regional_preprocessed])
    global_mean = np.mean(all_data)
    global_std = np.std(all_data)
    if np.isclose(global_std, 0):
        global_std = 1e-8
    for key in regional_preprocessed:
        regional_preprocessed[key] = (regional_preprocessed[key] - global_mean) / global_std

    min_length = min(len(regional_preprocessed[region]) for region in regional_preprocessed)
    n_window_samples = int(window_length_sec * new_sps_val)
    num_windows = min_length // n_window_samples

    if num_windows <= num_samples_per_file:
        selected_indices = list(range(num_windows))
    else:
        selected_indices = np.random.choice(num_windows, num_samples_per_file, replace=False)

    coeffs_dict = {"frontal": [], "motor_temporal": [], "parietal_occipital": []}
    for i in selected_indices:
        window_start = i * n_window_samples
        window_end = window_start + n_window_samples
        window_data = np.vstack([regional_preprocessed[region][window_start:window_end] for region in coeffs_dict])
        (decomposed_channels, scales, num_samples, normalized_data) = wavelet_decompose_window(
            window_data,
            wavelet='cmor1.5-1.0',
            scales=None,
            normalization=False,
            sampling_period=1.0 / new_sps_val
        )
        for idx, region in enumerate(coeffs_dict.keys()):
            coeffs_complex = decomposed_channels[idx].flatten()
            coeffs_real = coeffs_complex.real
            coeffs_imag = coeffs_complex.imag
            coeffs_combined = np.concatenate([coeffs_real, coeffs_imag])
            coeffs_dict[region].append(coeffs_combined)

    return coeffs_dict

async def download_file(s3_client, bucket, csv_key):
    response = await s3_client.get_object(Bucket=bucket, Key=csv_key)
    body = await response['Body'].read()
    return csv_key, body

async def collect_coeffs_from_s3_async(csv_files, bucket, num_samples_per_file=10, window_length_sec=1.96, wvlet='db2', level=4):
    all_coeffs = {"frontal": [], "motor_temporal": [], "parietal_occipital": []}
    async with aioboto3.Session().client("s3") as s3_client:
        tasks = [download_file(s3_client, bucket, csv_file) for csv_file in csv_files]
        results = await asyncio.gather(*tasks)
        csv_data = {key: pd.read_csv(io.BytesIO(body)) for key, body in results}

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_csv_for_coeffs, key, bucket, None, window_length_sec, wvlet, level, num_samples_per_file)
                       for key in csv_data.keys()]
            for future in futures:
                coeffs = future.result()
                for region in all_coeffs:
                    all_coeffs[region].extend(coeffs[region])
    return all_coeffs

def collect_coeffs_from_s3(csv_files, bucket, num_samples_per_file=10, window_length_sec=1.96, wvlet='db2', level=4):
    return asyncio.run(collect_coeffs_from_s3_async(csv_files, bucket, num_samples_per_file, window_length_sec, wvlet, level))

def train_kmeans_models(all_coeffs, num_clusters=256, output_folder="/tmp"):
    kmeans_models = {}
    for region, coeffs_list in all_coeffs.items():
        coeffs_array = np.vstack(coeffs_list)
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
        kmeans.fit(coeffs_array)
        model_path = os.path.join(output_folder, f"kmeans_{region}.pkl")
        joblib.dump(kmeans, model_path)
        kmeans_models[region] = model_path
    return kmeans_models

if __name__ == "__main__":
    folders = list_s3_folders()[0:3]
    csv_files = []
    for i, folder in enumerate(folders):
        print(f"{i}/{len(folders)}: Looking into folder: {folder}")
        files = list_csv_files_in_folder(folder)
        csv_files.extend(files)
    print(f"Done listing. Total files: {len(csv_files)}")
    all_coeffs = collect_coeffs_from_s3(csv_files, "dataframes--use1-az6--x-s3", num_samples_per_file=200, window_length_sec=2, wvlet='db2', level=4)
    kmeans_model_paths = train_kmeans_models(all_coeffs, num_clusters=2048)
    s3 = boto3.client("s3")
    for region, path in kmeans_model_paths.items():
        s3.upload_file(path, "dataframes--use1-az6--x-s3", f"kmeans_models/kmeans_{region}.pkl")
        os.remove(path)