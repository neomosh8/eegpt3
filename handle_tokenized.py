import os
import boto3

def upload_folder_to_s3(local_folder_path: str, bucket_name: str, s3_prefix: str):
    """
    Uploads the contents of a local folder (and its subfolders) to an S3 bucket.

    :param local_folder_path: Path to the local folder whose contents should be uploaded.
    :param bucket_name: Name of the S3 bucket.
    :param s3_prefix: Prefix inside the bucket under which the files will be uploaded.
    """
    s3_client = boto3.client("s3")

    # Gather all files in the local_folder_path
    all_files = []
    for root, dirs, files in os.walk(local_folder_path):
        for file_name in files:
            local_path = os.path.join(root, file_name)
            all_files.append(local_path)

    total_files = len(all_files)
    if total_files == 0:
        print("No files found to upload.")
        return

    print(f"Found {total_files} files to upload.")

    # Upload each file
    for idx, local_path in enumerate(all_files, start=1):
        # Construct the relative path (so the S3 object keys mirror the local folder structure)
        relative_path = os.path.relpath(local_path, start=local_folder_path)
        s3_path = os.path.join(s3_prefix, relative_path).replace("\\", "/")

        print(f"Uploading file {idx}/{total_files}: {local_path} --> s3://{bucket_name}/{s3_path}")
        s3_client.upload_file(local_path, bucket_name, s3_path)

    print(f"Upload completed. Total files uploaded: {total_files}.")


def download_folder_from_s3(bucket_name: str, s3_prefix: str, local_folder_path: str):
    """
    Downloads all objects from an S3 prefix into a local folder, recreating the prefix structure locally.

    :param bucket_name: Name of the S3 bucket.
    :param s3_prefix: Prefix of objects to download from the S3 bucket.
    :param local_folder_path: Path to the local folder where objects should be saved.
    """
    s3_client = boto3.client("s3")
    paginator = s3_client.get_paginator("list_objects_v2")

    # First, collect all object keys under the prefix
    object_keys = []
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if "Contents" not in page:
            continue
        for obj in page["Contents"]:
            key = obj["Key"]
            # Skip "directory" placeholders if they appear
            if key.endswith("/"):
                continue
            object_keys.append(key)

    total_objects = len(object_keys)
    if total_objects == 0:
        print("No files found in S3 to download.")
        return

    print(f"Found {total_objects} files in s3://{bucket_name}/{s3_prefix} to download.")

    # Download each object
    for idx, key in enumerate(object_keys, start=1):
        relative_path = os.path.relpath(key, start=s3_prefix)
        local_path = os.path.join(local_folder_path, relative_path)

        # Create any necessary local directories
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        print(f"Downloading file {idx}/{total_objects}: s3://{bucket_name}/{key} --> {local_path}")
        s3_client.download_file(bucket_name, key, local_path)

    print(f"Download completed. Total files downloaded: {total_objects}.")

if __name__ == "__main__":
    # Example usage:
    # 1. Upload the folder ./local_shards to S3
    # upload_folder_to_s3(
    #     local_folder_path="./local_shards",
    #     bucket_name="dataframes--use1-az6--x-s3",
    #     s3_prefix="uploads/local_shards"
    # )

    # 2. Download the prefix uploads/local_shards from S3 into a local folder
    download_folder_from_s3(
        bucket_name="dataframes--use1-az6--x-s3",
        s3_prefix="uploads/local_shards/",
        local_folder_path="./local_shards"
    )
    3.
    # upload_folder_to_s3(
    #     local_folder_path="./validation_datasets_imageNet",
    #     bucket_name="dataframes--use1-az6--x-s3",
    #     s3_prefix="uploads/validation_datasets_imageNet"
    # )
    # 4. Download the prefix uploads/local_shards from S3 into a local folder
    download_folder_from_s3(
        bucket_name="dataframes--use1-az6--x-s3",
        s3_prefix="uploads/validation_datasets_imageNet/",
        local_folder_path="./validation_datasets_imageNet"
    )
