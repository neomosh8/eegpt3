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

    # Walk through all subdirectories and files under local_folder_path
    for root, dirs, files in os.walk(local_folder_path):
        for file in files:
            local_path = os.path.join(root, file)

            # Construct the relative path (so the S3 object keys mirror local folder structure)
            relative_path = os.path.relpath(local_path, start=local_folder_path)
            s3_path = os.path.join(s3_prefix, relative_path).replace("\\", "/")

            print(f"Uploading {local_path} to s3://{bucket_name}/{s3_path}")
            s3_client.upload_file(local_path, bucket_name, s3_path)


def download_folder_from_s3(bucket_name: str, s3_prefix: str, local_folder_path: str):
    """
    Downloads all objects from an S3 prefix into a local folder, recreating the prefix structure locally.

    :param bucket_name: Name of the S3 bucket.
    :param s3_prefix: Prefix of objects to download from the S3 bucket.
    :param local_folder_path: Path to the local folder where objects should be saved.
    """
    s3_client = boto3.client("s3")
    paginator = s3_client.get_paginator("list_objects_v2")

    # Iterate over all pages of S3 object listings
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if "Contents" not in page:
            continue
        for obj in page["Contents"]:
            key = obj["Key"]

            # Skip "directory" placeholders if they appear
            if key.endswith("/"):
                continue

            # Recreate the S3 folder structure locally
            relative_path = os.path.relpath(key, start=s3_prefix)
            local_path = os.path.join(local_folder_path, relative_path)

            # Make sure the local directories exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            print(f"Downloading s3://{bucket_name}/{key} to {local_path}")
            s3_client.download_file(bucket_name, key, local_path)


if __name__ == "__main__":
    # Example usage:
    # 1. Upload the folder ./local_shards to S3
    upload_folder_to_s3(
        local_folder_path="./local_shards",
        bucket_name="dataframes--use1-az6--x-s3",
        s3_prefix="uploads/local_shards"
    )

    # # 2. Download the prefix uploads/local_shards from S3 into a local folder
    # download_folder_from_s3(
    #     bucket_name="dataframes--use1-az6--x-s3",
    #     s3_prefix="uploads/local_shards",
    #     local_folder_path="./downloaded_local_shards"
    # )
