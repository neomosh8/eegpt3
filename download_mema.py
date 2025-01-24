import os
import boto3
import gdown
from urllib.parse import urlparse

# If you want a nicer progress bar, install tqdm and uncomment these:
# from tqdm import tqdm

SOURCE_BUCKET_NAME = "eegfiles"
FOLDER_URL = "https://drive.google.com/drive/folders/1G3vnCz1C3HsYoRipt6BT2v8ZfDn6y6r2"  # Public folder link
LOCAL_DOWNLOAD_DIR = "MEMA_dataset_downloaded"  # Temporary local directory for downloads
S3_PREFIX = "MEMA_dataset"  # S3 folder prefix


def gather_local_files(base_path):
    """
    Recursively gather all files under base_path.
    Returns a list of absolute file paths.
    """
    all_files = []
    for root, dirs, files in os.walk(base_path):
        for f in files:
            full_path = os.path.join(root, f)
            all_files.append(full_path)
    return all_files


def upload_files_to_s3(file_paths, bucket_name, s3_prefix):
    """
    Uploads each file in file_paths to the specified S3 bucket under s3_prefix.
    Prints progress in terms of file count and bytes transferred.
    """
    s3_client = boto3.client("s3")

    total_files = len(file_paths)
    # Calculate total bytes for all files
    total_size_bytes = sum(os.path.getsize(fp) for fp in file_paths)

    uploaded_bytes = 0

    print(f"Found {total_files} files to upload. Total size: {total_size_bytes / (1024 ** 3):.2f} GB\n")

    for idx, file_path in enumerate(file_paths, start=1):
        file_size = os.path.getsize(file_path)

        # Construct an S3 key that preserves relative path structure.
        # We find the relative path from the LOCAL_DOWNLOAD_DIR so subfolders remain intact in S3.
        rel_path = os.path.relpath(file_path, start=LOCAL_DOWNLOAD_DIR)
        s3_key = os.path.join(s3_prefix, rel_path).replace("\\", "/")  # Ensure forward slashes

        # Upload the file
        s3_client.upload_file(file_path, bucket_name, s3_key)

        # Update and print progress
        uploaded_bytes += file_size
        current_gb = uploaded_bytes / (1024 ** 3)
        total_gb = total_size_bytes / (1024 ** 3)
        print(
            f"Uploaded file {idx}/{total_files} | "
            f"{current_gb:.2f} / {total_gb:.2f} GB completed.",
            end="\r"
        )
    print("\nUpload to S3 completed.")


def main():
    # 1. Download all files/folders from the Google Drive folder
    print(f"Downloading folder from Google Drive:\n{FOLDER_URL}\n")
    # gdown will create a local folder with the folder name by default,
    # but you can override that by passing 'output=LOCAL_DOWNLOAD_DIR'.
    # For clarity, let's explicitly name the local download directory:
    downloaded_paths = gdown.download_folder(
        url=FOLDER_URL,
        output=LOCAL_DOWNLOAD_DIR,
        quiet=False,  # Shows gdown's own progress
        use_cookies=False
    )

    # 'downloaded_paths' may contain both files and folders (if subfolders exist).
    # We gather everything into a flat list of file paths:
    all_downloaded_files = gather_local_files(LOCAL_DOWNLOAD_DIR)

    # 2. Upload files to S3
    print("\nStarting upload to S3...")
    upload_files_to_s3(all_downloaded_files, SOURCE_BUCKET_NAME, S3_PREFIX)

    # (Optional) Clean up local downloads if desired:
    # import shutil
    # shutil.rmtree(LOCAL_DOWNLOAD_DIR)


if __name__ == "__main__":
    main()
