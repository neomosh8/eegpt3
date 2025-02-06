import requests
import boto3
from tqdm import tqdm


# --- Helper Class to Track Progress with tqdm ---
class ProgressReader:
    def __init__(self, stream, total, desc=""):
        self.stream = stream
        self.total = total
        self.pbar = tqdm(total=total, unit='B', unit_scale=True, desc=desc)

    def read(self, amt=None):
        data = self.stream.read(amt)
        self.pbar.update(len(data))
        return data

    def __getattr__(self, attr):
        # Forward any other attribute/method calls to the underlying stream.
        return getattr(self.stream, attr)

    def close(self):
        self.pbar.close()
        if hasattr(self.stream, "close"):
            self.stream.close()


# --- Configuration ---

# Zenodo record ID
ZENODO_RECORD_ID = "7893847"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

# S3 configuration
S3_BUCKET = "dataframes--use1-az6--x-s3"  # Your S3 bucket name
S3_BASE_FOLDER = "attention fintune"  # Base folder in S3

# --- Retrieve Zenodo Record Metadata ---
response = requests.get(ZENODO_API_URL)
if response.status_code != 200:
    raise Exception(f"Failed to fetch Zenodo record: HTTP {response.status_code}")

record = response.json()

# --- Initialize S3 Client ---
s3_client = boto3.client("s3")

# --- Process Each File in the Zenodo Record ---
for file_info in record.get("files", []):
    # Retrieve the file name (adjust the key name if necessary)
    file_name = file_info.get("key") or file_info.get("filename")

    # Retrieve the download URL from the links dictionary
    links = file_info.get("links", {})
    download_url = links.get("download")

    if not download_url:
        print(f"Download URL not found for file '{file_name}'. Available link keys: {list(links.keys())}")
        # Fallback to using the 'self' link if available
        download_url = links.get("self")
        if download_url:
            print(f"Using 'self' link as fallback for file '{file_name}'.")
        else:
            print(f"Skipping file '{file_name}' as no download URL is available.")
            continue

    print(f"Downloading '{file_name}' from {download_url}...")
    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()  # Ensure we got a valid response
        r.raw.decode_content = True  # Enable proper decoding of the raw stream

        # Get the total file size from the header (if provided)
        total_bytes = int(r.headers.get("Content-Length", 0))
        if total_bytes:
            print(f"Expected file size: {total_bytes} bytes")
        else:
            print("Content-Length header not provided.")

        # Wrap the raw stream with our progress tracker
        progress_stream = ProgressReader(r.raw, total=total_bytes, desc=file_name)

        # Construct the S3 key using the desired folder structure
        s3_key = f"{S3_BASE_FOLDER}/{ZENODO_RECORD_ID}/{file_name}"
        print(f"Uploading '{file_name}' to s3://{S3_BUCKET}/{s3_key} ...")

        # Upload the streamed file directly to S3
        s3_client.upload_fileobj(progress_stream, S3_BUCKET, s3_key)

        # Close the progress bar
        progress_stream.close()

    print(f"Uploaded '{file_name}' successfully!\n")
