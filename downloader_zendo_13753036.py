import requests
import boto3


# --- Helper Class to Track Bytes Read ---
class ProgressReader:
    def __init__(self, stream):
        self.stream = stream
        self.bytes_read = 0

    def read(self, amt=None):
        data = self.stream.read(amt)
        self.bytes_read += len(data)
        return data

    def __getattr__(self, attr):
        # Forward any other attribute/method calls to the underlying stream.
        return getattr(self.stream, attr)


# --- Configuration ---

# Zenodo record ID (change as needed)
ZENODO_RECORD_ID = "13753036"
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

        # Optionally print the expected file size if provided by the server
        content_length = r.headers.get("Content-Length")
        if content_length:
            print(f"Expected file size: {content_length} bytes")

        # Wrap the raw stream to track the number of bytes read
        progress_stream = ProgressReader(r.raw)

        # Construct the S3 key to include the desired folder structure
        s3_key = f"{S3_BASE_FOLDER}/{ZENODO_RECORD_ID}/{file_name}"
        print(f"Uploading '{file_name}' to s3://{S3_BUCKET}/{s3_key} ...")

        # Upload the streamed file directly to S3
        s3_client.upload_fileobj(progress_stream, S3_BUCKET, s3_key)

        # Report the total number of bytes read (downloaded and uploaded)
        print(f"Uploaded '{file_name}' successfully! Total bytes transferred: {progress_stream.bytes_read}\n")
