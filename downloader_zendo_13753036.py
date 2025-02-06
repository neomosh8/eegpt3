import requests
import boto3

# --- Configuration ---
# Use the Zenodo record ID you want (e.g., "13753036" or "4694668")
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
    # Get the file name (adjust the key name if necessary)
    file_name = file_info.get("key") or file_info.get("filename")

    # Retrieve the download URL from the links dictionary
    links = file_info.get("links", {})
    download_url = links.get("download")

    if not download_url:
        # Print available keys for debugging
        print(f"Download URL not found for file '{file_name}'. Available link keys: {list(links.keys())}")
        # Optionally, try using 'self' as a fallback
        download_url = links.get("self")
        if download_url:
            print(f"Using 'self' link as fallback for file '{file_name}'.")
        else:
            print(f"Skipping file '{file_name}' as no download URL is available.")
            continue  # Skip this file if no suitable link is found

    print(f"Downloading '{file_name}' from {download_url}...")

    # Stream download the file from Zenodo
    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()  # Raise an error for bad status codes
        r.raw.decode_content = True

        # Construct the S3 key using the desired folder structure
        s3_key = f"{S3_BASE_FOLDER}/{ZENODO_RECORD_ID}/{file_name}"
        print(f"Uploading '{file_name}' to s3://{S3_BUCKET}/{s3_key} ...")

        # Upload the streamed file directly to S3
        s3_client.upload_fileobj(r.raw, S3_BUCKET, s3_key)

    print(f"Uploaded '{file_name}' successfully!\n")
