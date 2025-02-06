import os
import requests
import boto3
from tqdm import tqdm
import concurrent.futures

# --- Helper Class for Progress Reporting ---
class ProgressReader:
    """
    Wraps a stream (e.g. r.raw from a requests response) to update a tqdm progress bar on each read.
    """
    def __init__(self, stream, total, desc=""):
        self.stream = stream
        self.total = total
        self.pbar = tqdm(total=total, unit='B', unit_scale=True, desc=desc)

    def read(self, amt=None):
        data = self.stream.read(amt)
        self.pbar.update(len(data))
        return data

    def __getattr__(self, attr):
        # Pass-through for any attribute/method not explicitly overridden.
        return getattr(self.stream, attr)

    def close(self):
        self.pbar.close()
        if hasattr(self.stream, "close"):
            self.stream.close()


# --- Main Processing Function ---
def process_record(record_url):
    """
    Processes a single Zenodo record:
      - Fetches record metadata.
      - For each file in the record, downloads the file (with progress bar) and uploads to S3.
    """
    # Extract the record ID from the URL (e.g. "https://zenodo.org/records/7893847" -> "7893847")
    record_id = record_url.rstrip("/").split("/")[-1]
    zenodo_api_url = f"https://zenodo.org/api/records/{record_id}"

    print(f"[Record {record_id}] Fetching metadata from {zenodo_api_url} ...")
    response = requests.get(zenodo_api_url)
    if response.status_code != 200:
        print(f"[Record {record_id}] ERROR: Failed to fetch record (HTTP {response.status_code}).")
        return

    record = response.json()

    # Initialize an S3 client (each process should create its own client)
    s3_bucket = "dataframes--use1-az6--x-s3"  # <-- REPLACE with your bucket name if needed
    s3_base_folder = "attention fintune"
    s3_client = boto3.client("s3")

    # Process each file listed in the record metadata
    for file_info in record.get("files", []):
        file_name = file_info.get("key") or file_info.get("filename")
        links = file_info.get("links", {})
        download_url = links.get("download")
        if not download_url:
            print(f"[Record {record_id}] Download URL not found for file '{file_name}'. Available keys: {list(links.keys())}")
            download_url = links.get("self")
            if download_url:
                print(f"[Record {record_id}] Using 'self' link as fallback for file '{file_name}'.")
            else:
                print(f"[Record {record_id}] Skipping file '{file_name}' as no download URL is available.")
                continue

        print(f"[Record {record_id}] Downloading '{file_name}' from {download_url} ...")
        try:
            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()  # Abort on HTTP errors
                r.raw.decode_content = True

                # Get total file size (if provided by the server)
                total_bytes = int(r.headers.get("Content-Length", 0))
                if total_bytes:
                    print(f"[Record {record_id}] Expected file size: {total_bytes} bytes")
                else:
                    print(f"[Record {record_id}] No Content-Length header provided for file '{file_name}'.")

                # Wrap the raw stream with our progress tracker
                progress_stream = ProgressReader(r.raw, total=total_bytes, desc=f"{record_id}-{file_name}")

                # Construct the S3 key: attention fintune/<record_id>/<file_name>
                s3_key = f"{s3_base_folder}/{record_id}/{file_name}"
                print(f"[Record {record_id}] Uploading '{file_name}' to s3://{s3_bucket}/{s3_key} ...")
                s3_client.upload_fileobj(progress_stream, s3_bucket, s3_key)
                progress_stream.close()
                print(f"[Record {record_id}] Uploaded '{file_name}' successfully!\n")
        except Exception as e:
            print(f"[Record {record_id}] ERROR processing file '{file_name}': {e}")


# --- Main Entry Point ---
def main():
    # List of Zenodo record URLs to process
    record_urls = [
        "https://zenodo.org/records/1199011",
        "https://zenodo.org/records/4518754"
        # "https://zenodo.org/records/6395564",
        # "https://zenodo.org/records/2536267",
        # "https://zenodo.org/records/12734987",
        # "https://zenodo.org/records/5016646",
        # "https://zenodo.org/records/10803229",
        # "https://zenodo.org/records/3745593",
        # "https://zenodo.org/records/4385970",
        # "https://zenodo.org/records/5512578",
        # "https://zenodo.org/records/30084",
        # "https://zenodo.org/records/10518106",
        # "https://zenodo.org/records/8173495",
        # "https://zenodo.org/records/7650679",
        # "https://zenodo.org/records/7795585",
        # "https://zenodo.org/records/4004271",
        # "https://zenodo.org/records/14732590",
        # "https://zenodo.org/records/4987915",
        # "https://zenodo.org/records/10980117",
        # "https://zenodo.org/records/1199011",
        # "https://zenodo.org/records/4537751"
    ]

    # Use the maximum number of available cores.
    max_workers = os.cpu_count() or 4

    # Use ProcessPoolExecutor to run downloads in parallel (each record is processed in its own process).
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_record, url) for url in record_urls]

        # Wait for all tasks to complete and catch exceptions.
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print("An error occurred:", exc)


if __name__ == '__main__':
    main()
