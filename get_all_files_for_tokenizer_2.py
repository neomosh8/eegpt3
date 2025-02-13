import boto3

s3 = boto3.client('s3')

def gather_coeffs_from_single_folder(
    bucket_name,
    prefix='output/',                  # The folder to look in
    local_output='coeffs_combined.txt' # Local file to write
):
    """
    Looks ONLY in the `prefix` folder inside `bucket_name` for any file
    that is a text file and whose name includes '_coeffs_'.
    Concatenates them locally, then uploads the combined file to the
    *root* of that bucket as 'combined_coeffs.txt'.
    """

    paginator = s3.get_paginator('list_objects_v2')
    total_matches_found = 0
    all_text_chunks = []
    page_count = 0

    print(f"Looking for files in s3://{bucket_name}/{prefix} ...")

    # Paginate over objects in the specified folder
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        page_count += 1
        print(f"→ Reading objects page {page_count} for prefix '{prefix}'...")

        for obj in page.get('Contents', []):
            key = obj['Key']
            # Adjusted matching: the file must be a text file and contain '_coeffs_' in its name.
            if key.endswith('.txt') and '_coeffs_' in key:
                total_matches_found += 1
                print(f"   Found match #{total_matches_found}: {key}")

                # Read file content
                file_obj = s3.get_object(Bucket=bucket_name, Key=key)
                file_content = file_obj['Body'].read().decode('utf-8')
                all_text_chunks.append(file_content)

    # Write all matching contents to a local file
    with open(local_output, 'w', encoding='utf-8') as f:
        for chunk in all_text_chunks:
            f.write(chunk)

    print(f"\nDone! Found {total_matches_found} files with '_coeffs_' in the name in {prefix}.")
    print(f"Local file created: {local_output}")

    # Optional: Upload the combined file back to the root of the bucket.
    # Remove this block if you do not wish to upload.
    upload_key = 'combined_coeffs.txt'  # File name at the root of the bucket
    s3.upload_file(local_output, bucket_name, upload_key)
    print(f"Uploaded combined file back to s3://{bucket_name}/{upload_key}")


# Example usage:
if __name__ == "__main__":
    gather_coeffs_from_single_folder(
        bucket_name='dataframes--use1-az6--x-s3',
        prefix='output/',  # Folder containing the files
        local_output='coeffs_combined.txt'
    )
