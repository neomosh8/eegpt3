import boto3

s3 = boto3.client('s3')


def list_s3_folders_with_paginator(bucket_name, prefix=''):
    """
    Lists subfolders in the specified S3 bucket/prefix using a paginator,
    printing progress along the way.
    """
    paginator = s3.get_paginator('list_objects_v2')
    result_prefixes = []
    page_count = 0

    print(f"Listing subfolders in S3 bucket '{bucket_name}' with prefix='{prefix}'...")

    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter='/'):
        page_count += 1
        print(f"  → Reading 'folders' page {page_count}...")

        # Each 'page' can contain multiple CommonPrefixes (i.e., subfolders)
        common_prefixes = page.get('CommonPrefixes', [])
        for pfx in common_prefixes:
            folder = pfx.get('Prefix').rstrip('/')
            print(f"     Found folder: {folder}")
            result_prefixes.append(folder)

    print(f"\nTotal top-level folders found: {len(result_prefixes)}\n")
    return result_prefixes


def gather_coeffs_from_s3_with_paginator(bucket_name, prefix='', local_output='coeffs_combined.txt'):
    """
    Gathers all text files ending in '_coeffs.txt' under each subfolder of the given prefix,
    concatenates them, and saves the result locally to local_output.
    Also prints how many matching files were found, and shows progress.
    """
    # First, get the list of top-level folders
    top_level_folders = list_s3_folders_with_paginator(bucket_name, prefix)

    total_matches_found = 0
    all_text_chunks = []

    print("Gathering '_coeffs.txt' files from each folder...\n")

    # Go through each folder and list objects with a paginator
    for idx, folder in enumerate(top_level_folders, start=1):
        print(f"[{idx}/{len(top_level_folders)}] Processing folder: '{folder}'")

        # We'll keep track of pages for debugging
        page_count = 0

        # Create a paginator for listing objects inside this folder
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name, Prefix=f"{folder}/"):
            page_count += 1
            print(f"   → Reading objects page {page_count} in folder '{folder}'...")

            # 'Contents' might not be in page if no objects
            for obj in page.get('Contents', []):
                key = obj['Key']
                # Check for matching filename
                if key.endswith('_coeffs.txt'):
                    total_matches_found += 1
                    print(f"      Found match: {key}")
                    file_obj = s3.get_object(Bucket=bucket_name, Key=key)
                    file_content = file_obj['Body'].read().decode('utf-8')
                    all_text_chunks.append(file_content)

    # Write all matching contents to local_output
    with open(local_output, 'w', encoding='utf-8') as f:
        for chunk in all_text_chunks:
            f.write(chunk)
            f.write('\n')  # optional extra newline as a separator

    print(f"\nDone! Found {total_matches_found} matches ending with '_coeffs.txt'.")
    print(f"Results have been written to '{local_output}'.")


# Example usage
if __name__ == "__main__":
    gather_coeffs_from_s3_with_paginator(
        bucket_name='dataframes--use1-az6--x-s3',
        prefix='',  # or "some_prefix/"
        local_output='combined_coeffs.txt'
    )
