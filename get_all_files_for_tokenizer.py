import boto3

s3 = boto3.client('s3')

def list_s3_folders(bucket_name='dataframes--use1-az6--x-s3', prefix=''):
    """
    Lists subfolders in the specified S3 bucket/prefix by using '/' as a delimiter.
    """
    response = s3.list_objects_v2(
        Bucket=bucket_name,
        Prefix=prefix,
        Delimiter='/'
    )
    # Extract folder names (strip trailing '/')
    folders = [
        pfx.get('Prefix').rstrip('/')
        for pfx in response.get('CommonPrefixes', [])
    ]
    return folders

def gather_coeffs_from_s3(
    bucket_name='dataframes--use1-az6--x-s3',
    prefix='',  # empty means look at top-level directories in the bucket
    local_output='coeffs_combined.txt'
):
    """
    Gathers all text files ending in '_coeffs.txt' under each subfolder of the given prefix,
    concatenates them, and saves the result locally to local_output.
    Also prints how many matching files were found.
    """
    top_level_folders = list_s3_folders(bucket_name, prefix)

    all_text_chunks = []
    total_matches_found = 0

    for folder in top_level_folders:
        # List all objects under that folder
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=f"{folder}/")
        if 'Contents' not in response:
            # If there's no 'Contents', then folder is empty
            continue

        for obj in response['Contents']:
            key = obj['Key']
            # Check if the file ends with "_coeffs.txt"
            if key.endswith('_coeffs.txt'):
                total_matches_found += 1
                # Download the file content
                file_obj = s3.get_object(Bucket=bucket_name, Key=key)
                file_content = file_obj['Body'].read().decode('utf-8')
                all_text_chunks.append(file_content)

    # Write the concatenated text to a local file
    with open(local_output, 'w', encoding='utf-8') as f:
        for chunk in all_text_chunks:
            f.write(chunk)
            f.write('\n')  # optional separator

    print(f"Found {total_matches_found} matches ending with '_coeffs.txt'.")

# Example usage
if __name__ == "__main__":
    gather_coeffs_from_s3(
        bucket_name='dataframes--use1-az6--x-s3',
        prefix='',  # or "some_prefix/" if you want a deeper path
        local_output='combined_coeffs.txt'
    )
