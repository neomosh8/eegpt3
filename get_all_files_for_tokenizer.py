import boto3

s3 = boto3.client('s3')

def list_s3_folders_with_paginator(bucket_name, prefix=''):
    paginator = s3.get_paginator('list_objects_v2')
    result_prefixes = []
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter='/'):
        common_prefixes = page.get('CommonPrefixes', [])
        for pfx in common_prefixes:
            result_prefixes.append(pfx.get('Prefix').rstrip('/'))
    return result_prefixes

def gather_coeffs_from_s3_with_paginator(bucket_name, prefix='', local_output='coeffs_combined.txt'):
    top_level_folders = list_s3_folders_with_paginator(bucket_name, prefix)

    total_matches_found = 0
    all_text_chunks = []

    # For each folder, use the paginator again
    for folder in top_level_folders:
        for page in s3.get_paginator('list_objects_v2').paginate(Bucket=bucket_name, Prefix=f"{folder}/"):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('_coeffs.txt'):
                    total_matches_found += 1
                    file_obj = s3.get_object(Bucket=bucket_name, Key=key)
                    file_content = file_obj['Body'].read().decode('utf-8')
                    all_text_chunks.append(file_content)

    with open(local_output, 'w', encoding='utf-8') as f:
        for chunk in all_text_chunks:
            f.write(chunk + '\n')

    print(f"Found {total_matches_found} matches ending with '_coeffs.txt'.")

if __name__ == "__main__":
    gather_coeffs_from_s3_with_paginator(
        bucket_name='dataframes--use1-az6--x-s3',
        prefix='',
        local_output='combined_coeffs.txt'
    )
