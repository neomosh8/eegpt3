import os
import random
from Tokenizer_new_arch import StreamingPassBasedWordLevelBPETokenizer  # Adjust import as needed


def pick_random_sample(file_path, n, sample_chunk_size=1024 * 1024):
    """
    Pick a random contiguous sequence of n tokens from the file.
    Since the file is one long line, this function:
      - Chooses a random byte offset.
      - Advances to the next token boundary.
      - Reads enough data to extract n tokens.

    Parameters:
      file_path: Path to the huge file.
      n: Number of tokens to extract.
      sample_chunk_size: How many bytes to read after seeking (default 1MB).

    Returns:
      A string containing n space-separated tokens.
    """
    file_size = os.path.getsize(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        # Choose a random offset (ensure it's not the very end)
        offset = random.randint(0, file_size - 100)
        f.seek(offset)
        # If we're not at the start, skip to the next space to avoid a partial token.
        if offset != 0:
            char = f.read(1)
            while char and not char.isspace():
                char = f.read(1)
            # Now skip any additional spaces.
            while char and char.isspace():
                char = f.read(1)
        # Now read a chunk (e.g. 1MB) and split tokens.
        data = f.read(sample_chunk_size)
        tokens = data.split()
        if len(tokens) < n:
            # If not enough tokens, read from the beginning of the file.
            f.seek(0)
            data = f.read(sample_chunk_size)
            tokens = data.split()
        sample_tokens = tokens[:n]
        return " ".join(sample_tokens)


if __name__ == "__main__":
    # Parameters:
    training_file = "coeffs_combined.txt"
    num_merges = 3  # example value; adjust as needed
    num_passes = 3  # number of passes (if not provided, same as num_merges)
    chunk_size = 1024 * 1024 * 1024  # 1 GB chunk size
    chunk_size = 1024 * 1024   # 1 GB chunk size

    # Instantiate the tokenizer.
    tokenizer = StreamingPassBasedWordLevelBPETokenizer()

    print(f"Starting training on {training_file} ...")
    tokenizer.train(training_file, num_merges=num_merges, num_passes=num_passes, chunk_size=chunk_size)

    # Pick a random sample text from the file: choose n tokens at random.
    n = 10  # For example, pick a sample of 10 tokens.
    sample_text = pick_random_sample(training_file, n)
    print("\n--- Random Sample Text (from file) ---")
    print("Sample text:", sample_text)

    # Test encoding and decoding on the sample text.
    encoded_ids = tokenizer.encode(sample_text)
    decoded_text = tokenizer.decode(encoded_ids)
    print("\n--- Encoding / Decoding Test ---")
    print("Encoded token IDs:", encoded_ids)
    print("Decoded text:", decoded_text)

    # Save the trained model in the 'neotokenizer_2' folder.
    save_folder = "neotokenizer_2"
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, "tokenizer_model.json")
    tokenizer.save(save_path)
    print("\nModel saved to:", save_path)
