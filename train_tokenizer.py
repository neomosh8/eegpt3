import os

# Assume the StreamingPassBasedWordLevelBPETokenizer class is defined above or imported.
from Tokenizer_new_arch import StreamingPassBasedWordLevelBPETokenizer

if __name__ == "__main__":
    # Parameters (adjust num_merges and num_passes as needed)
    training_file = "coeffs_combined.txt"
    num_merges = 10000      # maximum number of merge operations (example value)
    num_passes = 100      # maximum number of passes (if not specified, same as num_merges)
    chunk_size = 10240 * 10240  # 1 MB chunks for streaming

    # Instantiate the tokenizer.
    tokenizer = StreamingPassBasedWordLevelBPETokenizer()

    print(f"Starting training on {training_file} ...")
    tokenizer.train(training_file, num_merges=num_merges, num_passes=num_passes, chunk_size=chunk_size)

    # Test encoding and decoding on a sample text.
    sample_text = "D23 A42 F223 D34 D23 A42"
    encoded_ids = tokenizer.encode(sample_text)
    decoded_text = tokenizer.decode(encoded_ids)
    print("\n--- Encoding / Decoding Test ---")
    print("Sample text:", sample_text)
    print("Encoded token IDs:", encoded_ids)
    print("Decoded text:", decoded_text)

    # Save the trained model in the 'neotokenizer_2' folder.
    save_folder = "neotokenizer_2"
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, "tokenizer_model.json")
    tokenizer.save(save_path)
    print("\nModel saved to:", save_path)
