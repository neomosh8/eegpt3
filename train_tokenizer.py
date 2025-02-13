import os
import random

# Assume BPETokenizer is already defined and imported as shown previously.
from Tokenizer_new_arch import BPETokenizer

# 1. Instantiate and train the tokenizer.
tokenizer = BPETokenizer()
training_file = "coeffs_combined.txt"
print("Training tokenizer on '{}' ...".format(training_file))
tokenizer.train(training_file, num_merges=10000, chunk_size=10000)

# 2. Compute compression metrics.
#    We define two metrics:
#      - Average number of characters per word (from the raw file).
#      - Average number of tokens per word after BPE segmentation.
#    Their ratio (chars per token) gives an idea of how much the words got “compressed.”
unique_words = set()
total_chars = 0
total_word_count = 0

with open(training_file, "r", encoding="utf-8") as f:
    for line in f:
        words = line.strip().split()
        for word in words:
            unique_words.add(word)
            total_chars += len(word)
            total_word_count += 1

avg_chars_per_word = total_chars / total_word_count if total_word_count > 0 else 0

# Compute average tokens per word over the unique words.
total_tokens = 0
for word in unique_words:
    # Note: tokenizer.encode() inserts a space token between words.
    # Since we're encoding a single word, there should be no spaces.
    token_ids = tokenizer.encode(word)
    # Remove any potential space token (if present).
    token_count = sum(1 for tid in token_ids if tid != tokenizer.token2id.get(" ", -1))
    total_tokens += token_count

avg_tokens_per_word = total_tokens / len(unique_words) if unique_words else 0
compression_ratio = avg_chars_per_word / avg_tokens_per_word if avg_tokens_per_word > 0 else 0

print("\nCompression metrics:")
print("  - Average characters per word: {:.2f}".format(avg_chars_per_word))
print("  - Average tokens per word after BPE segmentation: {:.2f}".format(avg_tokens_per_word))
print("  - Compression ratio (chars per token): {:.2f}".format(compression_ratio))

# 3. Test a random full-word token from the training data.
#    (During training, words that were seen as a single token are recorded in tokenizer.full_words.)
n = 5  # Change this to any desired length.
candidates = [w for w in tokenizer.full_words if len(w) == n]
if candidates:
    random_word = random.choice(candidates)
    print("\nRandom full-word token of length {}: '{}'".format(n, random_word))
    encoded = tokenizer.encode(random_word)
    print("Encoded token IDs:", encoded)
    decoded = tokenizer.decode(encoded)
    print("Decoded word:", decoded)
else:
    print("\nNo full-word token of length {} was found in the training data.".format(n))

# 4. Save the trained tokenizer in a folder named 'neotokenizer_2'.
save_folder = "neotokenizer_2"
os.makedirs(save_folder, exist_ok=True)
save_path = os.path.join(save_folder, "tokenizer.json")
tokenizer.save(save_path)
print("\nTokenizer saved to:", save_path)
