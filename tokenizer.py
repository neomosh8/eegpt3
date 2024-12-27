import collections
import sys


def read_tokens_from_file(filepath):
    """
    Read a file and split on whitespace into tokens.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    # Basic whitespace split:
    tokens = text.split()
    return tokens


def get_top_bigrams(tokens, num_merges=10):
    """
    Count bigram frequencies and return the top `num_merges` bigrams.
    Return as a list of tuples (bigram, frequency).
    """
    bigram_counts = collections.Counter()
    for i in range(len(tokens) - 1):
        bigram = (tokens[i], tokens[i + 1])
        bigram_counts[bigram] += 1

    # Get most common bigrams
    top_bigrams = bigram_counts.most_common(num_merges)
    return top_bigrams  # list of ((tokenA, tokenB), freq)


def apply_bpe_merges(tokens, merges):
    """
    Given a list of merges (each is ((A,B), freq)),
    turn every occurrence of (A, B) in the token list into a single token "A_B".

    We'll do a single pass for simplicity:
      - scan left-to-right, whenever we see (A, B) we merge them once.
    In real BPE, you'd often do repeated passes or more complex logic,
    but this is a basic demonstration.
    """
    merged_tokens = []
    skip_next = False

    # Convert merges list into a set of pairs for quick lookup
    # We only care about the pair (A, B), ignoring frequency for merging
    merge_pairs = set(m[0] for m in merges)

    i = 0
    while i < len(tokens):
        if skip_next:
            skip_next = False
            i += 1
            continue

        if i < len(tokens) - 1:
            pair = (tokens[i], tokens[i + 1])
            if pair in merge_pairs:
                # Merge them
                merged_tokens.append(tokens[i] + "_" + tokens[i + 1])
                skip_next = True  # skip the next token because it's merged
            else:
                merged_tokens.append(tokens[i])
        else:
            # last token, just append
            merged_tokens.append(tokens[i])
        i += 1

    return merged_tokens


def run_length_encode(tokens):
    """
    Perform a simple run-length encoding:
    If we have T repeated k times, represent it as [T, R(k-1)]
    with a special 'R' token plus the count.

    E.g. T T T T => T R3  (meaning T repeated 4 times).

    We'll represent R(k) as a single token: 'R=<k>'
    """
    if not tokens:
        return []

    rle_tokens = []

    current_token = tokens[0]
    count = 1

    for t in tokens[1:]:
        if t == current_token:
            count += 1
        else:
            # flush the previous run
            rle_tokens.append(current_token)
            if count > 1:
                rle_tokens.append(f"R={count - 1}")
            # reset
            current_token = t
            count = 1

    # flush last run
    rle_tokens.append(current_token)
    if count > 1:
        rle_tokens.append(f"R={count - 1}")

    return rle_tokens


def run_length_decode(rle_tokens):
    """
    Decode the tokens created by run_length_encode.

    Whenever we see a token like 'R=3', we repeat the previous token 3 more times.
    """
    decoded = []
    for i, tok in enumerate(rle_tokens):
        if tok.startswith("R="):
            # get the repeat count
            repeat_count = int(tok.split("=")[1])
            # repeat the last real token
            if decoded:  # sanity check
                last_tok = decoded[-1]
                decoded.extend([last_tok] * repeat_count)
        else:
            decoded.append(tok)
    return decoded


def undo_bpe_merges(tokens):
    """
    Undo merges by splitting on the underscore '_'.
    If a token was originally "A_B", we return ["A", "B"].
    If it doesn't have '_', we keep it as is.
    """
    expanded = []
    for tok in tokens:
        if "_" in tok:
            parts = tok.split("_")
            expanded.extend(parts)
        else:
            expanded.append(tok)
    return expanded


def main():
    # -------------------------------------------------------------------------
    # 1) Read and tokenize input text
    # -------------------------------------------------------------------------

    filepath = 'quantized_coeffs.txt'
    num_merges = int(sys.argv[2]) if len(sys.argv) > 2 else 400

    tokens = read_tokens_from_file(filepath)
    original_len = len(tokens)
    print("Original token count:", original_len)

    # -------------------------------------------------------------------------
    # 2) Learn top bigrams (mini BPE merges)
    # -------------------------------------------------------------------------
    top_bigrams = get_top_bigrams(tokens, num_merges=num_merges)
    print(f"Top {num_merges} bigrams:", top_bigrams)

    # -------------------------------------------------------------------------
    # 3) Apply BPE merges
    # -------------------------------------------------------------------------
    bpe_tokens = apply_bpe_merges(tokens, top_bigrams)
    bpe_len = len(bpe_tokens)

    # Calculate BPE compression ratio
    if bpe_len > 0:
        bpe_compression_ratio = original_len / bpe_len
    else:
        bpe_compression_ratio = 1.0  # fallback if empty

    print(f"After BPE merges, token count: {bpe_len} "
          f"(BPE compression ratio: {bpe_compression_ratio:.2f})")

    # -------------------------------------------------------------------------
    # 4) Apply Run-Length Encoding
    # -------------------------------------------------------------------------
    rle_tokens = run_length_encode(bpe_tokens)
    rle_len = len(rle_tokens)

    # RLE compression ratio relative to BPE
    if rle_len > 0:
        rle_compression_ratio_bpe = bpe_len / rle_len
        overall_compression_ratio = original_len / rle_len
    else:
        rle_compression_ratio_bpe = 1.0
        overall_compression_ratio = 1.0

    print(f"After RLE, token count: {rle_len} "
          f"(RLE compression vs BPE: {rle_compression_ratio_bpe:.2f}; "
          f"overall: {overall_compression_ratio:.2f})")

    # Optionally, compute percentage of tokens reduced overall
    overall_reduction_percent = (1 - (rle_len / original_len)) * 100
    print(f"Overall tokens reduced by: {overall_reduction_percent:.2f}%")

    print("\nSample of encoded tokens:", rle_tokens[:50], "...")

    # -------------------------------------------------------------------------
    # 5) Decode back
    # -------------------------------------------------------------------------
    decoded_bpe_tokens = run_length_decode(rle_tokens)
    final_tokens = undo_bpe_merges(decoded_bpe_tokens)

    # -------------------------------------------------------------------------
    # 6) Verify correctness & show final results
    # -------------------------------------------------------------------------
    final_len = len(final_tokens)
    print("\nDecoded token count:", final_len)

    if final_tokens == tokens:
        print("Round-trip successful! Decoded tokens match the original.")
    else:
        print("WARNING: Decoded tokens do NOT match the original!")

    # Print a small sample
    print("\nOriginal sample:", tokens[:50], "...")
    print("Final decoded sample:", final_tokens[:50], "...")


if __name__ == "__main__":
    main()
