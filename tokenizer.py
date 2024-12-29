import collections
import sys
import os
import json
import concurrent.futures


def read_tokens_from_file(filepath):
    """
    Read a file and split on whitespace into tokens.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    # Basic whitespace split:
    tokens = text.split()
    return tokens


def count_bigrams_in_chunk(chunk, offset_token=None):
    """
    Count bigrams within a chunk of tokens.
    If offset_token is provided, count the bigram that crosses the boundary
    (offset_token, chunk[0]) if the chunk is non-empty.
    Returns a collections.Counter for the bigrams.
    """
    bigram_counts = collections.Counter()
    # If there's an offset_token from the previous chunk's boundary
    # then form a bigram with the first token in this chunk
    if offset_token is not None and chunk:
        bigram_counts[(offset_token, chunk[0])] += 1

    for i in range(len(chunk) - 1):
        bigram = (chunk[i], chunk[i + 1])
        bigram_counts[bigram] += 1
    return bigram_counts



class BPE_RLE_Tokenizer:
    """
    Class that encapsulates:
      1) Train BPE merges (parallel bigram counting).
      2) Apply BPE merges in a single pass.
      3) Apply run-length encoding.
      4) Decode run-length encoding, undo BPE merges.
      5) Save/load merges from disk.
      6) Build/save/load vocabulary and encode/decode to/from integer IDs.
    """

    def __init__(self, merges=None, token2id=None):
        """
        :param merges: a list of ((tokenA, tokenB), frequency) if already trained.
        :param token2id: an optional dictionary {token: int} if you already have a vocab.
        """
        self.merges = merges if merges is not None else []

        # For vocab
        self.token2id = token2id if token2id is not None else {}
        # Derive reverse mapping if token2id is given
        self.id2token = {idx: tok for tok, idx in self.token2id.items()}

    def train(self, tokens, num_merges=10, max_workers=None):
        """
        1. Learn top bigrams (like BPE).
        2. Store merges internally so we can do `encode()` later.

        :param tokens: list of tokens (strings)
        :param num_merges: number of merges (bigrams) to learn
        :param max_workers: number of worker threads/processes for parallel execution
        """
        n = len(tokens)
        if n == 0:
            return []

        chunk_count = max_workers
        if chunk_count is None:
            chunk_count = min(4, n)

        chunk_size = (n + chunk_count - 1) // chunk_count  # ceiling division

        futures = []
        counters = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=chunk_count) as executor:
            for i in range(0, n, chunk_size):
                chunk = tokens[i:i + chunk_size]
                offset_token = tokens[i - 1] if i > 0 else None
                futures.append(executor.submit(count_bigrams_in_chunk, chunk, offset_token))

            # Collect results
            for f in concurrent.futures.as_completed(futures):
                counters.append(f.result())

        # Combine all bigram counters
        global_bigram_counts = collections.Counter()
        for c in counters:
            global_bigram_counts.update(c)

        # Pick the top `num_merges` from the aggregated bigram counts
        top_bigrams = global_bigram_counts.most_common(num_merges)
        self.merges = top_bigrams
        return top_bigrams

    def apply_bpe_merges(self, tokens):
        """
        Given self.merges (list of ((A,B), freq)),
        turn every occurrence of (A, B) in the token list into "A_B".
        Single-pass scanning from left-to-right.
        """
        if not self.merges:
            # No merges to apply
            return tokens

        merge_pairs = set(m[0] for m in self.merges)

        merged_tokens = []
        skip_next = False
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
                    skip_next = True
                else:
                    merged_tokens.append(tokens[i])
            else:
                # Last token
                merged_tokens.append(tokens[i])
            i += 1

        return merged_tokens

    def run_length_encode(self, tokens):
        """
        E.g. T T T T => T R=3
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
                rle_tokens.append(current_token)
                if count > 1:
                    rle_tokens.append(f"R={count - 1}")
                current_token = t
                count = 1

        # flush last run
        rle_tokens.append(current_token)
        if count > 1:
            rle_tokens.append(f"R={count - 1}")

        return rle_tokens

    def run_length_decode(self, rle_tokens):
        """
        Reverse of run_length_encode.
        """
        decoded = []
        for tok in rle_tokens:
            if tok.startswith("R="):
                repeat_count = int(tok.split("=")[1])
                if decoded:  # sanity check
                    last_tok = decoded[-1]
                    decoded.extend([last_tok] * repeat_count)
            else:
                decoded.append(tok)
        return decoded

    def undo_bpe_merges(self, tokens):
        """
        If a token was "A_B", split to ["A", "B"].
        """
        expanded = []
        for tok in tokens:
            if "_" in tok:
                parts = tok.split("_")
                expanded.extend(parts)
            else:
                expanded.append(tok)
        return expanded

    def encode(self, tokens, as_ids=False):
        """
        1) apply BPE merges
        2) run-length encode
        3) (optionally) convert to IDs if `as_ids=True`
        """
        bpe_tokens = self.apply_bpe_merges(tokens)
        rle_tokens = self.run_length_encode(bpe_tokens)
        if as_ids:
            if not self.token2id:
                raise ValueError("No vocabulary found. Call build_vocab() or load_vocab() first.")
            # Convert each string token into its integer ID, falling back to <UNK> if needed
            unk_id = self.token2id.get("<UNK>", None)
            encoded_ids = []
            for t in rle_tokens:
                if t in self.token2id:
                    encoded_ids.append(self.token2id[t])
                elif unk_id is not None:
                    encoded_ids.append(unk_id)
                else:
                    raise ValueError(
                        f"Token '{t}' not in vocab and <UNK> is not defined in the vocab."
                    )
            return encoded_ids
        else:
            return rle_tokens

    def decode(self, encoded, from_ids=False):
        """
        Reverse of `encode`:
        1) either interpret `encoded` as IDs -> convert to string tokens
        2) run-length decode
        3) undo BPE merges
        """
        if from_ids:
            if not self.id2token:
                raise ValueError("No reverse vocab found. Call build_vocab() or load_vocab() first.")
            # Convert IDs back to string tokens
            rle_tokens = [self.id2token[i] for i in encoded]
        else:
            # Assume already string tokens
            rle_tokens = encoded

        decoded_bpe_tokens = self.run_length_decode(rle_tokens)
        final_tokens = self.undo_bpe_merges(decoded_bpe_tokens)
        return final_tokens

    # ---------------------------------------------------------------------
    # BPE merges save/load
    # ---------------------------------------------------------------------
    def save_merges(self, filepath):
        """
        Save self.merges to a JSON file so we can reuse them later.
        """
        data_to_save = [
            {"pair": list(pair), "freq": freq} for (pair, freq) in self.merges
        ]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)

    def load_merges(self, filepath):
        """
        Load merges from a JSON file and store them in self.merges.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.merges = [((item["pair"][0], item["pair"][1]), item["freq"]) for item in data]

    # ---------------------------------------------------------------------
    # Vocabulary build/save/load
    # ---------------------------------------------------------------------
    def build_vocab(self, tokens, add_unk=True):
        """
        Builds a vocabulary (token->id) from a list of final tokens (e.g. after BPE+RLE)
        or from the raw tokens if you prefer.
        :param tokens: A list of string tokens.
        :param add_unk: If True, reserve an <UNK> token in the vocab for unknown tokens.
        """
        unique_tokens = list(dict.fromkeys(tokens))  # preserves order, removes duplicates

        # Optionally add a special <UNK> token at index 0
        start_idx = 0
        if add_unk and "<UNK>" not in unique_tokens:
            self.token2id = {"<UNK>": 0}
            start_idx = 1
        else:
            self.token2id = {}

        # Populate the vocab
        for i, tok in enumerate(unique_tokens, start=start_idx):
            self.token2id[tok] = i

        # Build the reverse mapping
        self.id2token = {idx: tok for tok, idx in self.token2id.items()}

    def save_vocab(self, filepath):
        """
        Save the current vocabulary (token2id) to JSON.
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.token2id, f, ensure_ascii=False, indent=2)

    def load_vocab(self, filepath):
        """
        Load token2id from a JSON file and rebuild id2token.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            self.token2id = json.load(f)

        # Convert string keys to int if necessary (depending how you saved it).
        # If you saved it as {token: int_id}, this is okay. If you saved as {str_id: token}, invert it.
        # Here, we assume we have {token: int_id}.
        self.id2token = {idx: tok for tok, idx in self.token2id.items()}

def main():
    # Example usage:
    filepath = 'quantized_coeffs.txt'
    tokens = read_tokens_from_file(filepath)
    tokenizer = BPE_RLE_Tokenizer()
    merges = tokenizer.train(tokens,num_merges=5000)
    tokenizer.save_merges("neo_tokenizer/merges.json")
    final_tokens = tokenizer.encode(tokens)  # by default returns strings with BPE+RLE
    tokenizer.build_vocab(final_tokens, add_unk=True)
    tokenizer.save_vocab("neo_tokenizer/vocab.json")


    tokens_as_ids = tokenizer.encode(tokens[0:30], as_ids=True)
    print(tokens_as_ids)
    decoded_tokens = tokenizer.decode(tokens_as_ids, from_ids=True)
    print(decoded_tokens)

    if tokens[0:30] == decoded_tokens:
        print("GOOD!")
    else:
        print("RIDI")

if __name__ == "__main__":
    main()
