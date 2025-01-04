import collections
import sys
import os
import json
import concurrent.futures

from mne.io import read_raw


def read_tokens_from_file(filepath):
    """
    Read a file and split on whitespace into tokens.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    # Basic whitespace split:
    tokens = text.strip().split()
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

    def apply_bpe_merges_with_alignment(self, tokens, indices):
        """
        Same logic as apply_bpe_merges(...), but also returns a parallel
        list of 'merged_indices' so we know which original token indices
        contributed to each newly merged token.

        :param tokens: List of string tokens
        :param indices: List of integer indices (same length as tokens),
                        e.g. [0,1,2,3,...] or lists-of-indices
        :return: (merged_tokens, merged_indices)
        """
        if not self.merges:
            # no merges to do, just return as-is
            return tokens, indices

        merge_pairs = set(m[0] for m in self.merges)

        merged_tokens = []
        merged_indices = []

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
                    # BPE merge
                    new_token = tokens[i] + "_" + tokens[i + 1]
                    merged_tokens.append(new_token)

                    # the new token is formed from the union of indices[i], indices[i+1]
                    if isinstance(indices[i], list):
                        left_indices = indices[i]
                    else:
                        left_indices = [indices[i]]

                    if isinstance(indices[i+1], list):
                        right_indices = indices[i+1]
                    else:
                        right_indices = [indices[i+1]]

                    merged_indices.append(left_indices + right_indices)

                    skip_next = True
                else:
                    # no merge
                    merged_tokens.append(tokens[i])
                    merged_indices.append(indices[i])
            else:
                # last token in the list
                merged_tokens.append(tokens[i])
                merged_indices.append(indices[i])
            i += 1

        return merged_tokens, merged_indices

    def run_length_encode_with_alignment(self, tokens, indices):
        """
        Same logic as run_length_encode(...), but also merges the alignment indices.
        If we detect repeated tokens [T, T, T], we'll do "T, R=2" but also combine
        their 'indices' in some way.
        """
        if not tokens:
            return [], []

        rle_tokens = []
        rle_indices = []

        current_token = tokens[0]
        current_idx_list = indices[0] if isinstance(indices[0], list) else [indices[0]]
        count = 1

        for t, idx in zip(tokens[1:], indices[1:]):
            idx_list = idx if isinstance(idx, list) else [idx]
            if t == current_token:
                # same token => run
                count += 1
                # accumulate the alignment indices too
                current_idx_list.extend(idx_list)
            else:
                # flush the previous run
                rle_tokens.append(current_token)
                if count > 1:
                    rle_tokens.append(f"R={count-1}")

                rle_indices.append(current_idx_list)
                if count > 1:
                    # put a placeholder or special notation for the run-length token
                    rle_indices.append(current_idx_list)  # or store some separate structure

                # reset
                current_token = t
                current_idx_list = idx_list
                count = 1

        # flush last run
        rle_tokens.append(current_token)
        rle_indices.append(current_idx_list)
        if count > 1:
            rle_tokens.append(f"R={count-1}")
            rle_indices.append(current_idx_list)

        return rle_tokens, rle_indices

    def encode_with_alignment(self, raw_tokens):
        """
        1) Apply BPE merges (with alignment)
        2) Run-length encode (with alignment)
        3) Return (final_rle_tokens, final_rle_indices)
           final_rle_indices[i] => list of original token indices that contributed
           to final token i.
        """
        # Step 0: build initial indices = [0, 1, 2, ..., len(raw_tokens)-1]
        positions = list(range(len(raw_tokens)))

        # Step 1: BPE merges with alignment
        merged_tokens, merged_positions = self.apply_bpe_merges_with_alignment(raw_tokens, positions)

        # Step 2: run-length encode with alignment
        rle_tokens, rle_positions = self.run_length_encode_with_alignment(merged_tokens, merged_positions)

        return rle_tokens, rle_positions
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
    def save_alignment(self, alignment_positions, filepath):
        """
        Save the alignment (list of lists) to a file.
        Each final token has a list of original indices. We'll store them in e.g. CSV style.
        """
        with open(filepath, "w", encoding="utf-8") as f:
            for pos_list in alignment_positions:
                # pos_list might be [12, 13, 14], or a single int, etc.
                if isinstance(pos_list, list):
                    s = ",".join(map(str, pos_list))
                else:
                    s = str(pos_list)
                f.write(s + "\n")
def apply_alignment_to_channels(channel_array, alignment_positions, combine_mode="first"):
    """
    channel_array: a list/array of shape [num_original_tokens].
    alignment_positions: the list of lists that we saved/loaded from "my_alignment_positions.txt"
    combine_mode: how we combine multiple original channel values that merged together.
                  Could be "first", "mean", "sum", etc.

    Returns: final_channel_array (length = len(alignment_positions))
    """
    final_channels = []
    for pos_list in alignment_positions:
        if not isinstance(pos_list, list):
            pos_list = [pos_list]

        values = [channel_array[p] for p in pos_list]

        if combine_mode == "first":
            final_val = values[0]
        elif combine_mode == "mean":
            final_val = sum(values) / len(values)
        elif combine_mode == "sum":
            final_val = sum(values)
        else:
            raise ValueError("Unknown combine_mode")

        final_channels.append(final_val)

    return final_channels
def save_channels_as_text(channel_array, output_filepath):
    """
    Write the final channel array to a text file with space-delimited entries.
    """
    with open(output_filepath, "w", encoding="utf-8") as f:
        # Convert each channel value to string, then join by space
        line = " ".join(str(ch) for ch in channel_array)
        f.write(line + "\n")


def main():
    # Example usage:
    filepath = 'quantized_coeffs.txt'
    tokens = read_tokens_from_file(filepath)
    print("num of tokens: ",len(tokens))
    # tokenizer = BPE_RLE_Tokenizer()
    # merges = tokenizer.train(tokens,num_merges=4000)
    # tokenizer.save_merges("neo_tokenizer/merges.json")
    # final_tokens = tokenizer.encode(tokens)  # by default returns strings with BPE+RLE
    # tokenizer.build_vocab(final_tokens, add_unk=True)
    # tokenizer.save_vocab("neo_tokenizer/vocab.json")
    # final_rle_tokens, final_rle_positions = tokenizer.encode_with_alignment(tokens)
    # # Save
    # tokenizer.save_alignment(final_rle_positions, "my_alignment_positions.txt")
    tokenizer = BPE_RLE_Tokenizer()
    tokenizer.load_merges("neo_tokenizer/merges.json")
    tokenizer.load_vocab("neo_tokenizer/vocab.json")
    keke= tokenizer.encode(tokens,as_ids=True)
    final_rle_tokens, final_rle_positions = tokenizer.encode_with_alignment(tokens)
    tokenizer.save_alignment(final_rle_positions, "my_alignment_positions.txt")
    final_channels = apply_alignment_to_channels(tokens, final_rle_positions, combine_mode="first")
    save_channels_as_text(final_channels, 'final_channels.txt')

    print(len(final_channels),len(keke))
    # raw_channels = read_tokens_from_file('quantized_channels.txt')
    #
    # assert len(raw_channels) == len(tokens), "Mismatch!"
    #
    # # 4) Convert your channel array to final shape
    # final_channels = apply_alignment_to_channels(raw_channels, final_rle_positions, combine_mode="first")
    # save_channels_as_text(final_channels,'final_channels.txt')
    # print("Original length:", len(tokens))
    # print("Final length after merges/RLE:", len(final_rle_tokens), len(final_channels))

if __name__ == "__main__":
    main()
