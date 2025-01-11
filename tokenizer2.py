import collections
import sys
import os
import json

from tqdm import tqdm

import boto3

s3 = boto3.client('s3')
def read_tokens_in_chunks(filepath, chunk_size=1024 * 1024):
    """
    Generator that reads a large file in stream fashion and yields tokens in
    reasonably sized lists. A tqdm progress bar monitors how many bytes have
    been read so far.

    :param filepath: The text file path.
    :param chunk_size: How many bytes to read at a time.
    :yield: A list of tokens (strings) for each chunk.
    """
    file_size = os.path.getsize(filepath)
    buffer = ""

    # 'unit_scale=True' auto-scales bytes to KB/MB, etc.
    # 'desc' is the label for the progress bar.
    with open(filepath, 'r', encoding='utf-8') as f, \
            tqdm(total=file_size, unit='B', unit_scale=True, desc="Reading file") as pbar:

        while True:
            # Read a chunk (in bytes; chunk_size default: 1 MB).
            chunk = f.read(chunk_size)
            if not chunk:
                # no more data
                break

            # Update the progress bar by the number of *raw* bytes read.
            # If your file has multi-byte characters, you might want to do:
            #   pbar.update(len(chunk.encode('utf-8')))
            # But for ASCII or mostly single-byte text, pbar.update(len(chunk)) is fine.
            pbar.update(len(chunk))

            buffer += chunk
            # Split by whitespace
            tokens = buffer.strip().split()
            yield tokens

            # We reset the buffer. If you must handle partial tokens across
            # chunk boundaries (e.g. a token straddling two chunks), you'd need
            # more logic to preserve partial tokens. For simplicity, we consume
            # everything in this design.
            buffer = ""


def count_bigrams_in_chunk(chunk_tokens, offset_token=None):
    """
    Count bigrams within a chunk of tokens.
    If offset_token is provided, count the bigram that crosses the boundary
    (offset_token, chunk_tokens[0]) if the chunk is non-empty.
    Returns a collections.Counter for the bigrams.
    """
    bigram_counts = collections.Counter()
    if offset_token is not None and chunk_tokens:
        bigram_counts[(offset_token, chunk_tokens[0])] += 1

    for i in range(len(chunk_tokens) - 1):
        bigram = (chunk_tokens[i], chunk_tokens[i + 1])
        bigram_counts[bigram] += 1

    return bigram_counts


class BPE_RLE_Tokenizer:
    """
    Class that encapsulates:
      1) Train BPE merges (with bigram counting in a streaming-friendly manner).
      2) Apply BPE merges in a single pass.
      3) Apply run-length encoding (RLE).
      4) Decode run-length encoding, undo BPE merges.
      5) Save/load merges from disk.
      6) Build/save/load vocabulary and encode/decode to/from integer IDs.
      7) Optional streaming encode for large files.
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

    def train(self, filepath, num_merges=10, chunk_size=1024 * 1024):
        """
        Learn top bigrams (like a single-pass BPE).
        We'll stream through the file in chunks to build a global bigram count.
        Then we pick the top `num_merges`.

        :param filepath: path to large text file
        :param num_merges: how many merges to keep
        :param chunk_size: how many bytes to read at once
        """
        global_bigram_counts = collections.Counter()
        offset_token = None

        for chunk_tokens in read_tokens_in_chunks(filepath, chunk_size=chunk_size):
            if not chunk_tokens:
                continue
            # Count bigrams in this chunk
            chunk_counter = count_bigrams_in_chunk(chunk_tokens, offset_token=offset_token)
            global_bigram_counts.update(chunk_counter)
            # Remember the last token for boundary crossing with the next chunk
            offset_token = chunk_tokens[-1]

        # Pick the top `num_merges`
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

                    if isinstance(indices[i + 1], list):
                        right_indices = indices[i + 1]
                    else:
                        right_indices = [indices[i + 1]]

                    merged_indices.append(left_indices + right_indices)

                    skip_next = True
                else:
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
        If we detect repeated tokens [T, T, T], we do "T, R=2" while combining
        their corresponding 'indices'.
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
                rle_indices.append(current_idx_list)
                if count > 1:
                    rle_tokens.append(f"R={count - 1}")
                    rle_indices.append(current_idx_list)

                # reset
                current_token = t
                current_idx_list = idx_list
                count = 1

        # flush the last run
        rle_tokens.append(current_token)
        rle_indices.append(current_idx_list)
        if count > 1:
            rle_tokens.append(f"R={count - 1}")
            rle_indices.append(current_idx_list)

        return rle_tokens, rle_indices

    def save_alignment(self, alignment_positions, filepath):
        """
        Save the alignment (list of lists) to a file.
        Each final token has a list of original indices.
        We'll store them in a simple line-by-line format (CSV style).
        """
        with open(filepath, "w", encoding="utf-8") as f:
            for pos_list in alignment_positions:
                if isinstance(pos_list, list):
                    s = ",".join(map(str, pos_list))
                else:
                    s = str(pos_list)
                f.write(s + "\n")

    def encode_with_alignment(self, raw_tokens, as_ids=False, alignment_filepath=None):
        """
        1) Apply BPE merges (with alignment).
        2) Run-length encode (with alignment).
        3) Optionally convert to IDs if 'as_ids=True'.
        4) If 'alignment_filepath' is provided, save the alignment.
        5) Return (encoded_output, alignment_positions).

        :param raw_tokens: list of raw string tokens to encode
        :param as_ids: if True, convert the final RLE tokens to integer IDs
                       (requires a built or loaded vocab).
        :param alignment_filepath: if provided, save alignment to disk at this path.
        :return: (encoded_output, alignment_positions) where:
                 - encoded_output is either list of tokens or list of IDs
                 - alignment_positions is a list of lists indicating which
                   original token indices contributed to each output token
        """
        # Step 0: initial indices = [0, 1, 2, ..., len(raw_tokens)-1]
        positions = list(range(len(raw_tokens)))

        # Step 1: BPE merges with alignment
        merged_tokens, merged_positions = self.apply_bpe_merges_with_alignment(raw_tokens, positions)

        # Step 2: run-length encode with alignment
        rle_tokens, rle_positions = self.run_length_encode_with_alignment(merged_tokens, merged_positions)

        # Step 3: Optionally convert RLE tokens to IDs
        if as_ids:
            if not self.token2id:
                raise ValueError("No vocabulary found. Build or load vocab before encoding as IDs.")
            unk_id = self.token2id.get("<UNK>", None)
            encoded_output = []
            for t in rle_tokens:
                if t in self.token2id:
                    encoded_output.append(self.token2id[t])
                elif unk_id is not None:
                    encoded_output.append(unk_id)
                else:
                    raise ValueError(f"Token '{t}' not in vocab and <UNK> not defined.")
        else:
            encoded_output = rle_tokens

        # Step 4: If a filepath is given, save the alignment
        if alignment_filepath is not None:
            self.save_alignment(rle_positions, alignment_filepath)

        return encoded_output, rle_positions

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
        Builds a vocabulary (token->id) from a list of final tokens.
        :param tokens: A list of string tokens (e.g. after BPE+RLE).
        :param add_unk: If True, reserve an <UNK> token in the vocab for unknown tokens.
        """
        unique_tokens = list(dict.fromkeys(tokens))  # preserves order, removes duplicates

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
        self.id2token = {idx: tok for tok, idx in self.token2id.items()}

    # ---------------------------------------------------------------------
    # OPTIONAL: Streaming encode with tqdm monitoring
    # ---------------------------------------------------------------------
    def encode_streaming(self, input_filepath, output_filepath, chunk_size=1024 * 1024):
        """
        Stream a large file from disk, apply BPE merges + RLE, and write
        the resulting tokens (as text) to `output_filepath`.
        A tqdm progress bar monitors how many bytes have been processed.

        NOTE: If you want them as IDs, you can modify this to convert tokens to IDs
              and write them out in some numeric format.
        """
        file_size = os.path.getsize(input_filepath)

        with open(input_filepath, 'r', encoding='utf-8') as fin, \
                open(output_filepath, 'w', encoding='utf-8') as fout, \
                tqdm(total=file_size, unit='B', unit_scale=True, desc="Encoding file") as pbar:

            buffer = ""
            while True:
                chunk = fin.read(chunk_size)
                if not chunk:
                    # no more data
                    break

                pbar.update(len(chunk))

                buffer += chunk
                chunk_tokens = buffer.strip().split()
                if not chunk_tokens:
                    buffer = ""
                    continue

                encoded_tokens = self.encode(chunk_tokens, as_ids=False)
                fout.write(" ".join(encoded_tokens) + " ")
                buffer = ""  # reset buffer
                # If partial tokens across chunk boundaries matter,
                # you'd do more advanced leftover handling here.

    def decode_with_alignment(self, encoded, channels, from_ids=False):
        """
        Decodes `encoded` tokens (or IDs) back to the raw tokens while also expanding the
        `channels` array so that every decoded subtoken has the correct channel.

        :param encoded: list of tokens (strings) or IDs (ints), exactly like in `decode(...)`.
        :param channels: list of integers, same length as `encoded`, representing the channel
                         for each encoded token.
        :param from_ids: if True, interpret `encoded` as vocabulary IDs; else interpret as tokens.

        :return: (decoded_tokens, expanded_channels)
                 where decoded_tokens is the list of fully expanded raw tokens,
                 and expanded_channels is a list of the same length, with the corresponding channel.
        """

        if len(encoded) != len(channels):
            raise ValueError("encoded and channels must have the same length.")

        # -------------------------------------------------------------------------
        # Step 1: Convert IDs -> RLE tokens if needed
        # -------------------------------------------------------------------------
        if from_ids:
            if not self.id2token:
                raise ValueError("No reverse vocab found. Call build_vocab() or load_vocab() first.")
            # Convert each ID back to its string token
            rle_tokens = [self.id2token[i] for i in encoded]
        else:
            # Assume already string tokens
            rle_tokens = encoded

        # -------------------------------------------------------------------------
        # Step 2: Run-length decode WITH channel expansion
        # -------------------------------------------------------------------------
        # We'll maintain two lists: `decoded_bpe_tokens` and `decoded_bpe_channels`.
        decoded_bpe_tokens = []
        decoded_bpe_channels = []

        for token, ch in zip(rle_tokens, channels):
            if token.startswith("R="):
                # This indicates a run-length code, e.g. "R=3".
                repeat_count = int(token.split("=")[1])
                if not decoded_bpe_tokens:
                    # This would be unusual, but we can handle or raise an error.
                    raise ValueError("Encountered R=... but no previous token to repeat.")

                # The last token/channel that was appended:
                last_token = decoded_bpe_tokens[-1]
                last_ch = decoded_bpe_channels[-1]

                # We replicate that last token `repeat_count` more times
                for _ in range(repeat_count):
                    decoded_bpe_tokens.append(last_token)
                    decoded_bpe_channels.append(last_ch)
            else:
                # A normal token
                decoded_bpe_tokens.append(token)
                decoded_bpe_channels.append(ch)

        # -------------------------------------------------------------------------
        # Step 3: Undo BPE merges WITH channel expansion
        #         e.g. "A_B" => ["A", "B"], replicating the channel for both pieces
        # -------------------------------------------------------------------------
        final_tokens = []
        final_channels = []
        for token, ch in zip(decoded_bpe_tokens, decoded_bpe_channels):
            if "_" in token:
                # e.g. "D23_D44" => ["D23", "D44"]
                sub_parts = token.split("_")
                for sp in sub_parts:
                    final_tokens.append(sp)
                    final_channels.append(ch)
            else:
                final_tokens.append(token)
                final_channels.append(ch)

        return final_tokens, final_channels


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

def main():
    # Example usage:
    # large_text_file = s3.get_object(Bucket="dataframes--use1-az6--x-s3", Key="combined_coeffs.txt")["Body"].read().decode("utf-8")
    large_text_file = 'coeffs_combined.txt'
    tokenizer = BPE_RLE_Tokenizer()

    # 1) TRAIN BPE merges on a large file in streaming mode
    #    We learn the top 4000 bigrams, with a 1MB chunk size (adjust as needed).
    merges = tokenizer.train(large_text_file, num_merges=10000, chunk_size=32*4096 * 32*4096)
    tokenizer.save_merges("neo_tokenizer/merges.json")

    # 2) Demonstrate how you'd build the vocab using a (relatively small) snippet
    #    of data. Typically you'd do a second pass (or store the BPE+RLE tokens
    #    from the entire file) to build your vocab comprehensively.
    with open(large_text_file, "r", encoding="utf-8") as f:
        small_text_snippet = f.read()
    snippet_tokens = small_text_snippet.strip().split()

    # Apply BPE+RLE to that snippet
    final_tokens = tokenizer.encode(snippet_tokens)
    # Build vocab
    tokenizer.build_vocab(final_tokens, add_unk=True)
    tokenizer.save_vocab("neo_tokenizer/vocab.json")

    # 3) Example of encode/decode round trip with IDs on a small slice
    slice_tokens = snippet_tokens[:30]
    tokens_as_ids = tokenizer.encode(slice_tokens, as_ids=True)
    decoded_tokens = tokenizer.decode(tokens_as_ids, from_ids=True)

    print("Tokens as IDs:", tokens_as_ids)
    print("Decoded Tokens:", decoded_tokens)
    if slice_tokens == decoded_tokens:
        print("Round trip success!")
    else:
        print("Mismatch in round trip...")

    # 4) OPTIONAL: Streaming encode the entire file to some output
    # tokenizer.encode_streaming(large_text_file, "encoded_output.txt", chunk_size=1024*1024)


if __name__ == "__main__":
    main()
