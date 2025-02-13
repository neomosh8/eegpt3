import os
import json
from collections import defaultdict
from tqdm import tqdm


class StreamingPassBasedWordLevelBPETokenizer:
    def __init__(self):
        # List of learned merge operations (each is a tuple, e.g. ("D23", "A42"))
        self.merges = []
        # Dictionary mapping merge pair to its rank (order learned)
        self.bpe_ranks = {}
        # Vocabulary: token -> id and inverse mapping
        self.token2id = {}
        self.id2token = {}
        # Working file for training (a temporary file on disk)
        self.working_file = None

    def _copy_file(self, src, dst, chunk_size=1024 * 1024 * 1024):
        """Copy a large file in chunks."""
        total_size = os.path.getsize(src)
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst, tqdm(
                total=total_size, unit="B", unit_scale=True, desc="Copying file"
        ) as pbar:
            while True:
                buf = fsrc.read(chunk_size)
                if not buf:
                    break
                fdst.write(buf)
                pbar.update(len(buf))

    def stream_tokens(self, file_path, chunk_size=1024 * 1024 * 1024):
        """
        Generator that yields tokens from the file.
        The file is assumed to be one long line with tokens separated by spaces.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            leftover = ""
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                data = leftover + chunk
                tokens = data.split(" ")
                # If the chunk ends in the middle of a token, hold the last part.
                if data and not data[-1].isspace():
                    leftover = tokens.pop()
                else:
                    leftover = ""
                for token in tokens:
                    if token:
                        yield token
            if leftover:
                yield leftover

    def compute_pair_frequencies(self, file_path, chunk_size=1024 * 1024 * 1024):
        """
        Compute the frequency of each adjacent token pair from the file,
        with progress monitoring.
        Returns a dictionary mapping (token1, token2) to frequency.
        """
        pair_freq = defaultdict(int)
        total_size = os.path.getsize(file_path)
        with open(file_path, "r", encoding="utf-8") as f, tqdm(
                total=total_size, unit="B", unit_scale=True, desc="Counting pairs"
        ) as pbar:
            leftover = ""
            prev = None
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                # Update the progress bar.
                pbar.update(len(chunk.encode("utf-8")))
                data = leftover + chunk
                tokens = data.split(" ")
                if data and not data[-1].isspace():
                    leftover = tokens.pop()
                else:
                    leftover = ""
                for token in tokens:
                    if token:
                        if prev is not None:
                            pair = (prev, token)
                            pair_freq[pair] += 1
                        prev = token
            if leftover and prev is not None:
                pair = (prev, leftover)
                pair_freq[pair] += 1
        return pair_freq

    def merge_file_multi(self, input_file, output_file, pairs_to_merge, chunk_size=1024 * 1024 * 1024):
        """
        Reads tokens from input_file, merges all occurrences of any token pair in
        pairs_to_merge, and writes the new token sequence to output_file.
        """
        total_size = os.path.getsize(input_file)
        with open(input_file, "r", encoding="utf-8") as fin, \
                open(output_file, "w", encoding="utf-8") as fout, \
                tqdm(total=total_size, unit="B", unit_scale=True, desc="Merging tokens") as pbar:

            leftover = ""
            merged_tokens = []
            while True:
                chunk = fin.read(chunk_size)
                if not chunk:
                    break
                pbar.update(len(chunk.encode("utf-8")))
                data = leftover + chunk
                tokens = data.split(" ")
                if data and not data[-1].isspace():
                    leftover = tokens.pop()
                else:
                    leftover = ""
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) in pairs_to_merge:
                        merged_tokens.append(tokens[i] + "_" + tokens[i + 1])
                        i += 2
                    else:
                        merged_tokens.append(tokens[i])
                        i += 1
                    if len(merged_tokens) >= 100000:
                        fout.write(" ".join(merged_tokens) + " ")
                        merged_tokens = []
            if leftover:
                merged_tokens.append(leftover)
            if merged_tokens:
                fout.write(" ".join(merged_tokens))

    def build_vocab_from_file(self, file_path, chunk_size=1024 * 1024 * 1024):
        """
        Build the final vocabulary (set of unique tokens) by streaming through the file.
        """
        vocab_set = set()
        for token in self.stream_tokens(file_path, chunk_size=chunk_size):
            vocab_set.add(token)
        # Add an unknown token.
        vocab_set.add("<unk>")
        self.token2id = {token: idx for idx, token in enumerate(sorted(vocab_set))}
        self.id2token = {idx: token for token, idx in self.token2id.items()}

    def _apply_merges(self, tokens):
        """
        Apply the learned merges (in order) to a list of tokens in memory.
        """
        tokens = tokens[:]
        changed = True
        while changed:
            changed = False
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1:
                    pair = (tokens[i], tokens[i + 1])
                    if pair in self.bpe_ranks:
                        new_tokens.append(tokens[i] + "_" + tokens[i + 1])
                        i += 2
                        changed = True
                        continue
                new_tokens.append(tokens[i])
                i += 1
            tokens = new_tokens
        return tokens

    def encode(self, text):
        """
        Encode a text string (space-separated tokens) into a list of token IDs.
        The learned BPE merges are applied.
        """
        tokens = text.strip().split()
        merged_tokens = self._apply_merges(tokens)
        unk_id = self.token2id.get("<unk>")
        return [self.token2id.get(tok, unk_id) for tok in merged_tokens]

    def decode(self, token_ids):
        """
        Decode a list of token IDs back into a string.
        Merged tokens (joined by '_') are split back into their original tokens.
        """
        tokens = [self.id2token.get(tid, "<unk>") for tid in token_ids]
        output_tokens = []
        for tok in tokens:
            if "_" in tok:
                output_tokens.extend(tok.split("_"))
            else:
                output_tokens.append(tok)
        return " ".join(output_tokens)

    def save(self, file_path):
        """
        Save the current BPE model to a JSON file.
        """
        data = {
            "merges": [list(pair) for pair in self.merges],
            "token2id": self.token2id,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print("Model saved to", file_path)

    def load(self, file_path):
        """
        Load a BPE model from a JSON file.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.merges = [tuple(pair) for pair in data["merges"]]
        self.bpe_ranks = {pair: idx for idx, pair in enumerate(self.merges)}
        self.token2id = data["token2id"]
        self.id2token = {int(idx): token for token, idx in self.token2id.items()}
        print("Model loaded from", file_path)

    def add_quantization_tokens(self, resolution=80):
        """
        Generate all possible tokens from the quantization logic and add them
        to the vocabulary. This ensures that any token produced by your quantization
        function (like 'A0', 'B4', etc.) is present.
        """
        ranges = [
            {'id': 'A', 'start': -5, 'end': -3, 'proportion': 0.05},
            {'id': 'B', 'start': -3, 'end': -2, 'proportion': 0.10},
            {'id': 'C', 'start': -2, 'end': -1, 'proportion': 0.15},
            {'id': 'D', 'start': -1, 'end': 1, 'proportion': 0.40},
            {'id': 'E', 'start': 1, 'end': 2, 'proportion': 0.15},
            {'id': 'F', 'start': 2, 'end': 3, 'proportion': 0.10},
            {'id': 'G', 'start': 3, 'end': 5, 'proportion': 0.05},
        ]
        cumulative = 0
        for r in ranges:
            r['tokens'] = int(round(r['proportion'] * resolution))
            r['token_start'] = cumulative
            cumulative += r['tokens']
        for r in ranges:
            for quant_level in range(r['tokens']):
                global_quant_index = r['token_start'] + quant_level
                token = f"{r['id']}{global_quant_index}"
                if token not in self.token2id:
                    new_id = len(self.token2id)
                    self.token2id[token] = new_id
                    self.id2token[new_id] = token

    def train(self, file_path, num_merges=1000, num_passes=None, chunk_size=1024 * 1024 * 1024):
        """
        Train the tokenizer on a huge file (one long line) by performing multiple merges per pass.
        In each pass, the tokenizer:
          - Streams through the current working file to compute adjacent token pair frequencies.
          - Selects the top frequently occurring pairs (up to the remaining target merges).
          - Merges all those pairs simultaneously.
          - Writes a new working file with the merged tokens.
        This allows tokens that were merged in previous passes to be merged further.
        """
        if num_passes is None:
            num_passes = num_merges

        # Create a working copy of the training file.
        self.working_file = "temp_working_file.txt"
        print("Copying original file to working file...")
        self._copy_file(file_path, self.working_file, chunk_size=chunk_size)

        merges_done = 0
        pass_num = 0
        while pass_num < num_passes and merges_done < num_merges:
            pass_num += 1
            print(f"\n--- Pass {pass_num} ---")
            pair_freq = self.compute_pair_frequencies(self.working_file, chunk_size=chunk_size)
            if not pair_freq:
                print("No pairs found. Stopping training.")
                break
            # Sort pairs by frequency descending and keep those with frequency >= 2.
            sorted_pairs = sorted(pair_freq.items(), key=lambda x: x[1], reverse=True)
            sorted_pairs = [(pair, freq) for pair, freq in sorted_pairs if freq >= 2]
            if not sorted_pairs:
                print("No pair appears more than once. Stopping training.")
                break

            merges_remaining = num_merges - merges_done
            selected_pairs = [pair for pair, freq in sorted_pairs[:merges_remaining]]
            if not selected_pairs:
                print("No eligible pairs to merge in this pass.")
                break

            # Record the selected pairs.
            for pair in selected_pairs:
                self.merges.append(pair)
            merges_done += len(selected_pairs)
            self.bpe_ranks = {pair: idx for idx, pair in enumerate(self.merges)}
            print(f"Pass {pass_num}: Merging {len(selected_pairs)} pairs. Total merges so far: {merges_done}")

            new_working_file = self.working_file + ".new"
            self.merge_file_multi(self.working_file, new_working_file, set(selected_pairs), chunk_size=chunk_size)
            os.remove(self.working_file)
            os.rename(new_working_file, self.working_file)

        print("\nTraining passes complete. Total merges learned:", len(self.merges))
        print("Building final vocabulary from merged file...")
        self.build_vocab_from_file(self.working_file, chunk_size=chunk_size)
        print("Final vocabulary size:", len(self.token2id))
        # Add quantization tokens to avoid any unknown tokens later.
        self.add_quantization_tokens(resolution=80)
