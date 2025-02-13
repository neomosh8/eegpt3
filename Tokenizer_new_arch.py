import os
import json
import shutil
from collections import defaultdict
from tqdm import tqdm


class StreamingPassBasedWordLevelBPETokenizer:
    def __init__(self):
        # List of learned merge operations (each is a tuple, e.g. ("D23", "A42"))
        self.merges = []
        # Dictionary mapping merge pair to its rank (order learned)
        self.bpe_ranks = {}
        # Vocabulary mapping token -> id (and its inverse)
        self.token2id = {}
        self.id2token = {}
        # Name of the working file used during training.
        self.working_file = None

    def _copy_file(self, src, dst, chunk_size=1024 * 1024):
        """Copy a large file in chunks."""
        total_size = os.path.getsize(src)
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst, tqdm(total=total_size, unit="B", unit_scale=True,
                                                                    desc="Copying file") as pbar:
            while True:
                buf = fsrc.read(chunk_size)
                if not buf:
                    break
                fdst.write(buf)
                pbar.update(len(buf))

    def stream_tokens(self, file_path, chunk_size=1024 * 1024):
        """
        Generator that yields tokens from a huge file.
        The file is assumed to be one long line with tokens separated by spaces.
        It reads in chunks and makes sure not to break tokens.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            leftover = ""
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                data = leftover + chunk
                # Split on spaces.
                tokens = data.split(" ")
                # If the chunk did not end with a space, the last token may be incomplete.
                if data and not data[-1].isspace():
                    leftover = tokens.pop()  # hold back the last (possibly incomplete) token
                else:
                    leftover = ""
                for token in tokens:
                    if token:  # skip empty strings
                        yield token
            if leftover:
                yield leftover

    def compute_pair_frequencies(self, file_path, chunk_size=1024 * 1024):
        """
        Compute the frequency of each adjacent token pair from the file,
        with progress monitoring.
        Returns a dictionary mapping (token1, token2) to frequency.
        """
        pair_freq = defaultdict(int)
        total_size = os.path.getsize(file_path)
        with open(file_path, "r", encoding="utf-8") as f, tqdm(total=total_size, unit="B", unit_scale=True,
                                                               desc="Counting pairs") as pbar:
            leftover = ""
            prev = None
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                # Update progress bar with the actual number of bytes read.
                pbar.update(len(chunk.encode("utf-8")))
                data = leftover + chunk
                tokens = data.split(" ")
                # If the chunk ends in the middle of a token, hold it for the next round.
                if data and not data[-1].isspace():
                    leftover = tokens.pop()
                else:
                    leftover = ""
                for token in tokens:
                    if token:  # Skip empty tokens.
                        if prev is not None:
                            pair = (prev, token)
                            pair_freq[pair] += 1
                        prev = token
            # Process any remaining token.
            if leftover and prev is not None:
                pair = (prev, leftover)
                pair_freq[pair] += 1
        return pair_freq

    def merge_file(self, input_file, output_file, best_pair, chunk_size=1024 * 1024):
        """
        Reads tokens from input_file, merges occurrences of best_pair,
        and writes the new token sequence (as one long line) to output_file.
        Displays a progress bar during processing.
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
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
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

    def _apply_merges(self, tokens):
        """
        Apply all learned merges (in order) to a list of tokens.
        (This is used for encoding new text in memory.)
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

    def build_vocab_from_file(self, file_path, chunk_size=1024 * 1024):
        """
        Build the final vocabulary (set of unique tokens) by streaming through the file.
        """
        vocab_set = set()
        for token in self.stream_tokens(file_path, chunk_size):
            vocab_set.add(token)
        # Always add an unknown token.
        vocab_set.add("<unk>")
        # Create token2id and id2token mappings.
        self.token2id = {token: idx for idx, token in enumerate(sorted(vocab_set))}
        self.id2token = {idx: token for token, idx in self.token2id.items()}

    def train(self, file_path, num_merges=10000, num_passes=None, chunk_size=1024 * 1024):
        """
        Train the tokenizer on a huge file (with one long line).
        The training works in passes:
          - In each pass the file is streamed to compute adjacent token pair frequencies.
          - The most frequent pair is merged (via a second pass that writes a new file).
          - This is repeated until num_merges (or num_passes) is reached.

        Parameters:
          file_path: Path to the original training file.
          num_merges: Maximum number of merge operations to perform.
          num_passes: Maximum number of passes to run (if None, use num_merges).
          chunk_size: Number of bytes to read per chunk.
        """
        if num_passes is None:
            num_passes = num_merges

        # Create a working copy of the training file so as not to load the original 13GB file into memory.
        self.working_file = "temp_working_file.txt"
        print("Copying original file to working file...")
        self._copy_file(file_path, self.working_file, chunk_size=chunk_size)

        merges_done = 0
        for pass_num in range(1, num_passes + 1):
            print(f"\n--- Pass {pass_num} ---")
            pair_freq = self.compute_pair_frequencies(self.working_file, chunk_size=chunk_size)
            if not pair_freq:
                print("No pairs found. Stopping training.")
                break
            # Select the most frequent pair.
            best_pair = max(pair_freq, key=pair_freq.get)
            freq = pair_freq[best_pair]
            if freq < 2:
                print("No pair appears more than once. Stopping training.")
                break
            self.merges.append(best_pair)
            merges_done += 1
            self.bpe_ranks = {pair: idx for idx, pair in enumerate(self.merges)}
            print(f"Pass {pass_num}: Merging {best_pair} (frequency: {freq}). Total merges so far: {merges_done}")
            if merges_done >= num_merges:
                print("Reached maximum number of merges.")
                break
            # Create a new file with the best pair merged.
            new_working_file = self.working_file + ".new"
            self.merge_file(self.working_file, new_working_file, best_pair, chunk_size=chunk_size)
            # Replace the old working file with the new merged file.
            os.remove(self.working_file)
            os.rename(new_working_file, self.working_file)

        print("\nTraining passes complete. Total merges learned:", len(self.merges))
        print("Building final vocabulary from merged file...")
        self.build_vocab_from_file(self.working_file, chunk_size=chunk_size)
        print("Final vocabulary size:", len(self.token2id))
        # (Optionally, you may remove the working file here if it’s no longer needed.)
        # os.remove(self.working_file)

    def encode(self, text):
        """
        Encode a text string (space-separated tokens) into a list of token IDs.
        The learned BPE merges are applied in memory.
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
        Save the learned BPE model (merges and vocabulary) to a JSON file.
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
