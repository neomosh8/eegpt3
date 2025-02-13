import os
import json
from collections import defaultdict
from tqdm import tqdm


class PassBasedWordLevelBPETokenizer:
    def __init__(self):
        # List of learned merge operations (each is a tuple, e.g. ("D23", "A42"))
        self.merges = []
        # Dictionary mapping merge pair to its rank (order in which it was learned)
        self.bpe_ranks = {}
        # Vocabulary mapping final token -> id and its inverse.
        self.token2id = {}
        self.id2token = {}

    def _apply_merges(self, tokens):
        """
        Given a list of tokens, apply all learned merges in order.
        This method scans repeatedly until no merge applies.
        """
        # Make a copy to avoid modifying the input.
        tokens = tokens[:]
        changed = True
        while changed:
            changed = False
            i = 0
            new_tokens = []
            while i < len(tokens):
                # If there is a next token, check if the pair has been merged.
                if i < len(tokens) - 1:
                    pair = (tokens[i], tokens[i + 1])
                    if pair in self.bpe_ranks:
                        # Merge the pair (join with an underscore) and skip the next token.
                        new_tokens.append(tokens[i] + "_" + tokens[i + 1])
                        i += 2
                        changed = True
                        continue
                # Otherwise, just keep the token.
                new_tokens.append(tokens[i])
                i += 1
            tokens = new_tokens
        return tokens

    def _read_and_apply_merges(self, file_path, chunk_size):
        """
        Read the training file line by line (with a progress bar), split each line into tokens,
        apply the current merges to the token list, and build a vocabulary dictionary mapping
        token sequences (as tuples) to frequency.
        """
        vocab = defaultdict(int)
        total_size = os.path.getsize(file_path)
        with open(file_path, "r", encoding="utf-8") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, desc="Reading corpus"
        ) as pbar:
            for line in f:
                encoded_line = line.encode("utf-8")
                pbar.update(len(encoded_line))
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                # Apply all merges learned so far.
                tokens = self._apply_merges(tokens)
                vocab[tuple(tokens)] += 1
        return vocab

    def _get_stats(self, vocab):
        """
        Given a vocabulary (mapping token sequence to frequency), count the frequency of each
        adjacent token pair.
        """
        pairs = defaultdict(int)
        for token_seq, freq in vocab.items():
            tokens = list(token_seq)
            for i in range(len(tokens) - 1):
                pairs[(tokens[i], tokens[i + 1])] += freq
        return pairs

    def train(self, file_path, num_merges=10000, num_passes=None, chunk_size=1024):
        """
        Train the tokenizer on a file where each line is a sequence of space–separated tokens.
        The training works in passes: each pass reads the entire file, applies merges learned so far,
        computes adjacent token pair frequencies, and (if possible) merges the most frequent pair.

        Parameters:
          file_path: Path to the training text file.
          num_merges: Maximum number of merge operations (across all passes).
          num_passes: Maximum number of passes to perform. If None, num_passes = num_merges.
          chunk_size: Number of bytes to update the progress bar (for file reading).
        """
        if num_passes is None:
            num_passes = num_merges

        learned_merges = []
        current_merge_count = 0

        # Each pass re-reads the file, applying merges learned so far.
        for p in range(num_passes):
            vocab = self._read_and_apply_merges(file_path, chunk_size)
            pairs = self._get_stats(vocab)
            if not pairs:
                print("No more pairs to merge. Stopping at pass", p + 1)
                break
            # Select the most frequent adjacent token pair.
            best_pair = max(pairs, key=pairs.get)
            freq = pairs[best_pair]
            if freq < 1:
                # If the best pair does not appear, nothing to merge.
                print("Best pair frequency is 0. Stopping.")
                break
            learned_merges.append(best_pair)
            current_merge_count += 1
            print(f"Pass {p + 1}: Merging {best_pair} with frequency {freq}")
            if current_merge_count >= num_merges:
                print("Reached maximum number of merges.")
                break

        self.merges = learned_merges
        self.bpe_ranks = {pair: idx for idx, pair in enumerate(self.merges)}

        # Build final vocabulary by reading the file one last time with all merges applied.
        final_vocab = self._read_and_apply_merges(file_path, chunk_size)
        token_set = set()
        for token_seq in final_vocab:
            token_set.update(token_seq)
        token_set.add("<unk>")  # Add an unknown token.
        self.token2id = {token: idx for idx, token in enumerate(sorted(token_set))}
        self.id2token = {idx: token for token, idx in self.token2id.items()}
        print("Training complete. Final vocabulary size:", len(self.token2id))

    def encode(self, text):
        """
        Encode a text string (space-separated tokens) into a list of token IDs.
        The current merges are applied.
        """
        tokens = text.strip().split()
        merged_tokens = self._apply_merges(tokens)
        unk_id = self.token2id.get("<unk>")
        token_ids = [self.token2id.get(tok, unk_id) for tok in merged_tokens]
        return token_ids

    def decode(self, token_ids):
        """
        Decode a list of token IDs back into a string.
        Merged tokens (joined by '_') are split back into their original tokens.
        """
        tokens = [self.id2token.get(tid, "<unk>") for tid in token_ids]
        output_tokens = []
        for tok in tokens:
            # If the token contains an underscore, split it.
            if "_" in tok:
                output_tokens.extend(tok.split("_"))
            else:
                output_tokens.append(tok)
        return " ".join(output_tokens)

    def save(self, file_path):
        """
        Save the BPE model to a JSON file.
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
