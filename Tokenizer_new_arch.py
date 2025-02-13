import os
import re
import json
from collections import defaultdict
from tqdm import tqdm


class BPETokenizer:
    def __init__(self):
        # List of merge operations (each is a tuple of two symbols)
        self.merges = []
        # Dictionary mapping each merge pair to its rank (lower = earlier merge)
        self.bpe_ranks = {}
        # Final vocabulary mapping token -> id
        self.token2id = {}
        self.id2token = {}
        # Set of “full‐word” tokens (i.e. words that during training were learned as one token)
        self.full_words = set()

    def train(self, file_path, num_merges=10000, chunk_size=10000):
        """
        Train the BPE tokenizer from a text file.

        Parameters:
          file_path: Path to the training text file.
          num_merges: Maximum number of merge operations (or passes) to perform.
          chunk_size: Number of bytes (approx.) to process per chunk.
        """
        # Build frequency dictionary of words from the file.
        # (Assumes one or more space‐separated tokens per line.)
        word_freqs = defaultdict(int)
        total_size = os.path.getsize(file_path)
        with open(file_path, "r", encoding="utf-8") as f, tqdm(total=total_size, unit='B', unit_scale=True,
                                                               desc="Reading corpus") as pbar:
            for line in f:
                encoded_line = line.encode("utf-8")
                pbar.update(len(encoded_line))
                line = line.strip()
                if not line:
                    continue
                for word in line.split():
                    word_freqs[word] += 1

        # Build the “initial” vocabulary:
        # Each word is represented as a tuple of its characters, with a special end-of-word marker.
        vocab = {}
        for word, freq in word_freqs.items():
            # For BPE training we represent each word as a tuple of characters, and add '</w>' to mark word end.
            # (This marker helps us later to recover full words.)
            vocab[tuple(word) + ("</w>",)] = freq

        # Perform BPE merge operations.
        for i in tqdm(range(num_merges), desc="Performing BPE merges"):
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            # Select the most frequent adjacent pair.
            best = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best, vocab)
            self.merges.append(best)

        # Create the bpe_ranks dictionary.
        self.bpe_ranks = {pair: i for i, pair in enumerate(self.merges)}

        # Build the final vocabulary (set of subword tokens) and record full-word tokens.
        # We “reconstruct” each training word with the learned merges.
        token_set = set()
        for word in word_freqs.keys():
            # Apply BPE to the word
            tokens = self._bpe(word)
            # If the entire word was merged into a single token (i.e. no split), record it.
            if len(tokens) == 1 and tokens[0].endswith("</w>"):
                full_token = tokens[0].replace("</w>", "")
                self.full_words.add(full_token)
                token_set.add(full_token)
            else:
                for tok in tokens:
                    # Remove the end-of-word marker for tokens that ended a word.
                    if tok.endswith("</w>"):
                        tok = tok[:-4]
                    token_set.add(tok)
        # We also add an explicit space token to mark word boundaries during encoding.
        token_set.add(" ")
        # And include an unknown token.
        token_set.add("<unk>")

        # Assign unique ids to each token.
        self.token2id = {token: idx for idx, token in enumerate(sorted(token_set))}
        self.id2token = {idx: token for token, idx in self.token2id.items()}

    def _get_stats(self, vocab):
        """
        Count frequency of all adjacent symbol pairs in the vocabulary.

        Parameters:
          vocab: A dict mapping word (tuple of symbols) to frequency.

        Returns:
          A dict mapping symbol pairs (tuples) to their aggregated frequency.
        """
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = list(word)
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_vocab(self, pair, vocab_in):
        """
        Merge all occurrences of the given symbol pair in the vocabulary.

        Parameters:
          pair: A tuple of symbols (e.g. ('a', 'b')) to merge.
          vocab_in: Current vocabulary (dict mapping tuple of symbols to frequency).

        Returns:
          A new vocabulary with the pair merged.
        """
        vocab_out = {}
        bigram = pair
        for word, freq in vocab_in.items():
            new_word = []
            i = 0
            word = list(word)
            while i < len(word):
                # If the pair is found, merge the two symbols.
                if i < len(word) - 1 and (word[i], word[i + 1]) == bigram:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            vocab_out[tuple(new_word)] = freq
        return vocab_out

    def _bpe(self, word):
        """
        Apply the learned BPE merges to segment a word.

        Parameters:
          word: A string (a single word).

        Returns:
          A list of subword tokens (some ending with '</w>' to mark word-end).
        """
        # Start with the list of characters plus the end-of-word marker.
        word_symbols = list(word) + ["</w>"]
        while True:
            # Get list of adjacent pairs.
            pairs = [(word_symbols[i], word_symbols[i + 1]) for i in range(len(word_symbols) - 1)]
            # Find the pair that has been merged (if any) with the lowest rank.
            candidate = None
            min_rank = None
            for pair in pairs:
                if pair in self.bpe_ranks:
                    rank = self.bpe_ranks[pair]
                    if min_rank is None or rank < min_rank:
                        min_rank = rank
                        candidate = pair
            if candidate is None:
                break  # No more merges to apply.
            # Merge the first occurrence of candidate in a left-to-right pass.
            new_word = []
            i = 0
            while i < len(word_symbols):
                if i < len(word_symbols) - 1 and (word_symbols[i], word_symbols[i + 1]) == candidate:
                    new_word.append(word_symbols[i] + word_symbols[i + 1])
                    i += 2
                else:
                    new_word.append(word_symbols[i])
                    i += 1
            word_symbols = new_word
        return word_symbols

    def encode(self, text):
        """
        Encode a text string into a flat list of token ids.

        The text is first split into words (by whitespace). For each word, if it was
        seen in training as a full word, it is encoded as a single token; otherwise the
        learned BPE rules are applied.

        A special space token is inserted between words.
        """
        token_ids = []
        words = text.strip().split()
        space_id = self.token2id.get(" ")
        unk_id = self.token2id.get("<unk>")
        for i, word in enumerate(words):
            if word in self.full_words:
                token = word
                token_ids.append(self.token2id.get(token, unk_id))
            else:
                # Apply BPE segmentation.
                sub_tokens = self._bpe(word)
                for sub in sub_tokens:
                    if sub.endswith("</w>"):
                        sub = sub[:-4]
                    token_ids.append(self.token2id.get(sub, unk_id))
            # Insert a space token between words (except after the last word)
            if i < len(words) - 1:
                token_ids.append(space_id)
        return token_ids

    def decode(self, token_ids):
        """
        Decode a flat list of token ids back into a string.

        This routine assumes that a space token was inserted between words during encoding.
        It concatenates subword tokens (which were produced during BPE segmentation) and
        then splits on the special space token.
        """
        tokens = [self.id2token.get(tid, "") for tid in token_ids]
        words = []
        current_word = ""
        for tok in tokens:
            if tok == " ":
                # A space token indicates the end of a word.
                if current_word:
                    words.append(current_word)
                    current_word = ""
            else:
                current_word += tok
        if current_word:
            words.append(current_word)
        return " ".join(words)

    def load(self, file_path):
        """
        Load a saved BPE model from a JSON file.

        The file is expected to contain:
          - "merges": a list of pairs (each pair is a list of two strings)
          - "token2id": the vocabulary mapping token to id
          - "full_words": (optional) list of full-word tokens.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.merges = [tuple(pair) for pair in data["merges"]]
        self.bpe_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.token2id = data["token2id"]
        # Note: keys of token2id are strings; ensure id2token maps integer ids to tokens.
        self.id2token = {int(idx): tok for tok, idx in self.token2id.items()}
        self.full_words = set(data.get("full_words", []))

    def save(self, file_path):
        """
        Save the current BPE model to a JSON file.
        """
        data = {
            "merges": [list(pair) for pair in self.merges],
            "token2id": self.token2id,
            "full_words": list(self.full_words)
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
