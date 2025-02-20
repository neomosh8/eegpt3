import json
from typing import Dict, List


def quantize_number(z_value, resolution=80):
    """
    Quantize a z-scored value into a token: e.g. 'C23', 'D150', etc.
    with finer granularity near -1 < z < 1, and coarser outside.
    """
    z_clamped = max(min(z_value, 5), -5)
    ranges = [
        {'id': 'A', 'start': -5, 'end': -3, 'proportion': 0.05},
        {'id': 'B', 'start': -3, 'end': -2, 'proportion': 0.10},
        {'id': 'C', 'start': -2, 'end': -1, 'proportion': 0.15},
        {'id': 'D', 'start': -1, 'end': 1, 'proportion': 0.40},
        {'id': 'E', 'start': 1, 'end': 2, 'proportion': 0.15},
        {'id': 'F', 'start': 2, 'end': 3, 'proportion': 0.10},
        {'id': 'G', 'start': 3, 'end': 5, 'proportion': 0.05},
    ]
    for r in ranges:
        r['tokens'] = int(round(r['proportion'] * resolution))
    cumulative = 0
    for r in ranges:
        r['token_start'] = cumulative
        cumulative += r['tokens']
    for r in ranges:
        if r['start'] <= z_clamped < r['end']:
            range_id = r['id']
            start = r['start']
            end = r['end']
            tokens_in_range = r['tokens']
            token_offset = r['token_start']
            break
    else:
        range_id = 'G'
        start = 3
        end = 5
        tokens_in_range = ranges[-1]['tokens']
        token_offset = ranges[-1]['token_start']
    if tokens_in_range <= 1:
        quant_level = 0
    else:
        q = (z_clamped - start) / (end - start)
        quant_level = int(round(q * (tokens_in_range - 1)))
    global_quant_index = token_offset + quant_level
    token = f"{range_id}{global_quant_index}"
    return token


class SimpleQuantizerTokenizer:
    def __init__(self, resolution=80):
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.resolution = resolution
        self._build_vocab()

    def _build_vocab(self):
        """Build vocabulary by generating all possible tokens from quantize_number."""
        # Generate tokens across the full range [-5, 5] with fine steps
        step_size = 0.01  # Small step to ensure we cover all quantization levels
        z_values = [i * step_size for i in range(int(-5 / step_size), int(5 / step_size) + 1)]
        z_values.append(5.0)  # Ensure edge case z=5 is included

        tokens = set()
        for z in z_values:
            token = quantize_number(z, self.resolution)
            tokens.add(token)

        # Assign IDs to tokens
        self.next_id = 0
        for token in sorted(tokens):  # Sort for consistency
            self.token_to_id[token] = self.next_id
            self.id_to_token[self.next_id] = token
            self.next_id += 1

        # Add UNK and |trial| tokens at the end
        for special_token in ["UNK", "|trial|"]:
            if special_token not in self.token_to_id:
                self.token_to_id[special_token] = self.next_id
                self.id_to_token[self.next_id] = special_token
                self.next_id += 1
    def encode(self, text: str) -> List[int]:
        """Convert space-separated token string to list of IDs."""
        tokens = text.strip().split()
        return [self.token_to_id.get(token, self.next_id - 1) for token in tokens]  # Last ID for unknown

    def decode(self, ids: List[int]) -> str:
        """Convert list of IDs back to space-separated token string."""
        tokens = [self.id_to_token.get(id_, "UNK") for id_ in ids]  # "UNK" for unknown IDs
        return " ".join(tokens)

    def save_vocab(self, filepath: str):
        """Save vocabulary to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

    def load_vocab(self, filepath: str):
        """Load vocabulary from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.token_to_id = json.load(f)
            self.id_to_token = {int(id_): token for token, id_ in self.token_to_id.items()}
            self.next_id = max(self.id_to_token.keys()) + 1 if self.id_to_token else 0

    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.token_to_id)

