import os
import regex as re
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm

from cs336_basics.constants import *


class BPETrainer:
    @staticmethod
    def init_vocab(special_tokens: List[str]) -> Tuple[Dict[int, bytes], int]:
        """
        Initialize vocabulary with single-byte tokens and special tokens.
        """
        vocab = {i: bytes([i]) for i in range(256)}
        next_id = 256

        for token in special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in vocab.values():
                vocab[next_id] = token_bytes
                next_id += 1

        return vocab, next_id

    @staticmethod
    def pre_tokenize(input_path: str | os.PathLike, special_tokens: List[str]) -> Dict[Tuple[bytes, ...], int]:
        """
        Tokenize text into byte-level tuples and count their frequencies.
        """
        with open(input_path, "r", encoding="utf-8") as file:
            text = file.read()

        chunks = re.split("|".join(map(re.escape, special_tokens)), text)
        token_counts = defaultdict(int)

        for chunk in tqdm(chunks, desc="Pre-tokenizing"):
            for match in re.finditer(PAT, chunk):
                word = match.group(0)
                token = tuple(bytes([b]) for b in word.encode("utf-8"))
                token_counts[token] += 1

        return token_counts

    @staticmethod
    def count_byte_pairs(token_counts: Dict[Tuple[bytes, ...], int]) -> Dict[Tuple[bytes, bytes], int]:
        """
        Count the frequency of adjacent byte pairs across all tokens.
        """
        pair_counts = defaultdict(int)
        for token_seq, count in token_counts.items():
            for i in range(len(token_seq) - 1):
                pair = (token_seq[i], token_seq[i + 1])
                pair_counts[pair] += count
        return pair_counts

    @staticmethod
    def select_best_pair(pair_counts: Dict[Tuple[bytes, bytes], int]) -> Tuple[bytes, bytes]:
        """
        Select the most frequent byte pair (deterministic if tied).
        """
        max_count = max(pair_counts.values())
        candidates = [pair for pair, count in pair_counts.items() if count == max_count]
        return max(candidates)

    @staticmethod
    def apply_merge(
        token_counts: Dict[Tuple[bytes, ...], int],
        pair_to_merge: Tuple[bytes, bytes],
        new_token: bytes
    ) -> None:
        """
        Apply the merge operation to all token sequences.
        """
        changes = []

        for token_seq, count in token_counts.items():
            merged_seq = []
            i = 0
            changed = False
            while i < len(token_seq):
                if i < len(token_seq) - 1 and token_seq[i:i + 2] == pair_to_merge:
                    merged_seq.append(new_token)
                    i += 2
                    changed = True
                else:
                    merged_seq.append(token_seq[i])
                    i += 1
            if changed:
                changes.append((token_seq, tuple(merged_seq), count))

        for old_seq, new_seq, count in changes:
            token_counts[new_seq] += count
            del token_counts[old_seq]
