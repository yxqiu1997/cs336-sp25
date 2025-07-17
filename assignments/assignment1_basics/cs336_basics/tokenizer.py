from typing import Iterable

from cs336_basics.tokenizer_utils import pre_tokenize


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []


    @classmethod
    def from_files(
        cls,
        path: str,
        special_tokens: list[str] | None = None,
    ):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special tokens.
        """
        pass


    def encode(
        self,
        text: str
    ) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        # Build a reverse vocab: bytes -> int
        vocab_reversed = {v: k for k, v in self.vocab.items()}

        # Pre-tokenize the input text, preserving special tokens
        byte_pre_tokens = pre_tokenize(text, self.special_tokens, drop_special_token=False)

        # Encode special tokens once to avoid repeated encoding
        byte_special_tokens = set(token.encode('utf-8') for token in self.special_tokens)

        # Initialize the output list of pre-token ID lists
        pre_tokens = []

        # Convert each byte-level pre-token into a list of token IDs
        for pre_token in byte_pre_tokens:
            if pre_token in byte_special_tokens:
                # Special token: directly map to token ID
                pre_tokens.append([vocab_reversed[pre_token]])
            else:
                # Regular token: split by byte and map each to token ID
                pre_tokens.append([vocab_reversed[bytes([b])] for b in pre_token])

        # Apply merge rules to each pre-token list
        for i, pre_token in enumerate(pre_tokens):
            for merge in self.merges:
                new_index = vocab_reversed[merge[0] + merge[1]]
                j = 0
                merged = []
                while j < len(pre_token):
                    if (j < len(pre_token) - 1 and
                            (self.vocab[pre_token[j]], self.vocab[pre_token[j + 1]]) == merge):
                        # If current and next tokens match a merge rule, merge them
                        merged.append(new_index)
                        j += 2
                    else:
                        # Otherwise, keep the current token
                        merged.append(pre_token[j])
                        j += 1
                pre_token = merged
            pre_tokens[i] = pre_token

        # Flatten the list of token ID lists into a single token sequence
        tokens = [token for pre_token in pre_tokens for token in pre_token]
        return tokens


    def encode_iterable(
        self,
        iterable: Iterable[str]
    ) -> Iterable[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        """
        for line in iterable:
            token_ids = self.encode(line)
            for token_id in token_ids:
                yield token_id


    def decode(
        self,
        ids: list[int]
    ) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        vocab_size = len(self.vocab)
        # Pre-encode once to avoid repeated work
        replacement_token = "\uFFFD".encode('utf-8')

        # Use a list to collect tokens efficiently instead of concatenating bytes
        tokens = [
            self.vocab[token_id] if token_id < vocab_size else replacement_token
            for token_id in ids
        ]

        # Join bytes only once for efficiency, decode with 'replace' to ensure fallback
        return b''.join(tokens).decode('utf-8', errors='replace')
