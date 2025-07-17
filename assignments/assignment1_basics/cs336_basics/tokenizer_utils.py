import regex as re

from cs336_basics.constants import PAT


def split_by_special_tokens(
    text: str,
    special_tokens: list[str]
) -> list[str]:
    """
    Split the input text based on special tokens.

    Examples:
        text = "Hello<|end|>World"
        special_tokens = ["<|end|>"]
        => ["Hello", "<|end|>", "World"]
    """
    # Sort by length to avoid substring matching conflicts
    # ensuring that the regular expression matches longer, complete tokens first
    special_tokens_sorted = sorted(special_tokens, key=lambda x: -len(x))

    if not special_tokens_sorted:
        return [text]

    pattern = "|".join(re.escape(token) for token in special_tokens_sorted)
    return re.split(f"({pattern})", text)


def pre_tokenize(
    text: str,
    special_tokens: list[str],
    drop_special_token: bool = True
) -> list[bytes]:
    """
    Split input text into UTF-8 encoded byte-level pre-tokens.

    - Special tokens are preserved as whole.
    - Other parts are tokenized using a regex-based tokenizer and converted to bytes.

    Args:
        text (str): The input string.
        special_tokens (list[str]): List of special tokens to isolate.
        drop_special_token (bool): If True, special tokens are excluded from result.

    Returns:
        list[bytes]: List of encoded bytes.
    """
    parts = split_by_special_tokens(text, special_tokens)

    token_list = []
    for part in parts:
        if part in special_tokens:
            if not drop_special_token:
                token_list.append([part.encode("utf-8")])
        else:
            str_tokens = re.findall(PAT, part)
            byte_tokens = [token.encode("utf-8") for token in str_tokens]
            token_list.append(byte_tokens)

    # Flatten the list of token lists
    return [token for sublist in token_list for token in sublist]
