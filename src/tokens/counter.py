"""Token counting utilities."""

from typing import Any

import tiktoken

# Cache for encoders
_ENCODERS: dict[str, Any] = {}


def get_encoder(encoding_name: str = "cl100k_base") -> Any:
    """Get a tiktoken encoder, with caching."""
    if encoding_name not in _ENCODERS:
        _ENCODERS[encoding_name] = tiktoken.get_encoding(encoding_name)
    return _ENCODERS[encoding_name]


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in a string."""
    encoder = get_encoder(encoding_name)
    return len(encoder.encode(text))


def count_tokens_batch(texts: list[str], encoding_name: str = "cl100k_base") -> list[int]:
    """Count tokens for a batch of strings."""
    encoder = get_encoder(encoding_name)
    return [len(encoder.encode(text)) for text in texts]


def estimate_message_tokens(
    messages: list[dict[str, str]],
    encoding_name: str = "cl100k_base",
) -> int:
    """
    Estimate token count for a list of chat messages.

    This is an approximation - actual token count may vary by model.
    """
    encoder = get_encoder(encoding_name)
    total = 0

    for message in messages:
        # Each message has overhead for role tokens
        total += 4  # Approximate overhead per message

        for key, value in message.items():
            total += len(encoder.encode(str(value)))

    total += 2  # Conversation overhead

    return total


def get_token_breakdown(text: str, encoding_name: str = "cl100k_base") -> dict[str, Any]:
    """Get detailed token breakdown for a string."""
    encoder = get_encoder(encoding_name)
    tokens = encoder.encode(text)

    return {
        "total_tokens": len(tokens),
        "token_ids": tokens,
        "decoded_tokens": [encoder.decode([t]) for t in tokens],
        "char_to_token_ratio": len(text) / max(1, len(tokens)),
    }
