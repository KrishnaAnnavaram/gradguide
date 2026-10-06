"""Token counting, so chunk sizes are measured in tokens rather than characters.

``SimpleTokenizer`` (default, no dependencies) counts words and punctuation marks,
which tracks sub-word tokenizers closely enough for sizing chunks. Install the
``tokens`` extra and set ``GRADGUIDE_TOKENIZER=tiktoken`` to count real BPE tokens.
"""
from __future__ import annotations

import re
from typing import Protocol

_PIECES = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int:
        ...


class SimpleTokenizer:
    name = "simple"

    def count(self, text: str) -> int:
        return len(_PIECES.findall(text))


class TiktokenTokenizer:
    def __init__(self, encoding: str = "cl100k_base"):
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError("install the 'tokens' extra to use tiktoken") from exc
        self._enc = tiktoken.get_encoding(encoding)
        self.name = f"tiktoken:{encoding}"

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))


def make_tokenizer(kind: str) -> TokenCounter:
    if kind in ("simple", ""):
        return SimpleTokenizer()
    if kind == "tiktoken":
        return TiktokenTokenizer()
    raise ValueError(f"unknown tokenizer: {kind!r}")
