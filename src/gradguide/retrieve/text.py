"""Search tokenisation (for BM25, the hashing embedder and the lexical reranker).

This only affects search terms; stored chunk text is never lower-cased or stripped.
"""
from __future__ import annotations

import re

TOKENIZER_VERSION = 1  # bump when this changes so persisted BM25 statistics are rebuilt

_WORD = re.compile(r"[a-z0-9]+(?:['.][a-z0-9]+)*")

STOPWORDS = frozenset(
    "a an and are as at be been but by can could do does for from had has have how i if in into is it its "
    "may me my of on or our should so than that the their them then there these they this to up us was we "
    "what when where which who whom why will with would you your".split()
)


def _fold(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def terms(text: str) -> list[str]:
    """Lower-cased whole-word search terms with stopwords removed and plurals folded."""
    return [_fold(t) for t in _WORD.findall(text.lower()) if t not in STOPWORDS]
