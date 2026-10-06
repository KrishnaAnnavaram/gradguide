"""Versioned prompt templates and message assembly.

Templates live in ``generate/templates/<kind>_<version>.txt``; the version used is
recorded on every answer. Passage text is untrusted: tags that could close or
forge a passage block are neutralised before it is placed in the prompt.
"""
from __future__ import annotations

import html
import re
from functools import lru_cache
from importlib import resources
from typing import Sequence

from gradguide.ingest.tokens import SimpleTokenizer, TokenCounter
from gradguide.types import Hit

_TAGS = re.compile(r"</?\s*(passage|passages|question)\b[^>]*>", re.I)
_MARKERS = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")


@lru_cache(maxsize=None)
def load_template(kind: str, version: str) -> str:
    name = f"{kind}_{version}.txt"
    try:
        return resources.files("gradguide.generate").joinpath("templates", name).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"no prompt template {name!r}") from exc


def neutralise(text: str) -> str:
    """Defuse markup in retrieved text that could break out of its passage block."""
    return _TAGS.sub(lambda m: m.group(0).replace("<", "(").replace(">", ")"), text)


def render_passages(hits: Sequence[Hit]) -> str:
    blocks = []
    for n, hit in enumerate(hits, start=1):
        c = hit.chunk
        blocks.append(f'<passage id="{n}" source="{html.escape(c.source)}" section="{html.escape(c.heading)}">\n'
                      f"{neutralise(c.text)}\n</passage>")
    return "\n".join(blocks)


def history_messages(history: Sequence[tuple[str, str]], budget_tokens: int,
                     tok: TokenCounter | None = None) -> list[dict[str, str]]:
    """Most recent exchanges that fit in ``budget_tokens`` (oldest dropped first)."""
    tok = tok or SimpleTokenizer()
    kept: list[dict[str, str]] = []
    used = 0
    for question, answer in reversed(list(history)):
        answer = _MARKERS.sub("", answer)
        cost = tok.count(question) + tok.count(answer)
        if used + cost > budget_tokens:
            break
        kept[:0] = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        used += cost
    return kept


def build_messages(question: str, hits: Sequence[Hit], version: str = "v1", fallback_contact: str = "",
                   history: Sequence[tuple[str, str]] = (), history_tokens: int = 400) -> list[dict[str, str]]:
    system = load_template("system", version).format(fallback_contact=fallback_contact or "an advisor")
    user = load_template("user", version).format(question=neutralise(question.strip()),
                                                 passages=render_passages(hits))
    return [{"role": "system", "content": system},
            *history_messages(history, history_tokens),
            {"role": "user", "content": user}]
