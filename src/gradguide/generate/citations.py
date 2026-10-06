"""Validate ``[n]`` citation markers against the passages shown to the model."""
from __future__ import annotations

import re
from typing import Sequence

from gradguide.types import Citation, Hit

_MARK = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def cited_numbers(text: str) -> list[int]:
    out: list[int] = []
    for group in _MARK.findall(text):
        for n in (int(x) for x in group.split(",")):
            if n not in out:
                out.append(n)
    return out


def _snippet(text: str, width: int = 180) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 3].rstrip() + "..."


def attach_citations(answer: str, hits: Sequence[Hit]) -> tuple[str, list[Citation]]:
    """Drop markers that point at no passage; if none remain, cite every passage shown."""
    n_passages = len(hits)

    def repair(match: re.Match) -> str:
        valid = [x.strip() for x in match.group(1).split(",") if 1 <= int(x) <= n_passages]
        return "[" + ", ".join(valid) + "]" if valid else ""

    text = _MARK.sub(repair, answer)
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", re.sub(r"[ \t]{2,}", " ", text)).strip()
    numbers = [n for n in cited_numbers(text) if 1 <= n <= n_passages] or list(range(1, n_passages + 1))
    citations = [Citation(n, hits[n - 1].chunk.chunk_id, hits[n - 1].chunk.source, hits[n - 1].chunk.heading,
                          _snippet(hits[n - 1].chunk.text)) for n in numbers]
    return text, citations


def sources_block(citations: Sequence[Citation]) -> str:
    if not citations:
        return ""
    rows = [f"[{c.marker}] {c.source}" + (f" - {c.heading}" if c.heading else "") for c in citations]
    return "Sources:\n" + "\n".join(rows)
