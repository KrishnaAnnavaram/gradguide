"""Structure-aware chunking sized in tokens.

* Every FAQ question/answer pair is its own chunk (split only if enormous).
* Headed sections are packed paragraph by paragraph up to ``max_tokens``;
  newlines are preserved, so paragraph and list boundaries really are the
  split points. Over-long paragraphs fall back to sentences, then to words.
* Consecutive chunks of one section share ``overlap_tokens`` of trailing text.
* A chunk's id is a hash of its source, heading and text: re-indexing the same
  content yields the same ids, and edited content gets new ones.
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable

from gradguide.ingest.sources import SourceInfo
from gradguide.ingest.tokens import SimpleTokenizer, TokenCounter
from gradguide.types import Chunk, Document

_SENTENCES = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")


def chunk_id(source: str, heading: str, text: str) -> str:
    return hashlib.sha256(f"{source}\x1f{heading}\x1f{text}".encode("utf-8")).hexdigest()[:16]


def _split_to_budget(text: str, budget: int, tok: TokenCounter) -> list[str]:
    if tok.count(text) <= budget:
        return [text]
    pieces: list[str] = []
    current = ""
    parts = _SENTENCES.split(text)
    if len(parts) == 1:
        parts = text.split()
    for part in parts:
        if tok.count(part) > budget:  # a single giant sentence: recurse on its words
            pieces.extend(p for p in _split_to_budget(" ".join(part.split()), budget, tok) if p)
            continue
        candidate = f"{current} {part}".strip()
        if current and tok.count(candidate) > budget:
            pieces.append(current)
            current = part
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _tail(text: str, overlap: int, tok: TokenCounter) -> str:
    if overlap <= 0:
        return ""
    words = text.split()
    tail: list[str] = []
    for word in reversed(words):
        if tok.count(" ".join([word, *tail])) > overlap:
            break
        tail.insert(0, word)
    return " ".join(tail)


def pack(text: str, max_tokens: int, overlap_tokens: int, tok: TokenCounter | None = None) -> list[str]:
    tok = tok or SimpleTokenizer()
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap must be smaller than the chunk size")
    budget = max_tokens - overlap_tokens
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        if paragraph.strip():
            units.extend(_split_to_budget(paragraph.strip(), budget, tok))
    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}" if current else unit
        if current and tok.count(candidate) > max_tokens:
            chunks.append(current)
            carry = _tail(current, overlap_tokens, tok)
            current = f"{carry}\n\n{unit}" if carry else unit
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def chunk_document(doc: Document, max_tokens: int = 220, overlap_tokens: int = 30,
                   tok: TokenCounter | None = None, info: SourceInfo | None = None) -> list[Chunk]:
    tok = tok or SimpleTokenizer()
    title = (info.title if info and info.title else doc.title)
    updated = (info.updated if info else "") or str(doc.metadata.get("updated", ""))
    chunks: list[Chunk] = []
    for section in doc.sections:
        if section.kind == "faq":
            pieces = _split_to_budget(section.text, max_tokens - tok.count(section.heading), tok)
            texts = [f"Q: {section.heading}\nA: {piece}" for piece in pieces]
        else:
            texts = pack(section.text, max_tokens, overlap_tokens, tok)
        for text in texts:
            chunks.append(Chunk(
                chunk_id=chunk_id(doc.source, section.heading, text),
                source=doc.source,
                title=title,
                heading=section.heading,
                text=text,
                kind=section.kind,
                n_tokens=tok.count(text),
                updated=updated,
            ))
    return chunks


def chunk_corpus(docs: Iterable[Document], max_tokens: int = 220, overlap_tokens: int = 30,
                 tok: TokenCounter | None = None, manifest: dict[str, SourceInfo] | None = None) -> list[Chunk]:
    """Chunk all documents; identical chunks (same id) are kept once."""
    manifest = manifest or {}
    seen: set[str] = set()
    out: list[Chunk] = []
    for doc in docs:
        for chunk in chunk_document(doc, max_tokens, overlap_tokens, tok, manifest.get(doc.source)):
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                out.append(chunk)
    return out
