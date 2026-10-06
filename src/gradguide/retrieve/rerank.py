"""Second-stage re-rankers.

``LexicalReranker`` is a cheap, dependency-free scorer (query-term coverage of
the passage, with a bonus for FAQ questions that match the query). The
``CrossEncoderReranker`` uses a sentence-transformers cross-encoder when the
``rerank`` extra is installed. Both return scores in which higher is better.
"""
from __future__ import annotations

from typing import Protocol, Sequence

from gradguide.retrieve.text import terms
from gradguide.types import Chunk


class Reranker(Protocol):
    name: str

    def score(self, query: str, chunks: Sequence[Chunk]) -> list[float]:
        ...


class LexicalReranker:
    name = "lexical"

    def score(self, query: str, chunks: Sequence[Chunk]) -> list[float]:
        q = set(terms(query))
        if not q:
            return [0.0] * len(chunks)
        scores = []
        for chunk in chunks:
            body = set(terms(chunk.text))
            head = set(terms(chunk.heading))
            coverage = len(q & (body | head)) / len(q)
            heading_match = len(q & head) / len(q)
            scores.append(coverage + 0.5 * heading_match)
        return scores


class CrossEncoderReranker:
    def __init__(self, model: str):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError("install the 'rerank' extra to use a cross-encoder") from exc
        self._model = CrossEncoder(model)
        self.name = f"cross-encoder:{model}"

    def score(self, query: str, chunks: Sequence[Chunk]) -> list[float]:  # pragma: no cover - heavy
        pairs = [(query, f"{c.heading}\n{c.text}") for c in chunks]
        return [float(s) for s in self._model.predict(pairs)]


def make_reranker(kind: str, model: str = "") -> Reranker | None:
    if kind in ("none", "off", ""):
        return None
    if kind == "lexical":
        return LexicalReranker()
    if kind in ("cross-encoder", "cross_encoder"):
        return CrossEncoderReranker(model)
    raise ValueError(f"unknown reranker: {kind!r}")
