"""Hybrid retriever: BM25 + dense candidates, reciprocal-rank fusion, optional re-ranking.

Modes (all real code paths, so ablations compare actual systems):
``bm25``, ``vector``, ``hybrid`` (RRF) and ``hybrid+rerank``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from gradguide.index.store import Index
from gradguide.providers.embeddings import Embedder
from gradguide.retrieve.fusion import rrf
from gradguide.retrieve.rerank import Reranker
from gradguide.retrieve.text import terms
from gradguide.types import Hit

MODES = ("bm25", "vector", "hybrid", "hybrid+rerank")


def term_coverage(query: str, text: str) -> float:
    """Share of the query's search terms that occur in ``text`` (0..1)."""
    wanted = set(terms(query))
    return len(wanted & set(terms(text))) / len(wanted) if wanted else 0.0


def relevance(query: str, hit: Hit) -> float:
    """Evidence score used for abstention: the better of lexical coverage and cosine similarity."""
    return max(term_coverage(query, hit.chunk.search_text), hit.vector_score)


class Retriever:
    def __init__(self, index: Index, embedder: Embedder, reranker: Reranker | None = None,
                 candidate_k: int = 20, rrf_k: int = 60):
        if index.embedder_name != embedder.name:
            raise ValueError(f"index built with {index.embedder_name!r}, but queries use {embedder.name!r}; "
                             "run `gradguide index --force`")
        self.index, self.embedder, self.reranker = index, embedder, reranker
        self.candidate_k, self.rrf_k = candidate_k, rrf_k

    @property
    def default_mode(self) -> str:
        return "hybrid+rerank" if self.reranker is not None else "hybrid"

    def search(self, query: str, k: int = 4, mode: str | None = None) -> list[Hit]:
        mode = mode or self.default_mode
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "hybrid+rerank" and self.reranker is None:
            raise ValueError("hybrid+rerank needs a reranker")
        if not query.strip() or not self.index.chunks:
            return []

        lexical = self.index.bm25.score_all(query)[: self.candidate_k]
        qvec = self.embedder.embed([query])[0]
        cosines = self.index.vectors @ qvec
        dense_order = np.argsort(-cosines, kind="stable")[: self.candidate_k]
        dense = [(int(i), float(cosines[i])) for i in dense_order]
        bm25_of = dict(lexical)

        if mode == "bm25":
            ranked = [(i, s, {"bm25": r}) for r, (i, s) in enumerate(lexical, 1)]
        elif mode == "vector":
            ranked = [(i, s, {"vector": r}) for r, (i, s) in enumerate(dense, 1)]
        else:
            ranked = rrf({"bm25": [i for i, _ in lexical], "vector": [i for i, _ in dense]}, k=self.rrf_k)

        hits = [Hit(self.index.chunks[i], float(score), bm25_of.get(i, 0.0), float(cosines[i]), None, dict(ranks))
                for i, score, ranks in ranked]
        if mode == "hybrid+rerank":
            pool = hits[: self.candidate_k]
            scores = self.reranker.score(query, [h.chunk for h in pool])
            for hit, s in zip(pool, scores):
                hit.rerank_score = float(s)
            # stable sort: ties keep their fused order
            hits = sorted(pool, key=lambda h: -(h.rerank_score or 0.0))
        return hits[:k]


def best_relevance(query: str, hits: Sequence[Hit]) -> float:
    return max((relevance(query, h) for h in hits), default=0.0)
