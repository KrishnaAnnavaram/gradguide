"""Okapi BM25 with state that can be saved next to the vectors."""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Sequence

from gradguide.retrieve.text import terms


class BM25:
    def __init__(self, docs: Sequence[Counter[str]] | None = None, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs: list[Counter[str]] = list(docs or [])
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avgdl = sum(self.lengths) / len(self.lengths) if self.lengths else 1.0
        self.df: Counter[str] = Counter()
        for d in self.docs:
            self.df.update(d.keys())

    @classmethod
    def from_texts(cls, texts: Sequence[str], **kwargs: float) -> "BM25":
        return cls([Counter(terms(t)) for t in texts], **kwargs)

    def _idf(self, term: str) -> float:
        n, df = len(self.docs), self.df.get(term, 0)
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def score_all(self, query: str) -> list[tuple[int, float]]:
        q = set(terms(query))
        results = []
        for i, doc in enumerate(self.docs):
            total = 0.0
            denom_base = self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avgdl or 1.0))
            for term in q:
                tf = doc.get(term, 0)
                if tf:
                    total += self._idf(term) * tf * (self.k1 + 1) / (tf + denom_base)
            if total > 0:
                results.append((i, total))
        results.sort(key=lambda r: (-r[1], r[0]))
        return results

    def state(self) -> dict[str, Any]:
        return {"k1": self.k1, "b": self.b, "docs": [dict(d) for d in self.docs]}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "BM25":
        return cls([Counter(d) for d in state["docs"]], k1=state["k1"], b=state["b"])
