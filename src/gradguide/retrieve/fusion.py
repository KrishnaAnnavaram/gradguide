"""Reciprocal-rank fusion of several ranked lists.

``fused(d) = sum over lists L containing d of 1 / (k + rank_L(d))`` with ranks
starting at 1 and k = 60 by default. Rank-based fusion needs no score
calibration between BM25 (unbounded) and cosine similarity (-1..1).
"""
from __future__ import annotations

from typing import Hashable, Mapping, Sequence, TypeVar

Key = TypeVar("Key", bound=Hashable)


def rrf(ranked_lists: Mapping[str, Sequence[Key]], k: int = 60) -> list[tuple[Key, float, dict[str, int]]]:
    if k < 1:
        raise ValueError("k must be >= 1")
    totals: dict[Key, float] = {}
    positions: dict[Key, dict[str, int]] = {}
    order: list[Key] = []
    for list_name, items in ranked_lists.items():
        seen_here: set[Key] = set()
        for rank, item in enumerate(items, start=1):
            if item in seen_here:
                continue
            seen_here.add(item)
            if item not in totals:
                totals[item] = 0.0
                positions[item] = {}
                order.append(item)
            totals[item] += 1.0 / (k + rank)
            positions[item][list_name] = rank
    tie_break = {item: i for i, item in enumerate(order)}
    best_first = sorted(order, key=lambda item: (-totals[item], tie_break[item]))
    return [(item, totals[item], positions[item]) for item in best_first]
