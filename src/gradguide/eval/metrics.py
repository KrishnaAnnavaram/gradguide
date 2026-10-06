"""Retrieval and answer-level metrics.

Retrieval metrics take, for each ranked result, the set of gold targets it
satisfies. The answer-level checks are deliberately simple and named for what
they measure: whether the assistant abstained when it should, and whether its
citations include a gold source. They are not hallucination detectors.
"""
from __future__ import annotations

from typing import Sequence


def recall_at_k(matches: Sequence[set[int]], n_targets: int, k: int) -> float:
    if n_targets < 1:
        raise ValueError("need at least one gold target")
    found: set[int] = set()
    for m in matches[:k]:
        found.update(m)
    return len(found) / n_targets


def mrr(matches: Sequence[set[int]]) -> float:
    return next((1.0 / rank for rank, m in enumerate(matches, start=1) if m), 0.0)


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def abstention_scores(expected_answerable: Sequence[bool], abstained: Sequence[bool]) -> dict[str, float]:
    """Accuracy of the answer/abstain decision, plus the two error rates."""
    pairs = list(zip(expected_answerable, abstained))
    answerable = [a for e, a in pairs if e]
    unanswerable = [a for e, a in pairs if not e]
    return {
        "decision_accuracy": mean([float(e != a) for e, a in pairs]),
        "false_abstention_rate": mean([float(a) for a in answerable]),
        "missed_abstention_rate": mean([float(not a) for a in unanswerable]),
    }
