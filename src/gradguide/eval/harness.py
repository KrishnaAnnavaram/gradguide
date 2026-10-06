"""Evaluation harness with real ablations.

Every configuration in a report is actually executed: each retrieval mode runs
its own code path, and each chunk size gets its own freshly built index.

Gold-set format (JSONL), one question per line::

    {"id": "reg-01", "question": "...", "answerable": true,
     "relevant": [{"source": "registration.md", "heading": "Holds"}]}

A retrieved chunk satisfies a target if its source matches and its heading
contains the target heading (case-insensitive). Unanswerable questions have
``"answerable": false`` and no targets; they test abstention.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from gradguide.config import Settings
from gradguide.eval.metrics import abstention_scores, mean, mrr, recall_at_k
from gradguide.providers.embeddings import Embedder
from gradguide.providers.llm import ChatModel
from gradguide.retrieve.hybrid import MODES, Retriever
from gradguide.types import Chunk

KS = (1, 3, 5)


@dataclass
class GoldItem:
    qid: str
    question: str
    answerable: bool = True
    relevant: list[dict[str, str]] = field(default_factory=list)


def load_gold(path: str | Path) -> list[GoldItem]:
    items = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        item = GoldItem(row["id"], row["question"], bool(row.get("answerable", True)), list(row.get("relevant", [])))
        if item.answerable and not item.relevant:
            raise ValueError(f"line {n}: answerable question without relevant targets")
        items.append(item)
    return items


def satisfied(chunk: Chunk, targets: Sequence[dict[str, str]]) -> set[int]:
    return {i for i, t in enumerate(targets)
            if chunk.source == t["source"] and t.get("heading", "").lower() in chunk.heading.lower()}


def retrieval_report(retriever: Retriever, gold: Iterable[GoldItem], modes: Sequence[str] | None = None,
                     ks: Sequence[int] = KS) -> dict[str, dict[str, float]]:
    answerable = [g for g in gold if g.answerable]
    modes = modes or [m for m in MODES if m != "hybrid+rerank" or retriever.reranker is not None]
    report: dict[str, dict[str, float]] = {}
    for mode in modes:
        recalls: dict[int, list[float]] = {k: [] for k in ks}
        rrs: list[float] = []
        for item in answerable:
            hits = retriever.search(item.question, k=max(ks), mode=mode)
            matches = [satisfied(h.chunk, item.relevant) for h in hits]
            for k in ks:
                recalls[k].append(recall_at_k(matches, len(item.relevant), k))
            rrs.append(mrr(matches))
        report[mode] = {**{f"recall@{k}": mean(recalls[k]) for k in ks}, "mrr": mean(rrs)}
    return report


def answer_report(service, gold: Sequence[GoldItem]) -> dict[str, float]:
    """End-to-end checks: abstention decisions and whether citations include a gold source."""
    abstained, cited_gold = [], []
    for item in gold:
        answer = service.ask(item.question)
        abstained.append(answer.abstained)
        if item.answerable and not answer.abstained:
            cited_gold.append(float(any(
                c.source == t["source"] and t.get("heading", "").lower() in c.heading.lower()
                for c in answer.citations for t in item.relevant)))
    scores = abstention_scores([g.answerable for g in gold], abstained)
    scores["gold_source_cited"] = mean(cited_gold)
    return scores


def chunk_size_ablation(settings: Settings, gold: Sequence[GoldItem], sizes: Sequence[int],
                        embedder: Embedder | None = None, model: ChatModel | None = None,
                        mode: str = "hybrid") -> dict[int, dict[str, float]]:
    """Rebuild the index at each chunk size (in a temporary folder) and score retrieval."""
    from gradguide.service import GradGuide

    results: dict[int, dict[str, float]] = {}
    with tempfile.TemporaryDirectory(prefix="gradguide-ablation-") as tmp:
        for size in sizes:
            variant = Settings(**{**settings.__dict__, "chunk_tokens": size,
                                  "chunk_overlap_tokens": min(settings.chunk_overlap_tokens, size // 4),
                                  "index_dir": Path(tmp) / f"size-{size}"})
            svc = GradGuide(variant, embedder=embedder, model=model)
            row = retrieval_report(svc.retriever, gold, modes=[mode])[mode]
            row["chunks"] = float(len(svc.index.chunks))
            results[size] = row
    return results


def markdown_table(rows: dict, first_column: str) -> str:
    if not rows:
        return ""
    columns = list(next(iter(rows.values())).keys())
    out = [f"| {first_column} | " + " | ".join(columns) + " |", "|---|" + "---:|" * len(columns)]
    for name, row in rows.items():
        cells = [f"{row[c]:.0f}" if c == "chunks" else f"{row[c]:.3f}" for c in columns]
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(out)
