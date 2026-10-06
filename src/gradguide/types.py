"""Plain data types shared across the pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Section:
    """A structural unit produced by a loader: a headed section or one FAQ question/answer pair."""

    heading: str
    text: str
    kind: str = "section"   # section | faq


@dataclass(frozen=True)
class Document:
    source: str                       # path relative to the documents folder
    title: str
    sections: tuple[Section, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str        # content hash: stable across runs, changes when the text changes
    source: str
    title: str
    heading: str
    text: str
    kind: str = "section"
    n_tokens: int = 0
    updated: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(**data)

    @property
    def location(self) -> str:
        return f"{self.source} > {self.heading}" if self.heading else self.source

    @property
    def search_text(self) -> str:
        """Text used for indexing: title and heading give short chunks their context."""
        return f"{self.title}\n{self.heading}\n{self.text}"


@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25_score: float = 0.0
    vector_score: float = 0.0
    rerank_score: float | None = None
    ranks: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Citation:
    marker: int
    chunk_id: str
    source: str
    heading: str
    snippet: str


@dataclass
class Answer:
    question: str
    text: str
    citations: list[Citation]
    hits: list[Hit]
    abstained: bool
    model: str
    prompt_version: str
    retrieval_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.text,
            "abstained": self.abstained,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "retrieval_mode": self.retrieval_mode,
            "citations": [asdict(c) for c in self.citations],
        }
