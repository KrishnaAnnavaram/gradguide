from __future__ import annotations

from pathlib import Path

import pytest

from gradguide.config import Settings
from gradguide.providers.embeddings import HashingEmbedder

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DOCS = ROOT / "sample_docs"
GOLD = ROOT / "eval" / "gold_set.jsonl"


class CountingEmbedder(HashingEmbedder):
    def __init__(self):
        super().__init__()
        self.batches: list[int] = []

    def embed(self, texts):
        self.batches.append(len(texts))
        return super().embed(texts)


class RecordingModel:
    """Fake chat model that records every prompt it receives."""

    name = "fake:recording"

    def __init__(self, reply: str = "Answer [1]."):
        self.reply = reply
        self.prompts: list[list[dict[str, str]]] = []

    def chat(self, messages):
        self.prompts.append(messages)
        return self.reply


def make_settings(tmp_path: Path, **overrides: str) -> Settings:
    env = {"GRADGUIDE_HOME": str(tmp_path), "GRADGUIDE_DOCS_DIR": str(SAMPLE_DOCS),
           "GRADGUIDE_INDEX_DIR": "index", **overrides}
    return Settings.from_env(env=env)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def embedder() -> CountingEmbedder:
    return CountingEmbedder()
