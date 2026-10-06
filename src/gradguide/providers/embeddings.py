"""Embedders: one interface, an offline fake and two API-backed implementations."""
from __future__ import annotations

import hashlib
from typing import Protocol, Sequence

import numpy as np

from gradguide.providers.http import ProviderError, post_json
from gradguide.retrieve.text import terms


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """(n, dim) float32 array with unit-length rows."""
        ...


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    m = np.atleast_2d(np.asarray(matrix, dtype=np.float32))
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1.0, norms)


class HashingEmbedder:
    """Deterministic feature-hashing embedder (words, word pairs, character 4-grams).

    It needs no model and no network, which makes it ideal for tests and offline
    demos. It captures lexical overlap and spelling variants, not meaning.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hashing-v1-{dim}"

    def _index(self, feature: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.md5(feature.encode("utf-8"), usedforsecurity=False).digest()[:8], "big")
        return h % self.dim, 1.0 if h & 1 else -1.0

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        words = terms(text)
        features = [(w, 1.0) for w in words] + [(f"{a}|{b}", 0.6) for a, b in zip(words, words[1:])]
        for w in words:
            padded = f"^{w}$"
            features += [(padded[i:i + 4], 0.25) for i in range(max(len(padded) - 3, 1))]
        for feature, weight in features:
            i, sign = self._index(feature)
            vec[i] += sign * weight
        return vec

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return unit_rows(np.stack([self._vector(t) for t in texts]))


class OpenAICompatibleEmbedder:
    def __init__(self, model: str, base_url: str, api_key: str, batch: int = 64):
        if not api_key:
            raise ProviderError("OPENAI_API_KEY is required for the openai embedding provider")
        self.model, self.base_url, self.api_key, self.batch = model, base_url, api_key, batch
        self.name = f"openai:{model}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            reply = post_json(f"{self.base_url}/embeddings", {"model": self.model, "input": list(texts[i:i + self.batch])},
                              api_key=self.api_key)
            vectors += [row["embedding"] for row in sorted(reply["data"], key=lambda r: r["index"])]
        return unit_rows(np.asarray(vectors, dtype=np.float32))


class OllamaEmbedder:
    def __init__(self, model: str, host: str):
        self.model, self.host = model, host
        self.name = f"ollama:{model}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        reply = post_json(f"{self.host}/api/embed", {"model": self.model, "input": list(texts)})
        return unit_rows(np.asarray(reply["embeddings"], dtype=np.float32))
