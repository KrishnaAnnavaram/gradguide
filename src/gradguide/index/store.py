"""Idempotent, incremental index.

* Chunk ids are content hashes, so re-running ingestion on unchanged documents
  produces the same ids and nothing is duplicated.
* ``sync_index`` compares the fresh chunk set with the saved one: vectors of
  unchanged chunks are reused, only new or edited chunks are embedded, and chunks
  whose text disappeared are deleted.
* If the documents, chunking settings, tokenizer and embedder are all unchanged
  (same fingerprint), the saved index is loaded without re-chunking at all.
* Only compact artifacts are written: chunks (JSONL), BM25 term counts (JSON),
  vectors (``.npy``) and a manifest.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from gradguide.ingest.chunking import chunk_corpus
from gradguide.ingest.loaders import iter_paths, load_folder
from gradguide.ingest.sources import MANIFEST_NAME, load_manifest
from gradguide.ingest.tokens import SimpleTokenizer, TokenCounter
from gradguide.providers.embeddings import Embedder
from gradguide.retrieve.bm25 import BM25
from gradguide.retrieve.text import TOKENIZER_VERSION
from gradguide.types import Chunk

INDEX_FORMAT = 1


@dataclass
class Index:
    chunks: list[Chunk]
    vectors: np.ndarray
    bm25: BM25
    embedder_name: str
    manifest: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.position = {c.chunk_id: i for i, c in enumerate(self.chunks)}

    @classmethod
    def build(cls, chunks: Sequence[Chunk], vectors: np.ndarray, embedder_name: str, manifest: dict | None = None):
        return cls(list(chunks), vectors, BM25.from_texts([c.search_text for c in chunks]), embedder_name,
                   manifest or {})

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)

        def write(name: str, data: str) -> None:
            tmp = directory / f".{name}.tmp"
            tmp.write_text(data, encoding="utf-8")
            os.replace(tmp, directory / name)

        write("chunks.jsonl", "".join(json.dumps(c.to_dict(), ensure_ascii=False) + "\n" for c in self.chunks))
        write("bm25.json", json.dumps(self.bm25.state()))
        tmp_vec = directory / ".vectors.tmp.npy"
        np.save(tmp_vec, self.vectors)
        os.replace(tmp_vec, directory / "vectors.npy")
        write("manifest.json", json.dumps(self.manifest, indent=2))  # last: marks the index as complete

    @classmethod
    def load(cls, directory: Path) -> "Index":
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        lines = (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        chunks = [Chunk.from_dict(json.loads(line)) for line in lines if line.strip()]
        bm25 = BM25.from_state(json.loads((directory / "bm25.json").read_text(encoding="utf-8")))
        return cls(chunks, np.load(directory / "vectors.npy"), bm25, manifest["embedder"], manifest)


@dataclass
class SyncReport:
    added: int = 0
    removed: int = 0
    kept: int = 0
    loaded_from_cache: bool = False

    def __str__(self) -> str:
        if self.loaded_from_cache:
            return f"index up to date ({self.kept} chunks); nothing re-embedded"
        return f"{self.added} chunks embedded, {self.kept} reused, {self.removed} removed"


def _file_hashes(docs_dir: Path) -> dict[str, str]:
    files = iter_paths(docs_dir)
    manifest_file = docs_dir / MANIFEST_NAME
    if manifest_file.is_file():
        files.append(manifest_file)
    return {p.relative_to(docs_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def fingerprint(files: dict[str, str], settings_key: str) -> str:
    h = hashlib.sha256(settings_key.encode("utf-8"))
    for name in sorted(files):
        h.update(f"{name}={files[name]};".encode("utf-8"))
    return h.hexdigest()


def read_manifest(index_dir: Path) -> dict | None:
    try:
        return json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def sync_index(docs_dir: Path, index_dir: Path, embedder: Embedder, max_tokens: int = 220,
               overlap_tokens: int = 30, tok: TokenCounter | None = None, force: bool = False
               ) -> tuple[Index, SyncReport]:
    tok = tok or SimpleTokenizer()
    docs_dir, index_dir = Path(docs_dir), Path(index_dir)
    settings_key = (f"format={INDEX_FORMAT};terms={TOKENIZER_VERSION};tok={tok.name};"
                    f"size={max_tokens};overlap={overlap_tokens};embedder={embedder.name}")
    files = _file_hashes(docs_dir)
    fp = fingerprint(files, settings_key)

    previous_manifest = None if force else read_manifest(index_dir)
    if previous_manifest and previous_manifest.get("fingerprint") == fp:
        index = Index.load(index_dir)
        return index, SyncReport(kept=len(index.chunks), loaded_from_cache=True)

    reusable: dict[str, np.ndarray] = {}
    old_text: dict[str, str] = {}
    if previous_manifest and previous_manifest.get("embedder") == embedder.name:
        old = Index.load(index_dir)
        reusable = {c.chunk_id: old.vectors[i] for i, c in enumerate(old.chunks)}
        old_text = {c.chunk_id: c.search_text for c in old.chunks}

    chunks = chunk_corpus(load_folder(docs_dir), max_tokens, overlap_tokens, tok, load_manifest(docs_dir))
    # a vector is reused only if the embedded text (title + heading + text) is unchanged: a new title in
    # sources.toml keeps the chunk id but changes what was embedded
    to_embed = [c for c in chunks
                if c.chunk_id not in reusable or old_text.get(c.chunk_id) != c.search_text]
    for c in to_embed:
        reusable.pop(c.chunk_id, None)
    fresh = embedder.embed([c.search_text for c in to_embed]) if to_embed else None
    fresh_by_id = {c.chunk_id: fresh[i] for i, c in enumerate(to_embed)} if fresh is not None else {}

    if chunks:
        vectors = np.stack([reusable.get(c.chunk_id, fresh_by_id.get(c.chunk_id)) for c in chunks]).astype(np.float32)
    else:
        vectors = np.zeros((0, 1), dtype=np.float32)
    current_ids = {c.chunk_id for c in chunks}
    report = SyncReport(added=len(to_embed), removed=len(set(reusable) - current_ids),
                        kept=len(chunks) - len(to_embed))
    manifest = {
        "format": INDEX_FORMAT,
        "fingerprint": fp,
        "embedder": embedder.name,
        "tokenizer": tok.name,
        "chunk_tokens": max_tokens,
        "overlap_tokens": overlap_tokens,
        "chunks": len(chunks),
        "sources": files,
        "indexed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    index = Index.build(chunks, vectors, embedder.name, manifest)
    index.save(index_dir)
    return index, report
