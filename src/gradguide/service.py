"""Service layer shared by the CLI, the HTTP API and the UI."""
from __future__ import annotations

from typing import Sequence

from gradguide.config import Settings
from gradguide.generate.answer import Advisor
from gradguide.index.store import Index, SyncReport, sync_index
from gradguide.ingest.tokens import make_tokenizer
from gradguide.privacy.querylog import QueryLog
from gradguide.providers import build_chat_model, build_embedder
from gradguide.providers.embeddings import Embedder
from gradguide.providers.llm import ChatModel
from gradguide.retrieve.hybrid import Retriever
from gradguide.retrieve.rerank import Reranker, make_reranker
from gradguide.types import Answer

_FROM_SETTINGS = object()


class GradGuide:
    def __init__(self, settings: Settings, embedder: Embedder | None = None, model: ChatModel | None = None,
                 reranker: Reranker | None | object = _FROM_SETTINGS, force_reindex: bool = False):
        self.settings = settings
        self.embedder = embedder or build_embedder(settings)
        self.model = model or build_chat_model(settings)
        self.tokenizer = make_tokenizer(settings.tokenizer)
        self.index, self.sync_report = sync_index(
            settings.docs_dir, settings.index_dir, self.embedder, settings.chunk_tokens,
            settings.chunk_overlap_tokens, self.tokenizer, force=force_reindex)
        if reranker is _FROM_SETTINGS:
            reranker = make_reranker(settings.reranker, settings.rerank_model)
        self.retriever = Retriever(self.index, self.embedder, reranker, candidate_k=settings.candidate_k)
        self.advisor = Advisor(self.retriever, self.model, settings.top_k, settings.min_relevance,
                               settings.prompt_version, settings.fallback_contact, settings.history_tokens)
        self.query_log = QueryLog(settings.query_log_path, settings.query_log, settings.log_retention_days)

    @property
    def report(self) -> SyncReport:
        return self.sync_report

    def ask(self, question: str, history: Sequence[tuple[str, str]] = (), mode: str | None = None) -> Answer:
        answer = self.advisor.answer(question, history, mode)
        self.query_log.record(answer)
        return answer

    def describe(self) -> dict:
        idx: Index = self.index
        return {
            "chunks": len(idx.chunks),
            "sources": len({c.source for c in idx.chunks}),
            "embedder": idx.embedder_name,
            "model": self.model.name,
            "reranker": getattr(self.retriever.reranker, "name", "none"),
            "prompt_version": self.settings.prompt_version,
            "indexed_at": idx.manifest.get("indexed_at", ""),
        }
