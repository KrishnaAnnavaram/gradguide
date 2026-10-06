"""Retrieve -> (abstain | prompt -> model -> citations)."""
from __future__ import annotations

from typing import Sequence

from gradguide.generate.citations import attach_citations
from gradguide.generate.prompts import build_messages
from gradguide.providers.llm import ChatModel
from gradguide.retrieve.hybrid import Retriever, best_relevance
from gradguide.types import Answer

FALLBACK = ("I couldn't find this in the documents I have. Please contact {contact} - "
            "they can give you an answer for your specific program.")


class Advisor:
    def __init__(self, retriever: Retriever, model: ChatModel, top_k: int = 4, min_relevance: float = 0.3,
                 prompt_version: str = "v1", fallback_contact: str = "", history_tokens: int = 400):
        self.retriever, self.model = retriever, model
        self.top_k, self.min_relevance = top_k, min_relevance
        self.prompt_version, self.fallback_contact = prompt_version, fallback_contact
        self.history_tokens = history_tokens

    def answer(self, question: str, history: Sequence[tuple[str, str]] = (), mode: str | None = None) -> Answer:
        mode = mode or self.retriever.default_mode
        hits = self.retriever.search(question, k=self.top_k, mode=mode)
        if not hits or best_relevance(question, hits) < self.min_relevance:
            text = FALLBACK.format(contact=self.fallback_contact or "your graduate advising office")
            return Answer(question, text, [], hits, True, self.model.name, self.prompt_version, mode)
        messages = build_messages(question, hits, self.prompt_version, self.fallback_contact, history,
                                  self.history_tokens)
        text, citations = attach_citations(self.model.chat(messages), hits)
        return Answer(question, text, citations, hits, False, self.model.name, self.prompt_version, mode)
