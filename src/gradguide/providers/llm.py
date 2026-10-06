"""Chat models: one interface, an offline fake and two API-backed implementations.

Every implementation exposes ``name`` (provider:model), which is stored on each
answer, so evaluation results always say which model produced them.
"""
from __future__ import annotations

import re
from typing import Protocol

from gradguide.providers.http import ProviderError, post_json
from gradguide.retrieve.text import terms

Message = dict[str, str]

_PASSAGE = re.compile(r'<passage id="(\d+)"[^>]*>\n?(.*?)</passage>', re.S)


class ChatModel(Protocol):
    name: str

    def chat(self, messages: list[Message]) -> str:
        ...


class EchoModel:
    """Offline extractive stand-in for an LLM.

    It reads the passages from the final user message and, for the first few,
    quotes the sentence sharing the most words with the question, citing it.
    """

    name = "echo:extractive"

    def __init__(self, passages: int = 2):
        self.passages = passages
        self.history: list[list[Message]] = []

    def chat(self, messages: list[Message]) -> str:
        self.history.append(messages)
        content = messages[-1]["content"]
        match = re.search(r"<question>(.*?)</question>", content, re.S)
        wanted = set(terms(match.group(1) if match else ""))
        quotes = []
        for number, body in _PASSAGE.findall(content)[: self.passages]:
            # FAQ chunks look like "Q: ...\nA: ..."; quote from the answer, not the question
            body = re.sub(r"^Q:.*\nA:\s*", "", body.strip())
            sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", body) if s.strip()]
            if not sentences:
                continue
            best = max(sentences, key=lambda s: len(wanted & set(terms(s))))
            quotes.append(f"{best.strip()} [{number}]")
        if not quotes:
            return "The provided documents do not answer this question."
        return " ".join(quotes)


class OpenAICompatibleChat:
    def __init__(self, model: str, base_url: str, api_key: str, temperature: float = 0.1):
        if not api_key:
            raise ProviderError("OPENAI_API_KEY is required for the openai LLM provider")
        self.model, self.base_url, self.api_key, self.temperature = model, base_url, api_key, temperature
        self.name = f"openai:{model}"

    def chat(self, messages: list[Message]) -> str:
        reply = post_json(f"{self.base_url}/chat/completions",
                          {"model": self.model, "messages": messages, "temperature": self.temperature},
                          api_key=self.api_key)
        return (reply["choices"][0]["message"].get("content") or "").strip()


class OllamaChat:
    def __init__(self, model: str, host: str, temperature: float = 0.1):
        self.model, self.host, self.temperature = model, host, temperature
        self.name = f"ollama:{model}"

    def chat(self, messages: list[Message]) -> str:
        reply = post_json(f"{self.host}/api/chat", {"model": self.model, "messages": messages, "stream": False,
                                                     "options": {"temperature": self.temperature}})
        return (reply.get("message", {}).get("content") or "").strip()
