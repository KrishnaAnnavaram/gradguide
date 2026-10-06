"""Choose providers from Settings."""
from __future__ import annotations

from gradguide.config import Settings
from gradguide.providers.embeddings import Embedder, HashingEmbedder, OllamaEmbedder, OpenAICompatibleEmbedder
from gradguide.providers.http import ProviderError
from gradguide.providers.llm import ChatModel, EchoModel, OllamaChat, OpenAICompatibleChat


def build_embedder(settings: Settings) -> Embedder:
    match settings.embed_provider:
        case "hashing" | "offline":
            return HashingEmbedder()
        case "openai":
            return OpenAICompatibleEmbedder(settings.embed_model, settings.openai_base_url, settings.openai_api_key)
        case "ollama":
            return OllamaEmbedder(settings.embed_model, settings.ollama_host)
    raise ProviderError(f"unknown embedding provider {settings.embed_provider!r}")


def build_chat_model(settings: Settings) -> ChatModel:
    match settings.llm_provider:
        case "echo" | "offline":
            return EchoModel()
        case "openai":
            return OpenAICompatibleChat(settings.llm_model, settings.openai_base_url, settings.openai_api_key,
                                        settings.temperature)
        case "ollama":
            return OllamaChat(settings.llm_model, settings.ollama_host, settings.temperature)
    raise ProviderError(f"unknown LLM provider {settings.llm_provider!r}")


__all__ = ["ChatModel", "EchoModel", "Embedder", "HashingEmbedder", "ProviderError", "build_chat_model",
           "build_embedder"]
