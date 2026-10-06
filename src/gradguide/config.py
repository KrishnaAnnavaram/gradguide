"""Settings from environment variables (plus an optional .env file).

Every relative path is resolved once, against ``GRADGUIDE_HOME`` (default: the
directory the process started in), and stored as an absolute path. Nothing later
depends on the current working directory, so the CLI, the API and the UI find
the same index wherever they are launched from.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

PREFIX = "GRADGUIDE_"


def load_dotenv(path: str | os.PathLike = ".env") -> dict[str, str]:
    """Read KEY=VALUE lines into os.environ without overriding variables that are already set."""
    loaded: dict[str, str] = {}
    env_path = Path(path)
    if not env_path.is_file():
        return loaded
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


class _Env:
    def __init__(self, env: Mapping[str, str]):
        self.env = env

    def str(self, name: str, default: str) -> str:
        return (self.env.get(name) or "").strip() or default

    def int(self, name: str, default: int) -> int:
        raw = (self.env.get(name) or "").strip()
        return int(raw) if raw else default

    def float(self, name: str, default: float) -> float:
        raw = (self.env.get(name) or "").strip()
        return float(raw) if raw else default

    def bool(self, name: str, default: bool) -> bool:
        raw = (self.env.get(name) or "").strip().lower()
        if not raw:
            return default
        return raw in ("1", "true", "yes", "on")


@dataclass
class Settings:
    home: Path
    docs_dir: Path
    index_dir: Path

    llm_provider: str = "echo"            # echo | openai | ollama
    llm_model: str = "gpt-4o-mini"
    embed_provider: str = "hashing"       # hashing | openai | ollama
    embed_model: str = "text-embedding-3-small"
    reranker: str = "none"                # none | lexical | cross-encoder
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    tokenizer: str = "simple"             # simple | tiktoken
    temperature: float = 0.1

    chunk_tokens: int = 220
    chunk_overlap_tokens: int = 30
    top_k: int = 4
    candidate_k: int = 20
    min_relevance: float = 0.3
    history_tokens: int = 400
    prompt_version: str = "v1"
    fallback_contact: str = "your program's graduate advising office"

    query_log: bool = False
    log_retention_days: int = 30
    rate_limit_per_minute: int = 30
    api_url: str = ""

    openai_api_key: str = field(default="", repr=False)
    openai_base_url: str = "https://api.openai.com/v1"
    ollama_host: str = "http://localhost:11434"

    @property
    def query_log_path(self) -> Path:
        return self.index_dir / "query_log.jsonl"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, dotenv: bool = True) -> "Settings":
        if env is None:
            if dotenv:
                load_dotenv()
            env = os.environ
        e, p = _Env(env), PREFIX
        home = Path(e.str(p + "HOME", str(Path.cwd()))).expanduser().resolve()

        def path(name: str, default: str) -> Path:
            value = Path(e.str(p + name, default)).expanduser()
            return value if value.is_absolute() else (home / value).resolve()

        return cls(
            home=home,
            docs_dir=path("DOCS_DIR", "sample_docs"),
            index_dir=path("INDEX_DIR", ".gradguide"),
            llm_provider=e.str(p + "LLM_PROVIDER", "echo").lower(),
            llm_model=e.str(p + "LLM_MODEL", cls.llm_model),
            embed_provider=e.str(p + "EMBED_PROVIDER", "hashing").lower(),
            embed_model=e.str(p + "EMBED_MODEL", cls.embed_model),
            reranker=e.str(p + "RERANKER", cls.reranker).lower(),
            rerank_model=e.str(p + "RERANK_MODEL", cls.rerank_model),
            tokenizer=e.str(p + "TOKENIZER", cls.tokenizer).lower(),
            temperature=e.float(p + "TEMPERATURE", cls.temperature),
            chunk_tokens=e.int(p + "CHUNK_TOKENS", cls.chunk_tokens),
            chunk_overlap_tokens=e.int(p + "CHUNK_OVERLAP_TOKENS", cls.chunk_overlap_tokens),
            top_k=e.int(p + "TOP_K", cls.top_k),
            candidate_k=e.int(p + "CANDIDATE_K", cls.candidate_k),
            min_relevance=e.float(p + "MIN_RELEVANCE", cls.min_relevance),
            history_tokens=e.int(p + "HISTORY_TOKENS", cls.history_tokens),
            prompt_version=e.str(p + "PROMPT_VERSION", cls.prompt_version),
            fallback_contact=e.str(p + "FALLBACK_CONTACT", cls.fallback_contact),
            query_log=e.bool(p + "QUERY_LOG", False),
            log_retention_days=e.int(p + "LOG_RETENTION_DAYS", cls.log_retention_days),
            rate_limit_per_minute=e.int(p + "RATE_LIMIT_PER_MINUTE", cls.rate_limit_per_minute),
            api_url=e.str(p + "API_URL", "").rstrip("/"),
            openai_api_key=(env.get("OPENAI_API_KEY") or "").strip(),
            openai_base_url=e.str("OPENAI_BASE_URL", cls.openai_base_url).rstrip("/"),
            ollama_host=e.str("OLLAMA_HOST", cls.ollama_host).rstrip("/"),
        )
