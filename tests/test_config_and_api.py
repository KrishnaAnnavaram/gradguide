import numpy as np
import pytest
from conftest import RecordingModel

from gradguide.config import Settings, load_dotenv
from gradguide.providers import EchoModel, HashingEmbedder, ProviderError, build_chat_model, build_embedder
from gradguide.service import GradGuide


def test_offline_defaults(tmp_path):
    s = Settings.from_env(env={"GRADGUIDE_HOME": str(tmp_path)})
    assert isinstance(build_chat_model(s), EchoModel) and isinstance(build_embedder(s), HashingEmbedder)
    assert s.docs_dir == (tmp_path / "sample_docs").resolve()
    assert s.query_log is False and s.reranker == "none"


def test_openai_requires_key_and_unknown_provider_fails(tmp_path):
    base = {"GRADGUIDE_HOME": str(tmp_path)}
    with pytest.raises(ProviderError):
        build_chat_model(Settings.from_env(env={**base, "GRADGUIDE_LLM_PROVIDER": "openai"}))
    with pytest.raises(ProviderError):
        build_embedder(Settings.from_env(env={**base, "GRADGUIDE_EMBED_PROVIDER": "openai"}))
    with pytest.raises(ProviderError):
        build_chat_model(Settings.from_env(env={**base, "GRADGUIDE_LLM_PROVIDER": "nope"}))


def test_key_hidden_from_repr(tmp_path):
    s = Settings.from_env(env={"GRADGUIDE_HOME": str(tmp_path), "OPENAI_API_KEY": "placeholder-value"})
    assert "placeholder-value" not in repr(s)


def test_dotenv_does_not_override(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("GRADGUIDE_T1=file\nGRADGUIDE_T2=file\n", encoding="utf-8")
    monkeypatch.setenv("GRADGUIDE_T1", "shell")
    monkeypatch.delenv("GRADGUIDE_T2", raising=False)
    assert load_dotenv(tmp_path / ".env") == {"GRADGUIDE_T2": "file"}
    monkeypatch.delenv("GRADGUIDE_T2")


def test_hashing_embedder_deterministic_unit_vectors():
    a = HashingEmbedder().embed(["optional practical training", "graduation fee"])
    assert np.array_equal(a, HashingEmbedder().embed(["optional practical training", "graduation fee"]))
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert HashingEmbedder().embed([]).shape == (0, 384)


def test_api_ask_health_and_rate_limit(settings, embedder):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from gradguide.api import create_app

    settings.rate_limit_per_minute = 2
    client = TestClient(create_app(GradGuide(settings, embedder=embedder, model=RecordingModel("Nine [1]."))))
    assert client.get("/health").json()["model"] == "fake:recording"
    reply = client.post("/ask", json={"question": "How many credit hours is full-time?"})
    assert reply.status_code == 200
    body = reply.json()
    assert body["answer"] == "Nine [1]." and body["citations"][0]["marker"] == 1
    assert client.post("/ask", json={"question": ""}).status_code == 422  # rejected before counting
    assert client.post("/ask", json={"question": "Full-time hours?", "mode": "bogus"}).status_code == 422
    assert client.post("/ask", json={"question": "One more?"}).status_code == 429  # limit of 2 per minute
