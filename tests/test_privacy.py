"""Problem 10: every question, answer and retrieved chunk was logged forever."""
import json
from datetime import date, datetime, timedelta, timezone

from conftest import SAMPLE_DOCS, RecordingModel, make_settings

from gradguide.ingest.sources import load_manifest, stale_sources
from gradguide.privacy.querylog import QueryLog, redact
from gradguide.privacy.ratelimit import RateLimiter
from gradguide.service import GradGuide
from gradguide.types import Answer, Citation


def _answer(question="How do I appeal? Mail me at jane.doe@mail.example.org or 555-010-0199, id 12345678"):
    return Answer(question, "secret answer text", [Citation(1, "c1", "a.md", "H", "snippet")], [], False,
                  "fake:model", "v1", "hybrid")


def test_logging_is_off_by_default(settings, embedder):
    assert settings.query_log is False
    svc = GradGuide(settings, embedder=embedder, model=RecordingModel())
    svc.ask("When will my diploma be mailed?")
    assert not settings.query_log_path.exists()


def test_enabled_log_is_redacted_and_minimal(tmp_path):
    log = QueryLog(tmp_path / "log.jsonl", enabled=True)
    assert log.record(_answer())
    entry = json.loads((tmp_path / "log.jsonl").read_text(encoding="utf-8"))
    assert "jane.doe" not in entry["question"] and "[email]" in entry["question"]
    assert "0199" not in entry["question"] and "12345678" not in entry["question"]
    assert "answer" not in entry and "secret answer text" not in json.dumps(entry)
    assert entry["cited_chunks"] == ["c1"] and entry["model"] == "fake:model"


def test_retention_purges_old_records(tmp_path):
    log = QueryLog(tmp_path / "log.jsonl", enabled=True, retention_days=30)
    now = datetime(2026, 9, 1, tzinfo=timezone.utc)
    log.record(_answer("old question"), now=now - timedelta(days=45))
    log.record(_answer("new question"), now=now)
    lines = (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and "new question" in lines[0]


def test_service_logs_when_opted_in(tmp_path, embedder):
    settings = make_settings(tmp_path, GRADGUIDE_QUERY_LOG="on")
    GradGuide(settings, embedder=embedder, model=RecordingModel()).ask("When will my diploma be mailed?")
    assert len(settings.query_log_path.read_text(encoding="utf-8").splitlines()) == 1


def test_redact_keeps_ordinary_numbers():
    assert redact("Is the fee $75 for 9 credit hours?") == "Is the fee $75 for 9 credit hours?"


def test_rate_limiter_window():
    now = [0.0]
    limiter = RateLimiter(per_minute=2, clock=lambda: now[0])
    assert limiter.allow("a") and limiter.allow("a") and not limiter.allow("a")
    assert limiter.allow("b")
    now[0] = 61.0
    assert limiter.allow("a")
    assert RateLimiter(per_minute=0).allow("anyone")


def test_source_manifest_flags_stale_documents():
    manifest = load_manifest(SAMPLE_DOCS)
    assert manifest["registration.md"].owner == "Registrar"
    stale = {s.path for s in stale_sources(manifest, today=date(2026, 9, 1))}
    assert stale == {"international_students.md"}
