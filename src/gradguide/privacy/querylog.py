"""Opt-in, redacted query log with a retention period.

Logging is off unless ``GRADGUIDE_QUERY_LOG=on``. When enabled, each record keeps
only what is needed to improve retrieval: a redacted question, whether the
assistant abstained, the cited chunk ids, the model and prompt version. Answers
and retrieved document text are never written, and personal identifiers in the
question (e-mail addresses, phone numbers, long digit runs such as student or
record numbers) are masked. Records older than the retention period are purged
on every write and by ``gradguide purge-logs``.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from gradguide.types import Answer

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
_LONG_NUMBER = re.compile(r"\b\d{5,}\b")


def redact(text: str) -> str:
    text = _EMAIL.sub("[email]", text)
    text = _PHONE.sub("[phone]", text)
    return _LONG_NUMBER.sub("[number]", text)


class QueryLog:
    def __init__(self, path: Path, enabled: bool = False, retention_days: int = 30):
        self.path, self.enabled, self.retention_days = Path(path), enabled, retention_days

    def record(self, answer: Answer, now: datetime | None = None) -> bool:
        if not self.enabled:
            return False
        now = now or datetime.now(timezone.utc)
        entry = {
            "ts": now.isoformat(timespec="seconds"),
            "question": redact(answer.question),
            "abstained": answer.abstained,
            "cited_chunks": [c.chunk_id for c in answer.citations],
            "model": answer.model,
            "prompt_version": answer.prompt_version,
            "mode": answer.retrieval_mode,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.purge(now)
        return True

    def purge(self, now: datetime | None = None) -> int:
        """Delete records older than the retention period; returns how many were removed."""
        if not self.path.is_file():
            return 0
        now = now or datetime.now(timezone.utc)
        cutoff = now - timedelta(days=self.retention_days)
        keep, dropped = [], 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                ts = datetime.fromisoformat(json.loads(line)["ts"])
            except (ValueError, KeyError, json.JSONDecodeError):
                dropped += 1
                continue
            if ts >= cutoff:
                keep.append(line)
            else:
                dropped += 1
        if dropped:
            self.path.write_text("".join(f"{line}\n" for line in keep), encoding="utf-8")
        return dropped
