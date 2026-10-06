"""Optional source manifest (``sources.toml`` in the documents folder).

Each entry records who owns a document and how often it must be re-checked, so
stale advising content can be found before it misleads anyone::

    [[source]]
    path = "registration.md"
    title = "Registration and Enrollment"
    owner = "Graduate advising office"
    refresh_days = 180
    updated = "2026-08-15"
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

MANIFEST_NAME = "sources.toml"


@dataclass(frozen=True)
class SourceInfo:
    path: str
    title: str = ""
    owner: str = ""
    refresh_days: int = 0
    updated: str = ""

    def age_days(self, today: date) -> int | None:
        if not self.updated:
            return None
        return (today - date.fromisoformat(self.updated)).days

    def is_stale(self, today: date) -> bool:
        age = self.age_days(today)
        if self.refresh_days <= 0:
            return False
        return age is None or age > self.refresh_days


def load_manifest(docs_dir: str | Path) -> dict[str, SourceInfo]:
    path = Path(docs_dir) / MANIFEST_NAME
    if not path.is_file():
        return {}
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    out: dict[str, SourceInfo] = {}
    for entry in data.get("source", []):
        info = SourceInfo(
            path=str(entry["path"]),
            title=str(entry.get("title", "")),
            owner=str(entry.get("owner", "")),
            refresh_days=int(entry.get("refresh_days", 0)),
            updated=str(entry.get("updated", "")),
        )
        out[info.path] = info
    return out


def stale_sources(manifest: dict[str, SourceInfo], today: date | None = None) -> list[SourceInfo]:
    today = today or date.today()
    return [info for info in manifest.values() if info.is_stale(today)]
