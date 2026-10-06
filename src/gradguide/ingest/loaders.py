"""Loaders that turn files into structured sections without altering their text.

Text is kept exactly as written (case, URLs, "$" amounts, phone numbers); only
runs of spaces are tidied. FAQ files become one section per question/answer
pair instead of a serialised JSON string, so they are retrievable like any other
content.

Supported: Markdown / text (``#`` headings), FAQ JSON / JSONL, HTML (stdlib
parser) and PDF (optional ``pdf`` extra).
"""
from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

from gradguide.types import Document, Section

SUFFIXES = {".md", ".markdown", ".txt", ".json", ".jsonl", ".html", ".htm", ".pdf"}
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_QUESTION_KEYS = ("question", "q", "title", "prompt")
_ANSWER_KEYS = ("answer", "a", "response", "body", "text")


def _tidy(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    lines = text.lstrip("﻿").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    meta: dict[str, str] = {}
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return meta, "\n".join(lines[i + 1:])
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip().lower()] = value.strip().strip("'\"")
    return {}, text


def markdown_sections(text: str, default_heading: str) -> list[Section]:
    """Split at headings; the heading path ("Degree > Thesis option") becomes the section heading."""
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    in_code = False

    def flush() -> None:
        body = _tidy("\n".join(buffer))
        if body:
            heading = " > ".join(t for _, t in stack) or default_heading
            sections.append(Section(heading, body))
        buffer.clear()

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        match = None if in_code else _HEADING.match(line)
        if match:
            flush()
            depth = len(match.group(1))
            while stack and stack[-1][0] >= depth:
                stack.pop()
            stack.append((depth, match.group(2).strip()))
        else:
            buffer.append(line)
    flush()
    return sections


def _pick(record: dict[str, Any], keys: Iterable[str]) -> str:
    lowered = {str(k).lower(): v for k, v in record.items()}
    for key in keys:
        value = lowered.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value:
            return "\n".join(str(v) for v in value)
    return ""


def _flatten(value: Any, label: str = "") -> list[str]:
    """Render arbitrary JSON as readable "key: value" lines (never as a JSON string)."""
    if isinstance(value, dict):
        return [line for key, item in value.items() for line in _flatten(item, f"{label} {key}".strip())]
    if isinstance(value, list):
        return [line for item in value for line in _flatten(item, label)]
    return [f"{label}: {value}" if label else str(value)]


def faq_sections(data: Any) -> tuple[str, list[Section]]:
    """Accept a list of Q/A records, or an object holding one under a common key."""
    title = ""
    records = data
    if isinstance(data, dict):
        title = str(data.get("title", "") or "")
        for key in ("faqs", "faq", "questions", "items", "entries"):
            if isinstance(data.get(key), list):
                records = data[key]
                break
        else:
            records = [data]
    sections: list[Section] = []
    for record in records if isinstance(records, list) else []:
        if not isinstance(record, dict):
            continue
        question, answer = _pick(record, _QUESTION_KEYS), _pick(record, _ANSWER_KEYS)
        if question and answer:
            sections.append(Section(_tidy(question), _tidy(answer), kind="faq"))
        elif record:
            body = _tidy("\n".join(_flatten(record)))
            if body:
                sections.append(Section(question or title or "Record", body))
    return title, sections


class _HTMLToSections(HTMLParser):
    SKIP = {"script", "style", "nav", "footer", "header", "noscript"}
    HEADINGS = {"h1", "h2", "h3", "h4"}
    BLOCKS = {"p", "li", "div", "tr", "br", "section", "article", "dd", "dt"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.lines: list[str] = []
        self._skip = 0
        self._in_heading: str | None = None
        self._in_title = False
        self._href: str | None = None
        self._current: list[str] = []

    def _end_line(self) -> None:
        text = " ".join("".join(self._current).split())
        if text:
            self.lines.append(text)
        self._current = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self.HEADINGS:
            self._end_line()
            self._in_heading = tag
        elif tag in self.BLOCKS:
            self._end_line()
            if tag == "li":
                self._current.append("- ")
        elif tag == "a":
            self._href = dict(attrs).get("href")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in self.HEADINGS and self._in_heading:
            text = " ".join("".join(self._current).split())
            self._current = []
            if text:
                self.lines.append("#" * int(tag[1]) + " " + text)
            self._in_heading = None
        elif tag in self.BLOCKS:
            self._end_line()
        elif tag == "a":
            if self._href and self._href.startswith(("http://", "https://", "mailto:")):
                self._current.append(f" ({self._href.removeprefix('mailto:')})")
            self._href = None

    def handle_data(self, data):
        if self._skip:
            return
        if self._in_title:
            self.title += data.strip()
            return
        self._current.append(data)

    def close(self):
        super().close()
        self._end_line()


def html_to_markdown(html: str) -> tuple[str, str]:
    parser = _HTMLToSections()
    parser.feed(html)
    parser.close()
    return parser.title, "\n\n".join(parser.lines)


def _read_pdf(path: Path) -> list[Section]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError("Reading PDFs needs the optional extra: pip install 'gradguide[pdf]'") from exc
    sections = []
    for number, page in enumerate(PdfReader(str(path)).pages, start=1):
        text = _tidy(page.extract_text() or "")
        if text:
            sections.append(Section(f"Page {number}", text))
    return sections


def load_file(path: Path, root: Path) -> Document:
    relative = path.relative_to(root).as_posix()
    suffix = path.suffix.lower()
    default_title = path.stem.replace("_", " ").replace("-", " ").strip().capitalize()
    meta: dict[str, Any] = {}
    title = ""
    if suffix in (".md", ".markdown", ".txt"):
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        title = meta.get("title", "")
        sections = markdown_sections(body, title or default_title)
    elif suffix == ".json":
        title, sections = faq_sections(json.loads(path.read_text(encoding="utf-8")))
    elif suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        title, sections = faq_sections(rows)
    elif suffix in (".html", ".htm"):
        title, body = html_to_markdown(path.read_text(encoding="utf-8"))
        sections = markdown_sections(body, title or default_title)
    elif suffix == ".pdf":
        sections = _read_pdf(path)
    else:
        raise ValueError(f"unsupported file type: {path.name}")
    return Document(relative, title or default_title, tuple(sections), meta)


def iter_paths(root: Path) -> list[Path]:
    return [p for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix.lower() in SUFFIXES and p.name.lower() != "readme.md"]


def load_folder(root: str | Path) -> list[Document]:
    folder = Path(root)
    if not folder.is_dir():
        raise FileNotFoundError(f"document folder not found: {folder}")
    docs = [load_file(p, folder) for p in iter_paths(folder)]
    return [d for d in docs if d.sections]
