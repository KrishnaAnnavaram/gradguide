"""Text rendering for the terminal and the UI (HTML-escaped)."""
from __future__ import annotations

import html

from gradguide.generate.citations import sources_block
from gradguide.types import Answer


def plain(answer: Answer) -> str:
    block = sources_block(answer.citations)
    return f"{answer.text}\n\n{block}" if block else answer.text


def safe(text: str) -> str:
    return html.escape(text, quote=True)
