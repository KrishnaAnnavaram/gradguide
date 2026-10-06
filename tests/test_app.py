"""Headless smoke test of the Streamlit UI (skipped when the 'ui' extra is not installed)."""
from pathlib import Path

import pytest
from conftest import SAMPLE_DOCS

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "src" / "gradguide" / "app.py"


def test_ui_answers_with_sources_and_escapes_html(tmp_path, monkeypatch):
    monkeypatch.setenv("GRADGUIDE_HOME", str(tmp_path))
    monkeypatch.setenv("GRADGUIDE_DOCS_DIR", str(SAMPLE_DOCS))
    monkeypatch.delenv("GRADGUIDE_API_URL", raising=False)
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    assert not at.exception
    at.chat_input[0].set_value("When is the <b>FAFSA</b> priority deadline?").run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "<b>" not in text and "&lt;b&gt;" in text
    assert "March 15" in text
    assert any("echo:extractive" in c.value for c in at.caption)
