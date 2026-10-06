"""Streamlit UI. It holds no RAG logic: it calls the HTTP API when
``GRADGUIDE_API_URL`` is set, otherwise the same service in-process.

Run with ``gradguide ui``.
"""
from __future__ import annotations

import json
import urllib.request

import streamlit as st

from gradguide.config import Settings
from gradguide.render import safe


@st.cache_resource(show_spinner="Loading the index...")
def local_service():
    from gradguide.service import GradGuide

    return GradGuide(Settings.from_env())


def ask(settings: Settings, question: str, history: list[tuple[str, str]]) -> dict:
    if settings.api_url:
        body = json.dumps({"question": question, "history": history}).encode("utf-8")
        req = urllib.request.Request(f"{settings.api_url}/ask", data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    return local_service().ask(question, history).to_dict()


def main() -> None:
    settings = Settings.from_env()
    st.set_page_config(page_title="gradguide", layout="centered")
    st.title("gradguide")
    st.caption("Answers come only from the documents loaded into this assistant, with sources. "
               "Always confirm important decisions with your advisor.")
    if settings.query_log:
        st.info(f"Questions are logged in redacted form for {settings.log_retention_days} days "
                "to improve the assistant. Answers are not stored.")

    turns = st.session_state.setdefault("turns", [])
    question = st.chat_input("Ask about requirements, registration, graduation...")
    if question:
        history = [(t["question"], t["answer"]) for t in turns][-5:]
        with st.spinner("Searching the documents..."):
            turns.append(ask(settings, question, history))

    for turn in turns:
        with st.chat_message("user"):
            st.markdown(safe(turn["question"]))
        with st.chat_message("assistant"):
            st.markdown(safe(turn["answer"]))
            if turn["citations"]:
                with st.expander("Sources"):
                    for c in turn["citations"]:
                        st.markdown(f"**[{c['marker']}]** {safe(c['source'])} - {safe(c['heading'])}")
                        st.text(c["snippet"])
            st.caption(f"model: {safe(turn['model'])} | prompt {safe(turn['prompt_version'])} | "
                       f"retrieval: {safe(turn['retrieval_mode'])}")


main()
