import json

from conftest import SAMPLE_DOCS

from gradguide.ingest.chunking import chunk_corpus
from gradguide.ingest.loaders import faq_sections, html_to_markdown, load_file, load_folder, markdown_sections


def test_faq_json_becomes_question_answer_sections(tmp_path):
    # problem 2: FAQ JSON was json.dumps-ed, then every chunk with a brace was filtered out
    doc = load_file(SAMPLE_DOCS / "faq_financial_aid.json", SAMPLE_DOCS)
    assert doc.title == "Financial Aid FAQ"
    assert len(doc.sections) == 6 and all(s.kind == "faq" for s in doc.sections)
    chunks = chunk_corpus([doc])
    assert chunks and all("{" not in c.text and "}" not in c.text for c in chunks)
    assert chunks[0].text.startswith("Q: How many credit hours")


def test_faq_jsonl_and_alternative_keys(tmp_path):
    path = tmp_path / "faq.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [{"Q": "Where is the office?", "A": "Room 101."},
                                                      {"prompt": "Hours?", "response": "9 to 5."}]),
                    encoding="utf-8")
    doc = load_file(path, tmp_path)
    assert [(s.heading, s.text) for s in doc.sections] == [("Where is the office?", "Room 101."), ("Hours?", "9 to 5.")]


def test_non_faq_json_is_flattened_without_braces():
    _, sections = faq_sections({"office": {"name": "Records", "hours": ["Mon", "Tue"]}})
    assert len(sections) == 1
    body = sections[0].text
    assert "{" not in body and "office name: Records" in body and "office hours: Mon" in body


def test_text_is_preserved_exactly():
    # problem 8: the local pipeline lower-cased text and stripped "/", "$" and "()"
    chunks = chunk_corpus(load_folder(SAMPLE_DOCS))
    corpus = "\n".join(c.text for c in chunks)
    for needle in ("https://advising.example.edu/schedule", "$200", "(555) 010-0142", "finaid@example.edu",
                   "F-1", "I-20", "Curricular Practical Training (CPT)"):
        assert needle in corpus, needle


def test_html_keeps_headings_and_links_and_drops_boilerplate():
    title, body = html_to_markdown((SAMPLE_DOCS / "assistantships.html").read_text(encoding="utf-8"))
    assert title == "Graduate Assistantships"
    assert "## Tuition Benefits" in body
    assert "https://jobs.example.edu/graduate" in body and "gradassist@example.edu" in body
    for junk in ("console.log", "should not be indexed", "Home"):
        assert junk not in body


def test_markdown_sections_have_heading_paths_and_keep_lists():
    sections = markdown_sections("# Guide\n\n## Holds\n\n- Advising hold\n- Financial hold\n\n## Fees\n\nPay $75.", "x")
    assert [s.heading for s in sections] == ["Guide > Holds", "Guide > Fees"]
    assert sections[0].text == "- Advising hold\n- Financial hold"


def test_every_sample_format_loads():
    sources = {d.source for d in load_folder(SAMPLE_DOCS)}
    assert {"faq_financial_aid.json", "assistantships.html", "advising_contacts.txt", "registration.md"} <= sources
    assert "README.md" not in sources
