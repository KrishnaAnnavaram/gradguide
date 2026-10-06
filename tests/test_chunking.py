import pytest

from gradguide.ingest.chunking import chunk_document, chunk_id, pack
from gradguide.ingest.tokens import SimpleTokenizer
from gradguide.types import Document, Section

TOK = SimpleTokenizer()


def test_chunk_size_is_measured_in_tokens():
    # problem 3: "token" limits were really character counts
    paragraph = "Students must file a degree plan with an advisor before the twelfth credit hour."
    text = "\n\n".join([paragraph] * 30)
    chunks = pack(text, max_tokens=60, overlap_tokens=10)
    assert len(chunks) > 1
    assert all(TOK.count(c) <= 60 for c in chunks)


def test_paragraph_boundaries_are_the_split_points():
    # problem 3: newlines were collapsed before splitting, so paragraph separators never fired
    paras = [f"Paragraph {i} " + "word " * 20 for i in range(6)]
    chunks = pack("\n\n".join(p.strip() for p in paras), max_tokens=50, overlap_tokens=0)
    for chunk in chunks:
        for part in chunk.split("\n\n"):
            assert part.startswith("Paragraph")  # never cut in the middle of a paragraph


def test_overlap_carries_trailing_words():
    paras = [f"Section text {i} " + " ".join(f"w{i}x{j}" for j in range(15)) for i in range(5)]
    chunks = pack("\n\n".join(paras), max_tokens=40, overlap_tokens=8)
    for previous, current in zip(chunks, chunks[1:]):
        carried = current.split("\n\n")[0]
        assert previous.rstrip().endswith(carried)


def test_giant_sentence_falls_back_to_words():
    chunks = pack("x " * 500, max_tokens=50, overlap_tokens=5)
    assert all(TOK.count(c) <= 50 for c in chunks)
    with pytest.raises(ValueError):
        pack("text", max_tokens=10, overlap_tokens=10)


def test_chunk_ids_are_content_hashes():
    # problem 7: ids were doc_0..N, regenerated on every run
    doc = Document("a.md", "A", (Section("Intro", "Some text."),))
    first = chunk_document(doc)[0]
    again = chunk_document(doc)[0]
    edited = chunk_document(Document("a.md", "A", (Section("Intro", "Some new text."),)))[0]
    assert first.chunk_id == again.chunk_id == chunk_id("a.md", "Intro", "Some text.")
    assert edited.chunk_id != first.chunk_id


def test_faq_pairs_stay_together():
    doc = Document("f.json", "FAQ", (Section("When is the deadline?", "March 15.", kind="faq"),))
    chunk = chunk_document(doc)[0]
    assert chunk.text == "Q: When is the deadline?\nA: March 15." and chunk.kind == "faq"
