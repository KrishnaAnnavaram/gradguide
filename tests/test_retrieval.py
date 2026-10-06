import pytest
from conftest import SAMPLE_DOCS

from gradguide.index.store import sync_index
from gradguide.providers.embeddings import HashingEmbedder
from gradguide.retrieve.bm25 import BM25
from gradguide.retrieve.fusion import rrf
from gradguide.retrieve.hybrid import Retriever, term_coverage
from gradguide.retrieve.rerank import LexicalReranker
from gradguide.retrieve.text import terms


def test_rrf_formula_and_order():
    fused = rrf({"bm25": ["a", "b", "c"], "vector": ["c", "d"]}, k=60)
    score = {item: s for item, s, _ in fused}
    assert score["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert fused[0][0] == "c"
    assert dict((i, r) for i, _, r in fused)["c"] == {"bm25": 3, "vector": 1}


def test_rrf_gives_single_list_items_a_fair_chance():
    fused = rrf({"bm25": ["a", "b", "c"], "vector": ["x", "y", "z"]})
    assert {item for item, _, _ in fused[:2]} == {"a", "x"}
    with pytest.raises(ValueError):
        rrf({"a": [1]}, k=0)


def test_terms_and_bm25():
    assert terms("Holds, holds and HOLDS!") == ["hold", "hold", "hold"]
    assert "u.s" not in terms("needs") and terms("e-mail") == ["e", "mail"]
    bm = BM25.from_texts(["registration holds block enrollment", "graduation fee is $75"])
    assert bm.score_all("what blocks registration")[0][0] == 0
    assert BM25.from_state(bm.state()).score_all("fee") == bm.score_all("fee")


@pytest.fixture(scope="module")
def retriever(tmp_path_factory):
    index, _ = sync_index(SAMPLE_DOCS, tmp_path_factory.mktemp("idx"), HashingEmbedder())
    return Retriever(index, HashingEmbedder(), LexicalReranker())


def test_all_modes_are_real_code_paths(retriever):
    # problem 1: the "Vector vs Rerank" comparison was random labels, not two systems
    q = "What balance triggers a financial hold?"
    results = {m: retriever.search(q, k=5, mode=m) for m in ("bm25", "vector", "hybrid", "hybrid+rerank")}
    assert all(results.values())
    assert all(set(h.ranks) == {"bm25"} for h in results["bm25"])
    assert all(set(h.ranks) == {"vector"} for h in results["vector"])
    assert any(len(h.ranks) == 2 for h in results["hybrid"])
    assert all(h.rerank_score is not None for h in results["hybrid+rerank"])
    scores = [h.rerank_score for h in results["hybrid+rerank"]]
    assert scores == sorted(scores, reverse=True)


def test_faq_answer_is_retrievable(retriever):
    hits = retriever.search("When is the FAFSA priority deadline?", k=3, mode="hybrid")
    assert hits[0].chunk.source == "faq_financial_aid.json"


def test_retriever_rejects_mismatched_embedder(retriever):
    with pytest.raises(ValueError, match="index built with"):
        Retriever(retriever.index, HashingEmbedder(dim=64))
    with pytest.raises(ValueError):
        retriever.search("x", mode="random")


def test_term_coverage():
    assert term_coverage("graduation fee", "The graduation fee is $75.") == 1.0
    assert term_coverage("swimming pool", "The graduation fee is $75.") == 0.0
