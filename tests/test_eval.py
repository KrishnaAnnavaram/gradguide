"""Problems 1, 4 and 12: real ablations, honestly named metrics, retrieval metrics on a gold set."""
import pytest
from conftest import GOLD

from gradguide.eval.harness import answer_report, chunk_size_ablation, load_gold, retrieval_report, satisfied
from gradguide.eval.metrics import abstention_scores, mrr, recall_at_k
from gradguide.retrieve.hybrid import Retriever
from gradguide.retrieve.rerank import LexicalReranker
from gradguide.service import GradGuide


def test_recall_and_mrr():
    matches = [set(), {1}, {0}]
    assert recall_at_k(matches, 2, 1) == 0.0
    assert recall_at_k(matches, 2, 3) == 1.0
    assert mrr(matches) == 0.5 and mrr([set()]) == 0.0
    with pytest.raises(ValueError):
        recall_at_k(matches, 0, 1)


def test_abstention_scores():
    scores = abstention_scores([True, True, False, False], [False, True, True, False])
    assert scores == {"decision_accuracy": 0.5, "false_abstention_rate": 0.5, "missed_abstention_rate": 0.5}


def test_gold_set_targets_exist(settings, embedder):
    gold = load_gold(GOLD)
    assert len(gold) >= 30 and sum(not g.answerable for g in gold) >= 5
    svc = GradGuide(settings, embedder=embedder)
    for item in (g for g in gold if g.answerable):
        assert any(satisfied(c, item.relevant) for c in svc.index.chunks), item.qid


def test_reports_are_deterministic_and_cover_every_mode(settings, embedder):
    svc = GradGuide(settings, embedder=embedder)
    retriever = Retriever(svc.index, svc.embedder, LexicalReranker())
    gold = load_gold(GOLD)
    first = retrieval_report(retriever, gold)
    assert first == retrieval_report(retriever, gold)
    assert set(first) == {"bm25", "vector", "hybrid", "hybrid+rerank"}
    assert first["hybrid"]["recall@3"] >= 0.9 and first["hybrid"]["mrr"] >= 0.8


def test_answer_report_on_sample_set(settings, embedder):
    scores = answer_report(GradGuide(settings, embedder=embedder), load_gold(GOLD))
    assert set(scores) == {"decision_accuracy", "false_abstention_rate", "missed_abstention_rate",
                           "gold_source_cited"}
    assert scores["false_abstention_rate"] <= 0.1 and scores["gold_source_cited"] >= 0.8


def test_chunk_size_ablation_rebuilds_indexes(settings, embedder):
    results = chunk_size_ablation(settings, load_gold(GOLD), [40, 220], embedder=embedder)
    assert results[40]["chunks"] > results[220]["chunks"]
