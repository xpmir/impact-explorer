import math

import pytest

from impact_explorer.engine import AnalyzedQuery, Hit, SearchResult
from impact_explorer.evaluation import Label, evaluate


def result(*docids):
    hits = [Hit(rank=i + 1, docid=d, score=10.0 - i) for i, d in enumerate(docids)]
    return SearchResult(query=AnalyzedQuery("q", []), hits=hits)


QRELS = {"a": 1, "b": 0, "c": 2, "z": 1}


def test_labels_and_missed():
    evaluation = evaluate(result("a", "b", "x", "c"), QRELS, k=3)
    assert evaluation.labels == [Label.RELEVANT, Label.NON_RELEVANT, Label.UNJUDGED]
    # c is retrieved below the cutoff, z is not retrieved at all
    assert [(m.docid, m.rank) for m in evaluation.missed] == [("c", 4), ("z", None)]
    assert evaluation.missed[0].score == 7.0


def test_metrics():
    m = evaluate(result("a", "b", "x", "c"), QRELS, k=3).metrics
    assert m["P@3"] == pytest.approx(1 / 3)
    assert m["R@3"] == pytest.approx(1 / 3)
    assert m["judged@3"] == pytest.approx(2 / 3)
    assert m["R@depth"] == pytest.approx(2 / 3)
    idcg = 3 + 1 / math.log2(3) + 1 / math.log2(4)
    assert m["nDCG@3"] == pytest.approx(1 / idcg)


def test_no_qrels():
    evaluation = evaluate(result("a"), {}, k=10)
    assert evaluation.labels == [Label.UNJUDGED]
    assert evaluation.missed == []
    assert evaluation.metrics["nDCG@10"] == 0


def test_rank_metrics():
    m = evaluate(result("x", "a", "b", "c"), QRELS, k=3).metrics
    assert m["RR@3"] == pytest.approx(1 / 2)
    # Relevant at ranks 2 and 4, out of 3 relevant documents (z is missed)
    assert m["AP"] == pytest.approx((1 / 2 + 2 / 4) / 3)
    assert evaluate(result("x", "b"), QRELS, k=3).metrics["RR@3"] == 0
