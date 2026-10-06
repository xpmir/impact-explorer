"""Relevance labels and metrics for a ranked list."""

import math
from dataclasses import dataclass
from enum import StrEnum

from .engine import Hit, SearchResult


class Label(StrEnum):
    RELEVANT = "relevant"
    NON_RELEVANT = "non-relevant"
    UNJUDGED = "unjudged"


def label(qrels: dict[str, float], docid: str) -> Label:
    if docid not in qrels:
        return Label.UNJUDGED
    return Label.RELEVANT if qrels[docid] > 0 else Label.NON_RELEVANT


@dataclass
class Missed:
    docid: str
    relevance: float
    rank: int | None
    """Rank within the search depth (None if not retrieved at all)"""

    score: float | None


@dataclass
class Evaluation:
    k: int
    labels: list[Label]
    """Labels of the top-k hits"""

    missed: list[Missed]
    """Relevant documents not in the top-k"""

    metrics: dict[str, float]


def evaluate(result: SearchResult, qrels: dict[str, float], k: int) -> Evaluation:
    top = result.top(k)
    labels = [label(qrels, hit.docid) for hit in top]
    relevant = {docid: rel for docid, rel in qrels.items() if rel > 0}
    in_top = {hit.docid for hit in top}
    by_docid: dict[str, Hit] = {hit.docid: hit for hit in result.hits}

    missed = []
    for docid, rel in relevant.items():
        if docid in in_top:
            continue
        hit = by_docid.get(docid)
        missed.append(
            Missed(
                docid=docid,
                relevance=rel,
                rank=hit.rank if hit else None,
                score=hit.score if hit else None,
            )
        )
    missed.sort(key=lambda m: (m.rank is None, m.rank or 0, -m.relevance, m.docid))

    found = sum(1 for lbl in labels if lbl is Label.RELEVANT)
    dcg = sum(
        (2 ** qrels[hit.docid] - 1) / math.log2(rank + 2)
        for rank, hit in enumerate(top)
        if qrels.get(hit.docid, 0) > 0
    )
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum((2**rel - 1) / math.log2(rank + 2) for rank, rel in enumerate(ideal))

    first = next((i for i, lbl in enumerate(labels) if lbl is Label.RELEVANT), None)
    # Average precision over the whole search depth
    precisions, hits = [], 0
    for rank, hit in enumerate(result.hits, start=1):
        if qrels.get(hit.docid, 0) > 0:
            hits += 1
            precisions.append(hits / rank)
    ap = sum(precisions) / len(relevant) if relevant else 0.0

    metrics = {
        f"nDCG@{k}": dcg / idcg if idcg else 0.0,
        f"RR@{k}": 1 / (first + 1) if first is not None else 0.0,
        "AP": ap,
        f"P@{k}": found / k if k else 0.0,
        f"R@{k}": found / len(relevant) if relevant else 0.0,
        f"judged@{k}": (
            sum(1 for lbl in labels if lbl is not Label.UNJUDGED) / len(labels)
            if labels
            else 0.0
        ),
        "R@depth": (
            sum(1 for d in relevant if d in by_docid) / len(relevant)
            if relevant
            else 0.0
        ),
    }
    return Evaluation(k=k, labels=labels, missed=missed, metrics=metrics)
