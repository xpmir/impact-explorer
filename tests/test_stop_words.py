import pytest

from impact_explorer.engine import SearchEngine
from impact_explorer.query_syntax import Resolution, to_query_tree


def test_keep_stop_words(unfiltered):
    engine = SearchEngine.open(unfiltered)
    default = engine.analyze("the lazy dog")
    kept = engine.analyze("the lazy dog", keep_stop_words=True)
    assert [t.words for t in default.terms] == [["lazy"], ["dog"]]
    assert [t.words for t in kept.terms] == [["the"], ["lazy"], ["dog"]]
    assert kept.terms[0].df > 0

    words = engine.plain_words("the dog", keep_stop_words=True)
    assert [w.resolution for w in words] == [Resolution.TERM, Resolution.TERM]
    assert engine.plain_words("the dog")[0].resolution is Resolution.STOPWORD

    # Highlighting follows the query
    assert engine.matcher_for(True).terms("the")
    assert not engine.matcher_for(False).terms("the")


def test_phrases_with_stop_words(unfiltered):
    engine = SearchEngine.open(unfiltered)
    # "Foxes are quick": documents keep "are", so without it "foxes quick"
    # are not adjacent
    assert not engine.search("#1(foxes are quick)").hits
    kept = engine.search("#1(foxes are quick)", keep_stop_words=True)
    assert [h.docid for h in kept.hits] == ["d2"]


@pytest.mark.parametrize(
    "query",
    [
        "#1(lazy dog) fox",
        "#combine:0=2:2=0.5(#syn(cat fox) quick the #1(lazy dog))",
        "#band(fox #syn(cat dog)) quick",
        "#uw8(fox dog) cats",
        "#band(fox zzzunknown) dog",
        "#combine(#1(lazy zzq) fox)",
    ],
)
def test_query_tree_matches_impact_index(collection, query):
    """The tree resolved here gives the same results as impact-index's own
    parsing of the string"""
    engine = SearchEngine.open(collection)
    scored = engine.scored(collection.bm25)
    reference = scored.search_maxscore_query(query, 100)
    tree = scored.search_maxscore_query(to_query_tree(engine.parse(query)), 100)
    assert [(r.docid, r.score) for r in tree] == [(r.docid, r.score) for r in reference]


def test_empty_tree(collection):
    engine = SearchEngine.open(collection)
    assert to_query_tree(engine.parse("#band(zzz fox) the")) is None
    assert engine.search("#band(zzz fox)", keep_stop_words=True).hits == []
