import pytest

from impact_explorer.engine import SearchEngine
from impact_explorer.highlight import error_html, query_html
from impact_explorer.query_syntax import (
    Operator,
    QuerySyntaxError,
    Resolution,
    Word,
    is_structured,
    make_resolver,
    parse,
)

VOCAB = {"quick": 1, "fox": 2, "lazy": 3, "dog": 4}
resolve = make_resolver(
    lambda w: (VOCAB[w.lower()],) if w.lower() in VOCAB else (), {"the", "of"}
)


def test_is_structured():
    assert is_structured("#syn(a b)")
    assert is_structured("fox #1(lazy dog)")
    assert not is_structured("quick fox (lazy)")


def test_words_and_resolution():
    parsed = parse("Quick the zzz fox", resolve)
    assert [(w.text, w.resolution, w.term_id) for w in parsed.nodes] == [
        ("Quick", Resolution.TERM, 1),
        ("the", Resolution.STOPWORD, None),
        ("zzz", Resolution.UNKNOWN, None),
        ("fox", Resolution.TERM, 2),
    ]
    assert parsed.term_ids() == [1, 2]
    assert parsed.nodes[2].start == 10 and parsed.nodes[2].end == 13


def test_operators():
    parsed = parse(
        "#combine:0=2(fox #band(lazy #syn(dog zzz))) #uw8(quick fox)", resolve
    )
    combine, window = parsed.nodes
    assert isinstance(combine, Operator) and combine.weights == {0: 2.0}
    assert combine.label == "#combine:0=2"
    band = combine.children[1]
    assert band.name == "band" and band.dropped is None
    # #syn drops unknown words but still matches
    assert band.children[1].dropped is None
    assert window.width == 8 and not window.dropped
    assert parsed.uses_positions()
    assert parsed.term_ids() == [2, 3, 4, 1]


@pytest.mark.parametrize(
    "query, reason",
    [
        ("#1(lazy zzz)", "unknown word(s): zzz"),
        ("#1(the of)", "no indexed word"),
        ("#band(fox zzz)", "a child can never match: zzz"),
        ("#band(fox #1(zzz dog))", "a child can never match: #1"),
        ("#combine(zzz)", "no child can match"),
    ],
)
def test_dropped(query, reason):
    assert parse(query, resolve).nodes[0].dropped == reason


def test_stop_words_do_not_drop_phrases():
    phrase = parse("#1(lazy of dog)", resolve).nodes[0]
    assert phrase.dropped is None


@pytest.mark.parametrize(
    "query, message, span",
    [
        ("#syn(fox", "expected ')', found end of input", (0, 4)),
        ("fox)", "unexpected trailing token ')'", (3, 4)),
        ("#foo(x)", "unknown operator '#foo'", (0, 4)),
        ("#syn fox", "expected '(', found 'fox'", (5, 8)),
        ("#uw1(a b)", "window width must be >= 2", (0, 4)),
        ("#syn(#1(a b))", "operator '#1' not allowed inside #syn", (5, 7)),
        ("#combine:x=1(fox)", "bad combine index 'x'", (0, 12)),
        ("#combine:0=a(fox)", "bad combine weight 'a'", (0, 12)),
        ("#combine:0(fox)", "malformed combine weight spec '0'", (0, 10)),
    ],
)
def test_errors(query, message, span):
    with pytest.raises(QuerySyntaxError) as e:
        parse(query, resolve)
    assert e.value.message.startswith(message)
    assert (e.value.start, e.value.end) == span


def test_warnings():
    parsed = parse("#combine:3=2(fox) #1:x(lazy dog)", resolve)
    assert len(parsed.warnings) == 2
    assert "weight index 3" in parsed.warnings[0]
    assert "ignored" in parsed.warnings[1]


def test_html():
    parsed = parse("#band(fox zzz) the <b>", resolve)
    html = query_html(parsed.nodes, {2: "#fff"})
    assert 'class="op dropped"' in html
    assert 'class="unknown"' in html and 'class="stop"' in html
    assert "<b>" not in html
    assert error_html("a <b", 2, 4) == 'a <span class="error">&lt;b</span>'


def test_matches_impact_index_errors(collection):
    """The mirror rejects what impact-index rejects"""
    engine = SearchEngine.open(collection)
    scored = engine.scored(collection.bm25)
    for query in ["#syn(fox", "fox)", "#foo(x)", "#uw1(a b)", "#syn(#1(a b))"]:
        with pytest.raises(QuerySyntaxError):
            engine.parse(query)
        with pytest.raises(ValueError):
            scored.search_maxscore_query(query, 10)


def test_structured_search(collection):
    engine = SearchEngine.open(collection)
    assert engine.has_positions

    phrase = engine.search("#1(lazy dog)")
    assert [h.docid for h in phrase.hits] == ["d0"]
    assert [t.words for t in phrase.query.terms] == [["lazy"], ["dog"]]

    syn = engine.search("#syn(cat fox)")
    assert {h.docid for h in syn.hits} == {"d0", "d1", "d2", "d3"}

    band = engine.search("#band(quick fox)")
    assert {h.docid for h in band.hits} == {"d0", "d2"}

    assert engine.search("#band(fox zzzunknown)").hits == []


def test_plain_words(collection):
    engine = SearchEngine.open(collection)
    words = engine.plain_words("the foxes zzz")
    assert [w.resolution for w in words] == [
        Resolution.STOPWORD,
        Resolution.TERM,
        Resolution.UNKNOWN,
    ]
    assert isinstance(words[0], Word)
