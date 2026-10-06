from impact_explorer.engine import SearchEngine
from impact_explorer.highlight import (
    Segment,
    segments,
    snippet,
    term_colors,
    term_counts,
    to_html,
)


def test_segments_and_html(collection):
    engine = SearchEngine.open(collection)
    query = engine.analyze("cats are friends")
    terms = {t.term_id for t in query.terms}
    text = "Dogs and cats <b>are</b> pets & friends."
    segs = segments(text, terms, engine.matcher)

    assert "".join(s.text for s in segs) == text
    matched = [s.text for s in segs if s.term_id is not None]
    # "are" is a stop word: not in the query vector, so not highlighted
    assert matched == ["cats", "friends"]
    assert sum(term_counts(segs).values()) == 2

    html = to_html(segs, term_colors(query))
    assert "<b>" not in html and "&lt;b&gt;" in html and "&amp;" in html
    assert html.count("<mark") == 2


def test_snippet_keeps_matches():
    text = "x" * 500
    segs = [Segment(text), Segment("fox", 1), Segment(text), Segment("fox", 1)]
    out = snippet(segs, max_chars=200, context=20)
    assert [s.text for s in out if s.term_id] == ["fox", "fox"]
    assert sum(len(s.text) for s in out) < 200
    # The text ends with a match: no trailing ellipsis
    assert out[0].text == "… " and out[-1].text == "fox"


def test_snippet_without_matches():
    out = snippet([Segment("y" * 1000)], max_chars=100)
    assert out[0].text == "y" * 100
    assert out[-1].text == " …"


def test_short_text_is_unchanged():
    segs = [Segment("a "), Segment("fox", 1)]
    assert snippet(segs) == segs
