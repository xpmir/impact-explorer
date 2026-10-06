from impact_explorer.config import BM25Params
from impact_explorer.engine import Engines, SearchEngine


def test_search(collection):
    engine = SearchEngine.open(collection)
    result = engine.search("quick fox")

    assert [t.words for t in result.query.terms] == [["quick"], ["fox"]]
    assert {t.df for t in result.query.terms} == {2}
    assert {h.docid for h in result.hits} == {"d0", "d2"}
    assert [h.rank for h in result.hits] == [1, 2]
    assert result.ranks[result.hits[0].docid] == 1


def test_stemmed_words_share_a_term(collection):
    engine = SearchEngine.open(collection)
    query = engine.analyze("fox foxes")
    assert len(query.terms) == 1
    assert query.terms[0].words == ["fox", "foxes"]
    assert query.terms[0].weight == 2


def test_unknown_terms(collection):
    engine = SearchEngine.open(collection)
    result = engine.search("zzzunknown the")
    assert result.query.terms == []
    assert result.hits == []


def test_bm25_parameters_change_scores(collection):
    engine = SearchEngine.open(collection)
    default = engine.search("lazy").hits
    other = engine.search("lazy", params=BM25Params(k1=2.0, b=0.0)).hits
    assert [h.score for h in default] != [h.score for h in other]


def test_documents(collection):
    engine = SearchEngine.open(collection)
    d1, missing = engine.documents.get(["d1", "nope"])
    assert d1.title == "Cats"
    assert d1.text == "A lazy cat sleeps all day long."
    assert missing is None


def test_engines_invalidate(workspace):
    opened = []

    def opener(config):
        opened.append(config.name)
        return object()

    engines = Engines(workspace, opener=opener)
    first = engines.get("test")
    assert engines.get("test") is first
    workspace.add_dataset("test", "other")  # datasets do not affect the engine
    assert engines.get("test") is first
    workspace.put(workspace.collection("test"))
    assert engines.get("test") is not first
    assert opened == ["test", "test"]


def test_keyless_document_store(tmp_path):
    import impact_index

    from impact_explorer.config import CollectionConfig
    from impact_explorer.documents import ImpactDocumentSource

    folder = tmp_path / "store"
    folder.mkdir()
    builder = impact_index.DocumentStoreBuilder(str(folder))
    for text in ["zero", "one", "two"]:
        builder.add({}, text.encode())
    builder.build()

    config = CollectionConfig(
        name="c", index="i", docstore=str(folder), content_format="text"
    )
    source = ImpactDocumentSource(config)
    assert source.external_ids([2, 0]) == ["2", "0"]
    docs = source.get(["1", "7", "x"])
    assert (docs[0].docid, docs[0].text) == ("1", "one")
    assert docs[1] is None and docs[2] is None


def test_datamaestro_documents_must_be_a_store():
    from types import SimpleNamespace

    import pytest

    from impact_explorer.config import ConfigError
    from impact_explorer.documents import DatamaestroDocumentSource

    with pytest.raises(ConfigError, match="cannot fetch documents by id"):
        DatamaestroDocumentSource(SimpleNamespace(iter=lambda: iter([])), "x")


def test_record_to_document():
    from datamaestro_ir.data.base import SimpleTextItem

    from impact_explorer.documents import record_to_document

    doc = record_to_document({"id": "7", "text_item": SimpleTextItem("hello")})
    assert (doc.docid, doc.text, doc.title) == ("7", "hello", None)


def test_stems(collection):
    engine = SearchEngine.open(collection)
    query = engine.analyze("Foxes hunting quickly")
    by_word = {t.words[0]: t for t in query.terms}
    assert by_word["foxes"].stem == "fox"
    assert by_word["foxes"].label == "fox"
    # Verified stems map back to their term
    for term in query.terms:
        if term.stem_verified:
            assert engine.matcher.terms(term.stem) == (term.term_id,)
    assert by_word["foxes"].stem_verified


def test_unverified_stem(collection):
    # Porter is not idempotent: "univers" is stemmed again to "univ"
    engine = SearchEngine.open(collection)
    engine.matcher.terms = lambda word: {"university": (7,), "univers": (8,)}.get(
        word, ()
    )
    assert engine.stem(7, ["university"]) == ("univers", False)


def test_timings(collection):
    result = SearchEngine.open(collection).search("quick fox")
    assert set(result.timings) == {"analysis", "retrieval"}
    assert result.timings["retrieval"].wall > 0
    assert result.timings["retrieval"].cpu >= 0
