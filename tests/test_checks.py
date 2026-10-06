import impact_index

from impact_explorer.checks import (
    Status,
    check_dataset,
    check_documents,
    check_index,
    check_name,
)
from impact_explorer.config import CollectionConfig
from impact_explorer.topics import TopicSets


def config(tmp_path, **kwargs) -> CollectionConfig:
    return CollectionConfig(name="c", base=tmp_path, **kwargs)


def test_name():
    assert check_name("ok-1", False).status is Status.OK
    assert check_name("", False).status is Status.ERROR
    assert check_name("a b", False).status is Status.ERROR
    assert "already exists" in check_name("ok", True).message


def test_index(tmp_path, collection_folder):
    ok = check_index(config(tmp_path, index=str(collection_folder / "index")))
    assert ok.status is Status.OK
    assert "5 documents" in ok.message and "positions" in ok.message

    missing = check_index(config(tmp_path, index="nope"))
    assert missing.status is Status.ERROR and "No such folder" in missing.message

    assert check_index(config(tmp_path, index=str(tmp_path))).status is Status.ERROR


def test_documents(tmp_path, collection_folder):
    index = str(collection_folder / "index")
    found = check_documents(config(tmp_path, index=index))
    assert found.status is Status.OK
    assert "found" in found.message and "5 documents" in found.message

    # A keyless store with another number of documents
    store = tmp_path / "store"
    store.mkdir()
    builder = impact_index.DocumentStoreBuilder(str(store))
    builder.add({}, b"x")
    builder.build()
    other = check_documents(config(tmp_path, index=index, docstore="store"))
    assert other.status is Status.WARNING
    assert "document numbers" in other.message and "may not match" in other.message

    bad_key = check_documents(config(tmp_path, index=index, id_key="docno"))
    assert bad_key.status is Status.ERROR and "'docno'" in bad_key.message

    none = check_documents(config(tmp_path, index=str(tmp_path)))
    assert none.status is Status.ERROR


def test_unknown_dataset():
    result = check_dataset("no.such.dataset.anywhere", TopicSets())
    assert result.status is Status.ERROR
    assert "Unknown dataset" in result.message


def compat(collection_folder, tmp_path, qrels, declared=None, documents=None):
    from types import SimpleNamespace

    from impact_explorer.checks import check_compatibility
    from impact_explorer.topics import TopicSet

    collection = config(
        tmp_path,
        index=str(collection_folder / "index"),
        docstore=str(collection_folder / "docstore"),
        documents=documents,
    )
    adhoc = SimpleNamespace(documents=SimpleNamespace(id=declared))
    return check_compatibility(TopicSet("ds", {}, {"q": qrels}), adhoc, collection)


def test_compatibility_by_lookup(tmp_path, collection_folder):
    ok = compat(collection_folder, tmp_path, {"d0": 1, "d3": 0})
    assert ok.status is Status.OK and "2 of 2" in ok.message

    partial = compat(collection_folder, tmp_path, {"d0": 1, "x1": 1})
    assert partial.status is Status.WARNING and "1 of 2" in partial.message

    none = compat(collection_folder, tmp_path, {"x1": 1, "x2": 0})
    assert none.status is Status.ERROR and "another collection" in none.message


def test_compatibility_by_declaration(tmp_path, collection_folder):
    same = compat(collection_folder, tmp_path, {"d0": 1}, "a.docs@ir", "a.docs")
    assert same.status is Status.OK and "same documents" in same.message

    # Declared on other documents, and ids do not all match: error
    other = compat(
        collection_folder, tmp_path, {"d0": 1, "x": 1}, "b.docs@ir", "a.docs"
    )
    assert other.status is Status.ERROR and "built on b.docs" in other.message

    # Declared on other documents, but every id matches: only a warning
    alike = compat(collection_folder, tmp_path, {"d0": 1}, "b.docs@ir", "a.docs")
    assert alike.status is Status.WARNING
