import json

import pytest

from impact_explorer.config import (
    WORKSPACE_FILE,
    BM25Params,
    CollectionConfig,
    ConfigError,
    Workspace,
)


def test_roundtrip(tmp_path):
    workspace = Workspace(tmp_path)
    workspace.put(
        CollectionConfig(
            name="c1",
            index="indexes/c1",
            datasets=["ds.main"],
            bm25=BM25Params(k1=0.9, b=0.4, variant="lucene"),
        )
    )
    workspace.add_dataset("c1", "ds.other")
    workspace.add_dataset("c1", "ds.other")

    loaded = Workspace(tmp_path).collection("c1")
    assert loaded.datasets == ["ds.main", "ds.other"]
    assert loaded.bm25 == BM25Params(k1=0.9, b=0.4, variant="lucene")
    # Relative paths are resolved against the workspace folder
    assert loaded.index_path == tmp_path.resolve() / "indexes/c1"

    workspace.remove_dataset("c1", "ds.main")
    assert Workspace(tmp_path).collection("c1").datasets == ["ds.other"]

    workspace.remove("c1")
    assert Workspace(tmp_path).collections == {}


def test_change_listeners(tmp_path):
    workspace = Workspace(tmp_path)
    changed = []
    workspace.on_change(changed.append)
    workspace.put(CollectionConfig(name="c1", index="/x"))
    workspace.remove("c1")
    assert changed == ["c1", "c1"]


@pytest.mark.parametrize(
    "name, index", [("", "/x"), ("bad name", "/x"), ("a/b", "/x"), ("ok", "")]
)
def test_validation(tmp_path, name, index):
    with pytest.raises(ConfigError):
        Workspace(tmp_path).put(CollectionConfig(name=name, index=index))


def test_docstore_detection(tmp_path, collection_folder):
    collection = CollectionConfig(
        name="c", index=str(collection_folder / "index"), base=tmp_path
    )
    assert collection.docstore_path == collection_folder / "docstore"

    explicit = CollectionConfig(name="c", index="i", docstore="ds", base=tmp_path)
    assert explicit.docstore_path == tmp_path / "ds"

    datamaestro = CollectionConfig(name="c", index="i", documents="x", base=tmp_path)
    assert datamaestro.docstore_path is None


def test_unsupported_version(tmp_path):
    (tmp_path / WORKSPACE_FILE).write_text(json.dumps({"version": 99}))
    with pytest.raises(ConfigError, match="version"):
        Workspace(tmp_path)


def test_written_file(tmp_path):
    workspace = Workspace(tmp_path)
    workspace.put(CollectionConfig(name="c1", index="/x"))
    path = tmp_path / WORKSPACE_FILE
    assert path.read_text().endswith("}\n")
    assert path.stat().st_mode & 0o044  # readable by others, as with open()


def test_stop_words_option(tmp_path):
    import argparse

    from impact_explorer.cli import stop_words_option

    assert stop_words_option("terrier") == "terrier"
    assert stop_words_option("none") == []
    words = tmp_path / "stop.txt"
    words.write_text("the\n# comment\n\nof\n")
    assert stop_words_option(str(words)) == ["the", "of"]
    with pytest.raises(argparse.ArgumentTypeError):
        stop_words_option("nope")
