import fcntl
from pathlib import Path
from types import SimpleNamespace

import pytest

from impact_explorer import builds as builds_module
from impact_explorer.builds import BuildFiles, Builds, BuildSpec, BuildState, Runner
from impact_explorer.config import ConfigError, Workspace

from .conftest import DOCUMENTS


class FakeDocuments:
    """datamaestro-like documents, failing after ``fail_after`` documents"""

    def __init__(self, fail_after: int | None = None):
        self.fail_after = fail_after
        self.starts = []

    @property
    def documentcount(self):
        return len(DOCUMENTS)

    def iter_documents_from(self, start=0):
        self.starts.append(start)
        for index, document in enumerate(DOCUMENTS[start:], start=start):
            if self.fail_after is not None and index >= self.fail_after:
                raise OSError("network down")
            yield {
                "id": document.docid,
                "text_item": SimpleNamespace(text=document.text, title=document.title),
            }


@pytest.fixture
def fake(monkeypatch):
    """Runs builds in this process, with fake documents"""
    documents = FakeDocuments()
    prepared = []

    def dataset(runner):
        prepared.append(runner.spec.documents)
        return documents

    monkeypatch.setattr(Runner, "dataset", dataset)
    monkeypatch.setattr(builds_module, "CHECKPOINT_FREQUENCY", 2)
    return SimpleNamespace(documents=documents, prepared=prepared)


@pytest.fixture
def builds(tmp_path):
    return Builds(Workspace(tmp_path / "ws"))


def create(builds, **kwargs) -> BuildFiles:
    spec = BuildSpec(name="fake", documents="fake.documents", **kwargs)
    builds.create(spec, start=False)
    return builds.files("fake")


def run(builds):
    Runner(builds.workspace.folder, "fake").run()


def stored_ids(builds, folder="indexes/fake") -> list[str]:
    import impact_index

    store = impact_index.DocumentStore.load(
        str(builds.workspace.folder / folder / "docstore"), "mmap"
    )
    docs = store.get_by_number(list(range(store.num_documents())))
    return [doc.keys["id"] for doc in docs]


def test_build_and_register(builds, fake):
    files = create(builds, positions=True, datasets=["test.topics"])
    state, alive = builds.get("fake")
    assert state.status(alive) == "pending"
    run(builds)
    state = files.load()
    assert state.complete and state.status(False) == "done"
    assert state.stages["index"].done == len(DOCUMENTS)
    assert state.stages["docstore"].total == len(DOCUMENTS)
    assert stored_ids(builds) == [d.docid for d in DOCUMENTS]

    assert builds.register_completed() == ["fake"]
    assert builds.register_completed() == []
    collection = builds.workspace.collection("fake")
    # Inside the workspace: relative paths
    assert collection.index == "indexes/fake/index"
    assert collection.datasets == ["test.topics"]

    from impact_explorer.engine import SearchEngine

    result = SearchEngine.open(collection).search("quick fox")
    assert {hit.docid for hit in result.hits} == {"d0", "d2"}
    # A title is indexed and stored
    assert SearchEngine.open(collection).search("cats").hits[0].docid == "d1"


def test_resume_document_store(builds, fake):
    files = create(builds)
    fake.documents.fail_after = 3
    with pytest.raises(OSError):
        run(builds)
    state = files.load()
    assert state.status(False) == "failed"
    assert "network down" in state.error
    assert state.stages["prepare"].status == "done"
    assert state.stages["docstore"].status == "failed"
    assert state.stages["docstore"].done == 3

    fake.documents.fail_after = None
    run(builds)
    state = files.load()
    assert state.complete and state.error is None
    # Resumed at the last checkpoint (every 2 documents), not from scratch
    assert fake.documents.starts == [0, 2]
    assert state.stages["docstore"].resumed_from == 2
    assert stored_ids(builds) == [d.docid for d in DOCUMENTS]


def test_resume_index(builds, fake, monkeypatch):
    files = create(builds)

    def fail(*args, **kwargs):
        raise RuntimeError("disk full")

    import impact_index

    original = impact_index.BOWIndexBuilder
    monkeypatch.setattr(impact_index, "BOWIndexBuilder", fail)
    with pytest.raises(RuntimeError):
        run(builds)
    assert files.load().stages["index"].status == "failed"
    prepared = len(fake.prepared)

    monkeypatch.setattr(impact_index, "BOWIndexBuilder", original)
    run(builds)
    assert files.load().complete
    # Neither downloaded nor copied again
    assert len(fake.prepared) == prepared
    assert fake.documents.starts == [0]


def test_status():
    state = BuildState(spec=BuildSpec(name="x", documents="d"))
    assert state.status(alive=False) == "pending"
    state.stages["prepare"].status = "done"
    state.stages["docstore"].status = "running"
    assert state.status(alive=True) == "running"
    # The runner is gone without recording anything
    assert state.status(alive=False) == "interrupted"
    state.cancelled = True
    assert state.status(alive=False) == "cancelled"
    state.error = "boom"
    assert state.status(alive=False) == "failed"


def test_lock(builds, fake):
    files = create(builds)
    assert not files.alive()
    files.lock.parent.mkdir(parents=True, exist_ok=True)
    with open(files.lock, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert files.alive()
        with pytest.raises(RuntimeError, match="already running"):
            run(builds)
        with pytest.raises(ConfigError):
            builds.start("fake")
        with pytest.raises(ConfigError):
            builds.reset("fake")
    assert not files.alive()


def test_reset_and_remove(builds, fake):
    files = create(builds)
    run(builds)
    output = builds.workspace.folder / "indexes/fake"
    builds.reset("fake")
    assert files.load().status(False) == "pending"
    assert not (output / "index").exists()
    builds.remove("fake")
    assert builds.names() == []


def test_create_validation(builds):
    with pytest.raises(ConfigError):
        builds.create(BuildSpec(name="a b", documents="x"), start=False)
    with pytest.raises(ConfigError):
        builds.create(BuildSpec(name="a", documents=""), start=False)
    builds.create(BuildSpec(name="a", documents="x"), start=False)
    with pytest.raises(ConfigError, match="already exists"):
        builds.create(BuildSpec(name="a", documents="x"), start=False)


def test_stop_words_options():
    spec = BuildSpec(name="x", documents="d")
    assert spec.builder_options()["stop_words"] is None
    spec.stop_words = "none"
    assert spec.builder_options()["stop_words"] == []
    spec.stop_words = "terrier"
    assert spec.builder_options()["stop_words"] == "terrier"


def test_suggested_name():
    from impact_explorer.settings_ui import suggested_name

    assert suggested_name("com.microsoft.msmarco.passage.documents") == (
        "msmarco-passage"
    )
    assert suggested_name("org.beir.scifact.collection") == "beir-scifact"
    assert suggested_name("org.beir.nq") == "beir-nq"


def test_catalog():
    """Reads the installed datamaestro repositories (nothing is downloaded)"""
    from impact_explorer.catalog import load_catalog

    catalog = load_catalog()
    assert catalog.error is None
    if "org.beir.scifact.collection" not in catalog.collections:
        pytest.skip("datamaestro-ir BEIR datasets not installed")
    assert "org.beir.scifact.test" in catalog.topics_for("org.beir.scifact.collection")
    # Adhoc datasets bundling their documents are collections of their own
    assert catalog.topics_for("org.beir.nq") == ["org.beir.nq"]
    options = catalog.topic_options("org.beir.scifact.collection")
    assert options[: len(catalog.topics_for("org.beir.scifact.collection"))] == (
        catalog.topics_for("org.beir.scifact.collection")
    )
    assert sorted(options) == catalog.topics


def test_download_errors(monkeypatch):
    """datamaestro only logs download failures: they must stop the build"""
    import sys

    from datamaestro.context import Context

    from impact_explorer.builds import DownloadError, download

    class Wrapper:
        def download(self):
            print("Traceback (most recent call last):", file=sys.stderr)
            print("ModuleNotFoundError: No module named 'pandas'", file=sys.stderr)
            return False

    context = Context.instance()
    monkeypatch.setattr(type(context), "dataset", lambda self, i: Wrapper())
    with pytest.raises(DownloadError, match="No module named 'pandas'"):
        download("some.documents")

    def unknown(self, i):
        raise Exception(f"Dataset {i} not found")

    monkeypatch.setattr(type(context), "dataset", unknown)
    with pytest.raises(DownloadError, match="Unknown datamaestro dataset"):
        download("no.such.dataset")


def index_kind(folder):
    import json

    return json.loads((folder / "manifest.json").read_text())["index_kind"]


def test_compress(builds, fake):
    files = create(builds, positions=True, compress=True)
    run(builds)
    state = files.load()
    assert state.complete and state.stages["compress"].status == "done"
    output = builds.workspace.folder / "indexes/fake"
    # The uncompressed index is gone
    assert sorted(p.name for p in output.iterdir()) == ["docstore", "index"]
    assert index_kind(output / "index") == "compressed"
    builds.register_completed()

    from impact_explorer.engine import SearchEngine

    engine = SearchEngine.open(builds.workspace.collection("fake"))
    assert engine.has_positions
    assert {hit.docid for hit in engine.search("quick fox").hits} == {"d0", "d2"}


def test_compress_failure_keeps_the_index(builds, fake, monkeypatch):
    files = create(builds, compress=True)

    def fail(self, stage):
        raise RuntimeError("disk full")

    monkeypatch.setattr(Runner, "stage_compress", fail)
    with pytest.raises(RuntimeError):
        run(builds)
    assert files.load().stages["index"].status == "done"
    monkeypatch.undo()
    monkeypatch.setattr(Runner, "dataset", lambda runner: fake.documents)
    run(builds)
    assert files.load().complete
    # Only compressed again
    assert fake.documents.starts == [0]


def test_stages_without_compression():
    spec = BuildSpec(name="x", documents="d")
    assert spec.stages() == ("prepare", "docstore", "index")
    state = BuildState(spec=spec)
    for stage in spec.stages():
        state.stages[stage].status = "done"
    assert state.complete
    # States saved before the compression stage existed
    data = state.to_dict()
    del data["stages"]["compress"], data["spec"]["compress"]
    assert BuildState.from_dict(data).complete


def test_rebuild_index_only(builds, fake):
    files = create(builds)
    run(builds)
    spec = files.load().spec
    spec.compress, spec.pipeline = True, "terrier"
    builds.reset("fake", spec, keep_documents=True)
    state = files.load()
    assert state.stages["docstore"].status == "done"
    assert state.stages["index"].status == "pending"
    run(builds)
    assert files.load().complete and files.load().spec.pipeline == "terrier"
    # The documents were not copied again
    assert fake.documents.starts == [0]
    assert index_kind(builds.workspace.folder / "indexes/fake/index") == "compressed"

    other = BuildSpec(name="fake", documents="other")
    with pytest.raises(ConfigError, match="cannot change"):
        builds.reset("fake", other)


def test_remove(builds, fake):
    output = builds.workspace.folder / "indexes/fake"
    create(builds)
    run(builds)
    builds.register_completed()
    # Kept without deletion
    builds.remove_collection("fake", delete_files=False)
    assert output.exists() and builds.names() == []

    create(builds)
    run(builds)
    builds.remove("fake", delete_files=True)
    assert not output.exists()

    create(builds)
    run(builds)
    builds.register_completed()
    assert builds.collection_files("fake") == [output / "index", output / "docstore"]
    builds.remove_collection("fake", delete_files=True)
    assert not output.exists() and "fake" not in builds.workspace.collections


def test_log_tail(builds):
    files = builds.files("x")
    files.log.parent.mkdir(parents=True)
    files.log.write_text("--- 1 start\nold failure\n--- 2 start\nworking\n")
    assert files.log_tail() == "--- 2 start\nworking"


def test_option_labels():
    from impact_explorer.builds import PIPELINES, STOP_WORDS
    from impact_explorer.settings_ui import PIPELINE_LABELS, STOP_WORDS_LABELS

    assert tuple(PIPELINE_LABELS) == PIPELINES
    assert tuple(STOP_WORDS_LABELS) == STOP_WORDS


class FakeStoreDocuments:
    """datamaestro documents in an impact-index document store"""

    def __init__(self, path):
        import impact_index

        builder = impact_index.DocumentStoreBuilder(str(path))
        for document in DOCUMENTS:
            builder.add({"id": document.docid}, document.text.encode())
        builder.build()
        self.path = path
        self._store = impact_index.DocumentStore.load(str(path), "mmap")

    def iter_documents_from(self, start=0):
        for document in DOCUMENTS[start:]:
            yield {
                "id": document.docid,
                "text_item": SimpleNamespace(text=document.text),
            }

    def docid_internal2external(self, docid):
        return DOCUMENTS[docid].docid

    def document_ext(self, docid):
        document = next(d for d in DOCUMENTS if d.docid == docid)
        return {"id": docid, "text_item": SimpleNamespace(text=document.text)}

    def documents_ext(self, docids):
        return [self.document_ext(docid) for docid in docids]


def test_datamaestro_store(builds, tmp_path, monkeypatch):
    """datamaestro's own document store is used, not copied"""
    documents = FakeStoreDocuments(tmp_path / "dm-store")
    monkeypatch.setattr(Runner, "dataset", lambda runner: documents)
    monkeypatch.setattr(
        builds_module, "datamaestro_store", lambda d: getattr(d, "path", None)
    )
    files = create(builds, compress=True)
    run(builds)
    state = files.load()
    assert state.complete
    assert state.datamaestro_store == str(tmp_path / "dm-store")
    assert state.stages["docstore"].done == len(DOCUMENTS)
    output = builds.workspace.folder / "indexes/fake"
    assert sorted(p.name for p in output.iterdir()) == ["index"]

    collection = builds.collection(state)
    assert (collection.documents, collection.docstore) == ("fake.documents", None)

    from impact_explorer.documents import DatamaestroDocumentSource
    from impact_explorer.engine import SearchEngine

    collection.base = builds.workspace.folder
    engine = SearchEngine(collection, DatamaestroDocumentSource(documents))
    assert {hit.docid for hit in engine.search("quick fox").hits} == {"d0", "d2"}

    # Rebuilding the index only keeps using it
    builds.reset("fake", keep_documents=True)
    assert files.load().datamaestro_store == state.datamaestro_store
    run(builds)
    assert files.load().complete
    # Removing the collection never deletes datamaestro's files
    builds.register_completed()
    builds.remove_collection("fake", delete_files=True)
    assert (tmp_path / "dm-store").is_dir() and not output.exists()


def test_datamaestro_store_detection():
    from impact_explorer.builds import datamaestro_store

    assert datamaestro_store(FakeDocuments()) is None
    try:
        from datamaestro import prepare_dataset

        documents = prepare_dataset("co.huggingface.nano-beir.nfcorpus.documents")
    except Exception:
        pytest.skip("nano-beir datasets not available")
    if not Path(documents.path).is_dir():
        pytest.skip("nano-beir NFCorpus not downloaded")
    assert datamaestro_store(documents) == Path(documents.path)
