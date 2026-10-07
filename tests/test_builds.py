import fcntl
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
