import pytest

from impact_explorer.config import CollectionConfig, Workspace
from impact_explorer.documents import Document
from impact_explorer.indexer import build_collection
from impact_explorer.topics import Topic, TopicSet

pytest_plugins = ["nicegui.testing.user_plugin"]

DOCUMENTS = [
    Document("d0", "The quick brown fox jumps over the lazy dog."),
    Document("d1", "A lazy cat sleeps all day long.", title="Cats"),
    Document("d2", "Foxes are quick and clever animals; a fox hunts at night."),
    Document("d3", "Dogs and cats <b>are</b> pets & friends."),
    Document("d4", "Nothing relevant here, only weather reports."),
]

TOPICS = TopicSet(
    dataset_id="test.topics",
    topics={
        "q1": Topic(
            "q1",
            "quick fox",
            {"text": "quick fox", "description": "hunts at night"},
        ),
        "q2": Topic("q2", "lazy cats"),
    },
    qrels={
        "q1": {"d0": 1, "d2": 2, "d4": 0, "d9": 1},
        "q2": {"d1": 1, "d3": 0},
    },
)


@pytest.fixture(scope="session")
def collection_folder(tmp_path_factory):
    folder = tmp_path_factory.mktemp("collection")
    build_collection(DOCUMENTS, folder, positions=True)
    return folder


@pytest.fixture
def workspace(tmp_path, collection_folder) -> Workspace:
    workspace = Workspace(tmp_path / "workspace")
    workspace.put(
        CollectionConfig(
            name="test",
            index=str(collection_folder / "index"),
            datasets=["test.topics"],
        )
    )
    return workspace


@pytest.fixture
def collection(workspace) -> CollectionConfig:
    return workspace.collection("test")


@pytest.fixture(scope="session")
def unfiltered_folder(tmp_path_factory):
    """An index whose documents keep their stop words (only queries are
    filtered, as with PISA)"""
    folder = tmp_path_factory.mktemp("unfiltered")
    build_collection(DOCUMENTS, folder, pipeline="terrier-pisa", positions=True)
    return folder


@pytest.fixture
def unfiltered(tmp_path, unfiltered_folder) -> CollectionConfig:
    return CollectionConfig(
        name="unfiltered",
        index=str(unfiltered_folder / "index"),
        base=tmp_path,
    )
