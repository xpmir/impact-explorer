from types import SimpleNamespace

from datamaestro_ir.data.base import (
    AdhocAssessedTopic,
    SimpleAdhocAssessment,
    SimpleTextItem,
)

from impact_explorer.store import QuerySource, QueryStore, SavedQuery
from impact_explorer.topics import TopicSets, topic_set_from_adhoc


def test_store(tmp_path):
    path = tmp_path / "saved.json"
    store = QueryStore(path)
    q1 = store.save(SavedQuery(collection="a", text="quick fox", name="Fox"))
    q2 = store.save(
        SavedQuery(collection="b", text="lazy", source=QuerySource("ds", "q2"))
    )

    reloaded = QueryStore(path)
    assert [q.id for q in reloaded.list("a")] == [q1.id]
    assert reloaded.get(q2.id).source == QuerySource("ds", "q2")
    assert len(reloaded.list()) == 2

    reloaded.delete(q1.id)
    assert [q.id for q in QueryStore(path).list()] == [q2.id]


def test_topic_set_from_adhoc():
    topics = [
        {"id": "1", "text_item": SimpleTextItem(" quick fox \n")},
        {"id": "2", "text_item": SimpleTextItem("lazy")},
    ]
    assessments = [
        AdhocAssessedTopic(
            "1", [SimpleAdhocAssessment("d0", 1), SimpleAdhocAssessment("d4", 0)]
        )
    ]
    adhoc = SimpleNamespace(
        topics=SimpleNamespace(iter=lambda: iter(topics)),
        assessments=SimpleNamespace(iter=lambda: iter(assessments)),
    )
    topic_set = topic_set_from_adhoc("ds", adhoc)
    assert topic_set.topics["1"].text == "quick fox"
    assert topic_set.qrels == {"1": {"d0": 1.0, "d4": 0.0}}
    assert topic_set.relevant_count("1") == 1
    assert topic_set.relevant_count("2") == 0


def test_topic_sets_are_cached():
    calls = []
    sets = TopicSets(loader=lambda ds: calls.append(ds) or ds)
    assert sets.get("x") == sets.get("x") == "x"
    assert calls == ["x"]


def test_trec_topic_fields():
    from datamaestro_ir.data.formats import TrecTopic

    from impact_explorer.topics import Topic, TopicSet, text_fields

    item = TrecTopic("quick fox", "Find foxes.", "")
    assert text_fields(item) == {"text": "quick fox", "description": "Find foxes."}

    topic = Topic("1", "quick fox", text_fields(item))
    assert topic.query("description") == "Find foxes."
    assert topic.query("narrative") == "quick fox"  # missing: default text
    topics = TopicSet("ds", {"1": topic, "2": Topic("2", "x", {"text": "x"})})
    assert topics.field_names() == ["text", "description"]
