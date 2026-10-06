import time

from impact_explorer.batch import BatchRuns, RunKey, assessed_topics
from impact_explorer.config import BM25Params
from impact_explorer.engine import SearchEngine

from .conftest import TOPICS


def wait(run):
    for _ in range(500):
        if run.finished:
            return run
        time.sleep(0.01)
    raise AssertionError("batch did not finish")


def test_batch(collection):
    engine = SearchEngine.open(collection)
    runs = BatchRuns()
    key = RunKey.of("test", TOPICS.dataset_id, 10, 100, BM25Params())
    run = wait(runs.start(engine, TOPICS, key))

    assert run.complete and run.done == run.total == 2
    assert set(run.per_topic) == set(assessed_topics(TOPICS)) == {"q1", "q2"}
    # q1 "quick fox": d0 and d2 (relevant) are the only matches, d9 is missed
    assert run.per_topic["q1"]["RR@10"] == 1
    assert run.per_topic["q1"]["R@10"] == 2 / 3
    means = run.means()
    assert (
        means["P@10"] == (run.per_topic["q1"]["P@10"] + run.per_topic["q2"]["P@10"]) / 2
    )

    # Same key: the finished run is reused; invalidation drops it
    assert runs.start(engine, TOPICS, key) is run
    runs.invalidate("test")
    assert runs.get(key) is None


def test_topics_are_searched_as_text(collection):
    from impact_explorer.topics import Topic, TopicSet

    engine = SearchEngine.open(collection)
    topics = TopicSet("t", {"q": Topic("q", "#syn(fox")}, {"q": {"d0": 1}})
    key = RunKey.of("test", "t", 10, 100, BM25Params())
    run = wait(BatchRuns().start(engine, topics, key))
    assert run.error is None and run.per_topic["q"]["RR@10"] > 0


def test_batch_on_another_field(collection):
    engine = SearchEngine.open(collection)
    title = RunKey.of("test", TOPICS.dataset_id, 10, 100, BM25Params())
    description = RunKey.of(
        "test", TOPICS.dataset_id, 10, 100, BM25Params(), "description"
    )
    assert title != description
    runs = BatchRuns()
    by_title = wait(runs.start(engine, TOPICS, title))
    by_description = wait(runs.start(engine, TOPICS, description))
    # "hunts at night" only finds d2; "quick fox" finds d0 and d2
    assert by_description.per_topic["q1"]["R@10"] == 1 / 3
    assert by_title.per_topic["q1"]["R@10"] == 2 / 3


def test_persisted_runs(tmp_path, collection):
    from impact_explorer.batch import fingerprint

    tmp_path = tmp_path / "evaluations"

    engine = SearchEngine.open(collection)
    stamp = fingerprint(collection)
    key = RunKey.of("test", TOPICS.dataset_id, 10, 100, BM25Params(), "text", stamp)
    run = wait(BatchRuns(tmp_path).start(engine, TOPICS, key))
    assert run.complete and len(list(tmp_path.iterdir())) == 1

    # A new process finds it on disk...
    loaded = BatchRuns(tmp_path).get(key)
    assert loaded is not None and loaded.complete
    assert loaded.per_topic == run.per_topic
    # ...but not for another index or other parameters
    other = RunKey.of("test", TOPICS.dataset_id, 10, 100, BM25Params(), "text", "x")
    assert BatchRuns(tmp_path).get(other) is None


def test_saved_queries(tmp_path, collection):
    from impact_explorer.store import QuerySource, SavedQuery

    engine = SearchEngine.open(collection)
    runs = BatchRuns(tmp_path)
    key = RunKey.of("test", TOPICS.dataset_id, 10, 100, BM25Params())
    run = wait(runs.start(engine, TOPICS, key))

    query = SavedQuery(
        "test", "hunts at night", source=QuerySource(TOPICS.dataset_id, "q1")
    )
    results = runs.evaluate_saved(engine, TOPICS, run, [query])
    assert results[query.id].metrics["R@10"] == 1 / 3
    first = results[query.id]
    # Unchanged text: not evaluated again; and it is stored with the run
    assert runs.evaluate_saved(engine, TOPICS, run, [query])[query.id] is first
    assert BatchRuns(tmp_path).get(key).saved[query.id].text == "hunts at night"

    query.text = "quick fox"
    assert runs.evaluate_saved(engine, TOPICS, run, [query])[query.id] is not first
