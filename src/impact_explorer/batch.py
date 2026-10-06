"""Evaluation of every assessed topic of a dataset (query-level metrics).

Finished evaluations are stored in the workspace (``evaluations/``), keyed
by everything they depend on: parameters, query field, and a fingerprint of
the index and document settings (so a rebuilt index is never mixed up with
the previous one).
"""

import hashlib
import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import BM25Params, CollectionConfig, atomic_write_json
from .engine import SearchEngine
from .evaluation import evaluate
from .topics import DEFAULT_FIELD, TopicSet

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1


def fingerprint(collection: CollectionConfig) -> str:
    """Identifies the index and document settings a result depends on"""
    try:
        manifest = json.loads((collection.index_path / "manifest.json").read_text())
        created = manifest.get("created")
    except (OSError, ValueError):
        created = None
    docstore = collection.docstore_path
    data = {
        "index": str(collection.index_path),
        "created": created,
        "docstore": str(docstore) if docstore else None,
        "documents": collection.documents,
        "id_key": collection.id_key,
    }
    return hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class RunKey:
    """What a batch evaluation depends on"""

    collection: str
    dataset: str
    k: int
    depth: int
    bm25: tuple
    field: str = DEFAULT_FIELD
    """Topic field used as query"""

    fingerprint: str = ""
    """Index and document settings (see :func:`fingerprint`)"""

    stop_words: bool = True
    """Whether the index's stop words are removed from queries"""

    @staticmethod
    def of(
        collection: str,
        dataset: str,
        k: int,
        depth: int,
        params: BM25Params,
        field: str = DEFAULT_FIELD,
        fingerprint: str = "",
        stop_words: bool = True,
    ):
        bm25 = (params.k1, params.b, params.variant, params.k3)
        return RunKey(
            collection, dataset, k, max(depth, k), bm25, field, fingerprint, stop_words
        )

    def filename(self) -> str:
        data = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha1(data).hexdigest()[:20] + ".json"


@dataclass
class SavedResult:
    """Evaluation of a saved query (a reformulation of a topic)"""

    topic_id: str
    text: str
    metrics: dict[str, float]


@dataclass
class BatchRun:
    key: RunKey
    total: int
    done: int = 0
    per_topic: dict[str, dict[str, float]] = field(default_factory=dict)
    saved: dict[str, SavedResult] = field(default_factory=dict)
    """Saved query id -> result"""

    error: str | None = None
    finished: bool = False
    cancel: threading.Event = field(default_factory=threading.Event)

    @property
    def complete(self) -> bool:
        return self.finished and self.error is None and not self.cancel.is_set()

    def means(self) -> dict[str, float]:
        if not self.per_topic:
            return {}
        names = next(iter(self.per_topic.values())).keys()
        n = len(self.per_topic)
        return {
            name: sum(m[name] for m in self.per_topic.values()) / n for name in names
        }

    def to_dict(self) -> dict:
        return {
            "version": FORMAT_VERSION,
            "key": asdict(self.key),
            "total": self.total,
            "per_topic": self.per_topic,
            "saved": {qid: asdict(r) for qid, r in self.saved.items()},
        }

    @staticmethod
    def from_dict(data: dict) -> "BatchRun":
        key = data["key"]
        key = RunKey(**{**key, "bm25": tuple(key["bm25"])})
        run = BatchRun(key=key, total=data["total"], per_topic=data["per_topic"])
        run.saved = {
            qid: SavedResult(**result) for qid, result in data["saved"].items()
        }
        run.done = len(run.per_topic)
        run.finished = True
        return run


def assessed_topics(topic_set: TopicSet) -> list[str]:
    """Topics with at least one relevant document (as trec_eval does)"""
    return [tid for tid in topic_set.topics if topic_set.relevant_count(tid) > 0]


def run_batch(engine: SearchEngine, topic_set: TopicSet, run: BatchRun, on_done=None):
    """Evaluates topics one by one, updating ``run`` (meant for a thread)"""
    params = BM25Params(*run.key.bm25)
    try:
        for topic_id in assessed_topics(topic_set):
            if run.cancel.is_set():
                break
            # Topics are plain text, whatever characters they contain
            result = engine.search(
                topic_set.topics[topic_id].query(run.key.field),
                run.key.depth,
                params,
                structured=False,
                keep_stop_words=not run.key.stop_words,
            )
            evaluation = evaluate(result, topic_set.qrels[topic_id], run.key.k)
            run.per_topic[topic_id] = evaluation.metrics
            run.done += 1
    except Exception as e:
        run.error = str(e)
    finally:
        # Stored before being marked as finished, so that a finished run is
        # always on disk
        if on_done is not None and run.error is None and not run.cancel.is_set():
            on_done(run)
        run.finished = True


class BatchRuns:
    """Batch evaluations shared by all clients (one per key)"""

    def __init__(self, folder: Path | None = None):
        self.folder = Path(folder) if folder is not None else None
        self._runs: dict[RunKey, BatchRun] = {}
        self._lock = threading.RLock()

    def _path(self, key: RunKey) -> Path | None:
        return self.folder / key.filename() if self.folder is not None else None

    def get(self, key: RunKey) -> BatchRun | None:
        with self._lock:
            run = self._runs.get(key)
            if run is not None:
                return run
            path = self._path(key)
            if path is None or not path.exists():
                return None
            try:
                data = json.loads(path.read_text())
                if data.get("version") != FORMAT_VERSION:
                    return None
                run = BatchRun.from_dict(data)
            except Exception as e:
                logger.warning("Ignoring the evaluation cache %s: %s", path, e)
                return None
            if run.key != key:
                return None
            self._runs[key] = run
            return run

    def save(self, run: BatchRun):
        """Stores a complete evaluation in the workspace"""
        path = self._path(run.key)
        if path is None or run.error is not None or run.cancel.is_set():
            return
        try:
            atomic_write_json(path, run.to_dict())
        except OSError as e:
            logger.warning("Could not store the evaluation %s: %s", path, e)

    def start(self, engine: SearchEngine, topic_set: TopicSet, key: RunKey) -> BatchRun:
        """Starts (or joins) the evaluation for this key"""
        with self._lock:
            run = self.get(key)
            if run is not None and not (run.finished and not run.complete):
                return run
            run = BatchRun(key=key, total=len(assessed_topics(topic_set)))
            self._runs[key] = run
        threading.Thread(
            target=run_batch, args=(engine, topic_set, run, self.save), daemon=True
        ).start()
        return run

    def evaluate_saved(
        self, engine: SearchEngine, topic_set: TopicSet, run: BatchRun, queries
    ) -> dict[str, SavedResult]:
        """Evaluates saved queries (missing or modified since); returns the
        results of ``queries``"""
        params = BM25Params(*run.key.bm25)
        changed = False
        for query in queries:
            topic_id = query.source.topic_id
            current = run.saved.get(query.id)
            if current is not None and current.text == query.text:
                continue
            qrels = topic_set.qrels.get(topic_id, {})
            try:
                result = engine.search(
                    query.text,
                    run.key.depth,
                    params,
                    keep_stop_words=not run.key.stop_words,
                )
            except Exception as e:
                logger.info("Saved query %s cannot be run: %s", query.id, e)
                continue
            run.saved[query.id] = SavedResult(
                topic_id=topic_id,
                text=query.text,
                metrics=evaluate(result, qrels, run.key.k).metrics,
            )
            changed = True
        if changed:
            self.save(run)
        return {q.id: run.saved[q.id] for q in queries if q.id in run.saved}

    def invalidate(self, collection: str):
        """Drops the in-memory evaluations of a collection whose settings
        changed (stored ones stay valid: their key has a fingerprint)"""
        with self._lock:
            for key in [k for k in self._runs if k.collection == collection]:
                self._runs.pop(key).cancel.set()
