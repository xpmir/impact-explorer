"""Saved queries, each linked to a collection (JSON file)."""

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .config import atomic_write_json


@dataclass
class QuerySource:
    """Where a query comes from (a datamaestro topic)"""

    dataset: str
    topic_id: str
    field: str = "text"
    """Topic field used as query (e.g. description for TREC topics)"""


@dataclass
class SavedQuery:
    collection: str
    text: str
    name: str = ""
    note: str = ""
    source: QuerySource | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )

    @staticmethod
    def from_dict(data: dict) -> "SavedQuery":
        data = dict(data)
        source = data.pop("source", None)
        return SavedQuery(source=QuerySource(**source) if source else None, **data)


class QueryStore:
    VERSION = 1

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._queries: dict[str, SavedQuery] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text())
            if data.get("version") != self.VERSION:
                raise ValueError(
                    f"{self.path}: unsupported saved-query format version "
                    f"{data.get('version')!r} (expected {self.VERSION})"
                )
            for raw in data["queries"]:
                query = SavedQuery.from_dict(raw)
                self._queries[query.id] = query

    def list(self, collection: str | None = None) -> list[SavedQuery]:
        with self._lock:
            queries = list(self._queries.values())
        if collection is not None:
            queries = [q for q in queries if q.collection == collection]
        return sorted(queries, key=lambda q: q.created, reverse=True)

    def get(self, query_id: str) -> SavedQuery | None:
        return self._queries.get(query_id)

    def save(self, query: SavedQuery) -> SavedQuery:
        with self._lock:
            self._queries[query.id] = query
            self._write()
        return query

    def delete(self, query_id: str) -> None:
        with self._lock:
            if self._queries.pop(query_id, None) is not None:
                self._write()

    def _write(self):
        data = {
            "version": self.VERSION,
            "queries": [asdict(q) for q in self._queries.values()],
        }
        atomic_write_json(self.path, data)
