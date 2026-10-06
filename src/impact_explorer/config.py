"""Workspace: a folder holding the explorer settings and saved queries.

Layout::

    <workspace>/
        workspace.json       collections (index paths, datamaestro datasets)
        saved-queries.json   saved queries, each linked to a collection

Both files are managed from the UI. Relative paths in ``workspace.json``
are resolved against the workspace folder.
"""

import json
import os
import re
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

WORKSPACE_FILE = "workspace.json"
SAVED_QUERIES_FILE = "saved-queries.json"

NAME_RE = re.compile(r"^[\w.\-]+$")


class ConfigError(ValueError):
    pass


@dataclass
class BM25Params:
    k1: float = 1.2
    b: float = 0.75
    variant: str = "bm25"
    k3: float | None = None


@dataclass
class CollectionConfig:
    name: str
    """Unique name (saved queries refer to it)"""

    index: str
    """impact-index BOW index folder"""

    docstore: str | None = None
    """impact-index DocumentStore folder (document number == index docid).
    Defaults to a ``docstore`` folder next to or inside the index."""

    documents: str | None = None
    """datamaestro dataset id whose documents are indexed in iteration order
    (used instead of a docstore)"""

    datasets: list[str] = field(default_factory=list)
    """datamaestro IR (Adhoc) dataset ids providing topics and assessments"""

    id_key: str | None = "id"
    """DocumentStore key holding the external document id (for a store
    without keys, ids are the document numbers)"""

    content_format: str = "json"
    """DocumentStore content format: ``text`` or ``json``"""

    text_fields: list[str] = field(default_factory=lambda: ["text"])
    """JSON fields concatenated to form the document text"""

    title_field: str | None = "title"
    """JSON field (or DocumentStore key) used as document title"""

    bm25: BM25Params = field(default_factory=BM25Params)

    in_memory: bool = True

    base: Path = field(default=Path("."), repr=False, compare=False)
    """Folder against which relative paths are resolved"""

    def resolve(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.base / path

    @property
    def index_path(self) -> Path:
        return self.resolve(self.index)

    @property
    def docstore_path(self) -> Path | None:
        if self.docstore:
            return self.resolve(self.docstore)
        if self.documents:
            return None
        for candidate in (
            self.index_path / "docstore",
            self.index_path.parent / "docstore",
        ):
            if candidate.is_dir():
                return candidate
        return None

    def validate(self):
        if not NAME_RE.match(self.name or ""):
            raise ConfigError(
                f"Invalid collection name {self.name!r} "
                "(letters, digits, '.', '_' and '-' only)"
            )
        if not self.index:
            raise ConfigError(f"Collection {self.name!r}: the index path is required")
        if self.content_format not in ("text", "json"):
            raise ConfigError(
                f"Collection {self.name!r}: content_format must be 'text' or 'json'"
            )

    def to_dict(self) -> dict:
        data = asdict(self)
        del data["base"]
        return data

    @staticmethod
    def from_dict(data: dict, base: Path) -> "CollectionConfig":
        data = dict(data)
        bm25 = BM25Params(**data.pop("bm25", {}))
        try:
            return CollectionConfig(bm25=bm25, base=base, **data)
        except TypeError as e:
            raise ConfigError(f"Collection {data.get('name')!r}: {e}") from None


class Workspace:
    """Explorer settings persisted in a folder"""

    VERSION = 1

    def __init__(self, folder: Path):
        self.folder = Path(folder).expanduser().resolve()
        self.collections: dict[str, CollectionConfig] = {}
        self.rewriters: dict = {}
        """Rewriter name -> RewriterConfig"""
        self._lock = threading.RLock()
        self._listeners = []
        self._rewriter_listeners = []
        if self.settings_path.exists():
            self._load()

    @property
    def settings_path(self) -> Path:
        return self.folder / WORKSPACE_FILE

    @property
    def saved_queries(self) -> Path:
        return self.folder / SAVED_QUERIES_FILE

    def collection(self, name: str) -> CollectionConfig:
        try:
            return self.collections[name]
        except KeyError:
            raise ConfigError(f"Unknown collection {name!r}") from None

    def on_change(self, listener):
        """Registers ``listener(name)``, called when a collection changes"""
        self._listeners.append(listener)

    def _load(self):
        data = json.loads(self.settings_path.read_text())
        if data.get("version") != self.VERSION:
            raise ConfigError(
                f"{self.settings_path}: unsupported workspace format version "
                f"{data.get('version')!r} (expected {self.VERSION})"
            )
        for raw in data.get("collections", []):
            collection = CollectionConfig.from_dict(raw, self.folder)
            collection.validate()
            if collection.name in self.collections:
                raise ConfigError(f"Duplicate collection name {collection.name!r}")
            self.collections[collection.name] = collection
        from .rewriters import RewriterConfig

        for raw in data.get("rewriters", []):
            rewriter = RewriterConfig.from_dict(raw)
            self.rewriters[rewriter.name] = rewriter

    def save(self):
        with self._lock:
            data = {
                "version": self.VERSION,
                "collections": [c.to_dict() for c in self.collections.values()],
                "rewriters": [r.to_dict() for r in self.rewriters.values()],
            }
            atomic_write_json(self.settings_path, data)

    def put(self, collection: CollectionConfig):
        """Adds or replaces a collection (and saves the workspace)"""
        collection.validate()
        collection.base = self.folder
        with self._lock:
            self.collections[collection.name] = collection
            self.save()
        self._notify(collection.name)

    def remove(self, name: str):
        with self._lock:
            if self.collections.pop(name, None) is None:
                return
            self.save()
        self._notify(name)

    def add_dataset(self, name: str, dataset_id: str):
        with self._lock:
            collection = self.collection(name)
            if dataset_id not in collection.datasets:
                collection.datasets.append(dataset_id)
                self.save()

    def remove_dataset(self, name: str, dataset_id: str):
        with self._lock:
            collection = self.collection(name)
            if dataset_id in collection.datasets:
                collection.datasets.remove(dataset_id)
                self.save()

    def on_rewriter_change(self, listener):
        """Registers ``listener(name)``, called when a rewriter changes"""
        self._rewriter_listeners.append(listener)

    def put_rewriter(self, rewriter):
        """Adds or replaces a rewriter (and saves the workspace)"""
        rewriter.validate()
        with self._lock:
            self.rewriters[rewriter.name] = rewriter
            self.save()
        for listener in self._rewriter_listeners:
            listener(rewriter.name)

    def remove_rewriter(self, name: str):
        with self._lock:
            if self.rewriters.pop(name, None) is None:
                return
            self.save()
        for listener in self._rewriter_listeners:
            listener(name)

    def _notify(self, name: str):
        for listener in self._listeners:
            listener(name)


def atomic_write_json(path: Path, data):
    """Writes JSON so that a crash never leaves a truncated file"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fp:
            json.dump(data, fp, indent=2)
            fp.write("\n")
        # mkstemp creates owner-only files: use the usual permissions
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise
