"""datamaestro introspection: document collections that can be indexed, and
the IR datasets (topics + assessments) that go with them.

Only dataset definitions are read: nothing is downloaded or prepared.

- A *documents* dataset (``datamaestro_ir.data.Documents``) can be indexed;
  the ``Adhoc`` datasets referencing it provide its topics.
- An ``Adhoc`` dataset that references no documents dataset bundles its
  own documents (e.g. most BEIR datasets): it is both the collection to
  index and its topics.
"""

import inspect
import logging
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CatalogCollection:
    id: str
    """datamaestro id of the documents to index"""

    name: str = ""
    topics: list[str] = field(default_factory=list)
    """Adhoc datasets (topics + assessments) over these documents"""

    @property
    def label(self) -> str:
        name = " ".join(self.name.split())
        if len(name) > 60:
            name = name[:59] + "…"
        return f"{self.id} — {name}" if name else self.id


@dataclass
class Catalog:
    collections: dict[str, CatalogCollection] = field(default_factory=dict)
    """Indexable documents, by id"""

    topics: list[str] = field(default_factory=list)
    """All Adhoc datasets"""

    error: str | None = None
    """Why datamaestro could not be read (empty catalog)"""

    def topics_for(self, documents: str) -> list[str]:
        collection = self.collections.get(documents)
        return list(collection.topics) if collection else []

    def topic_options(self, documents: str | None) -> list[str]:
        """All Adhoc datasets, those of ``documents`` first"""
        first = self.topics_for(documents) if documents else []
        return first + [t for t in self.topics if t not in first]


def _is_subclass(value, base) -> bool:
    return inspect.isclass(value) and issubclass(value, base)


def _referenced_ids(dataset) -> list[str]:
    """Ids of the datasets referenced by a dataset definition"""
    from datamaestro.download import reference

    ids = []
    for resource in getattr(dataset, "resources", {}).values():
        if isinstance(resource, reference):
            target = resource.reference
            wrapper = getattr(target, "__dataset__", None) or getattr(
                target, "__datamaestro__", None
            )
            if (target_id := getattr(wrapper or target, "id", None)) is not None:
                ids.append(target_id)
    return ids


def load_catalog() -> Catalog:
    try:
        from datamaestro.context import Context
        from datamaestro_ir.data import Adhoc, Documents
    except ImportError as e:
        return Catalog(error=f"datamaestro is not available: {e}")

    catalog = Catalog()
    adhoc = []
    try:
        for dataset in Context.instance().datasets():
            try:
                configtype = dataset.configtype
                if _is_subclass(configtype, Documents):
                    catalog.collections[dataset.id] = CatalogCollection(
                        dataset.id, dataset.name or ""
                    )
                elif _is_subclass(configtype, Adhoc):
                    adhoc.append(dataset)
            except Exception:
                logger.debug("Skipping dataset %s", dataset, exc_info=True)
    except Exception as e:
        logger.exception("Could not list datamaestro datasets")
        return Catalog(error=f"Could not list datamaestro datasets: {e}")

    for dataset in adhoc:
        catalog.topics.append(dataset.id)
        try:
            documents = [
                i for i in _referenced_ids(dataset) if i in catalog.collections
            ]
        except Exception:
            logger.debug("Skipping references of %s", dataset, exc_info=True)
            documents = []
        if not documents:
            # Bundles its own documents
            catalog.collections[dataset.id] = CatalogCollection(
                dataset.id, dataset.name or "", topics=[dataset.id]
            )
        for documents_id in documents:
            catalog.collections[documents_id].topics.append(dataset.id)

    catalog.collections = dict(sorted(catalog.collections.items()))
    catalog.topics.sort()
    for collection in catalog.collections.values():
        collection.topics.sort()
    return catalog


_catalog: Catalog | None = None
_lock = threading.Lock()


def catalog(reload: bool = False) -> Catalog:
    """The (cached) catalog of the installed datamaestro repositories"""
    global _catalog
    with _lock:
        if _catalog is None or reload:
            _catalog = load_catalog()
        return _catalog
