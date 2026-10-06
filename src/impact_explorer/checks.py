"""Checks of collection settings, shown live in the settings dialog.

Checks never download anything: a datamaestro dataset whose files are not
there yet is reported as a warning.
"""

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .config import NAME_RE, CollectionConfig
from .engine import index_features
from .index_info import summary as index_summary


class Status(StrEnum):
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Check:
    status: Status
    message: str

    @staticmethod
    def ok(message: str) -> "Check":
        return Check(Status.OK, message)

    @staticmethod
    def warning(message: str) -> "Check":
        return Check(Status.WARNING, message)

    @staticmethod
    def error(message: str) -> "Check":
        return Check(Status.ERROR, message)


def check_name(name: str, taken: bool) -> Check:
    if not name:
        return Check.error("A name is required")
    if not NAME_RE.match(name):
        return Check.error("Only letters, digits, '.', '_' and '-' are allowed")
    if taken:
        return Check.error(f"A collection named {name!r} already exists")
    return Check.ok("")


def index_documents(path: Path) -> int | None:
    """Number of documents of a BOW index (None if unknown)"""
    try:
        import impact_index

        return impact_index.DocMetadata.load(str(path)).num_docs()
    except Exception:
        return None


def check_index(collection: CollectionConfig) -> Check:
    if not collection.index:
        return Check.error("The index path is required")
    path = collection.index_path
    if not path.is_dir():
        return Check.error(f"No such folder: {path}")
    if not (path / "manifest.json").exists():
        return Check.error("Not an impact-index index (no manifest.json)")
    if not (path / "vocab.fst").exists():
        return Check.error(
            "The index has no vocabulary: build it with BOWIndexBuilder "
            "(text analysis is needed for queries and highlighting)"
        )
    try:
        import impact_index

        index = impact_index.Index.load(str(path), False)
        n_terms = index.num_postings()
        index.analyzer()
    except Exception as e:
        return Check.error(f"Cannot open the index: {e}")

    parts = [f"{n_terms:,} terms"]
    n_docs = index_documents(path)
    if n_docs is not None:
        parts.append(f"{n_docs:,} documents")
    features = index_features(path)
    parts.append("positions" if "positions" in features else "no positions")
    try:
        manifest = json.loads((path / "manifest.json").read_text())
        parts.append(manifest.get("index_kind", "?"))
    except (OSError, ValueError):
        pass
    if details := index_summary(path):
        parts.append(details)
    if n_docs is None:
        return Check.warning(
            " · ".join(parts) + " — no document lengths, BM25 needs them"
        )
    return Check.ok(" · ".join(parts))


def check_documents(collection: CollectionConfig) -> Check:
    """Checks the document source, and that it matches the index"""
    n_index = index_documents(collection.index_path) if collection.index else None
    docstore = collection.docstore_path

    if docstore is not None:
        where = "" if collection.docstore else f"found {docstore} · "
        try:
            import impact_index

            store = impact_index.DocumentStore.load(str(docstore), "mmap")
            n_docs = store.num_documents()
            keys = store.key_names()
        except Exception as e:
            return Check.error(f"Cannot open the document store {docstore}: {e}")
        parts = [f"{where}{n_docs:,} documents"]
        status = Status.OK
        if collection.id_key and collection.id_key not in keys:
            if keys:
                return Check.error(
                    f"No {collection.id_key!r} key in the store "
                    f"(keys: {', '.join(keys)}); set the id key in Advanced"
                )
            parts.append("no keys: ids are document numbers")
        elif collection.id_key:
            parts.append(f"ids from the {collection.id_key!r} key")
        else:
            parts.append("ids are document numbers")
        if n_index is not None and n_index != n_docs:
            status = Status.WARNING
            parts.append(f"the index has {n_index:,} documents: docids may not match")
        return Check(status, " · ".join(parts))

    if collection.documents:
        try:
            from datamaestro.context import get_dataset

            documents = get_dataset(collection.documents)
            documents = getattr(documents, "documents", documents)
            if not hasattr(documents, "document_ext"):
                return Check.error(
                    f"{collection.documents} is not a document store "
                    "(documents cannot be fetched by id)"
                )
            count = documents.documentcount
        except FileNotFoundError as e:
            return Check.warning(f"Not downloaded yet ({e.filename})")
        except Exception as e:
            return Check.error(f"{collection.documents}: {e}")
        message = f"{count:,} documents" if count else "documents available"
        if n_index is not None and count and n_index != count:
            return Check.warning(
                f"{message} · the index has {n_index:,}: docids may not match"
            )
        return Check.ok(message)

    if collection.docstore:
        return Check.error(f"No such folder: {collection.resolve(collection.docstore)}")
    return Check.error(
        "No document store found next to the index: set a document store path "
        "or a datamaestro documents dataset"
    )


def _dataset_id(config) -> str | None:
    """The datamaestro id of a data configuration (without ``@repository``)"""
    try:
        data_id = config.id
    except Exception:
        return None
    return data_id.split("@", 1)[0] if isinstance(data_id, str) else None


def _check_source(collection: CollectionConfig):
    """The collection's document source, opened without downloading"""
    from .documents import DatamaestroDocumentSource, ImpactDocumentSource

    if collection.docstore_path is not None:
        return ImpactDocumentSource(collection)
    if collection.documents:
        from datamaestro.context import get_dataset

        definition = get_dataset(collection.documents)
        return DatamaestroDocumentSource(
            getattr(definition, "documents", definition), collection.documents
        )
    return None


SAMPLE_SIZE = 200


def check_compatibility(
    topic_set, adhoc, collection: CollectionConfig | None
) -> Check | None:
    """Checks that assessed documents belong to the collection

    Two signals: the documents the dataset declares (when both sides are
    datamaestro datasets), and a lookup of a sample of the assessed document
    ids in the collection. Ids alone can collide (numeric ids in particular),
    so a declared mismatch is an error unless every sampled id is found.
    """
    if collection is None:
        return None
    declared = _dataset_id(getattr(adhoc, "documents", None))
    mismatch = bool(
        collection.documents and declared and declared != collection.documents
    )

    # Relevant documents first: they matter most
    judged = sorted(
        {
            (rel <= 0, docid)
            for qrels in topic_set.qrels.values()
            for docid, rel in qrels.items()
        }
    )
    sample = [docid for _, docid in judged[:SAMPLE_SIZE]]
    found = None
    lookup_error = None
    if sample:
        try:
            source = _check_source(collection)
            if source is not None:
                found = sum(doc is not None for doc in source.get(sample))
        except Exception as e:
            lookup_error = e
    lookup = (
        f"{found} of {len(sample)} assessed documents found"
        if found is not None
        else None
    )

    if mismatch:
        message = f"built on {declared}, but the collection uses {collection.documents}"
        if found is not None and found == len(sample):
            return Check.warning(f"{message} ({lookup})")
        return Check.error(message + (f" ({lookup})" if lookup else ""))
    if found is None:
        if lookup_error is not None:
            return Check.warning(
                f"cannot check the assessed documents ({lookup_error})"
            )
        return Check.ok(f"same documents ({declared})") if declared else None
    if found == 0:
        return Check.error(
            f"none of {len(sample)} assessed documents is in the collection: "
            "this dataset is for another collection"
        )
    if found < len(sample):
        return Check.warning(f"only {lookup} in the collection")
    if declared and collection.documents:
        return Check.ok(f"same documents ({declared})")
    return Check.ok(f"{lookup} in the collection")


def check_dataset(
    dataset_id: str, topic_sets, collection: CollectionConfig | None = None
) -> Check:
    """Loads topics and assessments of a dataset (without downloading), and
    checks that they fit the collection"""
    from .topics import topic_set_from_adhoc

    try:
        from datamaestro.context import find_dataset, get_dataset

        find_dataset(dataset_id)
    except Exception as e:
        return Check.error(f"Unknown dataset ({e})")
    try:
        adhoc = get_dataset(dataset_id)
        if not hasattr(adhoc, "topics"):
            return Check.error(
                f"Not an IR (Adhoc) dataset: {type(adhoc).__name__.split('.')[0]}"
            )
        topic_set = topic_set_from_adhoc(dataset_id, adhoc)
    except FileNotFoundError as e:
        return Check.warning(
            f"Not downloaded yet ({e.filename}); it is downloaded on first use"
        )
    except Exception as e:
        return Check.error(str(e))
    topic_sets.put(topic_set)
    assessed = sum(1 for qrels in topic_set.qrels.values() if qrels)
    summary = f"{len(topic_set.topics):,} topics, {assessed:,} assessed"
    if not assessed:
        return Check.warning(f"{summary}, no assessments")

    compatibility = check_compatibility(topic_set, adhoc, collection)
    if compatibility is None:
        return Check.ok(summary)
    return Check(compatibility.status, f"{summary} · {compatibility.message}")
