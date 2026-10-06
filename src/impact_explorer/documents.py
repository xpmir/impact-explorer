"""Document sources: map index docids to external ids and document text."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .config import CollectionConfig, ConfigError


@dataclass
class Document:
    docid: str
    """External document id"""

    text: str
    title: str | None = None


class DocumentSource(Protocol):
    def external_ids(self, docids: Sequence[int]) -> list[str]:
        """Maps index (internal) docids to external ids"""

    def get(self, ext_ids: Sequence[str]) -> list[Document | None]:
        """Returns documents given their external ids (None if unknown)"""


class ImpactDocumentSource:
    """An impact-index DocumentStore whose document numbers are index docids"""

    def __init__(self, config: CollectionConfig):
        import impact_index

        self.config = config
        # Memory-mapped: stores can be much larger than the index
        self.store = impact_index.DocumentStore.load(str(config.docstore_path), "mmap")
        self.id_key = config.id_key
        if self.id_key and self.id_key not in self.store.key_names():
            if self.store.key_names():
                raise ConfigError(
                    f"Collection {config.name!r}: the document store has no "
                    f"{self.id_key!r} key (keys: {', '.join(self.store.key_names())})"
                )
            # A store without keys: ids are document numbers (e.g. MS MARCO)
            self.id_key = None

    def external_ids(self, docids: Sequence[int]) -> list[str]:
        if self.id_key is None:
            return [str(docid) for docid in docids]
        docs = self.store.get_by_number(list(docids))
        return [doc.keys[self.id_key] for doc in docs]

    def get(self, ext_ids: Sequence[str]) -> list[Document | None]:
        if self.id_key is not None:
            docs = self.store.get_by_key(self.id_key, list(ext_ids))
            return [None if doc is None else self._convert(doc) for doc in docs]

        count = self.store.num_documents()
        numbers = {}
        for ext_id in ext_ids:
            if ext_id.isdigit() and int(ext_id) < count:
                numbers[ext_id] = int(ext_id)
        docs = dict(
            zip(
                numbers,
                self.store.get_by_number(list(numbers.values())),
                strict=True,
            )
        )
        return [
            self._convert(docs[ext_id]) if ext_id in docs else None
            for ext_id in ext_ids
        ]

    def _convert(self, doc) -> Document:
        keys = doc.keys
        content = doc.content.decode("utf-8", errors="replace")
        title_field = self.config.title_field
        title = keys.get(title_field) if title_field else None
        if self.config.content_format == "json" and content.startswith("{"):
            data = json.loads(content)
            text = "\n\n".join(
                str(data[f]) for f in self.config.text_fields if data.get(f)
            )
            if title_field and data.get(title_field):
                title = str(data[title_field])
        else:
            text = content
        docid = keys[self.id_key] if self.id_key else str(doc.internal_id)
        return Document(docid=docid, text=text, title=title)


class DatamaestroDocumentSource:
    """A datamaestro document store, indexed in its iteration order"""

    def __init__(self, documents, dataset_id: str = "documents"):
        missing = [
            name
            for name in ("docid_internal2external", "document_ext")
            if not hasattr(documents, name)
        ]
        if missing:
            raise ConfigError(
                f"{dataset_id} ({type(documents).__name__.split('.')[0]}) cannot "
                "fetch documents by id, so it cannot be used as a document "
                "source; build an impact-index document store instead "
                "(impact-explorer index) and set it as the document store path"
            )
        self.documents = documents

    @staticmethod
    def from_id(dataset_id: str) -> "DatamaestroDocumentSource":
        from datamaestro import prepare_dataset
        from datamaestro.context import get_dataset

        # Checks the type before preparing, which may download the dataset
        definition = get_dataset(dataset_id)
        DatamaestroDocumentSource(
            getattr(definition, "documents", definition), dataset_id
        )
        dataset = prepare_dataset(dataset_id)
        return DatamaestroDocumentSource(
            getattr(dataset, "documents", dataset), dataset_id
        )

    def external_ids(self, docids: Sequence[int]) -> list[str]:
        return [self.documents.docid_internal2external(docid) for docid in docids]

    def get(self, ext_ids: Sequence[str]) -> list[Document | None]:
        ext_ids = list(ext_ids)
        try:
            records = self.documents.documents_ext(ext_ids)
        except Exception:
            # Some id is unknown (stores raise on missing ids): one by one
            records = []
            for ext_id in ext_ids:
                try:
                    records.append(self.documents.document_ext(ext_id))
                except Exception:
                    records.append(None)
        return [
            None if record is None else record_to_document(record, ext_id)
            for ext_id, record in zip(ext_ids, records, strict=True)
        ]


def record_to_document(record, ext_id: str | None = None) -> Document:
    """Converts a datamaestro-ir document record (``IDTextRecord``)"""
    if ext_id is None:
        ext_id = record["id"]
    text_item = record.get("text_item")
    text = text_item.text if text_item is not None else ""
    # Some formats (MS MARCO, NFCorpus, ...) expose a separate title
    title = getattr(text_item, "title", None) or None
    return Document(docid=ext_id, text=text, title=title)


def document_source(config: CollectionConfig) -> DocumentSource:
    if config.docstore_path is not None:
        return ImpactDocumentSource(config)
    if config.documents:
        return DatamaestroDocumentSource.from_id(config.documents)
    raise ConfigError(
        f"Collection {config.name!r}: no document store found; set a docstore "
        "path or a datamaestro documents dataset"
    )
