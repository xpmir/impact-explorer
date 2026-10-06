"""Builds a BOW index + document store from datamaestro documents."""

import json
import logging
from collections.abc import Iterable
from pathlib import Path

from .documents import Document

logger = logging.getLogger(__name__)


def build_collection(
    documents: Iterable[Document],
    folder: Path,
    *,
    pipeline: str | None = "pyserini",
    stop_words: str | list[str] | None = None,
    positions: bool = False,
    log_every: int = 100_000,
) -> int:
    """Indexes documents into ``folder/index`` and ``folder/docstore``

    Index docids are the document numbers in the store. Contents are stored
    as JSON with ``title`` and ``text`` fields. Returns the number of
    documents.
    """
    import impact_index

    folder = Path(folder)
    (folder / "docstore").mkdir(parents=True, exist_ok=True)
    options = {"stop_words": stop_words, "positions": positions}
    if pipeline is not None:
        options["pipeline"] = pipeline
    builder = impact_index.BOWIndexBuilder(
        str(folder / "index"), dtype="int32", **options
    )
    store = impact_index.DocumentStoreBuilder(str(folder / "docstore"))

    count = 0
    for docid, document in enumerate(documents):
        content = {"text": document.text}
        if document.title:
            content["title"] = document.title
        store.add({"id": document.docid}, json.dumps(content).encode("utf-8"))
        indexed = (
            f"{document.title}\n{document.text}" if document.title else document.text
        )
        builder.add_text(docid, indexed)
        count = docid + 1
        if count % log_every == 0:
            logger.info("Indexed %d documents", count)

    store.build()
    builder.build(in_memory=False)
    return count


def datamaestro_documents(dataset_id: str) -> Iterable[Document]:
    from datamaestro import prepare_dataset

    from .documents import record_to_document

    dataset = prepare_dataset(dataset_id)
    documents = getattr(dataset, "documents", dataset)
    for record in documents.iter_documents():
        yield record_to_document(record)
