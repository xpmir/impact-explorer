"""Index builds: datamaestro documents -> document store + BOW index.

A build runs in its own process (``impact-explorer build <workspace>
<name>``), so that it survives a restart of the interface and a crash does
not take the interface down. Its state is a JSON file that the runner
updates and the interface reads::

    <workspace>/builds/<name>.json   specification and state
    <workspace>/builds/<name>.lock   held (flock) while the runner is alive
    <workspace>/builds/<name>.log    runner output

The build is a sequence of stages; a resumed build skips completed stages:

- ``prepare``: downloads the datamaestro documents;
- ``docstore``: copies them into an impact-index document store, which
  checkpoints regularly, so an interrupted copy resumes from its last
  checkpoint; skipped when datamaestro already stores the documents in an
  impact-index document store (it is used as is);
- ``index``: builds the BOW index from the (local) document store; an
  interrupted index restarts from scratch (impact-index does not
  checkpoint the vocabulary and document lengths of BOW indexes), but
  never downloads or copies the documents again;
- ``compress`` (optional): compresses the BOW index (block-max bitpacked
  postings, lossless), then deletes the uncompressed one.

The interface registers the collection in the workspace once the build is
complete (the runner never writes ``workspace.json``, which the interface
keeps in memory).
"""

import contextlib
import fcntl
import io
import itertools
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import NAME_RE, CollectionConfig, ConfigError, atomic_write_json
from .documents import Document

logger = logging.getLogger(__name__)

BUILDS_FOLDER = "builds"
STAGES = ("prepare", "docstore", "index", "compress")
STAGE_LABELS = {
    "prepare": "Download documents",
    "docstore": "Document store",
    "index": "BOW index",
    "compress": "Compression",
}
INDEX_STAGES = ("index", "compress")
"""Stages run again when only the index is rebuilt"""
PIPELINES = ("pyserini", "terrier", "terrier-pisa")
STOP_WORDS = ("default", "lucene", "terrier", "none")
"""``default``: the pipeline's own list"""

CHECKPOINT_FREQUENCY = 50_000
"""Documents between two document store checkpoints"""

BATCH_SIZE = 4096
"""Documents indexed at once (tokenized in parallel)"""

SAVE_INTERVAL = 2.0
"""Seconds between two progress updates of the state file"""

LOG_RUN_MARKER = "--- "
"""Starts the log lines of a run"""


class Cancelled(Exception):
    pass


@dataclass
class BuildSpec:
    name: str
    """Build name, also the name of the registered collection"""

    documents: str
    """datamaestro dataset id of the documents"""

    output: str = ""
    """Output folder (``index`` and ``docstore`` inside; relative paths are
    resolved against the workspace; default: ``indexes/<name>``)"""

    pipeline: str = "pyserini"
    stop_words: str = "default"
    positions: bool = False
    """Stores token positions (for ``#1`` and ``#uwN`` queries)"""

    compress: bool = False
    """Compresses the index (and deletes the uncompressed one)"""

    datasets: list[str] = field(default_factory=list)
    """datamaestro IR datasets of the registered collection (topics and
    assessments)"""

    def validate(self):
        if not NAME_RE.match(self.name or ""):
            raise ConfigError(
                f"Invalid name {self.name!r} (letters, digits, '.', '_' and '-' only)"
            )
        if not self.documents:
            raise ConfigError("The documents dataset id is required")
        if self.pipeline not in PIPELINES:
            raise ConfigError(f"pipeline must be one of {', '.join(PIPELINES)}")
        if self.stop_words not in STOP_WORDS:
            raise ConfigError(f"stop words must be one of {', '.join(STOP_WORDS)}")

    def output_path(self, workspace: Path) -> Path:
        path = Path(self.output or f"indexes/{self.name}").expanduser()
        return path if path.is_absolute() else workspace / path

    def stages(self) -> tuple[str, ...]:
        return tuple(s for s in STAGES if s != "compress" or self.compress)

    def options_text(self) -> str:
        options = [self.pipeline, f"stop words {self.stop_words}"]
        if self.positions:
            options.append("positions")
        if self.compress:
            options.append("compressed")
        return ", ".join(options)

    def builder_options(self) -> dict:
        stop_words = {"default": None, "none": []}.get(self.stop_words, self.stop_words)
        return {
            "pipeline": self.pipeline,
            "stop_words": stop_words,
            "positions": self.positions,
        }


@dataclass
class StageState:
    status: str = "pending"
    """pending, running, done or failed (a running stage whose runner is
    gone was interrupted)"""

    done: int = 0
    """Documents processed"""

    total: int | None = None
    resumed_from: int = 0
    """Documents already processed when the current run started"""

    started: float | None = None
    """Start of the current run"""

    finished: float | None = None

    def rate(self, now: float | None = None) -> float | None:
        """Documents per second in the current run"""
        if self.started is None:
            return None
        elapsed = (self.finished or now or time.time()) - self.started
        processed = self.done - self.resumed_from
        return processed / elapsed if elapsed > 0 and processed > 0 else None


@dataclass
class BuildState:
    spec: BuildSpec
    stages: dict[str, StageState] = field(
        default_factory=lambda: {stage: StageState() for stage in STAGES}
    )
    pid: int | None = None
    error: str | None = None
    traceback: str | None = None
    cancelled: bool = False
    registered: bool = False
    """The collection has been added to the workspace"""

    datamaestro_store: str | None = None
    """datamaestro's own document store, used instead of a copy"""

    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)

    @property
    def complete(self) -> bool:
        return all(self.stages[s].status == "done" for s in self.spec.stages())

    def current_stage(self) -> str | None:
        for name in self.spec.stages():
            if self.stages[name].status != "done":
                return name
        return None

    def status(self, alive: bool) -> str:
        """running, done, failed, cancelled, interrupted or pending"""
        if alive:
            return "running"
        if self.complete:
            return "done"
        if self.error:
            return "failed"
        if self.cancelled:
            return "cancelled"
        if any(
            self.stages[s].status in ("running", "done") for s in self.spec.stages()
        ):
            return "interrupted"
        return "pending"

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "BuildState":
        data = dict(data)
        spec = BuildSpec(**data.pop("spec"))
        stages = {name: StageState() for name in STAGES}
        stages.update({k: StageState(**v) for k, v in data.pop("stages", {}).items()})
        return BuildState(spec=spec, stages=stages, **data)


class BuildFiles:
    """The files of a build in a workspace"""

    def __init__(self, workspace: Path, name: str):
        self.workspace = Path(workspace)
        self.name = name
        folder = self.workspace / BUILDS_FOLDER
        self.state = folder / f"{name}.json"
        self.lock = folder / f"{name}.lock"
        self.log = folder / f"{name}.log"

    def load(self) -> BuildState:
        return BuildState.from_dict(json.loads(self.state.read_text()))

    def save(self, state: BuildState):
        state.updated = time.time()
        atomic_write_json(self.state, state.to_dict())

    def alive(self) -> bool:
        """Whether a runner holds the lock"""
        if not self.lock.exists():
            return False
        with open(self.lock, "a") as fp:
            try:
                fcntl.flock(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(fp, fcntl.LOCK_UN)
            return False

    def log_tail(self, lines: int = 30) -> str:
        """The end of the log of the last run (earlier runs are not shown)"""
        if not self.log.exists():
            return ""
        with open(self.log, "rb") as fp:
            fp.seek(max(0, fp.seek(0, os.SEEK_END) - 16384))
            text = fp.read().decode("utf-8", errors="replace")
        text = text[text.rfind(LOG_RUN_MARKER) :] if LOG_RUN_MARKER in text else text
        return "\n".join(text.splitlines()[-lines:])


# --- Runner (in the build process)


def document_content(document: Document) -> bytes:
    """Document store content: JSON with ``text`` (and ``title``)"""
    content = {"text": document.text}
    if document.title:
        content["title"] = document.title
    return json.dumps(content).encode("utf-8")


def indexed_text(document: Document) -> str:
    if document.title:
        return f"{document.title}\n{document.text}"
    return document.text


def stored_text(content: bytes) -> str:
    """The indexed text of a :func:`document_content`"""
    data = json.loads(content)
    return indexed_text(Document("", data.get("text", ""), data.get("title")))


class Runner:
    """Runs the remaining stages of a build"""

    def __init__(self, workspace: Path, name: str):
        self.files = BuildFiles(workspace, name)
        self.state = self.files.load()
        self.spec = self.state.spec
        self.output = self.spec.output_path(self.files.workspace)
        self._dataset = None
        self._last_save = 0.0
        self._save_lock = threading.Lock()

    def save(self, force: bool = True):
        now = time.time()
        if force or now - self._last_save >= SAVE_INTERVAL:
            with self._save_lock:
                self.files.save(self.state)
            self._last_save = now

    def run(self):
        self.files.lock.parent.mkdir(parents=True, exist_ok=True)
        with open(self.files.lock, "a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(
                    f"Build {self.spec.name!r} is already running"
                ) from None
            self._run()

    def _run(self):
        def on_term(*_):
            raise Cancelled()

        signal.signal(signal.SIGTERM, on_term)
        self.state.pid = os.getpid()
        self.state.error = self.state.traceback = None
        self.state.cancelled = False
        self.save()
        stage = None
        try:
            for stage in self.spec.stages():
                if self.state.stages[stage].status == "done":
                    logger.info("Stage %s already done", stage)
                    continue
                current = self.state.stages[stage]
                current.status = "running"
                current.started, current.finished = time.time(), None
                current.resumed_from = current.done
                self.save()
                logger.info("Stage %s", stage)
                getattr(self, f"stage_{stage}")(current)
                current.status, current.finished = "done", time.time()
                self.save()
            logger.info("Build %s complete", self.spec.name)
        except Cancelled:
            logger.info("Build %s cancelled", self.spec.name)
            self.state.cancelled = True
            self.state.stages[stage].status = "pending"
            self.save()
        except BaseException as e:
            logger.exception("Build %s failed", self.spec.name)
            self.state.error = f"{type(e).__name__}: {e}"
            self.state.traceback = traceback.format_exc()
            self.state.stages[stage].status = "failed"
            self.save()
            raise

    def dataset(self):
        if self._dataset is None:
            from datamaestro import prepare_dataset

            download(self.spec.documents)
            dataset = prepare_dataset(self.spec.documents)
            self._dataset = getattr(dataset, "documents", dataset)
        return self._dataset

    def stage_prepare(self, stage: StageState):
        documents = self.dataset()
        self.state.datamaestro_store = None
        if (store := datamaestro_store(documents)) is not None:
            logger.info("Using datamaestro's document store %s", store)
            self.state.datamaestro_store = str(store)
        try:
            # A property or a method, depending on the dataset type
            count = documents.documentcount
            self.state.stages["docstore"].total = int(
                count() if callable(count) else count
            )
        except Exception:
            logger.info("Unknown number of documents")

    def documents_from(self, start: int) -> Iterator[Document]:
        from .documents import record_to_document

        documents = self.dataset()
        if hasattr(documents, "iter_documents_from"):
            records = documents.iter_documents_from(start)
        else:
            records = itertools.islice(documents.iter_documents(), start, None)
        for record in records:
            yield record_to_document(record)

    def stage_docstore(self, stage: StageState):
        import impact_index

        if self.state.datamaestro_store:
            # Nothing to copy
            stage.done = stage.total = len(self.store())
            return
        folder = self.output / "docstore"
        if stage.resumed_from == 0 and stage.done == 0:
            # A first run: drop leftovers of a removed build
            shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True, exist_ok=True)
        builder = impact_index.DocumentStoreBuilder(
            str(folder), checkpoint_frequency=CHECKPOINT_FREQUENCY
        )
        # Documents after the last checkpoint are copied again
        stage.done = stage.resumed_from = builder.num_documents()
        if stage.done:
            logger.info("Resuming the document store at %d documents", stage.done)
        self.save()
        for document in self.documents_from(stage.done):
            builder.add({"id": document.docid}, document_content(document))
            stage.done += 1
            self.save(force=False)
        builder.build()
        stage.total = stage.done

    def stage_index(self, stage: StageState):
        import impact_index

        store = self.store()
        folder = self.bow_folder()
        # BOW indexes cannot resume: start again from the document store
        shutil.rmtree(folder, ignore_errors=True)
        stage.total, stage.done, stage.resumed_from = len(store), 0, 0
        stage.started = time.time()
        self.save()
        builder = impact_index.BOWIndexBuilder(
            str(folder), dtype="int32", **self.spec.builder_options()
        )
        for start in range(0, stage.total, BATCH_SIZE):
            texts = store.texts(start, min(start + BATCH_SIZE, stage.total))
            builder.add_texts(list(enumerate(texts, start=start)))
            stage.done = start + len(texts)
            self.save(force=False)
        logger.info("Writing the index")
        self.save()
        builder.build(in_memory=False)

    def store(self) -> "LocalStore | DatamaestroStore":
        """The documents to index (document number = index docid)"""
        if self.state.datamaestro_store:
            return DatamaestroStore(self.dataset())
        return LocalStore(self.output / "docstore")

    def bow_folder(self) -> Path:
        """The (uncompressed) BOW index"""
        return self.output / ("index.raw" if self.spec.compress else "index")

    def stage_compress(self, stage: StageState):
        import impact_index

        raw, folder = self.bow_folder(), self.output / "index"
        if not raw.exists() and folder.exists():
            # Interrupted after the compressed index replaced the raw one
            return
        tmp = self.output / "index.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        index = impact_index.Index.load(str(raw), False)
        stage.total = stage.done = 0
        self.save()
        logger.info("Compressing the index")
        index.compress(str(tmp), in_memory=False)
        del index
        shutil.rmtree(folder, ignore_errors=True)
        tmp.rename(folder)
        shutil.rmtree(raw)


class LocalStore:
    """The document store copied by the build"""

    def __init__(self, folder: Path):
        import impact_index

        self.store = impact_index.DocumentStore.load(str(folder), "mmap")

    def __len__(self):
        return self.store.num_documents()

    def texts(self, start: int, end: int) -> list[str]:
        documents = self.store.get_by_number(list(range(start, end)))
        return [stored_text(doc.content) for doc in documents]


class DatamaestroStore:
    """The impact-index document store of a datamaestro dataset"""

    def __init__(self, documents):
        self.documents = documents

    def __len__(self):
        return self.documents._store.num_documents()

    def texts(self, start: int, end: int) -> list[str]:
        from .documents import record_to_document

        records = itertools.islice(
            self.documents.iter_documents_from(start), end - start
        )
        return [indexed_text(record_to_document(record)) for record in records]


def datamaestro_store(documents) -> Path | None:
    """The folder of the impact-index document store holding datamaestro
    documents, if any: its document numbers follow the iteration order,
    and it can fetch documents by id (no need for a copy)"""
    try:
        from datamaestro_ir.data import CompressedDocumentStore
    except ImportError:
        return None
    if not isinstance(documents, CompressedDocumentStore):
        return None
    path = Path(documents.path)
    return path if path.is_dir() else None


class DownloadError(RuntimeError):
    pass


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def download(dataset_id: str):
    """Downloads a datamaestro dataset (and the datasets it references)

    datamaestro only logs download failures (``prepare_dataset`` returns the
    dataset anyway): they are raised here, with their cause (the last line
    of the traceback datamaestro prints)."""
    from datamaestro.context import Context

    try:
        wrapper = Context.instance().dataset(dataset_id)
    except Exception:
        raise DownloadError(f"Unknown datamaestro dataset {dataset_id!r}") from None
    captured = io.StringIO()
    with contextlib.redirect_stderr(_Tee(sys.stderr, captured)):
        success = wrapper.download()
    if not success:
        lines = [line for line in captured.getvalue().splitlines() if line.strip()]
        cause = f": {lines[-1].strip()}" if lines else " (see the log)"
        raise DownloadError(f"Could not download {dataset_id}{cause}")


INDEX_FOLDERS = ("index", "index.raw", "index.tmp")


def delete_output(output: Path):
    """Deletes the output of a build"""
    for sub in INDEX_FOLDERS + ("docstore",):
        shutil.rmtree(output / sub, ignore_errors=True)
    with contextlib.suppress(OSError):
        output.rmdir()


def run_build(workspace: Path, name: str):
    Runner(workspace, name).run()


# --- Manager (in the interface)


class Builds:
    """Builds of a workspace: creation, launching, state, registration"""

    def __init__(self, workspace):
        self.workspace = workspace
        self.folder = workspace.folder / BUILDS_FOLDER
        self._lock = threading.Lock()
        self._listeners = []

    def files(self, name: str) -> BuildFiles:
        return BuildFiles(self.workspace.folder, name)

    def names(self) -> list[str]:
        if not self.folder.is_dir():
            return []
        return sorted(p.stem for p in self.folder.glob("*.json"))

    def get(self, name: str) -> tuple[BuildState, bool]:
        """The state of a build, and whether its runner is alive"""
        files = self.files(name)
        return files.load(), files.alive()

    def create(self, spec: BuildSpec, start: bool = True) -> BuildState:
        spec.validate()
        if spec.name in self.workspace.collections:
            raise ConfigError(f"A collection named {spec.name!r} already exists")
        if self.files(spec.name).state.exists():
            raise ConfigError(f"A build named {spec.name!r} already exists")
        state = BuildState(spec=spec)
        self.files(spec.name).save(state)
        if start:
            self.start(spec.name)
        return state

    def start(self, name: str):
        """Launches (or resumes) a build in a new process"""
        files = self.files(name)
        if files.alive():
            raise ConfigError(f"Build {name!r} is already running")
        files.log.parent.mkdir(parents=True, exist_ok=True)
        with open(files.log, "a") as log:
            log.write(f"{LOG_RUN_MARKER}{time.strftime('%Y-%m-%d %H:%M:%S')} start\n")
            log.flush()
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "impact_explorer.cli",
                    "build",
                    str(self.workspace.folder),
                    name,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                # Survives a restart of the interface
                start_new_session=True,
            )

    def cancel(self, name: str):
        state, alive = self.get(name)
        if alive and state.pid:
            os.kill(state.pid, signal.SIGTERM)

    def reset(
        self, name: str, spec: BuildSpec | None = None, keep_documents: bool = False
    ):
        """Forgets the progress and deletes the output (restart from scratch)

        :param spec: new options (same name and documents)
        :param keep_documents: keeps the document store (only the index is
            built again)
        """
        files = self.files(name)
        if files.alive():
            raise ConfigError(f"Build {name!r} is running")
        state = files.load()
        if spec is not None:
            if (spec.name, spec.documents) != (state.spec.name, state.spec.documents):
                raise ConfigError("The name and documents of a build cannot change")
            spec.validate()
        else:
            spec = state.spec
        output = state.spec.output_path(self.workspace.folder)
        kept = {}
        if keep_documents and state.stages["docstore"].status == "done":
            kept = {s: state.stages[s] for s in STAGES if s not in INDEX_STAGES}
        subs = INDEX_FOLDERS + (() if kept else ("docstore",))
        for sub in subs:
            shutil.rmtree(output / sub, ignore_errors=True)
        new = BuildState(spec=spec, created=state.created)
        new.stages.update(kept)
        if kept:
            new.datamaestro_store = state.datamaestro_store
        files.save(new)

    def remove(self, name: str, delete_files: bool = False):
        """Forgets a build, and deletes its output if ``delete_files``"""
        files = self.files(name)
        if files.alive():
            raise ConfigError(f"Build {name!r} is running")
        if delete_files:
            delete_output(files.load().spec.output_path(self.workspace.folder))
        for path in (files.state, files.lock, files.log):
            path.unlink(missing_ok=True)

    def collection_files(self, name: str) -> list[Path]:
        """Files of a collection (index, document store): existing folders"""
        collection = self.workspace.collection(name)
        folders = [collection.index_path]
        if collection.docstore and collection.docstore_path is not None:
            folders.append(collection.docstore_path)
        return [f for f in folders if f.exists()]

    def remove_collection(self, name: str, delete_files: bool):
        """Removes a collection from the workspace, with its build, and
        deletes its files if ``delete_files``"""
        folders = self.collection_files(name) if delete_files else []
        for build in self.names():
            if build == name:
                self.remove(build, delete_files=delete_files)
        self.workspace.remove(name)
        for folder in folders:
            shutil.rmtree(folder, ignore_errors=True)
            with contextlib.suppress(OSError):
                # The output folder of a build, now empty
                folder.parent.rmdir()

    def collection(self, state: BuildState) -> CollectionConfig:
        spec = state.spec
        output = spec.output_path(self.workspace.folder)
        try:
            # Relative to the workspace when inside it (movable workspace)
            output = output.resolve().relative_to(self.workspace.folder)
        except ValueError:
            pass
        if state.datamaestro_store:
            # Documents fetched through datamaestro
            return CollectionConfig(
                name=spec.name,
                index=str(output / "index"),
                documents=spec.documents,
                datasets=list(spec.datasets),
            )
        return CollectionConfig(
            name=spec.name,
            index=str(output / "index"),
            docstore=str(output / "docstore"),
            datasets=list(spec.datasets),
        )

    def on_registered(self, listener):
        """Registers ``listener(name)``, called when a build is registered"""
        self._listeners.append(listener)

    def register_completed(self) -> list[str]:
        """Adds the collections of completed builds to the workspace"""
        registered = []
        with self._lock:
            for name in self.names():
                files = self.files(name)
                try:
                    state = files.load()
                except (OSError, ValueError, TypeError):
                    continue
                if not state.complete or state.registered or files.alive():
                    continue
                self.workspace.put(self.collection(state))
                state.registered = True
                files.save(state)
                registered.append(name)
                logger.info("Registered collection %s", name)
        for name in registered:
            for listener in self._listeners:
                listener(name)
        return registered

    def watch(self, interval: float = 2.0):
        """Registers completed builds in the background"""

        def loop():
            while True:
                try:
                    self.register_completed()
                except Exception:
                    logger.exception("Could not register completed builds")
                time.sleep(interval)

        threading.Thread(target=loop, daemon=True, name="builds").start()


def iter_states(builds: Builds) -> Iterable[tuple[str, BuildState, bool]]:
    for name in builds.names():
        try:
            state, alive = builds.get(name)
        except (OSError, ValueError, TypeError):
            continue
        yield name, state, alive
