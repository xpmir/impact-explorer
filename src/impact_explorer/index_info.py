"""What an index is made of, read from its own files (no loading)

- ``manifest.json``: kind, format, codecs, block size, features;
- ``analyzer.cbor``: text analysis (tokenizer, stemmer, stop words), from
  which the build pipeline is recognized;
- ``information.cbor``: value type of a forward index;
- ``docmeta.cbor``: number of documents.
"""

import json
from dataclasses import dataclass
from pathlib import Path

KINDS = {
    "forward": "forward (raw postings, uncompressed)",
    "compressed": "compressed (block-compressed postings)",
    "split": "split (impact quantiles over an inner index)",
    "seismic": "seismic (approximate)",
}

DTYPES = {"I32": "int32", "I64": "int64", "F32": "float32"}

PIPELINES = {
    # name: (tokenizer, stemmer, index stop words, query-only stop words,
    # position gaps), as BOWIndexBuilder defines them
    "pyserini": ("lucene-english", "porter", "lucene", None, True),
    "terrier": ("pisa-english", "snowball", "terrier", None, False),
    "terrier-pisa": ("pisa-english", "snowball", None, "terrier", False),
}


@dataclass
class Detail:
    section: str
    name: str
    value: str


def _cbor(path: Path):
    import cbor2

    with path.open("rb") as f:
        return cbor2.load(f)


def _size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def recognize_pipeline(analyzer: dict) -> tuple[str | None, list[str]]:
    """The pipeline an analyzer was built with, and what was overridden"""
    tokenizer = analyzer.get("tokenizer")
    has_stops = bool(analyzer.get("stop_words_list"))
    query_stops = bool(analyzer.get("query_stop_words_list"))
    if tokenizer == "lucene-english":
        name = "pyserini"
    elif tokenizer == "pisa-english":
        name = "terrier-pisa" if not has_stops and query_stops else "terrier"
    else:
        return None, []
    _, stemmer, family, _, gaps = PIPELINES[name]
    overrides = []
    if analyzer.get("stemmer") != stemmer:
        overrides.append(f"stemmer {analyzer.get('stemmer')} (default {stemmer})")
    actual = analyzer.get("stop_words_family") if has_stops else None
    if name != "terrier-pisa" and actual != family:
        overrides.append(
            f"stop words {actual or ('custom' if has_stops else 'none')} "
            f"(default {family})"
        )
    if "position_gaps" in analyzer and analyzer["position_gaps"] != gaps:
        overrides.append(f"position gaps {analyzer['position_gaps']}")
    return name, overrides


def describe_analyzer(analyzer: dict) -> list[Detail]:
    details = []
    name, overrides = recognize_pipeline(analyzer)
    if name:
        value = name + (f" — overrides: {'; '.join(overrides)}" if overrides else "")
        details.append(Detail("Text analysis", "Pipeline", value))
    details.append(
        Detail("Text analysis", "Tokenizer", analyzer.get("tokenizer", "standard"))
    )
    stemmer = analyzer.get("stemmer", "?")
    if stemmer == "snowball":
        stemmer += f" ({analyzer.get('language', '?')})"
    details.append(Detail("Text analysis", "Stemmer", stemmer))
    stops = analyzer.get("stop_words_list") or []
    if stops:
        family = analyzer.get("stop_words_family") or "custom"
        mode = analyzer.get("stop_words_filter_mode")
        value = f"{family}, {len(stops)} words" + (f", checked {mode}" if mode else "")
    else:
        value = "none (every word is indexed)"
    details.append(Detail("Text analysis", "Stop words (index)", value))
    query_stops = analyzer.get("query_stop_words_list") or []
    if query_stops:
        details.append(
            Detail(
                "Text analysis",
                "Stop words (queries only)",
                f"{len(query_stops)} words",
            )
        )
    for key, label in (
        ("english_possessive_filter", "Possessive filter ('s)"),
        ("position_gaps", "Position gaps for stop words"),
    ):
        if key in analyzer:
            details.append(
                Detail("Text analysis", label, "yes" if analyzer[key] else "no")
            )
    return details


def index_details(path: Path) -> list[Detail]:
    """Everything known about the index at ``path`` (missing files are
    skipped)"""
    path = Path(path)
    details: list[Detail] = []
    try:
        manifest = json.loads((path / "manifest.json").read_text())
    except (OSError, ValueError):
        manifest = {}
    kind = manifest.get("index_kind")
    builder = manifest.get("builder") or {}
    if kind:
        details.append(Detail("Index", "Kind", KINDS.get(kind, kind)))
    if "format_version" in manifest:
        details.append(
            Detail("Index", "Format version", str(manifest["format_version"]))
        )
    if builder.get("library_version"):
        details.append(
            Detail("Index", "Built with", f"impact-index {builder['library_version']}")
        )
    if manifest.get("created"):
        details.append(Detail("Index", "Created", manifest["created"]))
    features = manifest.get("features") or []
    details.append(
        Detail(
            "Index",
            "Positions",
            "yes (phrases and windows)" if "positions" in features else "no",
        )
    )
    other = [f for f in features if f != "positions"]
    if other:
        details.append(Detail("Index", "Other features", ", ".join(other)))

    # Storage
    if builder.get("codecs"):
        details.append(Detail("Storage", "Codecs", builder["codecs"]))
    elif kind == "forward":
        details.append(Detail("Storage", "Codecs", "none (raw postings)"))
    if builder.get("block_size"):
        details.append(Detail("Storage", "Block size", str(builder["block_size"])))
    try:
        info = _cbor(path / "information.cbor")
        dtype = info[0] if isinstance(info, list) and info else None
        if isinstance(dtype, str):
            details.append(
                Detail(
                    "Storage",
                    "Values",
                    f"{DTYPES.get(dtype, dtype)} (term frequencies; BM25 is "
                    "computed at query time)",
                )
            )
    except Exception:
        pass
    if kind == "split" and (path / "inner").is_dir():
        for detail in index_details(path / "inner"):
            if detail.section in ("Index", "Storage"):
                details.append(Detail("Inner index", detail.name, detail.value))
    try:
        details.append(Detail("Storage", "Size on disk", human_size(_size(path))))
    except OSError:
        pass

    try:
        docmeta = _cbor(path / "docmeta.cbor")
        details.append(Detail("Index", "Documents", f"{docmeta['num_docs']:,}"))
    except Exception:
        pass

    try:
        details.extend(describe_analyzer(_cbor(path / "analyzer.cbor")))
    except Exception:
        pass
    order = ["Index", "Storage", "Inner index", "Text analysis"]
    return sorted(details, key=lambda d: order.index(d.section))


def summary(path: Path) -> str:
    """One line: pipeline and storage"""
    parts = []
    values = {(d.section, d.name): d.value for d in index_details(path)}
    if pipeline := values.get(("Text analysis", "Pipeline")):
        parts.append(f"pipeline {pipeline.split(' — ')[0]}")
    if codecs := values.get(("Storage", "Codecs")):
        parts.append(f"codecs {codecs}")
    return " · ".join(parts)
