"""Command line: serve a workspace, or index a datamaestro collection."""

import argparse
import logging
from pathlib import Path


def serve(args):
    from nicegui import ui

    from .config import Workspace
    from .ui import Services, create_app

    workspace = Workspace(args.workspace)
    workspace.folder.mkdir(parents=True, exist_ok=True)
    create_app(Services(workspace))
    ui.run(
        host=args.host,
        port=args.port,
        title="impact-explorer",
        reload=False,
        show=args.show,
    )


def stop_words_option(value: str) -> str | list[str] | None:
    """Parses --stop-words: a family name, none, or a word list file"""
    if value in ("lucene", "terrier"):
        return value
    if value == "none":
        return []
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(
            f"{value!r} is neither lucene, terrier, none nor a file"
        )
    words = [w.strip() for w in path.read_text().splitlines()]
    return [w for w in words if w and not w.startswith("#")]


def index(args):
    from .config import CollectionConfig, Workspace
    from .indexer import build_collection, datamaestro_documents

    folder = Path(args.output)
    count = build_collection(
        datamaestro_documents(args.documents),
        folder,
        pipeline=args.pipeline,
        stop_words=args.stop_words,
        positions=args.positions,
    )
    logging.info("Indexed %d documents into %s", count, folder)

    if args.workspace:
        workspace = Workspace(args.workspace)
        workspace.put(
            CollectionConfig(
                name=args.name or folder.name,
                index=str((folder / "index").resolve()),
                docstore=str((folder / "docstore").resolve()),
                datasets=args.dataset or [],
            )
        )
        logging.info("Added collection to workspace %s", workspace.folder)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="impact-explorer")
    parser.add_argument("--debug", action="store_true")
    commands = parser.add_subparsers(required=True)

    p = commands.add_parser("serve", help="Starts the web interface")
    p.add_argument(
        "workspace",
        nargs="?",
        default=".",
        help="Workspace folder (settings and saved queries; created if needed)",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--show", action="store_true", help="Opens a browser")
    p.set_defaults(func=serve)

    p = commands.add_parser(
        "index", help="Builds a BOW index + document store from datamaestro documents"
    )
    p.add_argument("documents", help="datamaestro dataset id (documents or Adhoc)")
    p.add_argument("output", help="Output folder (index/ and docstore/ inside)")
    p.add_argument(
        "--pipeline",
        default="pyserini",
        choices=["pyserini", "terrier", "terrier-pisa"],
        help="impact-index text pipeline",
    )
    p.add_argument(
        "--stop-words",
        default=None,
        type=stop_words_option,
        help="Stop list: lucene (33 words), terrier (733 words), none, or a "
        "file with one word per line (default: the pipeline's own; "
        "terrier-pisa filters queries only)",
    )
    p.add_argument(
        "--positions",
        action="store_true",
        help="Stores token positions (needed for #1 and #uwN queries)",
    )
    p.add_argument(
        "--workspace", help="Also registers the collection in this workspace"
    )
    p.add_argument("--name", help="Collection name (default: output folder name)")
    p.add_argument(
        "--dataset",
        action="append",
        help="datamaestro IR dataset providing topics/assessments (repeatable)",
    )
    p.set_defaults(func=index)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    args.func(args)


if __name__ == "__main__":
    main()
