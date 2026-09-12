"""Samyak command-line interface.

Current commands:
  samyak corpus <path>          Corpus Intelligence analysis
  samyak view                   Local workspace (run history and model catalog)
  samyak model update openai    Refresh the local OpenAI model catalog
  samyak model update anthropic Refresh the local Anthropic model catalog
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from samyak.__version__ import __version__
from samyak.corpus.config import AnalysisConfig
from samyak.corpus.discovery import CorpusPathError
from samyak.corpus.pipeline import analyze_corpus, default_progress_callback
from samyak.corpus.report import REPORT_FORMATS, render_report
from samyak.model.update import (
    SUPPORTED_PROVIDERS,
    CatalogUpdateResult,
    update_anthropic_catalog,
    update_openai_catalog,
)
from samyak.server.app import DEFAULT_PORT, ViewerBindError, serve_viewer
from samyak.store.filesystem import FileRunStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="samyak",
        description=(
            "Samyak by DataAIHub — AI engineering tooling for building reliable AI "
            "applications. Local by default; `samyak model update` is the only network "
            "command. No API keys or accounts required."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")

    defaults = AnalysisConfig()
    corpus = subparsers.add_parser(
        "corpus",
        help="Analyze a local document corpus (Corpus Intelligence)",
        description=(
            "Analyze your document corpus before it becomes a RAG problem. "
            "Corpus Intelligence inspects document collections used in AI and RAG systems."
        ),
    )
    corpus.add_argument(
        "path",
        help="Path to a local directory containing documents to analyze",
    )
    corpus.add_argument(
        "--format",
        choices=REPORT_FORMATS,
        default="text",
        help="Output format: text, json, or html (default: text)",
    )
    corpus.add_argument(
        "--output",
        "-o",
        help="Optional path to write the report (default: stdout)",
    )
    corpus.add_argument(
        "--small-chars",
        type=int,
        default=None,
        help=(
            "Very-small document threshold in characters "
            f"(default: {defaults.small_document_chars})"
        ),
    )
    corpus.add_argument(
        "--large-chars",
        type=int,
        default=None,
        help=(
            "Very-large document threshold in characters "
            f"(default: {defaults.large_document_chars})"
        ),
    )
    corpus.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress messages on stderr",
    )
    corpus.add_argument(
        "--save",
        action="store_true",
        help="Save this run to the local Samyak cache for `samyak view`",
    )
    corpus.set_defaults(handler=_run_corpus)

    view = subparsers.add_parser(
        "view",
        help="Open the local Samyak workspace",
        description=(
            "Start a local web viewer for saved corpus analysis runs and the "
            "model catalog. Browse Model Intelligence after "
            "`samyak model update openai` or `samyak model update anthropic`. "
            "The server binds to 127.0.0.1. Data stays on this machine."
        ),
    )
    view.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to listen on (default: {DEFAULT_PORT})",
    )
    view.add_argument(
        "--no-open",
        action="store_true",
        help="Print the URL but do not open a browser",
    )
    view.set_defaults(handler=_run_view)

    model = subparsers.add_parser(
        "model",
        help="Local model catalog (Model Intelligence)",
        description=(
            "Refresh the local model catalog from official documentation. "
            "Browse the catalog in `samyak view`. "
            "`samyak model update` is the network command. "
            "No API key is required."
        ),
    )
    model.set_defaults(handler=_run_model, model_parser=model)
    model_sub = model.add_subparsers(dest="model_command")
    update = model_sub.add_parser(
        "update",
        help="Refresh the local catalog from official documentation",
        description=(
            "Fetch official provider documentation and merge that provider into "
            "the local model catalog overlay. No API key is required. "
            "This command uses the network."
        ),
    )
    update.add_argument(
        "provider",
        metavar="PROVIDER",
        help="Provider to refresh (openai or anthropic)",
    )
    update.set_defaults(handler=_run_model_update)

    return parser


def _run_corpus(args: argparse.Namespace) -> int:
    defaults = AnalysisConfig()
    try:
        config = AnalysisConfig(
            small_document_chars=(
                args.small_chars if args.small_chars is not None else defaults.small_document_chars
            ),
            large_document_chars=(
                args.large_chars if args.large_chars is not None else defaults.large_document_chars
            ),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    progress_callback = None
    if not args.no_progress and config.progress_every > 0:
        progress_callback = functools.partial(
            default_progress_callback,
            every=config.progress_every,
            stream=sys.stderr,
        )

    created_at = datetime.now(UTC)
    started = time.perf_counter()
    try:
        report = analyze_corpus(
            args.path,
            config=config,
            progress_callback=progress_callback,
        )
    except CorpusPathError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    completed_at = datetime.now(UTC)
    duration = time.perf_counter() - started

    save_failed = False
    if args.save:
        corpus_label, corpus_path = _local_corpus_location(args.path)
        try:
            stored = FileRunStore().save_run(
                report,
                duration_seconds=duration,
                created_at=created_at,
                completed_at=completed_at,
                corpus_label=corpus_label,
                corpus_path=corpus_path,
            )
        except OSError as exc:
            print(f"error: failed to save run: {exc}", file=sys.stderr)
            save_failed = True
        else:
            print(f"Saved run {stored.run_id}. View with: samyak view", file=sys.stderr)

    rendered = render_report(report, args.format)

    if args.output:
        output_path = Path(args.output)
        try:
            output_path.write_text(rendered, encoding="utf-8")
        except OSError as exc:
            print(f"error: failed to write output: {exc}", file=sys.stderr)
            return 2
    else:
        sys.stdout.write(rendered)

    return 2 if save_failed else 0


def _local_corpus_location(raw: str) -> tuple[str, str]:
    """Return (basename, absolute path) without extra symlink resolution.

    Analysis already resolves the corpus root to read files. The saved path is
    the local location the user asked to analyze, made absolute so two folders
    with the same name can be distinguished later.
    """
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = path.absolute()
    return path.name, str(path)


def _run_view(args: argparse.Namespace) -> int:
    if not 1 <= args.port <= 65535:
        print("error: port must be between 1 and 65535", file=sys.stderr)
        return 2

    store = FileRunStore()
    try:
        serve_viewer(
            store,
            host="127.0.0.1",
            port=args.port,
            open_browser=not args.no_open,
        )
    except ViewerBindError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def _run_model(args: argparse.Namespace) -> int:
    parser = getattr(args, "model_parser", None)
    if parser is not None:
        parser.print_help()
    return 2


def _run_model_update(args: argparse.Namespace) -> int:
    provider = str(args.provider).strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        supported = ", ".join(SUPPORTED_PROVIDERS)
        print(f"error: unknown provider; supported: {supported}", file=sys.stderr)
        return 2
    updater = update_openai_catalog if provider == "openai" else update_anthropic_catalog
    label = "OpenAI" if provider == "openai" else "Anthropic"
    try:
        result = updater()
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not result.ok or not result.committed:
        print(f"error: {result.error or f'{label} catalog update failed'}", file=sys.stderr)
        if result.previous_existed:
            print("The existing model catalog was not changed.", file=sys.stderr)
        else:
            print("No local catalog was written.", file=sys.stderr)
        return 2
    sys.stdout.write(_render_update_result(result, label=label))
    return 0


def _render_update_result(result: CatalogUpdateResult, *, label: str) -> str:
    status = "partial" if result.partial else "complete"
    lines = [
        f"Updated {label} model catalog.",
        f"Models: {result.provider_model_count}",
        f"Status: {status}",
    ]
    if result.partial:
        lines.append("Some model pages could not be retrieved.")
        lines.append(f"Notices: {len(result.notices)}")
    lines.append(f"Catalog: {result.catalog_path}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help()
        return 2

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 2

    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
