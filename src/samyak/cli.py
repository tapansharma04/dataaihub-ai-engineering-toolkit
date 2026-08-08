"""Samyak command-line interface.

Current commands:
  samyak corpus <path>   Corpus Intelligence analysis

Future capabilities may add additional top-level subcommands.
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

from samyak.__version__ import __version__
from samyak.corpus.config import AnalysisConfig
from samyak.corpus.discovery import CorpusPathError
from samyak.corpus.pipeline import analyze_corpus, default_progress_callback
from samyak.corpus.report import render_json_report, render_text_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="samyak",
        description=(
            "Samyak by DataAIHub — AI engineering tooling for building reliable AI "
            "applications. Runs entirely offline; no API keys or network access required."
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
        choices=("text", "json"),
        default="text",
        help="Output format (default: text)",
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
    corpus.set_defaults(handler=_run_corpus)

    return parser


def _run_corpus(args: argparse.Namespace) -> int:
    defaults = AnalysisConfig()
    config = AnalysisConfig(
        small_document_chars=(
            args.small_chars if args.small_chars is not None else defaults.small_document_chars
        ),
        large_document_chars=(
            args.large_chars if args.large_chars is not None else defaults.large_document_chars
        ),
    )

    progress_callback = None
    if not args.no_progress and config.progress_every > 0:
        progress_callback = functools.partial(
            default_progress_callback,
            every=config.progress_every,
            stream=sys.stderr,
        )

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

    rendered = render_json_report(report) if args.format == "json" else render_text_report(report)

    if args.output:
        output_path = Path(args.output)
        try:
            output_path.write_text(rendered, encoding="utf-8")
        except OSError as exc:
            print(f"error: failed to write output: {exc}", file=sys.stderr)
            return 2
    else:
        sys.stdout.write(rendered)

    return 0


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
