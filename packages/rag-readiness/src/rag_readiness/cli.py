"""Command-line interface for rag-readiness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag_readiness.__version__ import __version__
from rag_readiness.config import AnalysisConfig
from rag_readiness.discovery import CorpusPathError
from rag_readiness.pipeline import analyze_corpus
from rag_readiness.report import render_json_report, render_text_report


def build_parser() -> argparse.ArgumentParser:
    defaults = AnalysisConfig()
    parser = argparse.ArgumentParser(
        prog="rag-readiness",
        description=(
            "Analyze a local document corpus for RAG readiness before embedding or indexing. "
            "Runs entirely offline; no API keys or network access required."
        ),
    )
    parser.add_argument(
        "corpus",
        nargs="?",
        help="Path to a local directory containing documents to analyze",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Optional path to write the report (default: stdout)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--small-chars",
        type=int,
        default=None,
        help=(
            "Very-small document threshold in characters "
            f"(default: {defaults.small_document_chars})"
        ),
    )
    parser.add_argument(
        "--large-chars",
        type=int,
        default=None,
        help=(
            "Very-large document threshold in characters "
            f"(default: {defaults.large_document_chars})"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.corpus:
        parser.error("the following arguments are required: corpus")

    defaults = AnalysisConfig()
    config = AnalysisConfig(
        small_document_chars=(
            args.small_chars if args.small_chars is not None else defaults.small_document_chars
        ),
        large_document_chars=(
            args.large_chars if args.large_chars is not None else defaults.large_document_chars
        ),
    )

    try:
        report = analyze_corpus(args.corpus, config=config)
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


if __name__ == "__main__":
    raise SystemExit(main())
