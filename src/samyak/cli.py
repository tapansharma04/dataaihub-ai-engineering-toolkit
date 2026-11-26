"""Samyak command-line interface.

Current commands:
  samyak corpus <path>          Corpus Intelligence analysis
  samyak app check <path>       Application Readiness inspection
  samyak view                   Local workspace (run history and model catalog)
  samyak model update openai    Refresh the local OpenAI model catalog
  samyak model update anthropic Refresh the local Anthropic model catalog
  samyak model update google    Refresh the local Google Gemini API model catalog
  samyak model update fireworks Refresh the local Fireworks AI model catalog
  samyak model update together  Refresh the local Together AI model catalog
  samyak model update all       Refresh every supported provider sequentially
  samyak model history          Show documented model lifecycle history
  samyak model query            Query documented model capabilities locally
  samyak model export           Write a versioned JSON export of the local catalog
  samyak model diagnose-overrides
                                Compare archived observations; no catalog write
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from samyak.__version__ import __version__
from samyak.app.errors import ENGINE_EXIT_CODE, USAGE_EXIT_CODE, AppPathError
from samyak.app.pipeline import extract_application
from samyak.app.report import REPORT_FORMATS as APP_REPORT_FORMATS
from samyak.app.report import render_app_report
from samyak.corpus.config import AnalysisConfig
from samyak.corpus.discovery import CorpusPathError
from samyak.corpus.pipeline import analyze_corpus, default_progress_callback
from samyak.corpus.report import REPORT_FORMATS, render_report
from samyak.model.catalog import CatalogNotice
from samyak.model.compare import (
    ComparisonFact,
    ModelComparison,
    ModelComparisonError,
    compare_models,
    comparison_to_dict,
)
from samyak.model.errors import (
    CatalogDecodeError,
    CatalogSchemaError,
    CatalogStoreError,
    CatalogValidationError,
    HistorySchemaError,
    HistoryStoreError,
)
from samyak.model.export import (
    PublicationIntegrityStatus,
    PublicationStatus,
    build_model_export,
    model_export_to_json,
)
from samyak.model.facts import (
    ContextWindow,
    ContextWindowKind,
    Fact,
    FactStatus,
    LifecycleState,
    Modality,
)
from samyak.model.history import filter_history_events, render_lifecycle_history
from samyak.model.history_store import FileLifecycleHistoryStore
from samyak.model.identity import ModelIdentity, parse_samyak_id
from samyak.model.lifecycle import (
    LifecycleChange,
    LifecycleChangeType,
    is_unestablished,
)
from samyak.model.query import (
    FactEvidence,
    ModelQuery,
    QueryResult,
    explain_query_result,
    query_models,
)
from samyak.model.records import EvidenceDisposition
from samyak.model.source_archive import DEFAULT_ARCHIVE_KEEP
from samyak.model.store import FileCatalogStore
from samyak.model.update import (
    HISTORY_WRITE_FAILED_CODE,
    HISTORY_WRITE_FAILED_DETAIL,
    HISTORY_WRITE_FAILED_SUMMARY,
    SUPPORTED_PROVIDERS,
    CatalogUpdateResult,
    history_write_failure_code,
    is_history_write_failure_code,
    update_all_catalogs,
    update_anthropic_catalog,
    update_fireworks_catalog,
    update_google_catalog,
    update_openai_catalog,
    update_together_catalog,
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

    app = subparsers.add_parser(
        "app",
        help="Inspect a local AI application (Application Readiness)",
        description=(
            "Inspect a local AI application for model-lifecycle risks. "
            "`samyak app check` reads the project directory and the local "
            "model catalog if present. It is local, offline, and read-only. "
            "It does not refresh Model Intelligence and does not use the network."
        ),
    )
    app.set_defaults(handler=_run_app, app_parser=app)
    app_sub = app.add_subparsers(dest="app_command")
    check = app_sub.add_parser(
        "check",
        help="Inspect a local AI application project",
        description=(
            "Inspect a local AI application. Reads Python, JSON, and TOML in the "
            "project directory and reports model dependencies, unresolved AI "
            "operations, and lifecycle findings using the local Model Intelligence "
            "catalog if present. Local files only. Offline. Read-only: does not "
            "refresh the catalog, does not call provider APIs, and does not "
            "require an API key."
        ),
    )
    check.add_argument(
        "path",
        help="Path to a local application project directory",
    )
    check.add_argument(
        "--format",
        choices=APP_REPORT_FORMATS,
        default="text",
        help="Output format: text or json (default: text)",
    )
    check.add_argument(
        "--output",
        "-o",
        help="Optional path to write the report (default: stdout)",
    )
    check.set_defaults(handler=_run_app_check)

    view = subparsers.add_parser(
        "view",
        help="Open the local Samyak workspace",
        description=(
            "Start a local web viewer for saved corpus analysis runs and the "
            "model catalog. Browse Model Intelligence after "
            "`samyak model update openai`, `samyak model update anthropic`, "
            "`samyak model update google`, `samyak model update fireworks`, "
            "or `samyak model update together`. "
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
            "Inspect documented lifecycle history with `samyak model history`. "
            "Export the local catalog with `samyak model export`. "
            "Compare archived observations with `samyak model diagnose-overrides`. "
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
            "the local model catalog overlay. `all` refreshes every supported "
            "provider sequentially. No API key is required. "
            "This command uses the network. "
            "`--archive-sources` keeps a local diagnostic copy of fetched "
            "documentation and source failures. It is off by default and is "
            "not part of the catalog or export."
        ),
    )
    update.add_argument(
        "provider",
        metavar="PROVIDER",
        help=(f"Serving source to refresh ({', '.join(SUPPORTED_PROVIDERS)}, or all)"),
    )
    update.add_argument(
        "--archive-sources",
        action="store_true",
        help=(
            "Save fetched documentation and source-failure diagnostics under "
            "the model cache. Off by default. Does not change the catalog, "
            "history, or publication gate. "
            f"Keeps {DEFAULT_ARCHIVE_KEEP} completed refresh generations "
            "unless --archive-keep is set."
        ),
    )
    update.add_argument(
        "--archive-keep",
        type=int,
        default=DEFAULT_ARCHIVE_KEEP,
        metavar="N",
        help=(
            "Completed source-archive generations to retain with "
            "--archive-sources. "
            f"Default: {DEFAULT_ARCHIVE_KEEP}. Minimum: 1. "
            "Identical documents are stored once by SHA-256. A blob is deleted "
            "only when no retained generation still references it."
        ),
    )
    update.set_defaults(handler=_run_model_update)
    history = model_sub.add_parser(
        "history",
        help="Show documented model lifecycle history",
        description=(
            "Show previously observed documented lifecycle changes. "
            "This is local history of catalog updates, not current catalog "
            "state and not a prediction. No network access."
        ),
    )
    history.add_argument(
        "--provider",
        metavar="PROVIDER",
        help=f"Limit to one serving source ({', '.join(SUPPORTED_PROVIDERS)})",
    )
    history.add_argument(
        "--model",
        metavar="SAMYAK_ID",
        help="Limit to one model identity (provider:model-id)",
    )
    history.set_defaults(handler=_run_model_history)

    query = model_sub.add_parser(
        "query",
        help="Query documented model capabilities locally",
        description=(
            "Filter the local model catalog using verified typed facts. By default, "
            "only currently documented records are eligible; this command does not "
            "refresh providers or write the catalog."
        ),
    )
    query.add_argument(
        "--provider",
        action="append",
        choices=SUPPORTED_PROVIDERS,
        help="Provider to include; repeat to select multiple providers",
    )
    query.add_argument(
        "--documentation",
        action="append",
        choices=(
            "current_documented",
            "legacy_uncertified",
            "historical_reference_only",
            "not_observed_in_current_capture",
        ),
        help="Documentation category to add to the strict current-documentation default",
    )
    query.add_argument(
        "--include-legacy",
        action="store_true",
        help="Include legacy-uncertified records in addition to currently documented records",
    )
    query.add_argument("--lifecycle", choices=tuple(item.value for item in LifecycleState))
    query.add_argument("--api-access", type=_parse_boolean, metavar="BOOL")
    query.add_argument("--tool-calling", type=_parse_boolean, metavar="BOOL")
    query.add_argument("--structured-output", type=_parse_boolean, metavar="BOOL")
    query.add_argument(
        "--input-modality",
        action="append",
        choices=tuple(item.value for item in Modality),
        help="Required input modality; repeat to require all listed modalities",
    )
    query.add_argument(
        "--output-modality",
        action="append",
        choices=tuple(item.value for item in Modality),
        help="Required output modality; repeat to require all listed modalities",
    )
    query.add_argument("--min-context", type=_positive_integer, metavar="TOKENS")
    query.add_argument(
        "--context-kind",
        choices=("input", "output", "combined", "unknown"),
        help="Required context measurement kind when using --min-context",
    )
    query.add_argument("--format", choices=("text", "json"), default="text")
    query.add_argument(
        "--explain",
        action="store_true",
        help="Add evidence-safe matching and exclusion explanations",
    )
    query.set_defaults(handler=_run_model_query)

    compare = model_sub.add_parser(
        "compare",
        help="Compare two or three documented models locally",
        description=(
            "Compare saved typed evidence for exactly two or three explicit model IDs. "
            "By default, only currently documented records are eligible; this command "
            "does not refresh providers or write the catalog."
        ),
    )
    compare.add_argument("model_ids", nargs="+", metavar="MODEL_ID")
    compare.add_argument("--format", choices=("text", "json"), default="text")
    compare.add_argument(
        "--documentation",
        action="append",
        choices=(
            "current_documented",
            "legacy_uncertified",
            "historical_reference_only",
            "not_observed_in_current_capture",
        ),
        help="Documentation category to add to the strict current-documentation default",
    )
    compare.add_argument(
        "--include-legacy",
        action="store_true",
        help="Include legacy-uncertified records in addition to currently documented records",
    )
    compare.set_defaults(handler=_run_model_compare)

    export = model_sub.add_parser(
        "export",
        help="Write a versioned JSON export of the local model catalog",
        description=(
            "Write a read-only JSON export of the local model catalog and "
            "lifecycle history. The export records whether the snapshot is "
            "complete enough to publish. This command does not use the network "
            "and does not refresh or modify the catalog or history."
        ),
    )
    export.add_argument(
        "--output",
        "-o",
        help="Optional path to write the export (default: stdout)",
    )
    export.add_argument(
        "--require-complete",
        action="store_true",
        help=(
            "Exit with status 2 and do not write the export when the snapshot "
            "is not publishable as complete"
        ),
    )
    export.add_argument(
        "--require-publishable",
        action="store_true",
        help=(
            "Exit with status 2 and do not write the export when publication "
            "integrity is blocked; degraded optional-source coverage is allowed"
        ),
    )
    export.set_defaults(handler=_run_model_export)
    diagnose = model_sub.add_parser(
        "diagnose-overrides",
        help="Compare archived observations without changing the catalog",
        description=(
            "Re-parse selected archive generations and report supporting "
            "observations, overrides, and conflicts. Does not refresh providers "
            "or write the catalog, history, or archive."
        ),
    )
    diagnose.add_argument(
        "--provider",
        choices=SUPPORTED_PROVIDERS,
        help="Provider to diagnose. Requires --refresh-id",
    )
    diagnose.add_argument(
        "--refresh-id",
        help="Completed archive generation for --provider",
    )
    diagnose.add_argument(
        "--source-generation",
        action="append",
        default=None,
        metavar="PROVIDER=REFRESH_ID",
        help="Repeat for each provider generation. Not one combined snapshot",
    )
    diagnose.add_argument(
        "--model",
        metavar="SAMYAK_ID",
        help="Limit findings to one model identity (provider:model-id)",
    )
    diagnose.add_argument(
        "--fact",
        help="Limit findings to one catalog fact name",
    )
    diagnose.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text)",
    )
    diagnose.add_argument(
        "--archive-root",
        help="Archive root (default: the local model source archive)",
    )
    diagnose.add_argument(
        "--catalog",
        help="Read-only catalog file for reproducibility checks",
    )
    diagnose.add_argument(
        "--no-catalog",
        action="store_true",
        help="Do not read a catalog. Findings stay archive-only",
    )
    diagnose.set_defaults(handler=_run_model_diagnose_overrides)

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


def _run_app(args: argparse.Namespace) -> int:
    parser = getattr(args, "app_parser", None)
    if parser is not None:
        parser.print_help()
    return USAGE_EXIT_CODE


def _run_app_check(args: argparse.Namespace) -> int:
    try:
        extraction = extract_application(args.path)
    except AppPathError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return USAGE_EXIT_CODE
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return USAGE_EXIT_CODE
    except Exception as exc:
        print(f"error: unexpected analysis failure: {exc}", file=sys.stderr)
        return ENGINE_EXIT_CODE

    rendered = render_app_report(extraction.report, args.format)
    if args.output:
        output_path = Path(args.output)
        try:
            output_path.write_text(rendered, encoding="utf-8")
        except OSError as exc:
            print(f"error: failed to write output: {exc}", file=sys.stderr)
            return USAGE_EXIT_CODE
    else:
        sys.stdout.write(rendered)
    return 0


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


_UPDATE_ALL = "all"
_UPDATE_HANDLERS = {
    "openai": (update_openai_catalog, "OpenAI"),
    "anthropic": (update_anthropic_catalog, "Anthropic"),
    "google": (update_google_catalog, "Google"),
    "fireworks": (update_fireworks_catalog, "Fireworks"),
    "together": (update_together_catalog, "Together"),
}


def _provider_label(provider_id: str) -> str:
    handler = _UPDATE_HANDLERS.get(provider_id)
    if handler is None:
        return provider_id
    return handler[1]


def _run_model_update(args: argparse.Namespace) -> int:
    provider = str(args.provider).strip().lower()
    archive_sources = bool(args.archive_sources)
    archive_keep = args.archive_keep
    if archive_sources and (isinstance(archive_keep, bool) or not isinstance(archive_keep, int)):
        print("error: --archive-keep must be an integer", file=sys.stderr)
        return 2
    if archive_sources and archive_keep < 1:
        print("error: --archive-keep must be at least 1", file=sys.stderr)
        return 2
    if provider == _UPDATE_ALL:
        return _run_model_update_all(
            archive_sources=archive_sources,
            archive_keep=archive_keep,
        )
    if provider not in SUPPORTED_PROVIDERS:
        supported = ", ".join((*SUPPORTED_PROVIDERS, _UPDATE_ALL))
        print(f"error: unknown provider; supported: {supported}", file=sys.stderr)
        return 2
    updater, label = _UPDATE_HANDLERS[provider]
    try:
        result = updater(archive_sources=archive_sources, archive_keep=archive_keep)
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


def _run_model_update_all(
    *,
    archive_sources: bool = False,
    archive_keep: int = DEFAULT_ARCHIVE_KEEP,
) -> int:
    results = update_all_catalogs(
        progress=_report_update_all_progress,
        archive_sources=archive_sources,
        archive_keep=archive_keep,
    )
    sys.stdout.write(_render_update_all_results(results))
    if any(not item.ok or not item.committed for item in results):
        return 2
    return 0


def _report_update_all_progress(
    event: str,
    provider_id: str,
    index: int,
    total: int,
    result: CatalogUpdateResult | None,
) -> None:
    if event == "start":
        print(
            f"Updating {_provider_label(provider_id)} ({index}/{total}) ",
            end="",
            file=sys.stderr,
            flush=True,
        )
        return
    if event == "request":
        print(".", end="", file=sys.stderr, flush=True)
        return
    if event == "finish":
        failed = result is None or not result.ok or not result.committed
        print(" failed" if failed else " done", file=sys.stderr, flush=True)


def _parse_boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("must be true or false")


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _run_model_query(args: argparse.Namespace) -> int:
    """Run the typed query engine without triggering transaction recovery."""
    if args.context_kind and args.min_context is None:
        print("error: --context-kind requires --min-context", file=sys.stderr)
        return 2
    if args.min_context is not None and args.context_kind is None:
        print("error: --min-context requires --context-kind", file=sys.stderr)
        return 2
    dispositions: set[EvidenceDisposition | None] = {EvidenceDisposition.CURRENT_DOCUMENTED}
    for value in args.documentation or ():
        dispositions.add(_documentation_disposition(value))
    if args.include_legacy:
        dispositions.add(None)
    try:
        query = ModelQuery(
            providers=None if not args.provider else frozenset(args.provider),
            documentation_dispositions=frozenset(dispositions),
            lifecycle=None if args.lifecycle is None else LifecycleState(args.lifecycle),
            api_access=args.api_access,
            tool_calling=args.tool_calling,
            structured_output=args.structured_output,
            input_modalities=frozenset(Modality(item) for item in args.input_modality or ()),
            output_modalities=frozenset(Modality(item) for item in args.output_modality or ()),
            minimum_context_window=args.min_context,
            context_window_kind=(
                None if args.context_kind is None else ContextWindowKind(args.context_kind)
            ),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    catalog, error = _load_catalog_for_read_only_query()
    if error is not None:
        print(f"error: {error}", file=sys.stderr)
        return 2
    assert catalog is not None
    result = query_models(catalog.models, query)
    if args.format == "json":
        sys.stdout.write(_query_result_json(result, explain=args.explain))
    else:
        sys.stdout.write(_render_query_result(result, catalog.models, explain=args.explain))
    return 0


def _run_model_compare(args: argparse.Namespace) -> int:
    """Present the immutable comparison API without duplicate selection logic."""
    catalog, error = _load_catalog_for_read_only_query()
    if error is not None:
        print(f"error: {error}", file=sys.stderr)
        return 2
    assert catalog is not None
    documentation = tuple(_documentation_disposition(item) for item in args.documentation or ())
    try:
        comparison = compare_models(
            catalog.models,
            args.model_ids,
            include_legacy=args.include_legacy,
            documentation=documentation,
        )
    except ModelComparisonError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.format == "json":
        sys.stdout.write(
            json.dumps(comparison_to_dict(comparison), indent=2, ensure_ascii=False) + "\n"
        )
    else:
        sys.stdout.write(_render_model_comparison(comparison))
    return 0


def _render_model_comparison(comparison: ModelComparison) -> str:
    """Render facts as evidence, not a scorecard or capability ranking."""
    models = comparison.models
    headers = [model.samyak_id for model in models]
    rows = [
        ("Provider", [model.provider_id for model in models]),
        (
            "Documentation",
            [_disposition_name(model.documentation_disposition) for model in models],
        ),
        ("Detail", [model.detail_href for model in models]),
        ("Display name", [_format_comparison_fact(model.display_name) for model in models]),
        ("Lifecycle", [_format_comparison_fact(model.lifecycle) for model in models]),
        (
            "Availability scope",
            [_format_comparison_fact(model.availability_scope) for model in models],
        ),
        ("API access", [_format_comparison_fact(model.api_access) for model in models]),
        ("Tool calling", [_format_comparison_fact(model.tool_calling) for model in models]),
        (
            "Structured output",
            [_format_comparison_fact(model.structured_output) for model in models],
        ),
        (
            "Input modalities",
            [_format_comparison_fact(model.input_modalities) for model in models],
        ),
        (
            "Output modalities",
            [_format_comparison_fact(model.output_modalities) for model in models],
        ),
        ("Context window", [_format_comparison_fact(model.context_window) for model in models]),
    ]
    width = max(len(header) for header in headers)
    lines = ["Model comparison", "".join(f"  {header:<{width}}" for header in headers)]
    for label, values in rows:
        lines.append(label + ":")
        lines.extend(
            f"  {header:<{width}}  {value}" for header, value in zip(headers, values, strict=True)
        )
    lines.append("Comparability:")
    for item in comparison.dimensions:
        status = (
            "directly comparable" if item.comparable else f"not directly comparable — {item.reason}"
        )
        lines.append(f"  {item.name}: {status}")
    if comparison.warnings:
        lines.append("Warnings:")
        lines.extend(f"  {item}" for item in comparison.warnings)
    return "\n".join(lines) + "\n"


def _format_comparison_fact(fact: ComparisonFact) -> str:
    if fact.status is FactStatus.KNOWN:
        if fact.value is True:
            value = "Verified yes"
        elif fact.value is False:
            value = "Verified no"
        else:
            value = (
                _query_json_value(fact.value)
                if isinstance(fact.value, ContextWindow)
                else fact.value
            )
            value = f"Verified {value!r}"
    elif fact.status is FactStatus.CONFLICT:
        claims = ", ".join(f"{item.value!r} ({item.source_id})" for item in fact.claims)
        value = f"Conflicting evidence: {claims}"
    else:
        value = fact.status.value.replace("_", " ").capitalize()
    sources = ", ".join(fact.source_ids) or "no source reference recorded"
    verified = ", ".join(fact.verified_at) or "no verification timestamp"
    return f"{value}; source {sources}; verified {verified}"


def _documentation_disposition(value: str) -> EvidenceDisposition | None:
    if value == "legacy_uncertified":
        return None
    return EvidenceDisposition(value)


def _load_catalog_for_read_only_query():
    """Read a validated catalog, failing rather than recovering a transaction."""
    from samyak.model.serialize import catalog_from_json

    store = FileCatalogStore()
    pending = (
        ".catalog-history-transaction.json",
        ".catalog-history-catalog.stage",
        ".catalog-history-history.stage",
    )
    if any((store.models_dir / name).exists() for name in pending):
        return (
            None,
            "catalog/history transaction is pending; recover it with a writer before querying",
        )
    if not store.exists():
        return None, "no local model catalog"
    try:
        return catalog_from_json(store.catalog_path.read_text(encoding="utf-8")), None
    except CatalogSchemaError as exc:
        return None, str(exc)
    except (CatalogDecodeError, CatalogValidationError, OSError, UnicodeError):
        return None, "model catalog is corrupt"


def _query_result_json(result: QueryResult, *, explain: bool = False) -> str:
    payload = {
        "query": _query_to_dict(result.query),
        "documentation_eligibility": [
            _disposition_name(item)
            for item in sorted(result.query.documentation_dispositions, key=_disposition_name)
        ],
        "summary": {
            "total_records_considered": result.summary.total_records_considered,
            "documentation_eligible_records": result.summary.documentation_eligible_records,
            "matching_records": result.summary.matching_records,
            "excluded_records": result.summary.excluded_records,
            "exclusion_reason_counts": dict(result.summary.exclusion_reason_counts),
        },
        "matches": [
            {
                "samyak_id": item.samyak_id,
                "provider_id": item.provider_id,
                "documentation_disposition": _disposition_name(item.documentation_disposition),
                "lifecycle": _fact_evidence_dict(item.lifecycle),
                "matched_facts": [_fact_evidence_dict(fact) for fact in item.matched_facts],
            }
            for item in result.matches
        ],
        "exclusions": [
            {
                "samyak_id": item.samyak_id,
                "provider_id": item.provider_id,
                "documentation_disposition": _disposition_name(item.documentation_disposition),
                "reasons": list(item.reasons),
            }
            for item in result.exclusions
        ],
    }
    if explain:
        payload["explanations"] = {
            model_id: [
                {
                    "predicate": item.predicate,
                    "outcome": item.outcome,
                    "reason_code": item.reason_code,
                    "reason": item.reason,
                    "source_ids": list(item.source_ids),
                    "verified_at": list(item.verified_at),
                }
                for item in items
            ]
            for model_id, items in explain_query_result(result).items()
        }
    return json.dumps(payload, indent=2, ensure_ascii=False, default=_query_json_value) + "\n"


def _query_to_dict(query: ModelQuery) -> dict[str, object]:
    return {
        "providers": None if query.providers is None else sorted(query.providers),
        "documentation_dispositions": sorted(
            _disposition_name(item) for item in query.documentation_dispositions
        ),
        "lifecycle": None if query.lifecycle is None else query.lifecycle.value,
        "api_access": query.api_access,
        "tool_calling": query.tool_calling,
        "structured_output": query.structured_output,
        "input_modalities": sorted(item.value for item in query.input_modalities),
        "output_modalities": sorted(item.value for item in query.output_modalities),
        "minimum_context_window": query.minimum_context_window,
        "context_window_kind": (
            None if query.context_window_kind is None else query.context_window_kind.value
        ),
    }


def _fact_evidence_dict(fact: FactEvidence) -> dict[str, object]:
    return {
        "name": fact.name,
        "status": fact.status.value,
        "value": fact.value,
        "source_ids": list(fact.source_ids),
        "verified_at": list(fact.verified_at),
    }


def _query_json_value(value: object):
    if isinstance(value, LifecycleState | Modality | ContextWindowKind):
        return value.value
    if isinstance(value, ContextWindow):
        return value.to_dict()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _render_query_result(result: QueryResult, records, *, explain: bool = False) -> str:
    by_id = {record.samyak_id: record for record in records}
    lines = [
        "Model query",
        f"Total records: {result.summary.total_records_considered}",
        f"Documentation-eligible: {result.summary.documentation_eligible_records}",
        f"Matches: {result.summary.matching_records}",
        f"Excluded: {result.summary.excluded_records}",
        "Exclusion reasons:",
    ]
    if result.summary.exclusion_reason_counts:
        lines.extend(
            f"  {reason}: {count}" for reason, count in result.summary.exclusion_reason_counts
        )
    else:
        lines.append("  none")
    lines.append("Matches:")
    if not result.matches:
        lines.append("  none")
    for match in result.matches:
        record = by_id[match.samyak_id]
        display_name = (
            record.display_name.value if record.display_name.status is FactStatus.KNOWN else None
        )
        label = match.samyak_id if display_name is None else f"{match.samyak_id} ({display_name})"
        lines.append(
            f"  {match.provider_id} {label} [{_disposition_name(match.documentation_disposition)}]"
        )
        lines.append(f"    lifecycle: {_format_query_fact(match.lifecycle)}")
        for fact in match.matched_facts[1:]:
            lines.append(f"    {fact.name}: {_format_query_fact(fact)}")
    if explain:
        explanations = explain_query_result(result)
        lines.append(
            "Explanation: exclusion counts may overlap; excluded does not mean "
            "unavailable or incapable."
        )
        if not result.matches:
            lines.append("  No verified matches were found for the requested evidence predicates.")
        excluded_explanations = (
            (item.samyak_id, explanations[item.samyak_id]) for item in result.exclusions
        )
        sampled_exclusions = tuple(excluded_explanations)[:10]
        for model_id, items in sampled_exclusions:
            lines.append(f"  {model_id}:")
            lines.extend(f"    {item.reason_code}: {item.reason}" for item in items)
        if len(result.exclusions) > len(sampled_exclusions):
            omitted = len(result.exclusions) - len(sampled_exclusions)
            lines.append(f"  … {omitted} further excluded records omitted")
    return "\n".join(lines) + "\n"


def _format_query_fact(fact: FactEvidence) -> str:
    value = _query_json_value(fact.value) if isinstance(fact.value, ContextWindow) else fact.value
    source = ", ".join(fact.source_ids) or "no source"
    verified = ", ".join(fact.verified_at) or "no verification timestamp"
    return f"{fact.status.value} {value!r}; {source}; verified {verified}"


def _disposition_name(value: EvidenceDisposition | None) -> str:
    return "legacy_uncertified" if value is None else value.value


def _run_model_history(args: argparse.Namespace) -> int:
    provider = args.provider
    if provider is not None:
        provider = str(provider).strip().lower()
        if provider not in SUPPORTED_PROVIDERS:
            supported = ", ".join(SUPPORTED_PROVIDERS)
            print(f"error: unknown provider; supported: {supported}", file=sys.stderr)
            return 2
    samyak_id = args.model
    if samyak_id is not None:
        samyak_id = str(samyak_id).strip()
        try:
            parse_samyak_id(samyak_id)
        except CatalogValidationError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    try:
        history = FileLifecycleHistoryStore().load()
    except HistorySchemaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except HistoryStoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    events = filter_history_events(
        history.events,
        provider_id=provider,
        samyak_id=samyak_id,
    )
    sys.stdout.write(render_lifecycle_history(events))
    return 0


def _run_model_export(args: argparse.Namespace) -> int:
    """Read the local overlay and emit a publication-gated JSON export.

    Missing, corrupt, or unsupported catalog or history exits 2 and does not
    rewrite either file. ``--require-complete`` checks publication status
    before any write: an incomplete snapshot exits 2 with diagnostics on
    stderr and no JSON on stdout or ``--output``. Without that flag, a
    successfully loaded incomplete snapshot is written and exits 0.
    """
    catalog_store = FileCatalogStore()
    history_store = FileLifecycleHistoryStore(root=catalog_store.root)
    try:
        catalog = catalog_store.load()
    except CatalogSchemaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except CatalogStoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        history = history_store.load() if history_store.exists() else None
    except HistorySchemaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except HistoryStoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        export = build_model_export(catalog, history)
        rendered = model_export_to_json(export)
    except CatalogValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.require_complete and export.publication.status is not PublicationStatus.COMPLETE:
        print("error: model export is incomplete", file=sys.stderr)
        for reason in export.publication.reasons:
            print(reason, file=sys.stderr)
        return 2
    if (
        args.require_publishable
        and export.publication.integrity_status is not PublicationIntegrityStatus.PUBLISHABLE
    ):
        print("error: model export is not publishable", file=sys.stderr)
        for reason in export.publication.blocking_reasons:
            print(reason, file=sys.stderr)
        return 2

    if args.output:
        output_path = _resolve_export_path(Path(args.output))
        if output_path is None or _export_output_replaces_catalog(
            output_path, catalog_store, history_store
        ):
            print(
                "error: export output must not replace the model catalog or lifecycle history",
                file=sys.stderr,
            )
            return 2
        try:
            output_path.write_text(rendered, encoding="utf-8")
        except OSError as exc:
            print(f"error: failed to write output: {exc}", file=sys.stderr)
            return 2
    else:
        sys.stdout.write(rendered)
    return 0


def _run_model_diagnose_overrides(args: argparse.Namespace) -> int:
    """Re-parse selected archive generations. Does not write or refresh."""
    from samyak.model.override_diagnostic import diagnose_overrides, render_override_report
    from samyak.model.records import RECORD_FACT_NAMES
    from samyak.model.serialize import catalog_from_json
    from samyak.model.source_archive import ARCHIVE_DIRNAME

    generations, error = _diagnose_generations(args)
    if error is not None:
        print(f"error: {error}", file=sys.stderr)
        return 2
    model_id = None
    if args.model is not None:
        model_id = str(args.model).strip()
        try:
            parse_samyak_id(model_id)
        except CatalogValidationError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    fact_name = None
    if args.fact is not None:
        fact_name = str(args.fact).strip()
        if fact_name not in RECORD_FACT_NAMES:
            print(f"error: unknown fact {fact_name}", file=sys.stderr)
            return 2
    if args.catalog and args.no_catalog:
        print("error: --catalog and --no-catalog cannot be combined", file=sys.stderr)
        return 2
    for _provider, refresh_id in generations:
        if not _is_generation_name(refresh_id):
            print("error: refresh id is not a generation name", file=sys.stderr)
            return 2

    if args.archive_root:
        archive_root = Path(args.archive_root)
    else:
        archive_root = FileCatalogStore().models_dir / ARCHIVE_DIRNAME

    catalog = None
    catalog_unavailable = False
    if not args.no_catalog:
        if args.catalog:
            catalog_path = Path(args.catalog)
            if not catalog_path.is_file():
                print("error: catalog was not found", file=sys.stderr)
                return 2
            try:
                catalog = catalog_from_json(catalog_path.read_text(encoding="utf-8"))
            except (
                CatalogDecodeError,
                CatalogSchemaError,
                CatalogValidationError,
                OSError,
                UnicodeError,
            ) as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
        else:
            store = FileCatalogStore()
            if not store.exists():
                catalog_unavailable = True
            else:
                try:
                    catalog = store.load()
                except (CatalogSchemaError, CatalogStoreError) as exc:
                    print(f"error: {exc}", file=sys.stderr)
                    return 2

    report = diagnose_overrides(
        archive_root=archive_root,
        generations=generations,
        catalog=catalog,
        model_id=model_id,
        fact_name=fact_name,
        catalog_unavailable=catalog_unavailable,
    )
    sys.stdout.write(render_override_report(report, output_format=args.format))
    return 0


def _diagnose_generations(
    args: argparse.Namespace,
) -> tuple[list[tuple[str, str]], str | None]:
    generations: list[tuple[str, str]] = []
    if args.provider or args.refresh_id:
        if not args.provider or not args.refresh_id:
            return [], "--provider and --refresh-id must be used together"
        generations.append((str(args.provider), str(args.refresh_id).strip()))
    for item in args.source_generation or []:
        if str(item).count("=") != 1:
            return [], "--source-generation must be PROVIDER=REFRESH_ID"
        provider, refresh_id = str(item).split("=", 1)
        provider = provider.strip().lower()
        refresh_id = refresh_id.strip()
        if provider not in SUPPORTED_PROVIDERS:
            supported = ", ".join(SUPPORTED_PROVIDERS)
            return [], f"unknown provider; supported: {supported}"
        if not refresh_id:
            return [], "--source-generation is missing a refresh id"
        generations.append((provider, refresh_id))
    if not generations:
        return [], (
            "select a provider generation with --provider and --refresh-id or --source-generation"
        )
    seen: dict[str, str] = {}
    unique: list[tuple[str, str]] = []
    for provider, refresh_id in generations:
        previous = seen.get(provider)
        if previous is not None and previous != refresh_id:
            return [], f"{provider} was given two archive generations"
        if previous is None:
            seen[provider] = refresh_id
            unique.append((provider, refresh_id))
    return unique, None


def _is_generation_name(refresh_id: str) -> bool:
    if not refresh_id or refresh_id in {".", ".."} or refresh_id.startswith("."):
        return False
    return "/" not in refresh_id and "\\" not in refresh_id


def _export_output_replaces_catalog(
    output_path: Path,
    catalog_store: FileCatalogStore,
    history_store: FileLifecycleHistoryStore,
) -> bool:
    """Return whether ``output_path`` is one of the live catalog files.

    Comparison uses the store's catalog, backup, and history paths. Both sides
    are resolved so ``..`` and symlinks to those files are rejected. A path
    that cannot be resolved is refused rather than written.
    """
    candidate = _resolve_export_path(output_path)
    if candidate is None:
        return True
    protected = (
        catalog_store.catalog_path,
        catalog_store.backup_path,
        history_store.history_path,
    )
    for path in protected:
        resolved = _resolve_export_path(path)
        if resolved is None or candidate == resolved:
            return True
    return False


def _resolve_export_path(path: Path) -> Path | None:
    """Return the absolute path used both for protection checks and for writing.

    ``~`` is expanded and ``..`` and symlinks are resolved. The final component
    does not need to exist. The returned path is the file ``write_text`` uses,
    so validation and the write cannot refer to different locations.
    """
    try:
        return path.expanduser().resolve(strict=False)
    except (OSError, RuntimeError):
        return None


def _render_update_result(result: CatalogUpdateResult, *, label: str) -> str:
    status = "partial" if result.partial else "complete"
    catalog_notices = _catalog_notices(result.notices)
    lines = [
        f"Updated {label} model catalog.",
        f"Models: {result.provider_model_count}",
        f"Status: {status}",
    ]
    if result.partial:
        lines.append("Some model pages could not be retrieved.")
    if catalog_notices:
        lines.append(f"Notices: {len(catalog_notices)}")
    lines.append(f"Catalog: {result.catalog_path}")
    if _history_write_failed(result):
        lines.extend(_history_write_failed_lines())
    if result.lifecycle_changes:
        lines.append("")
        lines.extend(_render_lifecycle_changes(result.lifecycle_changes))
    return "\n".join(lines) + "\n"


def _render_update_all_results(results: tuple[CatalogUpdateResult, ...]) -> str:
    lines = [
        "Updating model catalogs",
        "------------------------------",
        "",
    ]
    updated = 0
    failed = 0
    lifecycle_count = 0
    for result in results:
        lines.append(_provider_label(result.provider_id))
        if result.ok and result.committed:
            updated += 1
            lifecycle_count += len(result.lifecycle_changes)
            lines.append("  Updated")
            lines.append(f"  Models: {result.provider_model_count}")
            if result.partial:
                lines.append("  Status: partial")
            if _history_write_failed(result):
                lines.extend(f"  {line}" for line in _history_write_failed_lines())
            lines.append(f"  Lifecycle changes: {len(result.lifecycle_changes)}")
            if result.lifecycle_changes:
                for change in result.lifecycle_changes:
                    lines.append("")
                    lines.extend(
                        f"    {line}" if line else "" for line in _render_lifecycle_change(change)
                    )
        else:
            failed += 1
            lines.append("  Failed")
            lines.append(f"  {result.error or 'catalog update failed'}")
            if result.previous_existed:
                lines.append("  The existing model catalog was not changed.")
            else:
                lines.append("  No local catalog was written.")
        lines.append("")
    updated_noun = "provider" if updated == 1 else "providers"
    failed_noun = "provider" if failed == 1 else "providers"
    change_noun = "change" if lifecycle_count == 1 else "changes"
    lines.extend(
        [
            "Update complete",
            f"  {updated} {updated_noun} updated",
            f"  {failed} {failed_noun} failed",
            f"  {lifecycle_count} lifecycle {change_noun} detected",
        ]
    )
    return "\n".join(lines) + "\n"


def _catalog_notices(notices: tuple[CatalogNotice, ...]) -> tuple[CatalogNotice, ...]:
    return tuple(notice for notice in notices if not is_history_write_failure_code(notice.code))


def _history_write_failed(result: CatalogUpdateResult) -> bool:
    """Warn for this update's failed append, not for an older unresolved marker.

    A bare ``HISTORY_WRITE_FAILED`` code is the previous single-notice form.
    A qualified code warns only when it names this provider and this catalog
    stamp. Another provider's marker, or an older stamp, stays on the catalog
    without a warning on this command.
    """
    if any(notice.code == HISTORY_WRITE_FAILED_CODE for notice in result.notices):
        return True
    if result.catalog is None:
        return False
    expected = history_write_failure_code(result.provider_id, result.catalog.generated_at)
    return any(notice.code == expected for notice in result.notices)


def _history_write_failed_lines() -> list[str]:
    return [
        f"Warning: {HISTORY_WRITE_FAILED_SUMMARY}",
        HISTORY_WRITE_FAILED_DETAIL,
    ]


def _render_lifecycle_changes(changes: tuple[LifecycleChange, ...]) -> list[str]:
    lines = [
        "Lifecycle changes",
        "------------------------------",
        "",
    ]
    for index, change in enumerate(changes):
        if index:
            lines.append("")
        lines.extend(_render_lifecycle_change(change))
    count = len(changes)
    noun = "change" if count == 1 else "changes"
    lines.append("")
    lines.append(f"{count} lifecycle {noun} detected.")
    return lines


def _render_lifecycle_change(change: LifecycleChange) -> list[str]:
    heading, evidence = _lifecycle_heading_and_evidence(change)
    lines = [heading, f"  {change.samyak_id}", f"  {evidence}"]
    extra = _lifecycle_change_details(change)
    lines.extend(f"  {line}" for line in extra)
    return lines


def _lifecycle_heading_and_evidence(change: LifecycleChange) -> tuple[str, str]:
    if change.change_type is LifecycleChangeType.ADDED:
        return "ADDED", "Model was added to the documented catalog."
    if change.change_type is LifecycleChangeType.REMOVED:
        return (
            "REMOVED FROM DOCUMENTED CATALOG",
            "Model was removed from the documented catalog.",
        )
    if change.change_type is LifecycleChangeType.LIFECYCLE_CHANGED:
        return _lifecycle_state_heading(change)
    if change.change_type is LifecycleChangeType.DEPRECATION_DATE_CHANGED:
        return "DEPRECATION DATE CHANGED", "Provider deprecation date changed."
    if change.change_type is LifecycleChangeType.RETIREMENT_DATE_CHANGED:
        return "RETIREMENT DATE CHANGED", "Provider retirement date changed."
    return _replacement_heading_and_evidence(change)


def _lifecycle_state_heading(change: LifecycleChange) -> tuple[str, str]:
    current = change.current.lifecycle if change.current is not None else None
    previous = change.previous.lifecycle if change.previous is not None else None
    heading = _lifecycle_label(current)
    if heading == "DEPRECATED":
        return heading, "Provider announced deprecation."
    if heading == "RETIRED":
        previous_label = _lifecycle_label(previous)
        if previous_label not in {"NOT ESTABLISHED", "RETIRED", "LIFECYCLE CHANGED"}:
            return heading, f"Lifecycle changed: {previous_label.lower()} → retired"
        return heading, "Provider announced retirement."
    if heading == "LEGACY":
        return heading, "Provider documented this model as legacy."
    if heading == "ACTIVE":
        return heading, "Provider documented this model as active."
    return "LIFECYCLE CHANGED", "Provider documented a lifecycle change."


def _replacement_heading_and_evidence(change: LifecycleChange) -> tuple[str, str]:
    previous = change.previous.replacement if change.previous is not None else None
    current = change.current.replacement if change.current is not None else None
    previous_missing = previous is None or is_unestablished(previous)
    current_missing = current is None or is_unestablished(current)
    if previous_missing and not current_missing:
        return "REPLACEMENT ADDED", "Provider documented a replacement."
    if not previous_missing and current_missing:
        return "REPLACEMENT REMOVED", "Provider no longer documented a replacement."
    return "REPLACEMENT CHANGED", "Provider documented a replacement change."


def _lifecycle_change_details(change: LifecycleChange) -> list[str]:
    if change.change_type is LifecycleChangeType.LIFECYCLE_CHANGED:
        previous = change.previous.lifecycle if change.previous is not None else None
        current = change.current.lifecycle if change.current is not None else None
        lines = [f"{_lifecycle_label(previous)} → {_lifecycle_label(current)}"]
        record = change.current
        if record is not None:
            if not is_unestablished(record.retirement_at):
                lines.append(f"Retirement: {_format_date_fact(record.retirement_at)}")
            if not is_unestablished(record.replacement):
                lines.append(f"Replacement: {_format_replacement_fact(record.replacement)}")
        return lines
    if change.change_type is LifecycleChangeType.DEPRECATION_DATE_CHANGED:
        previous = change.previous.deprecated_at if change.previous is not None else None
        current = change.current.deprecated_at if change.current is not None else None
        return [f"Deprecation: {_format_date_fact(previous)} → {_format_date_fact(current)}"]
    if change.change_type is LifecycleChangeType.RETIREMENT_DATE_CHANGED:
        previous = change.previous.retirement_at if change.previous is not None else None
        current = change.current.retirement_at if change.current is not None else None
        return [f"Retirement: {_format_date_fact(previous)} → {_format_date_fact(current)}"]
    if change.change_type is LifecycleChangeType.REPLACEMENT_CHANGED:
        previous = change.previous.replacement if change.previous is not None else None
        current = change.current.replacement if change.current is not None else None
        return [
            "Replacement: "
            f"{_format_replacement_fact(previous)} → {_format_replacement_fact(current)}"
        ]
    return []


def _lifecycle_label(fact: Fact | None) -> str:
    if fact is None or is_unestablished(fact):
        return "NOT ESTABLISHED"
    if fact.status is FactStatus.CONFLICT:
        return "CONFLICT"
    if fact.status is FactStatus.NOT_APPLICABLE:
        return "NOT APPLICABLE"
    if fact.status is FactStatus.KNOWN and isinstance(fact.value, LifecycleState):
        return fact.value.value.upper()
    return "LIFECYCLE CHANGED"


def _format_date_fact(fact: Fact | None) -> str:
    if fact is None or is_unestablished(fact):
        return "not established"
    if fact.status is FactStatus.KNOWN:
        return str(fact.value)
    if fact.status is FactStatus.CONFLICT:
        values = ", ".join(sorted(str(claim.value) for claim in fact.claims))
        return f"conflict ({values})"
    return fact.status.value


def _format_replacement_fact(fact: Fact | None) -> str:
    if fact is None or is_unestablished(fact):
        return "not established"
    if fact.status is FactStatus.KNOWN:
        return _format_identities(fact.value)
    if fact.status is FactStatus.CONFLICT:
        values = ", ".join(sorted(_format_identities(claim.value) for claim in fact.claims))
        return f"conflict ({values})"
    return fact.status.value


def _format_identities(value: object) -> str:
    if isinstance(value, tuple) and value and isinstance(value[0], ModelIdentity):
        return ", ".join(item.samyak_id for item in value)
    if isinstance(value, ModelIdentity):
        return value.samyak_id
    return str(value)


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
