"""Samyak command-line interface.

Current commands:
  samyak corpus <path>          Corpus Intelligence analysis
  samyak view                   Local workspace (run history and model catalog)
  samyak model update openai    Refresh the local OpenAI model catalog
  samyak model update anthropic Refresh the local Anthropic model catalog
  samyak model update google    Refresh the local Google Gemini API model catalog
  samyak model update fireworks Refresh the local Fireworks AI model catalog
  samyak model update together  Refresh the local Together AI model catalog
  samyak model update all       Refresh every supported provider sequentially
  samyak model history          Show documented model lifecycle history
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
from samyak.model.catalog import CatalogNotice
from samyak.model.errors import CatalogValidationError, HistorySchemaError, HistoryStoreError
from samyak.model.facts import Fact, FactStatus, LifecycleState
from samyak.model.history import filter_history_events, render_lifecycle_history
from samyak.model.history_store import FileLifecycleHistoryStore
from samyak.model.identity import ModelIdentity, parse_samyak_id
from samyak.model.lifecycle import (
    LifecycleChange,
    LifecycleChangeType,
    is_unestablished,
)
from samyak.model.update import (
    HISTORY_WRITE_FAILED_CODE,
    HISTORY_WRITE_FAILED_DETAIL,
    HISTORY_WRITE_FAILED_SUMMARY,
    SUPPORTED_PROVIDERS,
    CatalogUpdateResult,
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
            "This command uses the network."
        ),
    )
    update.add_argument(
        "provider",
        metavar="PROVIDER",
        help=(f"Serving source to refresh ({', '.join(SUPPORTED_PROVIDERS)}, or all)"),
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
    if provider == _UPDATE_ALL:
        return _run_model_update_all()
    if provider not in SUPPORTED_PROVIDERS:
        supported = ", ".join((*SUPPORTED_PROVIDERS, _UPDATE_ALL))
        print(f"error: unknown provider; supported: {supported}", file=sys.stderr)
        return 2
    updater, label = _UPDATE_HANDLERS[provider]
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


def _run_model_update_all() -> int:
    results = update_all_catalogs(progress=_report_update_all_progress)
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
    if _history_write_failed(result.notices):
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
            if _history_write_failed(result.notices):
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
    return tuple(notice for notice in notices if notice.code != HISTORY_WRITE_FAILED_CODE)


def _history_write_failed(notices: tuple[CatalogNotice, ...]) -> bool:
    return any(notice.code == HISTORY_WRITE_FAILED_CODE for notice in notices)


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
