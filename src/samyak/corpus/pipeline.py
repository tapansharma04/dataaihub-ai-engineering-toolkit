"""Orchestrate discovery, incremental loading, and analysis."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from statistics import median

from samyak.__version__ import __version__
from samyak.corpus.analyzers.accumulator import CorpusAccumulator
from samyak.corpus.analyzers.document_pass import process_document
from samyak.corpus.analyzers.findings_builder import build_findings
from samyak.corpus.config import AnalysisConfig
from samyak.corpus.discovery import discover_files, validate_corpus_root
from samyak.corpus.loaders import DocumentLoadError, load_document
from samyak.corpus.models import (
    SEVERITY_ORDER,
    AnalysisReport,
    CorpusSummary,
    Finding,
    Severity,
)

PRODUCT_NAME = "samyak"
CAPABILITY_NAME = "corpus"
SUPPORTED_FORMAT_HELP = ".txt, .md, .pdf, .html, .htm"

ProgressCallback = Callable[[int, int], None]


def analyze_corpus(
    corpus_path: str | Path,
    config: AnalysisConfig | None = None,
    *,
    progress_callback: ProgressCallback | None = None,
) -> AnalysisReport:
    """Analyze a local document corpus and return an :class:`AnalysisReport`.

    Documents are loaded and analyzed one at a time. Full text is not retained
    after each document's checks complete. Cross-document state is limited to
    content hashes, compact HTML fingerprints, counters, and bounded path samples.
    """
    cfg = config or AnalysisConfig()
    root = validate_corpus_root(Path(corpus_path))
    discovered = discover_files(root)

    supported = [item for item in discovered if item.supported]
    unsupported = [item for item in discovered if not item.supported]

    unsupported_by_ext: dict[str, int] = {}
    for item in unsupported:
        key = item.extension or "(none)"
        unsupported_by_ext[key] = unsupported_by_ext.get(key, 0) + 1

    acc = CorpusAccumulator(config=cfg)
    load_errors: list[str] = []
    total_supported = len(supported)

    for index, item in enumerate(supported, start=1):
        try:
            doc = load_document(item)
        except DocumentLoadError:
            load_errors.append(item.relative_path)
            if progress_callback is not None:
                progress_callback(index, total_supported)
            continue

        try:
            process_document(doc, cfg, acc)
        finally:
            # Drop the only strong reference to the loaded document text.
            del doc

        if progress_callback is not None:
            progress_callback(index, total_supported)

    char_counts = acc.char_counts
    summary = CorpusSummary(
        corpus_root=root.name,
        total_discovered_files=len(discovered),
        supported_files=len(supported),
        unsupported_files=len(unsupported),
        analyzed_documents=acc.analyzed_documents,
        load_errors=len(load_errors),
        total_characters=acc.total_chars,
        total_bytes=acc.total_bytes,
        average_characters=(acc.total_chars / acc.analyzed_documents)
        if acc.analyzed_documents
        else 0.0,
        median_characters=float(median(char_counts)) if char_counts else 0.0,
        min_characters=min(char_counts) if char_counts else None,
        max_characters=max(char_counts) if char_counts else None,
        unsupported_by_extension=dict(sorted(unsupported_by_ext.items())),
        analyzed_by_extension=dict(sorted(acc.analyzed_by_ext.items())),
        load_error_paths=tuple(load_errors[: cfg.max_affected_documents]),
    )

    findings: list[Finding] = build_findings(acc, cfg)

    if unsupported:
        sample = [item.relative_path for item in unsupported[: cfg.max_affected_documents]]
        findings.append(
            Finding(
                code="UNSUPPORTED_FILES",
                category="inventory",
                severity=Severity.INFO,
                title="Unsupported files skipped",
                message=(
                    f"{len(unsupported)} file(s) were discovered but skipped because "
                    f"their formats are not supported ({SUPPORTED_FORMAT_HELP})."
                ),
                why_it_matters=(
                    "Unsupported files are not analyzed. If they contain important "
                    "knowledge, they will be missing from any RAG index built only from "
                    "supported formats."
                ),
                recommendation=(
                    "Convert important unsupported documents to a supported format, "
                    "or extend loaders for additional formats."
                ),
                evidence={
                    "count": len(unsupported),
                    "by_extension": dict(sorted(unsupported_by_ext.items())),
                    "sample_paths": sample[: cfg.max_sample_paths],
                    "affected_document_count": len(unsupported),
                    "affected_documents_truncated": len(unsupported) > len(sample),
                },
                affected_documents=tuple(sample),
            )
        )

    if load_errors:
        sample = load_errors[: cfg.max_affected_documents]
        findings.append(
            Finding(
                code="LOAD_ERRORS",
                category="inventory",
                severity=Severity.MEDIUM,
                title="Documents failed to load",
                message=(
                    f"{len(load_errors)} supported file(s) could not be loaded. "
                    "Other documents were still analyzed."
                ),
                why_it_matters=(
                    "Files that fail to load are excluded from analysis and would also "
                    "fail naive ingestion pipelines. One bad file should not stop a corpus."
                ),
                recommendation=(
                    "Inspect encoding, encryption, or file integrity for the listed paths."
                ),
                evidence={
                    "count": len(load_errors),
                    "sample_paths": sample[: cfg.max_sample_paths],
                    "affected_document_count": len(load_errors),
                    "affected_documents_truncated": len(load_errors) > len(sample),
                },
                affected_documents=tuple(sample),
            )
        )

    findings.sort(key=lambda f: (SEVERITY_ORDER[f.severity], f.code, f.title))

    # Release large accumulator structures before returning.
    del acc

    return AnalysisReport(
        product=PRODUCT_NAME,
        capability=CAPABILITY_NAME,
        version=__version__,
        summary=summary,
        findings=findings,
        config=cfg.to_dict(),
    )


def default_progress_callback(current: int, total: int, *, every: int, stream=None) -> None:
    """Write progress to *stream* (default stderr) without flooding."""
    if every <= 0 or total <= 0:
        return
    out = stream if stream is not None else sys.stderr
    if current == 1:
        print(f"Analyzing {total} document(s)...", file=out, flush=True)
    if current == total or current % every == 0:
        print(f"{current:,} / {total:,}", file=out, flush=True)
