"""Orchestrate discovery, loading, and analysis."""

from __future__ import annotations

from pathlib import Path
from statistics import median

from rag_readiness.__version__ import __version__
from rag_readiness.analyzers import (
    analyze_chunkability,
    analyze_document_sizes,
    analyze_empty_documents,
    analyze_exact_duplicates,
    analyze_text_quality,
)
from rag_readiness.config import AnalysisConfig
from rag_readiness.discovery import discover_files, validate_corpus_root
from rag_readiness.loaders import DocumentLoadError, load_document
from rag_readiness.models import (
    SEVERITY_ORDER,
    AnalysisReport,
    CorpusSummary,
    Document,
    Finding,
    Severity,
)

UTILITY_NAME = "rag-readiness"


def analyze_corpus(
    corpus_path: str | Path,
    config: AnalysisConfig | None = None,
) -> AnalysisReport:
    """Analyze a local document corpus and return an :class:`AnalysisReport`."""
    cfg = config or AnalysisConfig()
    root = validate_corpus_root(Path(corpus_path))
    discovered = discover_files(root)

    supported = [item for item in discovered if item.supported]
    unsupported = [item for item in discovered if not item.supported]

    documents: list[Document] = []
    load_errors: list[str] = []
    for item in supported:
        try:
            documents.append(load_document(item))
        except DocumentLoadError:
            load_errors.append(item.relative_path)

    unsupported_by_ext: dict[str, int] = {}
    for item in unsupported:
        key = item.extension or "(none)"
        unsupported_by_ext[key] = unsupported_by_ext.get(key, 0) + 1

    char_counts = [doc.char_count for doc in documents]
    total_chars = sum(char_counts)
    total_bytes = sum(doc.size_bytes for doc in documents)

    summary = CorpusSummary(
        corpus_root=root.name,
        total_discovered_files=len(discovered),
        supported_files=len(supported),
        unsupported_files=len(unsupported),
        analyzed_documents=len(documents),
        load_errors=len(load_errors),
        total_characters=total_chars,
        total_bytes=total_bytes,
        average_characters=(total_chars / len(documents)) if documents else 0.0,
        median_characters=float(median(char_counts)) if char_counts else 0.0,
        min_characters=min(char_counts) if char_counts else None,
        max_characters=max(char_counts) if char_counts else None,
        unsupported_by_extension=dict(sorted(unsupported_by_ext.items())),
        load_error_paths=tuple(load_errors),
    )

    findings: list[Finding] = []
    findings.extend(analyze_empty_documents(documents, cfg))
    findings.extend(analyze_document_sizes(documents, cfg))
    findings.extend(analyze_exact_duplicates(documents, cfg))
    findings.extend(analyze_text_quality(documents, cfg))
    findings.extend(analyze_chunkability(documents, cfg))

    if unsupported:
        findings.append(
            Finding(
                code="UNSUPPORTED_FILES",
                category="inventory",
                severity=Severity.INFO,
                title="Unsupported files skipped",
                message=(
                    f"{len(unsupported)} file(s) were discovered but skipped because "
                    "their formats are not supported in v0.1 (.txt and .md only)."
                ),
                why_it_matters=(
                    "Unsupported files are not analyzed. If they contain important "
                    "knowledge, they will be missing from any RAG index built only from "
                    "supported formats."
                ),
                recommendation=(
                    "Convert important unsupported documents to .txt/.md for analysis, "
                    "or wait for additional format support in a later release."
                ),
                evidence={
                    "count": len(unsupported),
                    "by_extension": dict(sorted(unsupported_by_ext.items())),
                    "sample_paths": [
                        item.relative_path for item in unsupported[: cfg.max_sample_paths]
                    ],
                },
                affected_documents=tuple(item.relative_path for item in unsupported),
            )
        )

    if load_errors:
        findings.append(
            Finding(
                code="LOAD_ERRORS",
                category="inventory",
                severity=Severity.MEDIUM,
                title="Documents failed to load",
                message=f"{len(load_errors)} supported file(s) could not be loaded as text.",
                why_it_matters=(
                    "Files that fail to load are excluded from analysis and would also "
                    "fail naive ingestion pipelines."
                ),
                recommendation=(
                    "Inspect encoding or file integrity for the listed paths and re-export "
                    "as UTF-8 text when possible."
                ),
                evidence={
                    "count": len(load_errors),
                    "sample_paths": list(load_errors[: cfg.max_sample_paths]),
                },
                affected_documents=tuple(load_errors),
            )
        )

    findings.sort(key=lambda f: (SEVERITY_ORDER[f.severity], f.code, f.title))

    return AnalysisReport(
        utility=UTILITY_NAME,
        version=__version__,
        summary=summary,
        findings=findings,
        config=cfg.to_dict(),
    )
