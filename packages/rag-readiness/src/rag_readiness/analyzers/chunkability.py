"""Chunkability indicators — not a chunking engine."""

from __future__ import annotations

from rag_readiness.analyzers._helpers import meaningful_text, sample_paths
from rag_readiness.config import AnalysisConfig
from rag_readiness.models import Document, Finding, Severity


def analyze_chunkability(
    documents: list[Document],
    config: AnalysisConfig,
) -> list[Finding]:
    """Detect simple structural traits that may complicate chunking."""
    long_paragraph_docs: list[str] = []
    fragmented_docs: list[str] = []

    for doc in documents:
        if not meaningful_text(doc.text):
            continue
        if _has_long_paragraph(doc.text, config.long_paragraph_chars):
            long_paragraph_docs.append(doc.path)
        if _is_overly_fragmented(
            doc.text,
            config.short_line_chars,
            config.short_line_ratio,
            config.short_line_min_lines,
        ):
            fragmented_docs.append(doc.path)

    findings: list[Finding] = []

    if long_paragraph_docs:
        findings.append(
            Finding(
                code="LONG_UNINTERRUPTED_BLOCKS",
                category="chunkability",
                severity=Severity.MEDIUM,
                title="Long uninterrupted text blocks",
                message=(
                    f"{len(long_paragraph_docs)} document(s) contain paragraph-like "
                    f"blocks of at least {config.long_paragraph_chars} characters "
                    "without a blank-line break."
                ),
                why_it_matters=(
                    "Very long uninterrupted blocks make structure-aware chunking harder. "
                    "Fixed-size chunking may split mid-argument and hurt retrieval precision. "
                    "This check does not choose an optimal chunk size."
                ),
                recommendation=(
                    "Consider structure-aware splitting (headings, sections) or smaller "
                    "semantic units before indexing these documents."
                ),
                evidence={
                    "threshold_chars": config.long_paragraph_chars,
                    "count": len(long_paragraph_docs),
                    "sample_paths": sample_paths(long_paragraph_docs, config),
                },
                affected_documents=tuple(long_paragraph_docs),
            )
        )

    if fragmented_docs:
        findings.append(
            Finding(
                code="OVERLY_FRAGMENTED_LINES",
                category="chunkability",
                severity=Severity.LOW,
                title="Documents dominated by very short lines",
                message=(
                    f"{len(fragmented_docs)} document(s) are dominated by very short lines "
                    f"(≥{config.short_line_ratio:.0%} of non-empty lines shorter than "
                    f"{config.short_line_chars} characters)."
                ),
                why_it_matters=(
                    "Highly fragmented text (menus, TOC dumps, broken extraction) can yield "
                    "weak standalone chunks and noisy retrieval results."
                ),
                recommendation=(
                    "Review whether these files should be reformatted, merged into richer "
                    "sections, or excluded from indexing."
                ),
                evidence={
                    "short_line_chars": config.short_line_chars,
                    "short_line_ratio": config.short_line_ratio,
                    "count": len(fragmented_docs),
                    "sample_paths": sample_paths(fragmented_docs, config),
                },
                affected_documents=tuple(fragmented_docs),
            )
        )

    return findings


def _has_long_paragraph(text: str, threshold: int) -> bool:
    paragraphs = text.split("\n\n")
    return any(len(paragraph.strip()) >= threshold for paragraph in paragraphs)


def _is_overly_fragmented(
    text: str,
    short_line_chars: int,
    short_line_ratio: float,
    min_lines: int,
) -> bool:
    lines = [line for line in text.split("\n") if line.strip()]
    if len(lines) < min_lines:
        return False
    short = sum(1 for line in lines if len(line.strip()) < short_line_chars)
    return (short / len(lines)) >= short_line_ratio
