"""Lightweight deterministic text-quality / noise indicators."""

from __future__ import annotations

from collections import Counter

from samyak.corpus.analyzers._helpers import meaningful_text, sample_paths
from samyak.corpus.config import AnalysisConfig
from samyak.corpus.models import Document, Finding, Severity


def analyze_text_quality(
    documents: list[Document],
    config: AnalysisConfig,
) -> list[Finding]:
    """Flag documents with conservative noise heuristics.

    Findings are phrased as potential issues. Technical documents with code,
    markup, or symbols are not automatically labeled as bad.
    """
    findings: list[Finding] = []

    high_symbol: list[str] = []
    excessive_ws: list[str] = []
    repeated_lines: list[str] = []

    for doc in documents:
        text = doc.text
        if not meaningful_text(text):
            continue

        if doc.metadata.get("code_heavy"):
            # Legitimate technical documentation with large code samples.
            pass
        elif _high_symbol_ratio(text, config.high_symbol_ratio):
            high_symbol.append(doc.path)
        if _excessive_whitespace(text, config.excessive_whitespace_ratio):
            excessive_ws.append(doc.path)
        if _repeated_line_noise(
            text,
            config.repeated_line_ratio,
            config.repeated_line_min_lines,
        ):
            repeated_lines.append(doc.path)

    if high_symbol:
        findings.append(
            Finding(
                code="HIGH_SYMBOL_RATIO",
                category="text_quality",
                severity=Severity.LOW,
                title="Potential high symbol / non-alphanumeric ratio",
                message=(
                    f"{len(high_symbol)} document(s) have a high ratio of non-alphanumeric "
                    f"characters (threshold {config.high_symbol_ratio:.0%})."
                ),
                why_it_matters=(
                    "Unusually symbol-heavy text can indicate extraction noise, corrupted "
                    "exports, or binary-like content. Legitimate code and markup can also "
                    "trigger this heuristic, so treat matches as candidates for review."
                ),
                recommendation=(
                    "Spot-check flagged files. If content is corrupted or non-textual, "
                    "exclude or re-extract before indexing."
                ),
                evidence={
                    "threshold": config.high_symbol_ratio,
                    "count": len(high_symbol),
                    "sample_paths": sample_paths(high_symbol, config),
                },
                affected_documents=tuple(high_symbol),
            )
        )

    if excessive_ws:
        findings.append(
            Finding(
                code="EXCESSIVE_WHITESPACE",
                category="text_quality",
                severity=Severity.LOW,
                title="Potential excessive whitespace",
                message=(
                    f"{len(excessive_ws)} document(s) appear dominated by whitespace "
                    f"(threshold {config.excessive_whitespace_ratio:.0%})."
                ),
                why_it_matters=(
                    "Whitespace-heavy documents may be sparsely extracted, padded, or "
                    "poorly converted. Indexing them can yield low-value chunks."
                ),
                recommendation=(
                    "Inspect extraction quality and normalize whitespace if the content "
                    "is otherwise useful."
                ),
                evidence={
                    "threshold": config.excessive_whitespace_ratio,
                    "count": len(excessive_ws),
                    "sample_paths": sample_paths(excessive_ws, config),
                },
                affected_documents=tuple(excessive_ws),
            )
        )

    if repeated_lines:
        findings.append(
            Finding(
                code="REPEATED_LINE_CONTENT",
                category="text_quality",
                severity=Severity.MEDIUM,
                title="Potential repeated-line / boilerplate noise",
                message=(
                    f"{len(repeated_lines)} document(s) show heavy line repetition "
                    f"(one line accounting for ≥{config.repeated_line_ratio:.0%} of "
                    "non-empty lines)."
                ),
                why_it_matters=(
                    "Repeated lines often indicate logging dumps, copy-paste corruption, "
                    "or boilerplate-dominated files that reduce retrieval usefulness."
                ),
                recommendation=(
                    "Review flagged documents and deduplicate or trim repetitive content "
                    "before indexing when the repetition is not intentional."
                ),
                evidence={
                    "threshold_ratio": config.repeated_line_ratio,
                    "min_lines": config.repeated_line_min_lines,
                    "count": len(repeated_lines),
                    "sample_paths": sample_paths(repeated_lines, config),
                },
                affected_documents=tuple(repeated_lines),
            )
        )

    return findings


def _high_symbol_ratio(text: str, threshold: float) -> bool:
    non_ws = [ch for ch in text if not ch.isspace()]
    if len(non_ws) < 40:
        return False
    symbolish = sum(1 for ch in non_ws if not ch.isalnum())
    return (symbolish / len(non_ws)) >= threshold


def _excessive_whitespace(text: str, threshold: float) -> bool:
    if len(text) < 80:
        return False
    ws = sum(1 for ch in text if ch.isspace())
    return (ws / len(text)) >= threshold


def _repeated_line_noise(text: str, ratio: float, min_lines: int) -> bool:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) < min_lines:
        return False
    counts = Counter(lines)
    _most_common_line, most_common_count = counts.most_common(1)[0]
    return (most_common_count / len(lines)) >= ratio
