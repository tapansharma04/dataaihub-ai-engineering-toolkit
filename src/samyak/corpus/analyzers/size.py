"""Very small and very large document checks."""

from __future__ import annotations

from samyak.corpus.analyzers._helpers import meaningful_text, sample_paths
from samyak.corpus.config import AnalysisConfig
from samyak.corpus.models import Document, Finding, Severity


def analyze_document_sizes(
    documents: list[Document],
    config: AnalysisConfig,
) -> list[Finding]:
    """Flag unusually small or large documents using documented thresholds."""
    findings: list[Finding] = []

    small: list[tuple[str, int]] = []
    large: list[tuple[str, int]] = []

    for doc in documents:
        content = meaningful_text(doc.text)
        if not content:
            continue  # empty handled separately
        length = len(content)
        if length < config.small_document_chars:
            small.append((doc.path, length))
        if length >= config.large_document_chars:
            large.append((doc.path, length))

    if small:
        paths = [path for path, _ in small]
        findings.append(
            Finding(
                code="VERY_SMALL_DOCUMENTS",
                category="document_size",
                severity=Severity.MEDIUM,
                title="Very small / low-information documents",
                message=(
                    f"{len(small)} of {len(documents)} documents have fewer than "
                    f"{config.small_document_chars} characters of meaningful text."
                ),
                why_it_matters=(
                    "Very short documents may not be useful as independent retrieval "
                    "units. They can also indicate truncated exports or incomplete files. "
                    "Not every small document is bad—short policies, FAQs, or titles may "
                    "be intentional."
                ),
                recommendation=(
                    "Review small documents: merge related stubs, restore truncated "
                    "exports, or keep them only when they are intentionally concise."
                ),
                evidence={
                    "threshold_chars": config.small_document_chars,
                    "count": len(small),
                    "sample_paths": sample_paths(paths, config),
                    "sample_lengths": [length for _, length in small[: config.max_sample_paths]],
                },
                affected_documents=tuple(paths),
            )
        )

    if large:
        paths = [path for path, _ in large]
        findings.append(
            Finding(
                code="VERY_LARGE_DOCUMENTS",
                category="document_size",
                severity=Severity.MEDIUM,
                title="Very large documents",
                message=(
                    f"{len(large)} of {len(documents)} documents have at least "
                    f"{config.large_document_chars} characters of meaningful text."
                ),
                why_it_matters=(
                    "Very large documents often need deliberate parsing and chunking. "
                    "Naive whole-document embedding can lose precision; naive fixed "
                    "chunking can split important context awkwardly."
                ),
                recommendation=(
                    "Review parsing and chunking strategy for these documents "
                    "(structure-aware splits, section boundaries, or hierarchical retrieval)."
                ),
                evidence={
                    "threshold_chars": config.large_document_chars,
                    "count": len(large),
                    "sample_paths": sample_paths(paths, config),
                    "sample_lengths": [length for _, length in large[: config.max_sample_paths]],
                },
                affected_documents=tuple(paths),
            )
        )

    return findings
