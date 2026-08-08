"""Empty and whitespace-only document checks."""

from __future__ import annotations

from samyak.corpus.analyzers._helpers import meaningful_text, sample_paths
from samyak.corpus.config import AnalysisConfig
from samyak.corpus.models import Document, Finding, Severity


def analyze_empty_documents(
    documents: list[Document],
    config: AnalysisConfig,
) -> list[Finding]:
    """Detect documents with no meaningful text content."""
    empty_paths = [doc.path for doc in documents if not meaningful_text(doc.text)]
    if not empty_paths:
        return []

    return [
        Finding(
            code="EMPTY_DOCUMENTS",
            category="empty_documents",
            severity=Severity.HIGH,
            title="Empty documents",
            message=(
                f"{len(empty_paths)} of {len(documents)} documents contain no meaningful text."
            ),
            why_it_matters=(
                "Empty documents usually should not be indexed. They waste embedding and "
                "storage cost and can pollute retrieval with contentless chunks."
            ),
            recommendation=(
                "Remove empty documents from the corpus before indexing, or fix the "
                "extraction/export process that produced them."
            ),
            evidence={
                "count": len(empty_paths),
                "sample_paths": sample_paths(empty_paths, config),
            },
            affected_documents=tuple(empty_paths),
        )
    ]
