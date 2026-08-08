"""Exact duplicate detection via normalized content hashing."""

from __future__ import annotations

from collections import defaultdict

from samyak.corpus.analyzers._helpers import sample_paths
from samyak.corpus.analyzers.hashing import content_hash, normalize_for_hash, normalized_is_empty
from samyak.corpus.config import AnalysisConfig
from samyak.corpus.models import Document, Finding, Severity

__all__ = [
    "analyze_exact_duplicates",
    "content_hash",
    "normalize_for_hash",
]


def analyze_exact_duplicates(
    documents: list[Document],
    config: AnalysisConfig,
) -> list[Finding]:
    """Detect groups of documents with identical normalized content."""
    groups: dict[str, list[str]] = defaultdict(list)
    for doc in documents:
        if normalized_is_empty(doc.text):
            continue
        groups[content_hash(doc.text)].append(doc.path)

    duplicate_groups = [sorted(paths) for paths in groups.values() if len(paths) > 1]
    duplicate_groups.sort(key=lambda g: (-len(g), g[0]))

    if not duplicate_groups:
        return []

    affected_all = sorted({path for group in duplicate_groups for path in group})
    affected = affected_all[: config.max_affected_documents]
    group_summaries = [
        {
            "size": len(group),
            "representative_path": group[0],
            "paths": sample_paths(group, config),
        }
        for group in duplicate_groups[: config.max_sample_paths]
    ]

    return [
        Finding(
            code="EXACT_DUPLICATES",
            category="duplicates",
            severity=Severity.HIGH,
            title="Exact duplicate documents",
            message=(
                f"{len(affected_all)} documents belong to {len(duplicate_groups)} "
                "exact-duplicate group(s)."
            ),
            why_it_matters=(
                "Duplicate content can produce redundant retrieval results, inflate "
                "embedding and storage cost, and waste context window space when multiple "
                "near-identical chunks are retrieved together."
            ),
            recommendation=(
                "Review and remove duplicate documents before indexing unless "
                "duplication is intentional (for example, mirrored sources)."
            ),
            evidence={
                "duplicate_group_count": len(duplicate_groups),
                "affected_document_count": len(affected_all),
                "groups": group_summaries,
                "affected_documents_truncated": len(affected_all) > len(affected),
            },
            affected_documents=tuple(affected),
        )
    ]
