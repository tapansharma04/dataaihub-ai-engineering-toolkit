"""Exact duplicate detection via normalized content hashing."""

from __future__ import annotations

import hashlib
from collections import defaultdict

from rag_readiness.analyzers._helpers import sample_paths
from rag_readiness.config import AnalysisConfig
from rag_readiness.models import Document, Finding, Severity


def normalize_for_hash(text: str) -> str:
    """Normalize text for exact-duplicate comparison.

    Collapses line endings already handled at load time; strips leading/trailing
    whitespace and normalizes internal runs of whitespace to a single space so
    trivial formatting-only differences still count as duplicates.
    """
    collapsed = " ".join(text.split())
    return collapsed


def content_hash(text: str) -> str:
    """Return a stable SHA-256 hex digest of normalized text."""
    normalized = normalize_for_hash(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def analyze_exact_duplicates(
    documents: list[Document],
    config: AnalysisConfig,
) -> list[Finding]:
    """Detect groups of documents with identical normalized content."""
    groups: dict[str, list[str]] = defaultdict(list)
    for doc in documents:
        if not normalize_for_hash(doc.text):
            continue
        groups[content_hash(doc.text)].append(doc.path)

    duplicate_groups = [sorted(paths) for paths in groups.values() if len(paths) > 1]
    duplicate_groups.sort(key=lambda g: (-len(g), g[0]))

    if not duplicate_groups:
        return []

    affected = sorted({path for group in duplicate_groups for path in group})
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
                f"{len(affected)} documents belong to {len(duplicate_groups)} "
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
                "affected_document_count": len(affected),
                "groups": group_summaries,
            },
            affected_documents=tuple(affected),
        )
    ]
