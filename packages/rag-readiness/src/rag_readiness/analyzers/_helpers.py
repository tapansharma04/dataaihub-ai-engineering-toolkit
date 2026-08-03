"""Shared helpers for analyzers."""

from __future__ import annotations

from collections.abc import Sequence

from rag_readiness.config import AnalysisConfig


def sample_paths(paths: Sequence[str], config: AnalysisConfig) -> list[str]:
    """Return a bounded sample of paths for evidence sections."""
    limit = config.max_sample_paths
    return list(paths[:limit])


def meaningful_text(text: str) -> str:
    """Return stripped text used to decide emptiness / low information."""
    return text.strip()
