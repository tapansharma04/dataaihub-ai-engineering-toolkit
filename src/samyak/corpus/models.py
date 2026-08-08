"""Core data models for corpus analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class Severity(StrEnum):
    """Finding severity levels."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


SEVERITY_ORDER = {
    Severity.HIGH: 0,
    Severity.MEDIUM: 1,
    Severity.LOW: 2,
    Severity.INFO: 3,
}


@dataclass(frozen=True, slots=True)
class PageInfo:
    """Page-level extraction summary for paginated formats (e.g. PDF)."""

    index: int
    char_count: int
    line_count: int
    has_images: bool = False
    text_preview_chars: int = 0


@dataclass(frozen=True, slots=True)
class Document:
    """Loaded document ready for analysis.

    Paths are stored relative to the corpus root when possible so reports
    do not leak unnecessary absolute machine paths.

    ``text`` is always a normalized plain-text view for generic analyzers.
    Format-specific details live in ``metadata`` / ``pages``.
    """

    path: str
    filename: str
    extension: str
    text: str
    size_bytes: int
    char_count: int
    line_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
    pages: tuple[PageInfo, ...] = ()

    @property
    def format_family(self) -> str:
        ext = self.extension.lower()
        if ext in {".txt", ".md"}:
            return "text"
        if ext == ".pdf":
            return "pdf"
        if ext in {".html", ".htm"}:
            return "html"
        return "other"


@dataclass(frozen=True, slots=True)
class Finding:
    """An explainable corpus issue or observation."""

    code: str
    category: str
    severity: Severity
    title: str
    message: str
    why_it_matters: str
    recommendation: str
    evidence: dict[str, Any] = field(default_factory=dict)
    affected_documents: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["severity"] = self.severity.value
        data["affected_documents"] = list(self.affected_documents)
        return data


@dataclass(slots=True)
class CorpusSummary:
    """High-level corpus inventory statistics."""

    corpus_root: str
    total_discovered_files: int
    supported_files: int
    unsupported_files: int
    analyzed_documents: int
    load_errors: int
    total_characters: int
    total_bytes: int
    average_characters: float
    median_characters: float
    min_characters: int | None
    max_characters: int | None
    unsupported_by_extension: dict[str, int] = field(default_factory=dict)
    analyzed_by_extension: dict[str, int] = field(default_factory=dict)
    load_error_paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["load_error_paths"] = list(self.load_error_paths)
        return data


@dataclass(slots=True)
class AnalysisReport:
    """Complete analysis result for a corpus."""

    product: str
    capability: str
    version: str
    summary: CorpusSummary
    findings: list[Finding]
    config: dict[str, Any]

    def severity_counts(self) -> dict[str, int]:
        counts = {s.value: 0 for s in Severity}
        for finding in self.findings:
            counts[finding.severity.value] += 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "product": self.product,
            "capability": self.capability,
            "version": self.version,
            "config": self.config,
            "summary": self.summary.to_dict(),
            "findings": [f.to_dict() for f in self.findings],
            "severity_counts": self.severity_counts(),
        }
