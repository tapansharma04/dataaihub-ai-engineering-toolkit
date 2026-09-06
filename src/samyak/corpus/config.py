"""Analysis configuration and documented default thresholds."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def _require_non_negative_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _require_unit_interval(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number between 0 and 1 inclusive")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1 inclusive")


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Deterministic thresholds for corpus checks.

    All thresholds are heuristics intended to surface review candidates,
    not absolute rules that every corpus must satisfy.
    """

    # Documents with fewer than this many characters (after strip) are "very small".
    small_document_chars: int = 100

    # Documents with at least this many characters are "very large".
    large_document_chars: int = 100_000

    # Text-quality: flag if non-alphanumeric (excluding whitespace) exceeds this ratio.
    # Conservative to avoid labeling code/markdown-heavy docs as noisy.
    high_symbol_ratio: float = 0.55

    # Text-quality: flag if a single unique line accounts for at least this fraction
    # of non-empty lines (and there are enough lines to judge).
    repeated_line_ratio: float = 0.60
    repeated_line_min_lines: int = 8

    # Text-quality: flag if whitespace characters exceed this fraction of all chars.
    excessive_whitespace_ratio: float = 0.45

    # Chunkability: paragraph (blank-line separated) longer than this may need review.
    long_paragraph_chars: int = 5_000

    # Chunkability: if this fraction of non-empty lines are shorter than
    # short_line_chars, the document may be overly fragmented.
    short_line_chars: int = 20
    short_line_ratio: float = 0.70
    short_line_min_lines: int = 15

    # Max sample paths included in finding evidence (avoid huge reports).
    max_sample_paths: int = 10

    # Max paths retained on Finding.affected_documents (full counts stay in evidence).
    max_affected_documents: int = 50

    # Progress reporting cadence (0 disables). Progress always goes to stderr.
    progress_every: int = 10_000

    # PDF: page with fewer than this many chars is considered text-poor.
    pdf_empty_page_chars: int = 20
    # PDF: fraction of text-poor pages to flag as OCR/likely-scanned concern.
    pdf_text_poor_page_ratio: float = 0.40
    # PDF: minimum pages before page-distribution checks apply.
    pdf_min_pages_for_distribution: int = 3
    # PDF: header/footer line must appear on at least this fraction of pages.
    pdf_header_footer_page_ratio: float = 0.60
    # PDF: high page-count informational threshold.
    pdf_high_page_count: int = 100
    # PDF: table-like line heuristic (columns separated by 2+ spaces).
    pdf_table_like_line_ratio: float = 0.25
    pdf_table_like_min_lines: int = 20

    # HTML: if extracted content chars / raw file chars is below this, low main content.
    html_low_content_ratio: float = 0.05
    html_low_content_min_raw_bytes: int = 5_000
    # HTML: navigation/boilerplate token share of extracted text.
    html_boilerplate_token_ratio: float = 0.45
    # HTML: large document character threshold (structure-aware chunking).
    html_large_document_chars: int = 80_000

    def __post_init__(self) -> None:
        _require_non_negative_int("small_document_chars", self.small_document_chars)
        _require_non_negative_int("large_document_chars", self.large_document_chars)
        if self.large_document_chars < self.small_document_chars:
            raise ValueError("large_document_chars must be >= small_document_chars")

        _require_unit_interval("high_symbol_ratio", self.high_symbol_ratio)
        _require_unit_interval("repeated_line_ratio", self.repeated_line_ratio)
        _require_non_negative_int("repeated_line_min_lines", self.repeated_line_min_lines)
        _require_unit_interval("excessive_whitespace_ratio", self.excessive_whitespace_ratio)

        _require_non_negative_int("long_paragraph_chars", self.long_paragraph_chars)
        _require_non_negative_int("short_line_chars", self.short_line_chars)
        _require_unit_interval("short_line_ratio", self.short_line_ratio)
        _require_non_negative_int("short_line_min_lines", self.short_line_min_lines)

        _require_non_negative_int("max_sample_paths", self.max_sample_paths)
        _require_non_negative_int("max_affected_documents", self.max_affected_documents)
        _require_non_negative_int("progress_every", self.progress_every)

        _require_non_negative_int("pdf_empty_page_chars", self.pdf_empty_page_chars)
        _require_unit_interval("pdf_text_poor_page_ratio", self.pdf_text_poor_page_ratio)
        _require_non_negative_int(
            "pdf_min_pages_for_distribution", self.pdf_min_pages_for_distribution
        )
        _require_unit_interval("pdf_header_footer_page_ratio", self.pdf_header_footer_page_ratio)
        _require_non_negative_int("pdf_high_page_count", self.pdf_high_page_count)
        _require_unit_interval("pdf_table_like_line_ratio", self.pdf_table_like_line_ratio)
        _require_non_negative_int("pdf_table_like_min_lines", self.pdf_table_like_min_lines)

        _require_unit_interval("html_low_content_ratio", self.html_low_content_ratio)
        _require_non_negative_int(
            "html_low_content_min_raw_bytes", self.html_low_content_min_raw_bytes
        )
        _require_unit_interval("html_boilerplate_token_ratio", self.html_boilerplate_token_ratio)
        _require_non_negative_int("html_large_document_chars", self.html_large_document_chars)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
