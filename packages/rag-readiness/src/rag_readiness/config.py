"""Analysis configuration and documented default thresholds."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
