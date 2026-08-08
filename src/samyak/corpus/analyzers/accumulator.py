"""Bounded finding path accumulation for large corpora."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from samyak.corpus.config import AnalysisConfig


@dataclass
class PathBucket:
    """Collect affected paths with a hard cap on retained examples."""

    count: int = 0
    paths: list[str] = field(default_factory=list)
    extras: list[Any] = field(default_factory=list)

    def add(self, path: str, *, extra: Any | None = None, limit: int) -> None:
        self.count += 1
        if len(self.paths) < limit:
            self.paths.append(path)
        if extra is not None and len(self.extras) < limit:
            self.extras.append(extra)


@dataclass
class CorpusAccumulator:
    """Minimal cross-document state retained after each document is released."""

    config: AnalysisConfig
    buckets: dict[str, PathBucket] = field(default_factory=lambda: defaultdict(PathBucket))
    hash_to_paths: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    html_first_line_to_paths: dict[str, list[str]] = field(
        default_factory=lambda: defaultdict(list)
    )
    html_doc_count: int = 0
    analyzed_documents: int = 0
    char_counts: list[int] = field(default_factory=list)
    total_chars: int = 0
    total_bytes: int = 0
    analyzed_by_ext: dict[str, int] = field(default_factory=dict)

    @property
    def path_limit(self) -> int:
        return self.config.max_affected_documents

    def note(self, code: str, path: str, *, extra: Any | None = None) -> None:
        self.buckets[code].add(path, extra=extra, limit=self.path_limit)

    def note_hash(self, digest: str, path: str) -> None:
        self.hash_to_paths[digest].append(path)

    def note_html_first_line(self, line: str, path: str) -> None:
        self.html_first_line_to_paths[line].append(path)

    def record_document_stats(self, *, extension: str, char_count: int, size_bytes: int) -> None:
        self.analyzed_documents += 1
        self.char_counts.append(char_count)
        self.total_chars += char_count
        self.total_bytes += size_bytes
        key = extension or "(none)"
        self.analyzed_by_ext[key] = self.analyzed_by_ext.get(key, 0) + 1
