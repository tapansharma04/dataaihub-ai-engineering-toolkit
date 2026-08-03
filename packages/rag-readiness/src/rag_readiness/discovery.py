"""Filesystem discovery for corpus files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = frozenset({".txt", ".md"})


@dataclass(frozen=True, slots=True)
class DiscoveredFile:
    """A file found under the corpus root."""

    absolute_path: Path
    relative_path: str
    extension: str
    supported: bool


class CorpusPathError(ValueError):
    """Raised when the corpus path is invalid."""


def validate_corpus_root(path: Path) -> Path:
    """Resolve and validate that ``path`` is an existing directory."""
    if not path.exists():
        raise CorpusPathError(f"Corpus path does not exist: {path}")
    if not path.is_dir():
        raise CorpusPathError(f"Corpus path is not a directory: {path}")
    return path.resolve()


def discover_files(corpus_root: Path) -> list[DiscoveredFile]:
    """Discover files under ``corpus_root`` without following external symlinks.

    Uses ``os.walk(..., followlinks=False)``. Symlinked files whose targets
    resolve outside the corpus root are skipped.
    """
    root = validate_corpus_root(corpus_root)
    discovered: list[DiscoveredFile] = []

    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        for name in sorted(filenames):
            file_path = current / name
            if not _is_safe_file(file_path, root):
                continue
            relative = file_path.relative_to(root).as_posix()
            extension = file_path.suffix.lower()
            discovered.append(
                DiscoveredFile(
                    absolute_path=file_path,
                    relative_path=relative,
                    extension=extension,
                    supported=extension in SUPPORTED_EXTENSIONS,
                )
            )

    discovered.sort(key=lambda item: item.relative_path)
    return discovered


def _is_safe_file(file_path: Path, root: Path) -> bool:
    """Return True if the path is a regular file inside the corpus root."""
    try:
        if file_path.is_symlink():
            resolved = file_path.resolve()
            if not resolved.is_relative_to(root):
                return False
            return resolved.is_file()
        return file_path.is_file()
    except OSError:
        return False
