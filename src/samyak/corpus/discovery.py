"""Filesystem discovery for corpus files."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = frozenset({".txt", ".md", ".pdf", ".html", ".htm"})
TEXT_EXTENSIONS = frozenset({".txt", ".md"})
PDF_EXTENSIONS = frozenset({".pdf"})
HTML_EXTENSIONS = frozenset({".html", ".htm"})

WalkFn = Callable[..., Iterator[tuple[str, list[str], list[str]]]]
CheckPathFn = Callable[[Path, Path], "PathCheckResult"]


@dataclass(frozen=True, slots=True)
class DiscoveredFile:
    """A file found under the corpus root."""

    absolute_path: Path
    relative_path: str
    extension: str
    supported: bool


@dataclass(frozen=True, slots=True)
class DiscoveryAccessError:
    """A filesystem path that could not be inspected during discovery."""

    relative_path: str
    reason: str


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """Discovered files plus access failures that were not silently dropped."""

    files: list[DiscoveredFile]
    access_errors: tuple[DiscoveryAccessError, ...]


@dataclass(frozen=True, slots=True)
class PathCheckResult:
    """Outcome of inspecting one directory entry during discovery."""

    include: bool
    error_reason: str | None = None


class CorpusPathError(ValueError):
    """Raised when the corpus path is invalid."""


def validate_corpus_root(path: Path) -> Path:
    """Resolve and validate that ``path`` is an existing directory."""
    if not path.exists():
        raise CorpusPathError(f"Corpus path does not exist: {path}")
    if not path.is_dir():
        raise CorpusPathError(f"Corpus path is not a directory: {path}")
    return path.resolve()


def discover_files(
    corpus_root: Path,
    *,
    walk: WalkFn | None = None,
    check_path: CheckPathFn | None = None,
) -> DiscoveryResult:
    """Discover files under ``corpus_root`` without following external symlinks.

    Uses ``os.walk(..., followlinks=False)`` by default. Symlinked files whose
    targets resolve outside the corpus root are skipped (security), not treated
    as access errors. Filesystem access failures are recorded in
    ``access_errors`` instead of being discarded.
    """
    root = validate_corpus_root(corpus_root)
    walk_fn = walk if walk is not None else _walk
    check_fn = check_path if check_path is not None else _check_path
    discovered: list[DiscoveredFile] = []
    access_errors: list[DiscoveryAccessError] = []

    def on_walk_error(err: OSError) -> None:
        access_errors.append(
            DiscoveryAccessError(
                relative_path=_relative_from_oserror(err, root),
                reason="unreadable_directory",
            )
        )

    for dirpath, dirnames, filenames in walk_fn(root, followlinks=False, onerror=on_walk_error):
        # Deterministic descent if a walker yields unsorted directory names.
        dirnames.sort()
        current = Path(dirpath)
        for name in sorted(filenames):
            file_path = current / name
            checked = check_fn(file_path, root)
            if checked.error_reason is not None:
                access_errors.append(
                    DiscoveryAccessError(
                        relative_path=_relative_path(file_path, root),
                        reason=checked.error_reason,
                    )
                )
                continue
            if not checked.include:
                continue
            relative = _relative_path(file_path, root)
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
    access_errors.sort(key=lambda item: (item.relative_path, item.reason))
    return DiscoveryResult(files=discovered, access_errors=tuple(access_errors))


def _walk(
    root: Path,
    *,
    followlinks: bool = False,
    onerror: Callable[[OSError], None] | None = None,
) -> Iterator[tuple[str, list[str], list[str]]]:
    return os.walk(root, followlinks=followlinks, onerror=onerror)


def _check_path(file_path: Path, root: Path) -> PathCheckResult:
    """Return whether ``file_path`` is a safe regular file inside ``root``."""
    try:
        if file_path.is_symlink():
            resolved = file_path.resolve()
            if not resolved.is_relative_to(root):
                return PathCheckResult(include=False)
            if not resolved.is_file():
                return PathCheckResult(include=False)
            return PathCheckResult(include=True)
        if not file_path.is_file():
            return PathCheckResult(include=False)
        return PathCheckResult(include=True)
    except OSError:
        return PathCheckResult(include=False, error_reason="inaccessible")


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _relative_from_oserror(err: OSError, root: Path) -> str:
    filename = getattr(err, "filename", None)
    if not filename:
        return "."
    return _relative_path(Path(filename), root)
