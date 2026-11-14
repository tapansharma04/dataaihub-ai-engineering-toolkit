"""Project-tree discovery for Application Readiness.

Ignore semantics differ from Corpus Intelligence: generated, vendored, and
VCS directories are skipped rather than inventoried. Do not import corpus
discovery.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from samyak.app.errors import AppPathError
from samyak.app.models import (
    NOTICE_ANALYSIS_TRUNCATED,
    NOTICE_FILE_SKIPPED_BINARY,
    NOTICE_FILE_SKIPPED_TOO_LARGE,
    AppConfig,
    AppNotice,
)

ANALYZED_EXTENSIONS = frozenset({".py", ".json", ".toml"})

SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        ".pnpm-store",
        ".venv",
        "venv",
        "env",
        "__pypackages__",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        "dist",
        "build",
        "site-packages",
        "eggs",
        "target",
        "vendor",
        ".idea",
        ".vscode",
        ".cursor",
    }
)

SKIP_FILE_NAMES = frozenset({".DS_Store", "Thumbs.db"})

WalkFn = Callable[..., Iterator[tuple[str, list[str], list[str]]]]


@dataclass(frozen=True, slots=True)
class DiscoveredAppFile:
    """A candidate file that passed skip, symlink, size, and binary gates."""

    absolute_path: Path
    relative_path: str
    extension: str


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """Candidate files plus discovery notices. Source bytes are not retained."""

    files: tuple[DiscoveredAppFile, ...]
    notices: tuple[AppNotice, ...]
    skipped_oversized: int
    skipped_binary: int
    truncated: bool


def validate_project_root(path: Path) -> Path:
    """Resolve and validate that ``path`` is an existing directory."""
    if not path.exists():
        raise AppPathError(f"Application path does not exist: {path}")
    if not path.is_dir():
        raise AppPathError(f"Application path is not a directory: {path}")
    return path.resolve()


def discover_project(
    project_root: Path,
    config: AppConfig | None = None,
    *,
    walk: WalkFn | None = None,
) -> DiscoveryResult:
    """Discover analyzable files under ``project_root``.

    Uses ``os.walk(..., followlinks=False)`` by default. Directory symlinks are
    not descended. File symlinks whose targets resolve outside the project root,
    and broken symlinks, are skipped.
    """
    cfg = config or AppConfig()
    root = validate_project_root(project_root)
    walk_fn = walk if walk is not None else _walk
    discovered: list[DiscoveredAppFile] = []
    notices: list[AppNotice] = []
    skipped_oversized = 0
    skipped_binary = 0
    truncated = False
    oversized_notices = 0
    binary_notices = 0

    def on_walk_error(_err: OSError) -> None:
        return

    for dirpath, dirnames, filenames in walk_fn(root, followlinks=False, onerror=on_walk_error):
        dirnames[:] = sorted(name for name in dirnames if name not in SKIP_DIR_NAMES)
        current = Path(dirpath)
        if truncated:
            dirnames.clear()
            continue
        for name in sorted(filenames):
            if truncated:
                break
            if name in SKIP_FILE_NAMES:
                continue
            file_path = current / name
            relative = _relative_path(file_path, root)
            if not _include_file(file_path, root):
                continue
            extension = file_path.suffix.lower()
            if extension not in ANALYZED_EXTENSIONS:
                continue
            try:
                size = file_path.stat().st_size
            except OSError:
                continue
            if size > cfg.max_file_bytes:
                skipped_oversized += 1
                if oversized_notices < cfg.max_notices_per_code:
                    notices.append(
                        AppNotice(
                            code=NOTICE_FILE_SKIPPED_TOO_LARGE,
                            message=(
                                f"Skipped {relative}: file exceeds {cfg.max_file_bytes} bytes."
                            ),
                            path=relative,
                        )
                    )
                    oversized_notices += 1
                continue
            if _has_nul(file_path, cfg.nul_probe_bytes):
                skipped_binary += 1
                if binary_notices < cfg.max_notices_per_code:
                    notices.append(
                        AppNotice(
                            code=NOTICE_FILE_SKIPPED_BINARY,
                            message=f"Skipped {relative}: NUL byte in the first probe window.",
                            path=relative,
                        )
                    )
                    binary_notices += 1
                continue
            if len(discovered) >= cfg.max_analyzed_files:
                truncated = True
                dirnames.clear()
                notices.append(
                    AppNotice(
                        code=NOTICE_ANALYSIS_TRUNCATED,
                        message=(f"Analysis stopped after {cfg.max_analyzed_files} files."),
                    )
                )
                break
            discovered.append(
                DiscoveredAppFile(
                    absolute_path=file_path,
                    relative_path=relative,
                    extension=extension,
                )
            )

    discovered.sort(key=lambda item: item.relative_path)
    notices.sort(key=lambda item: (item.code, item.path or "", item.message))
    return DiscoveryResult(
        files=tuple(discovered),
        notices=tuple(notices),
        skipped_oversized=skipped_oversized,
        skipped_binary=skipped_binary,
        truncated=truncated,
    )


def _walk(
    root: Path,
    *,
    followlinks: bool = False,
    onerror: Callable[[OSError], None] | None = None,
) -> Iterator[tuple[str, list[str], list[str]]]:
    return os.walk(root, followlinks=followlinks, onerror=onerror)


def _include_file(file_path: Path, root: Path) -> bool:
    try:
        if file_path.is_symlink():
            resolved = file_path.resolve()
            if not resolved.is_relative_to(root):
                return False
            return resolved.is_file()
        return file_path.is_file()
    except OSError:
        return False


def _has_nul(file_path: Path, probe_bytes: int) -> bool:
    try:
        with file_path.open("rb") as handle:
            sample = handle.read(probe_bytes)
    except OSError:
        return False
    return b"\x00" in sample


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name
