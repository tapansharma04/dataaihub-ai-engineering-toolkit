"""Document loaders for supported text formats."""

from __future__ import annotations

from pathlib import Path

from rag_readiness.discovery import DiscoveredFile
from rag_readiness.models import Document


class DocumentLoadError(Exception):
    """Raised when a supported file cannot be loaded as text."""


def load_document(discovered: DiscoveredFile) -> Document:
    """Load a supported discovered file into a :class:`Document`."""
    if not discovered.supported:
        raise DocumentLoadError(f"Unsupported extension for loading: {discovered.extension}")
    return _load_text_file(discovered)


def _load_text_file(discovered: DiscoveredFile) -> Document:
    path = discovered.absolute_path
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DocumentLoadError(f"Failed to read file: {discovered.relative_path}") from exc

    try:
        text = raw.decode("utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        try:
            text = raw.decode("utf-8-sig")
            encoding = "utf-8-sig"
        except UnicodeDecodeError:
            try:
                text = raw.decode("latin-1")
                encoding = "latin-1"
            except UnicodeDecodeError as exc:
                raise DocumentLoadError(
                    f"Failed to decode text: {discovered.relative_path}"
                ) from exc

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    return Document(
        path=discovered.relative_path,
        filename=Path(discovered.relative_path).name,
        extension=discovered.extension,
        text=normalized,
        size_bytes=len(raw),
        char_count=len(normalized),
        line_count=len(lines),
        metadata={"encoding": encoding},
    )
