"""Plain text / Markdown loaders."""

from __future__ import annotations

from pathlib import Path

from samyak.corpus.discovery import DiscoveredFile
from samyak.corpus.loaders.errors import DocumentLoadError
from samyak.corpus.models import Document


def load_text_document(discovered: DiscoveredFile) -> Document:
    """Load a .txt or .md file as UTF-8/latin-1 text."""
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
        metadata={"encoding": encoding, "format": "text"},
    )
