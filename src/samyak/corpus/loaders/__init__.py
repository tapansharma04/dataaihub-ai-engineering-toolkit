"""Document loaders for supported formats."""

from __future__ import annotations

from samyak.corpus.discovery import (
    HTML_EXTENSIONS,
    PDF_EXTENSIONS,
    TEXT_EXTENSIONS,
    DiscoveredFile,
)
from samyak.corpus.loaders.errors import DocumentLoadError
from samyak.corpus.loaders.html import load_html_document
from samyak.corpus.loaders.pdf import load_pdf_document
from samyak.corpus.loaders.text import load_text_document
from samyak.corpus.models import Document

__all__ = ["DocumentLoadError", "load_document"]


def load_document(discovered: DiscoveredFile) -> Document:
    """Load a supported discovered file into a :class:`Document`."""
    if not discovered.supported:
        raise DocumentLoadError(f"Unsupported extension for loading: {discovered.extension}")
    ext = discovered.extension.lower()
    try:
        if ext in TEXT_EXTENSIONS:
            return load_text_document(discovered)
        if ext in PDF_EXTENSIONS:
            return load_pdf_document(discovered)
        if ext in HTML_EXTENSIONS:
            return load_html_document(discovered)
    except DocumentLoadError:
        raise
    except Exception as exc:  # noqa: BLE001 - convert any loader failure
        raise DocumentLoadError(f"Failed to load {discovered.relative_path}: {exc}") from exc
    raise DocumentLoadError(f"No loader registered for: {discovered.extension}")
