"""PDF loader using pypdf (local text-layer extraction; no OCR, no network)."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from samyak.corpus.discovery import DiscoveredFile
from samyak.corpus.loaders.errors import DocumentLoadError
from samyak.corpus.models import Document, PageInfo


@contextmanager
def _quiet_pypdf() -> Iterator[None]:
    """Keep parser warnings off stderr; load failures still raise DocumentLoadError."""
    logger = logging.getLogger("pypdf")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        logger.setLevel(previous)


def load_pdf_document(discovered: DiscoveredFile) -> Document:
    """Extract text and page-level stats from a local PDF.

    Security posture: local file only; no JavaScript execution; no remote
    resource fetching. Scanned/image-only pages yield little or no text and
    are detectable via page stats (OCR is not performed).
    """
    path = discovered.absolute_path
    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        raise DocumentLoadError(f"Failed to stat PDF: {discovered.relative_path}") from exc

    try:
        with path.open("rb") as handle, _quiet_pypdf():
            reader = PdfReader(handle, strict=False)
            if getattr(reader, "is_encrypted", False):
                try:
                    # Empty password unlock for owner-restricted but openable files.
                    reader.decrypt("")
                except Exception as exc:  # noqa: BLE001
                    raise DocumentLoadError(
                        f"Encrypted PDF cannot be opened: {discovered.relative_path}"
                    ) from exc

            pages: list[PageInfo] = []
            page_texts: list[str] = []
            first_edge: list[str] = []
            last_edge: list[str] = []
            image_page_count = 0
            line_count = 0

            for index, page in enumerate(reader.pages):
                try:
                    text = page.extract_text() or ""
                except Exception:  # noqa: BLE001 - treat page extraction failure as empty
                    text = ""
                text = text.replace("\r\n", "\n").replace("\r", "\n")
                has_images = _page_has_images(page)
                if has_images:
                    image_page_count += 1
                page_lines = text.split("\n") if text else []
                line_count += len(page_lines)
                stripped_lines = [ln.strip() for ln in page_lines if ln.strip()]
                if stripped_lines:
                    if 8 <= len(stripped_lines[0]) <= 120:
                        first_edge.append(stripped_lines[0][:120])
                    if 8 <= len(stripped_lines[-1]) <= 120:
                        last_edge.append(stripped_lines[-1][:120])
                pages.append(
                    PageInfo(
                        index=index,
                        char_count=len(text.strip()),
                        line_count=len(page_lines) if text else 0,
                        has_images=has_images,
                        text_preview_chars=min(len(text), 200),
                    )
                )
                page_texts.append(text)

            combined = "\n\n".join(page_texts)
            # Release per-page list promptly after join (combined holds the bytes we need).
            page_texts.clear()
            meta_info = _safe_metadata(reader)

            return Document(
                path=discovered.relative_path,
                filename=Path(discovered.relative_path).name,
                extension=discovered.extension,
                text=combined,
                size_bytes=size_bytes,
                char_count=len(combined),
                line_count=line_count if combined else 0,
                metadata={
                    "format": "pdf",
                    "page_count": len(pages),
                    "image_page_count": image_page_count,
                    "pdf_metadata": meta_info,
                    "empty_page_count": sum(1 for p in pages if p.char_count == 0),
                    "page_edge_lines": {"first": first_edge, "last": last_edge},
                },
                pages=tuple(pages),
            )
    except DocumentLoadError:
        raise
    except PdfReadError as exc:
        raise DocumentLoadError(f"Malformed or unreadable PDF: {discovered.relative_path}") from exc
    except Exception as exc:  # noqa: BLE001
        raise DocumentLoadError(f"Failed to open PDF: {discovered.relative_path}") from exc


def _page_has_images(page: object) -> bool:
    try:
        resources = page.get("/Resources")  # type: ignore[attr-defined]
        if resources is None:
            return False
        if hasattr(resources, "get_object"):
            resources = resources.get_object()
        xobject = resources.get("/XObject") if resources else None
        if xobject is None:
            return False
        if hasattr(xobject, "get_object"):
            xobject = xobject.get_object()
        for _name, obj in xobject.items():
            try:
                if hasattr(obj, "get_object"):
                    obj = obj.get_object()
                if obj.get("/Subtype") == "/Image":
                    return True
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        return False
    return False


def _safe_metadata(reader: PdfReader) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        meta = reader.metadata
    except Exception:  # noqa: BLE001
        return out
    if not meta:
        return out
    for key in ("/Title", "/Author", "/Subject", "/Creator", "/Producer"):
        value = meta.get(key)
        if value is not None:
            out[key.lstrip("/")] = str(value)[:500]
    return out
