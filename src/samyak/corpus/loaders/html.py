"""Local HTML loader (no network fetches; scripts/styles removed)."""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

from samyak.corpus.discovery import DiscoveredFile
from samyak.corpus.loaders.errors import DocumentLoadError
from samyak.corpus.models import Document

_BOILERPLATE_TAGS = frozenset({"nav", "footer", "header", "aside"})
_REMOVE_TAGS = frozenset({"script", "style", "noscript", "template", "iframe", "svg"})


def load_html_document(discovered: DiscoveredFile) -> Document:
    """Parse a local HTML file into structured plain text.

    Does not fetch external images, scripts, stylesheets, or URLs.
    """
    path = discovered.absolute_path
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DocumentLoadError(f"Failed to read HTML: {discovered.relative_path}") from exc

    try:
        html = raw.decode("utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        html = raw.decode("latin-1", errors="replace")
        encoding = "latin-1"

    # Parser-only; no network. html.parser avoids native deps.
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(_REMOVE_TAGS):
        tag.decompose()

    # Strip on* handlers and javascript: URLs if present (defense in depth).
    for tag in soup.find_all(True):
        if not isinstance(tag, Tag):
            continue
        attrs = dict(tag.attrs)
        for attr_name in list(attrs):
            lowered = attr_name.lower()
            if lowered.startswith("on"):
                del tag.attrs[attr_name]
                continue
            value = attrs.get(attr_name)
            if isinstance(value, str) and value.strip().lower().startswith("javascript:"):
                del tag.attrs[attr_name]

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    boilerplate_text = _collect_boilerplate(soup)
    main = soup.body if soup.body else soup
    content_text = _render_content(main)
    code_chars = _code_char_count(main)

    if title and title not in content_text[:200]:
        text = f"{title}\n\n{content_text}".strip()
    else:
        text = content_text.strip()

    text = re.sub(r"\n{3,}", "\n\n", text)
    boilerplate_chars = len(boilerplate_text)
    content_chars = len(text)

    return Document(
        path=discovered.relative_path,
        filename=Path(discovered.relative_path).name,
        extension=discovered.extension,
        text=text + "\n" if text else "",
        size_bytes=len(raw),
        char_count=len(text),
        line_count=len(text.split("\n")) if text else 0,
        metadata={
            "format": "html",
            "encoding": encoding,
            "title": title,
            "boilerplate_chars": boilerplate_chars,
            "content_chars": content_chars,
            "code_chars": code_chars,
            "code_heavy": bool(content_chars and (code_chars / max(content_chars, 1)) >= 0.25),
            "raw_to_content_ratio": (content_chars / len(raw)) if raw else 0.0,
        },
    )


def _collect_boilerplate(soup: BeautifulSoup) -> str:
    chunks: list[str] = []
    for tag_name in _BOILERPLATE_TAGS:
        for tag in soup.find_all(tag_name):
            chunks.append(tag.get_text(" ", strip=True))
    return "\n".join(c for c in chunks if c)


def _code_char_count(root: Tag | BeautifulSoup) -> int:
    total = 0
    for tag in root.find_all(["pre", "code"]):
        total += len(tag.get_text())
    return total


def _render_content(root: Tag | BeautifulSoup) -> str:
    parts: list[str] = []
    for element in root.children:
        _render_node(element, parts)
    return "\n".join(parts).strip()


def _render_node(node: object, parts: list[str]) -> None:
    if isinstance(node, NavigableString):
        text = str(node).strip()
        if text:
            parts.append(text)
        return
    if not isinstance(node, Tag):
        return
    name = node.name.lower() if node.name else ""
    if name in _REMOVE_TAGS or name in _BOILERPLATE_TAGS:
        return
    if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        text = node.get_text(" ", strip=True)
        if text:
            parts.append(text)
            parts.append("")
        return
    if name == "p":
        text = node.get_text(" ", strip=True)
        if text:
            parts.append(text)
            parts.append("")
        return
    if name in {"ul", "ol"}:
        for li in node.find_all("li", recursive=False):
            item = li.get_text(" ", strip=True)
            if item:
                parts.append(f"- {item}")
        parts.append("")
        return
    if name == "pre":
        code = node.get_text("\n", strip=False).rstrip()
        if code:
            parts.append(code)
            parts.append("")
        return
    if name == "table":
        for row in node.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in row.find_all(["th", "td"])]
            cells = [c for c in cells if c]
            if cells:
                parts.append(" | ".join(cells))
        parts.append("")
        return
    if name in {"br"}:
        parts.append("")
        return
    for child in node.children:
        _render_node(child, parts)
