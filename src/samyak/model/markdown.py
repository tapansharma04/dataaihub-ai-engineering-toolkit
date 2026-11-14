"""Labeled Markdown structure helpers. Not CSS or HTML presentation selectors."""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_CODE = re.compile(r"`([^`]+)`")
_BULLET = re.compile(r"^[-*][ \t]+(.+)$")
_TRAILING_FOOTNOTE = re.compile(
    r"(?:"
    r"\s*\^[^^]+\^"
    r"|"
    r"\s*\^\{?\[[^\]]+\]\}?"
    r")+$"
)


@dataclass(frozen=True, slots=True)
class MarkdownSection:
    level: int
    title: str
    body: str


def normalize_heading(title: str) -> str:
    return " ".join(title.strip().lower().split())


def strip_trailing_markdown_footnote(text: str) -> str:
    """Remove a trailing Markdown footnote marker from a label.

    Handles caret-wrapped footnotes used in official docs, including
    ``^{[*]}`` and ``^[...](url)^``. Interior text is left unchanged.
    """
    return _TRAILING_FOOTNOTE.sub("", text).rstrip()


def split_sections(markdown: str) -> tuple[MarkdownSection, ...]:
    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    sections: list[MarkdownSection] = []
    current_level = 0
    current_title = ""
    current_body: list[str] = []

    def flush() -> None:
        if current_title == "" and not current_body:
            return
        sections.append(
            MarkdownSection(
                level=current_level,
                title=current_title,
                body="\n".join(current_body).strip("\n"),
            )
        )

    for line in lines:
        match = _HEADING.match(line)
        if match is None:
            current_body.append(line)
            continue
        flush()
        current_level = len(match.group(1))
        current_title = match.group(2).strip()
        current_body = []
    flush()
    return tuple(sections)


def section_named(
    sections: tuple[MarkdownSection, ...], title: str, *, level: int | None = None
) -> MarkdownSection | None:
    wanted = normalize_heading(title)
    for section in sections:
        if normalize_heading(section.title) != wanted:
            continue
        if level is not None and section.level != level:
            continue
        return section
    return None


def has_heading(sections: tuple[MarkdownSection, ...], title: str) -> bool:
    return section_named(sections, title) is not None


def bullet_items(text: str) -> tuple[str, ...]:
    items: list[str] = []
    for line in text.split("\n"):
        match = _BULLET.match(line.strip())
        if match is not None:
            items.append(match.group(1).strip())
    return tuple(items)


def markdown_links(text: str) -> tuple[tuple[str, str], ...]:
    return tuple((match.group(1).strip(), match.group(2).strip()) for match in _LINK.finditer(text))


def code_spans(text: str) -> tuple[str, ...]:
    return tuple(match.group(1).strip() for match in _CODE.finditer(text) if match.group(1).strip())


def labeled_value(text: str, label: str) -> str | None:
    """Return the value of a 'Label: value' line or list item, if present."""
    wanted = normalize_heading(label)
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if line.startswith(("- ", "* ")):
            line = line[2:].strip()
        if ":" not in line:
            continue
        head, tail = line.split(":", 1)
        if normalize_heading(head) == wanted:
            value = tail.strip()
            return value or None
    return None


def pipe_tables(text: str) -> tuple[tuple[dict[str, str], ...], ...]:
    """Return GitHub-style pipe tables as row dicts keyed by header text."""
    tables: list[tuple[dict[str, str], ...]] = []
    lines = [line.strip() for line in text.split("\n")]
    index = 0
    while index < len(lines):
        if not _is_table_row(lines[index]):
            index += 1
            continue
        block: list[str] = []
        while index < len(lines) and _is_table_row(lines[index]):
            block.append(lines[index])
            index += 1
        parsed = _parse_table_block(block)
        if parsed:
            tables.append(parsed)
    return tuple(tables)


def _is_table_row(line: str) -> bool:
    return line.startswith("|") and line.endswith("|") and line.count("|") >= 2


def unwrap_blockquote_pipe_row(line: str) -> str:
    """Return a pipe-table row with leading blockquote markers removed.

    Each marker is ``>`` plus at most one following space or tab. Nested
    markers are removed only when the line that remains is a pipe row, so a
    quoted sentence that merely contains ``|`` stays quoted. Cell text,
    including a ``>`` or an escaped pipe, is left unchanged.
    """
    current = line
    while True:
        remainder = _strip_one_blockquote_marker(current)
        if remainder is None:
            return line
        if _is_table_row(remainder.strip()):
            return remainder
        if remainder.lstrip().startswith(">"):
            current = remainder
            continue
        return line


def _strip_one_blockquote_marker(line: str) -> str | None:
    if not line.startswith(">"):
        return None
    rest = line[1:]
    if rest.startswith((" ", "\t")):
        rest = rest[1:]
    return rest


def _parse_table_block(block: list[str]) -> tuple[dict[str, str], ...]:
    rows = [_split_row(line) for line in block]
    if len(rows) < 2:
        return ()
    headers = tuple(cell.strip() for cell in rows[0])
    body_start = 1
    if all(_is_separator_cell(cell) for cell in rows[1]):
        body_start = 2
    if not headers or body_start >= len(rows):
        return ()
    parsed: list[dict[str, str]] = []
    for cells in rows[body_start:]:
        if all(_is_separator_cell(cell) for cell in cells):
            continue
        row: dict[str, str] = {}
        for offset, header in enumerate(headers):
            if not header:
                continue
            row[header] = cells[offset].strip() if offset < len(cells) else ""
        if any(row.values()):
            parsed.append(row)
    return tuple(parsed)


def _split_row(line: str) -> tuple[str, ...]:
    """Split a pipe row on unescaped ``|`` delimiters.

    ``\\|`` stays inside the cell. ``\\\\|`` is an escaped backslash followed
    by a real column delimiter. A leading or trailing delimiter still produces
    the empty edge cell that table syntax uses, and that edge cell is dropped.
    """
    cells = list(_split_unescaped_pipes(line.strip()))
    if cells and cells[0] == "":
        cells = cells[1:]
    if cells and cells[-1] == "":
        cells = cells[:-1]
    return tuple(cells)


def _split_unescaped_pipes(text: str) -> tuple[str, ...]:
    cells: list[str] = []
    current: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        nxt = text[index + 1] if index + 1 < len(text) else ""
        if char == "\\" and nxt == "\\":
            current.append("\\")
            current.append("\\")
            index += 2
            continue
        if char == "\\" and nxt == "|":
            current.append("\\")
            current.append("|")
            index += 2
            continue
        if char == "|":
            cells.append("".join(current).strip())
            current = []
            index += 1
            continue
        current.append(char)
        index += 1
    cells.append("".join(current).strip())
    return tuple(cells)


def _is_separator_cell(cell: str) -> bool:
    stripped = cell.strip().replace(":", "").replace(" ", "")
    return bool(stripped) and set(stripped) <= {"-"}
