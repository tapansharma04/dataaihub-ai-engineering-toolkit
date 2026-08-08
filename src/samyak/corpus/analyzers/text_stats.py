"""Single-pass text statistics for large-document analysis."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class TextStats:
    """Compact statistics derived from one document body."""

    total_chars: int
    meaningful_length: int
    is_empty: bool
    whitespace_chars: int
    non_ws_chars: int
    symbol_chars: int
    non_empty_line_count: int
    short_line_count: int
    very_short_line_count: int
    table_like_line_count: int
    max_paragraph_length: int
    max_repeated_line_count: int
    first_non_empty_line: str | None
    content_hash: str | None


def meaningful_length(text: str) -> int:
    """Length of ``text.strip()`` without allocating the stripped copy."""
    n = len(text)
    start = 0
    while start < n and text[start].isspace():
        start += 1
    end = n
    while end > start and text[end - 1].isspace():
        end -= 1
    return end - start


def max_paragraph_length(text: str) -> int:
    """Maximum stripped length among ``\\n\\n``-separated blocks."""
    n = len(text)
    i = 0
    best = 0
    while True:
        start = i
        while i < n:
            if text[i] == "\n" and i + 1 < n and text[i + 1] == "\n":
                break
            i += 1
        end = i
        a = start
        b = end
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        length = b - a
        if length > best:
            best = length
        if i >= n:
            break
        i += 2
    return best


def has_long_paragraph(text: str, threshold: int) -> bool:
    """True if any blank-line-separated block strips to ``threshold``+ chars."""
    return max_paragraph_length(text) >= threshold


def _line_is_table_like(stripped: str) -> bool:
    if " | " in stripped:
        return True
    return "  " in stripped and len(stripped.split()) >= 4


def collect_text_stats(text: str, *, short_line_chars: int) -> TextStats:
    """Collect readiness statistics without large intermediate copies.

    For typical documents, uses a fast multi-step scan. For very large
    documents, uses a single forward pass that streams the content hash and
    avoids full-size normalized / line-list / paragraph-list allocations.
    """
    # Below this size, temporary intermediates are cheap relative to process overhead.
    # Volume-scale corpora use many sub-multi-MB files; reserve the slower
    # memory-conscious scanner for truly large individual documents.
    if len(text) < 5_000_000:
        return _collect_text_stats_small(text, short_line_chars=short_line_chars)
    return _collect_text_stats_large(text, short_line_chars=short_line_chars)


def _collect_text_stats_small(text: str, *, short_line_chars: int) -> TextStats:
    """Faster path for ordinary-sized documents (< ~5 MB)."""
    from collections import Counter

    from samyak.corpus.analyzers.hashing import content_hash_normalized, normalize_for_hash

    total = len(text)
    mlen = meaningful_length(text)
    is_empty = mlen == 0

    ws = 0
    non_ws = 0
    symbol = 0
    for ch in text:
        if ch.isspace():
            ws += 1
        else:
            non_ws += 1
            if not ch.isalnum():
                symbol += 1

    lines: list[str] = []
    short_lines = 0
    very_short = 0
    table_like = 0
    for raw in text.split("\n"):
        stripped = raw.strip()
        if not stripped:
            continue
        lines.append(stripped)
        length = len(stripped)
        if length < short_line_chars:
            short_lines += 1
        if length < 8:
            very_short += 1
        if _line_is_table_like(stripped):
            table_like += 1

    max_rep = Counter(lines).most_common(1)[0][1] if lines else 0
    first_line = lines[0][:160] if lines else None
    # C-backed normalize is fast and cheap under the small-doc size cap.
    normalized = "" if is_empty else normalize_for_hash(text)
    digest = content_hash_normalized(normalized) if normalized else None
    max_para = max((len(p.strip()) for p in text.split("\n\n")), default=0)

    return TextStats(
        total_chars=total,
        meaningful_length=mlen,
        is_empty=is_empty,
        whitespace_chars=ws,
        non_ws_chars=non_ws,
        symbol_chars=symbol,
        non_empty_line_count=len(lines),
        short_line_count=short_lines,
        very_short_line_count=very_short,
        table_like_line_count=table_like,
        max_paragraph_length=max_para,
        max_repeated_line_count=max_rep,
        first_non_empty_line=first_line,
        content_hash=digest,
    )


def _collect_text_stats_large(text: str, *, short_line_chars: int) -> TextStats:
    """Memory-conscious path for multi-megabyte documents."""
    total = len(text)
    mlen = meaningful_length(text)
    is_empty = mlen == 0

    ws = 0
    non_ws = 0
    symbol = 0
    non_empty_lines = 0
    short_lines = 0
    very_short = 0
    table_like = 0
    first_line: str | None = None
    max_rep = 0
    counts: dict[bytes, int] = {}

    hasher = hashlib.sha256()
    hash_first_token = True
    in_token = False
    token_start = 0

    para_start = 0
    max_para = 0
    prev_was_newline = False

    line_start = 0
    n = total

    def finish_token(end: int) -> None:
        nonlocal hash_first_token, in_token
        if not in_token:
            return
        if not hash_first_token:
            hasher.update(b" ")
        else:
            hash_first_token = False
        hasher.update(text[token_start:end].encode("utf-8"))
        in_token = False

    def finish_paragraph(end: int) -> None:
        nonlocal max_para, para_start
        a = para_start
        b = end
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        length = b - a
        if length > max_para:
            max_para = length

    def finish_line(end: int) -> None:
        nonlocal non_empty_lines, short_lines, very_short, table_like, first_line, max_rep
        raw = text[line_start:end]
        stripped = raw.strip()
        if not stripped:
            return
        non_empty_lines += 1
        length = len(stripped)
        if length < short_line_chars:
            short_lines += 1
        if length < 8:
            very_short += 1
        if _line_is_table_like(stripped):
            table_like += 1
        if first_line is None:
            first_line = stripped[:160]
        key = hashlib.blake2b(stripped.encode("utf-8"), digest_size=16).digest()
        count = counts.get(key, 0) + 1
        counts[key] = count
        if count > max_rep:
            max_rep = count

    i = 0
    while i < n:
        ch = text[i]
        if ch.isspace():
            ws += 1
            finish_token(i)
            if ch == "\n":
                finish_line(i)
                line_start = i + 1
                if prev_was_newline:
                    finish_paragraph(i - 1)
                    para_start = i + 1
                prev_was_newline = True
            else:
                prev_was_newline = False
        else:
            non_ws += 1
            if not ch.isalnum():
                symbol += 1
            if not in_token:
                in_token = True
                token_start = i
            prev_was_newline = False
        i += 1

    finish_token(n)
    finish_line(n)
    finish_paragraph(n)

    digest = None if (is_empty or hash_first_token) else hasher.hexdigest()

    return TextStats(
        total_chars=total,
        meaningful_length=mlen,
        is_empty=is_empty,
        whitespace_chars=ws,
        non_ws_chars=non_ws,
        symbol_chars=symbol,
        non_empty_line_count=non_empty_lines,
        short_line_count=short_lines,
        very_short_line_count=very_short,
        table_like_line_count=table_like,
        max_paragraph_length=max_para,
        max_repeated_line_count=max_rep,
        first_non_empty_line=first_line,
        content_hash=digest,
    )


def high_symbol_ratio(stats: TextStats, threshold: float) -> bool:
    if stats.non_ws_chars < 40:
        return False
    return (stats.symbol_chars / stats.non_ws_chars) >= threshold


def excessive_whitespace(stats: TextStats, threshold: float) -> bool:
    if stats.total_chars < 80:
        return False
    return (stats.whitespace_chars / stats.total_chars) >= threshold


def repeated_line_noise(stats: TextStats, ratio: float, min_lines: int) -> bool:
    if stats.non_empty_line_count < min_lines:
        return False
    return (stats.max_repeated_line_count / stats.non_empty_line_count) >= ratio


def overly_fragmented(
    stats: TextStats,
    short_line_ratio: float,
    min_lines: int,
) -> bool:
    if stats.non_empty_line_count < min_lines:
        return False
    return (stats.short_line_count / stats.non_empty_line_count) >= short_line_ratio
