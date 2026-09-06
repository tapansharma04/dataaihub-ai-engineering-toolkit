"""Equivalence tests for memory-conscious text helpers."""

from __future__ import annotations

from collections import Counter

from samyak.corpus.analyzers.hashing import (
    content_hash,
    content_hash_normalized,
    normalize_for_hash,
)
from samyak.corpus.analyzers.text_stats import (
    collect_text_stats,
    excessive_whitespace,
    has_long_paragraph,
    high_symbol_ratio,
    max_paragraph_length,
    meaningful_length,
    overly_fragmented,
    repeated_line_noise,
)
from samyak.corpus.config import AnalysisConfig


def test_content_hash_matches_normalize_join() -> None:
    samples = [
        "",
        "   \n\t  ",
        "hello",
        "  hello  world  ",
        "hello\nworld",
        "hello\n\n\nworld",
        "a\tb\tc",
        "café résumé",
        "αβγ  123",
        "line1\r\nline2",  # may already be normalized at load; still hash-equivalent
        "x" * 1000 + "\n\n" + "y" * 50,
    ]
    for text in samples:
        expected = content_hash_normalized(normalize_for_hash(text))
        assert content_hash(text) == expected, repr(text)


def test_meaningful_length_matches_strip() -> None:
    samples = ["", "  ", "\n\t", "abc", "  abc  ", "\nhello\n"]
    for text in samples:
        assert meaningful_length(text) == len(text.strip())


def test_paragraph_length_matches_split() -> None:
    samples = [
        "",
        "short",
        "a" * 100,
        "para one\n\npara two",
        "para one\n\n\n\npara two",
        "  padded  \n\n  also  ",
        "x" * 6000,
        "small\n\n" + ("b" * 6000),
    ]
    for text in samples:
        classic = max((len(p.strip()) for p in text.split("\n\n")), default=0)
        assert max_paragraph_length(text) == classic
        assert has_long_paragraph(text, 5000) == any(
            len(p.strip()) >= 5000 for p in text.split("\n\n")
        )


def test_stats_match_classic_heuristics() -> None:
    cfg = AnalysisConfig()
    samples = [
        "Normal prose about refunds and shipping timelines for customers.\n" * 5,
        "!!!!@@@@####$$$$%%%%^^^^&&&&****(((())))\n" * 20,
        "word   \n\n   " * 40 + "tail",
        ("same line\n" * 20) + "other\n",
        "\n".join(["x"] * 30),
        "a" * 6000,
    ]
    for text in samples:
        stats = collect_text_stats(text, short_line_chars=cfg.short_line_chars)

        non_ws = [ch for ch in text if not ch.isspace()]
        symbolish = sum(1 for ch in non_ws if not ch.isalnum())
        classic_symbol = len(non_ws) >= 40 and (symbolish / len(non_ws)) >= cfg.high_symbol_ratio
        assert high_symbol_ratio(stats, cfg.high_symbol_ratio) == classic_symbol

        classic_ws = (
            len(text) >= 80
            and (sum(1 for ch in text if ch.isspace()) / len(text))
            >= cfg.excessive_whitespace_ratio
        )
        assert excessive_whitespace(stats, cfg.excessive_whitespace_ratio) == classic_ws

        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if len(lines) < cfg.repeated_line_min_lines:
            classic_rep = False
        else:
            most = Counter(lines).most_common(1)[0][1]
            classic_rep = (most / len(lines)) >= cfg.repeated_line_ratio
        assert (
            repeated_line_noise(stats, cfg.repeated_line_ratio, cfg.repeated_line_min_lines)
            == classic_rep
        )

        if len(lines) < cfg.short_line_min_lines:
            classic_frag = False
        else:
            short = sum(1 for ln in lines if len(ln) < cfg.short_line_chars)
            classic_frag = (short / len(lines)) >= cfg.short_line_ratio
        assert (
            overly_fragmented(stats, cfg.short_line_ratio, cfg.short_line_min_lines) == classic_frag
        )

        assert stats.max_paragraph_length == max(
            (len(p.strip()) for p in text.split("\n\n")), default=0
        )
        if normalize_for_hash(text):
            assert stats.content_hash == content_hash_normalized(normalize_for_hash(text))
        else:
            assert stats.content_hash is None


def test_large_text_stats_path_hash_matches_normalize() -> None:
    """The memory-conscious scanner starts at 5 MB; keep it equivalent to normalize+hash."""
    text = "word " * 1_000_000
    assert len(text) >= 5_000_000
    stats = collect_text_stats(text, short_line_chars=20)
    expected = content_hash_normalized(normalize_for_hash(text))
    assert stats.content_hash == expected
    assert stats.content_hash == content_hash(text)
    assert stats.is_empty is False
    assert stats.non_empty_line_count == 1
