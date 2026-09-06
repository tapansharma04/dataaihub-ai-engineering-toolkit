"""Tests for AnalysisConfig validation."""

from __future__ import annotations

import pytest

from helpers import make_corpus
from samyak import AnalysisConfig
from samyak.cli import main


def test_default_config_is_valid() -> None:
    config = AnalysisConfig()
    assert config.small_document_chars == 100
    assert config.large_document_chars == 100_000
    assert config.progress_every == 10_000


def test_equal_size_thresholds_are_valid() -> None:
    config = AnalysisConfig(small_document_chars=50, large_document_chars=50)
    assert config.small_document_chars == 50
    assert config.large_document_chars == 50


def test_zero_progress_every_is_valid() -> None:
    config = AnalysisConfig(progress_every=0)
    assert config.progress_every == 0


def test_negative_small_document_chars_rejected() -> None:
    with pytest.raises(ValueError, match="small_document_chars"):
        AnalysisConfig(small_document_chars=-1)


def test_large_smaller_than_small_rejected() -> None:
    with pytest.raises(ValueError, match="large_document_chars"):
        AnalysisConfig(small_document_chars=200, large_document_chars=100)


def test_ratio_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match="high_symbol_ratio"):
        AnalysisConfig(high_symbol_ratio=1.5)
    with pytest.raises(ValueError, match="repeated_line_ratio"):
        AnalysisConfig(repeated_line_ratio=-0.1)


def test_negative_report_bounds_rejected() -> None:
    with pytest.raises(ValueError, match="max_sample_paths"):
        AnalysisConfig(max_sample_paths=-1)
    with pytest.raises(ValueError, match="max_affected_documents"):
        AnalysisConfig(max_affected_documents=-1)
    with pytest.raises(ValueError, match="progress_every"):
        AnalysisConfig(progress_every=-1)


def test_negative_other_thresholds_rejected() -> None:
    with pytest.raises(ValueError, match="long_paragraph_chars"):
        AnalysisConfig(long_paragraph_chars=-1)
    with pytest.raises(ValueError, match="pdf_high_page_count"):
        AnalysisConfig(pdf_high_page_count=-5)
    with pytest.raises(ValueError, match="html_large_document_chars"):
        AnalysisConfig(html_large_document_chars=-1)


def test_cli_rejects_inverted_size_thresholds(tmp_path, capsys) -> None:
    corpus = make_corpus(
        tmp_path,
        {"doc.txt": "CLI should reject large-chars smaller than small-chars.\n" * 3},
    )
    code = main(
        [
            "corpus",
            str(corpus),
            "--small-chars",
            "500",
            "--large-chars",
            "10",
            "--no-progress",
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "error:" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""
