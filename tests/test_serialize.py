"""Tests for AnalysisReport JSON reconstruction."""

from __future__ import annotations

from pathlib import Path

import pytest

from helpers import make_corpus
from samyak import analyze_corpus
from samyak.corpus.report import render_html_report, render_json_report
from samyak.corpus.serialize import ReportDecodeError, report_from_dict


def test_report_roundtrip_matches_to_dict(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {
            "ok.txt": "Roundtrip document with enough content about shipping policies.\n" * 3,
            "empty.txt": "",
        },
    )
    report = analyze_corpus(corpus)
    rebuilt = report_from_dict(report.to_dict())
    assert rebuilt.to_dict() == report.to_dict()
    assert rebuilt.findings == report.findings
    assert rebuilt.summary.corpus_root == report.summary.corpus_root


def test_reconstructed_report_renders_identical_html(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {
            "a.txt": "Shared body for HTML reconstruction checks about refunds.\n" * 3,
            "b.txt": "Shared body for HTML reconstruction checks about refunds.\n" * 3,
        },
    )
    report = analyze_corpus(corpus)
    rebuilt = report_from_dict(report.to_dict())
    assert render_html_report(rebuilt) == render_html_report(report)
    assert render_json_report(rebuilt) == render_json_report(report)


def test_report_from_dict_rejects_non_object() -> None:
    with pytest.raises(ReportDecodeError):
        report_from_dict([])  # type: ignore[arg-type]


def test_report_from_dict_rejects_missing_fields() -> None:
    with pytest.raises(ReportDecodeError, match="product"):
        report_from_dict({"capability": "corpus", "version": "0.1.0"})


def test_report_from_dict_rejects_unknown_severity(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {"doc.txt": "Severity validation document with enough content about billing.\n" * 3},
    )
    payload = analyze_corpus(corpus).to_dict()
    payload["findings"] = [
        {
            "code": "X",
            "category": "inventory",
            "severity": "CRITICAL",
            "title": "x",
            "message": "x",
            "why_it_matters": "x",
            "recommendation": "x",
            "evidence": {},
            "affected_documents": [],
        }
    ]
    with pytest.raises(ReportDecodeError, match="severity"):
        report_from_dict(payload)


def test_report_from_dict_rejects_duplicate_finding_codes(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {"doc.txt": "Duplicate-code reconstruction must fail for persisted reports.\n" * 3},
    )
    payload = analyze_corpus(corpus).to_dict()
    finding = {
        "code": "EMPTY_DOCUMENTS",
        "category": "size",
        "severity": "HIGH",
        "title": "Empty documents",
        "message": "x",
        "why_it_matters": "x",
        "recommendation": "x",
        "evidence": {},
        "affected_documents": [],
    }
    payload["findings"] = [finding, dict(finding)]
    with pytest.raises(ReportDecodeError, match="unique codes"):
        report_from_dict(payload)
