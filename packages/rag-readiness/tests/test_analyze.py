"""Tests for corpus analysis pipeline and CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from helpers import make_corpus, write_text
from rag_readiness import __version__, analyze_corpus
from rag_readiness.cli import main
from rag_readiness.config import AnalysisConfig
from rag_readiness.discovery import CorpusPathError, discover_files
from rag_readiness.report import render_json_report, render_text_report


def test_nonexistent_path(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(CorpusPathError):
        analyze_corpus(missing)


def test_empty_directory(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    report = analyze_corpus(empty)
    assert report.summary.total_discovered_files == 0
    assert report.summary.analyzed_documents == 0
    assert report.findings == []


def test_normal_txt_and_md(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {
            "guide.txt": "A useful guide about refunds and shipping timelines for customers.\n" * 3,
            "intro.md": "# Intro\n\nThis markdown file explains product basics clearly.\n",
        },
    )
    report = analyze_corpus(corpus)
    assert report.summary.analyzed_documents == 2
    assert report.summary.unsupported_files == 0
    codes = {f.code for f in report.findings}
    assert "EMPTY_DOCUMENTS" not in codes
    assert "EXACT_DUPLICATES" not in codes


def test_unsupported_files_do_not_crash(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {
            "ok.txt": "Enough text content for a normal document about policies and process.\n",
            "skip.pdf": "not a real pdf",
            "skip.docx": "not a real docx",
        },
    )
    report = analyze_corpus(corpus)
    assert report.summary.unsupported_files == 2
    assert report.summary.analyzed_documents == 1
    unsupported = next(f for f in report.findings if f.code == "UNSUPPORTED_FILES")
    assert unsupported.severity.value == "INFO"


def test_empty_and_whitespace_only(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {
            "empty.txt": "",
            "spaces.txt": "   \n\t\n  ",
            "ok.txt": "A normal document with enough content to avoid the small threshold.\n",
        },
    )
    report = analyze_corpus(corpus)
    empty = next(f for f in report.findings if f.code == "EMPTY_DOCUMENTS")
    assert empty.severity.value == "HIGH"
    assert empty.evidence["count"] == 2


def test_very_small_document(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {
            "tiny.txt": "Hi",
            "ok.txt": "A sufficiently long document about onboarding and customer support.\n",
        },
    )
    report = analyze_corpus(corpus)
    small = next(f for f in report.findings if f.code == "VERY_SMALL_DOCUMENTS")
    assert "tiny.txt" in small.affected_documents


def test_large_document(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {
            "big.txt": "word " * 30_000,
            "ok.txt": "A normal-sized document about shipping and returns policies.\n",
        },
    )
    config = AnalysisConfig(large_document_chars=10_000)
    report = analyze_corpus(corpus, config=config)
    large = next(f for f in report.findings if f.code == "VERY_LARGE_DOCUMENTS")
    assert "big.txt" in large.affected_documents


def test_exact_duplicates(tmp_path: Path) -> None:
    body = "Customer returns policy with enough unique wording to be meaningful content.\n"
    corpus = make_corpus(
        tmp_path,
        {
            "a.txt": body,
            "b.txt": body,
            "c.txt": "  " + body + "\n",
            "other.txt": "Completely different content about warehouse logistics and inventory.\n",
        },
    )
    report = analyze_corpus(corpus)
    dup = next(f for f in report.findings if f.code == "EXACT_DUPLICATES")
    assert dup.severity.value == "HIGH"
    assert dup.evidence["affected_document_count"] == 3
    assert "why_it_matters" in dup.to_dict()
    assert dup.recommendation


def test_repeated_noisy_content(tmp_path: Path) -> None:
    noisy = "ERROR LINE\n" * 20
    corpus = make_corpus(
        tmp_path,
        {
            "noise.txt": noisy,
            "ok.txt": "A calm document describing account recovery steps for users.\n",
        },
    )
    report = analyze_corpus(corpus)
    finding = next(f for f in report.findings if f.code == "REPEATED_LINE_CONTENT")
    assert "noise.txt" in finding.affected_documents


def test_chunkability_long_block(tmp_path: Path) -> None:
    # One huge paragraph without blank-line breaks
    block = ("This is a long sentence about architecture and retrieval. " * 120).strip()
    corpus = make_corpus(
        tmp_path,
        {
            "long.txt": block,
            "ok.txt": "Short but sufficient document about billing questions and invoices.\n",
        },
    )
    config = AnalysisConfig(long_paragraph_chars=500)
    report = analyze_corpus(corpus, config=config)
    finding = next(f for f in report.findings if f.code == "LONG_UNINTERRUPTED_BLOCKS")
    assert "long.txt" in finding.affected_documents


def test_chunkability_fragmented_lines(tmp_path: Path) -> None:
    fragmented = "\n".join(f"x{i}" for i in range(30))
    corpus = make_corpus(
        tmp_path,
        {
            "frag.txt": fragmented,
            "ok.txt": "A coherent paragraph about deployment checklists and rollback plans.\n",
        },
    )
    report = analyze_corpus(corpus)
    finding = next(f for f in report.findings if f.code == "OVERLY_FRAGMENTED_LINES")
    assert "frag.txt" in finding.affected_documents


def test_json_output_structure(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {"doc.txt": "Structured JSON report test document with adequate length.\n"},
    )
    report = analyze_corpus(corpus)
    payload = json.loads(render_json_report(report))
    assert payload["utility"] == "rag-readiness"
    assert payload["version"] == __version__
    assert "summary" in payload
    assert "findings" in payload
    assert "config" in payload
    assert "severity_counts" in payload
    assert payload["summary"]["analyzed_documents"] == 1
    # Relative path, not absolute machine path
    assert payload["summary"]["corpus_root"] == "corpus"


def test_text_report_contains_sections(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {
            "a.txt": "Alpha document with enough text for analysis about returns.\n",
            "b.txt": "Alpha document with enough text for analysis about returns.\n",
        },
    )
    report = analyze_corpus(corpus)
    text = render_text_report(report)
    assert "RAG Data Readiness Report" in text
    assert "Findings" in text
    assert "Summary" in text
    assert "HIGH" in text


def test_deterministic_analysis(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {
            "one.txt": "Document one with enough content for stable hashing tests.\n",
            "two.txt": "Document one with enough content for stable hashing tests.\n",
            "tiny.txt": "x",
        },
    )
    first = render_json_report(analyze_corpus(corpus))
    second = render_json_report(analyze_corpus(corpus))
    assert first == second


def test_cli_help() -> None:
    env = os.environ.copy()
    src = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "rag_readiness", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert "corpus" in result.stdout.lower()


def test_cli_version() -> None:
    env = os.environ.copy()
    src = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "rag_readiness", "--version"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert __version__ in result.stdout


def test_cli_main_json(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {"doc.txt": "CLI integration document with enough content for analysis.\n"},
    )
    output = tmp_path / "report.json"
    code = main([str(corpus), "--format", "json", "--output", str(output)])
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["utility"] == "rag-readiness"


def test_cli_invalid_path(tmp_path: Path) -> None:
    code = main([str(tmp_path / "missing")])
    assert code == 2


def test_discovery_skips_external_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    write_text(outside, "secret outside content that should not be ingested.\n")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    write_text(corpus / "inside.txt", "Inside corpus document with enough content here.\n")
    link = corpus / "linked.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks not supported in this environment")
    discovered = discover_files(corpus)
    rels = {item.relative_path for item in discovered}
    assert "inside.txt" in rels
    assert "linked.txt" not in rels


def test_no_fake_score_in_report(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {"doc.txt": "Ensure we do not invent a readiness score in v0.1 output.\n"},
    )
    report = analyze_corpus(corpus)
    text = render_text_report(report)
    payload = report.to_dict()
    assert "readiness_score" not in payload
    assert "RAG Readiness Score" not in text
    assert "/100" not in text
