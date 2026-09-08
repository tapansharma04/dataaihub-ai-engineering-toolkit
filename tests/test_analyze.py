"""Tests for corpus analysis pipeline and CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from helpers import make_corpus, write_text
from samyak import AnalysisConfig, CorpusPathError, __version__, analyze_corpus
from samyak.cli import main
from samyak.corpus.discovery import discover_files
from samyak.corpus.report import render_json_report, render_text_report


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
    assert report.summary.discovery_errors == 0
    assert report.findings == ()


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
            "skip.docx": "not a real docx",
            "skip.xlsx": "not a real spreadsheet",
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


def test_analyze_emits_unique_finding_codes(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {
            "empty.txt": "",
            "ok.txt": "A normal document with enough content about shipping policies.\n" * 3,
            "dup.txt": "Shared duplicate body about warehouse logistics for pairing.\n" * 3,
            "dup-copy.txt": "Shared duplicate body about warehouse logistics for pairing.\n" * 3,
            "skip.docx": "unsupported",
        },
    )
    report = analyze_corpus(corpus)
    codes = [finding.code for finding in report.findings]
    assert len(codes) == len(set(codes))


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
    assert payload["product"] == "samyak"
    assert payload["capability"] == "corpus"
    assert payload["version"] == __version__
    assert "utility" not in payload
    assert "summary" in payload
    assert "findings" in payload
    assert "config" in payload
    assert "severity_counts" in payload
    assert payload["summary"]["analyzed_documents"] == 1
    assert payload["summary"]["discovery_errors"] == 0
    assert payload["summary"]["load_errors"] == 0
    # Relative path, not absolute machine path
    assert payload["summary"]["corpus_root"] == "corpus"
    blob = json.dumps(payload)
    assert str(tmp_path) not in blob
    assert str(tmp_path.resolve()) not in blob


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
    assert "Samyak Corpus Intelligence Report" in text
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


def test_report_order_is_independent_of_creation_order(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    dup_body = "Shared duplicate body for ordering checks about policies.\n" * 3
    write_text(corpus / "z_last.txt", dup_body)
    write_text(corpus / "a_first.txt", dup_body)
    write_text(
        corpus / "m_mid.txt",
        "Unique mid document with enough content about shipping.\n" * 3,
    )
    report = analyze_corpus(corpus)
    dup = next(f for f in report.findings if f.code == "EXACT_DUPLICATES")
    assert list(dup.affected_documents) == ["a_first.txt", "z_last.txt"]
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}
    keys = [(order[f.severity.value], f.code, f.title) for f in report.findings]
    assert keys == sorted(keys)
    payload = json.loads(render_json_report(report))
    assert [item["code"] for item in payload["findings"]] == [f.code for f in report.findings]


def test_cli_main_json(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {"doc.txt": "CLI integration document with enough content for analysis.\n"},
    )
    output = tmp_path / "report.json"
    code = main(["corpus", str(corpus), "--format", "json", "--output", str(output)])
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["product"] == "samyak"
    assert payload["capability"] == "corpus"
    assert "utility" not in payload


def test_cli_invalid_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["corpus", str(tmp_path / "missing")])
    captured = capsys.readouterr()
    assert code == 2
    assert "error:" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_cli_relative_missing_path_keeps_user_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    code = main(["corpus", "not-here"])
    captured = capsys.readouterr()
    assert code == 2
    assert "not-here" in captured.err
    assert "Traceback" not in captured.err
    assert str(tmp_path) not in captured.err
    assert str(tmp_path.resolve()) not in captured.err


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
    rels = {item.relative_path for item in discovered.files}
    assert "inside.txt" in rels
    assert "linked.txt" not in rels
    assert discovered.access_errors == ()


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


def test_nested_directories(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {
            "root.txt": "Root-level document with enough content about policies.\n",
            "nested/dir/deep.txt": "Nested document with enough content about shipping.\n",
        },
    )
    report = analyze_corpus(corpus)
    assert report.summary.analyzed_documents == 2
    discovered = {item.relative_path for item in discover_files(corpus).files}
    assert discovered == {"nested/dir/deep.txt", "root.txt"}


def test_high_symbol_ratio(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {
            "symbols.txt": "@#$%^&*()_+[]{}|;:,.<>/" * 8,
            "ok.txt": "A normal document with enough content about refunds and shipping.\n",
        },
    )
    report = analyze_corpus(corpus)
    finding = next(f for f in report.findings if f.code == "HIGH_SYMBOL_RATIO")
    assert "symbols.txt" in finding.affected_documents


def test_excessive_whitespace(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {
            "padded.txt": "word" + (" " * 100) + "end",
            "ok.txt": "A normal document with enough content about warehouse logistics.\n",
        },
    )
    report = analyze_corpus(corpus)
    finding = next(f for f in report.findings if f.code == "EXCESSIVE_WHITESPACE")
    assert "padded.txt" in finding.affected_documents


def test_cli_small_chars(tmp_path: Path) -> None:
    body = "A" * 50
    corpus = make_corpus(
        tmp_path,
        {
            "mid.txt": body,
            "ok.txt": "A sufficiently long document about onboarding and customer support.\n",
        },
    )
    output = tmp_path / "small.json"
    code = main(
        [
            "corpus",
            str(corpus),
            "--format",
            "json",
            "--small-chars",
            "200",
            "--no-progress",
            "-o",
            str(output),
        ]
    )
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    codes = {item["code"] for item in payload["findings"]}
    assert "VERY_SMALL_DOCUMENTS" in codes
    assert payload["config"]["small_document_chars"] == 200


def test_cli_large_chars(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {
            "big.txt": "word " * 1_000,
            "ok.txt": "A normal-sized document about shipping and returns policies.\n",
        },
    )
    output = tmp_path / "large.json"
    code = main(
        [
            "corpus",
            str(corpus),
            "--format",
            "json",
            "--large-chars",
            "1000",
            "--no-progress",
            "-o",
            str(output),
        ]
    )
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    codes = {item["code"] for item in payload["findings"]}
    assert "VERY_LARGE_DOCUMENTS" in codes
    assert payload["config"]["large_document_chars"] == 1000


def test_cli_no_progress(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    corpus = make_corpus(
        tmp_path,
        {"doc.txt": "CLI --no-progress should leave stderr empty for a small corpus.\n" * 3},
    )
    code = main(["corpus", str(corpus), "--format", "json", "--no-progress"])
    captured = capsys.readouterr()
    assert code == 0
    json.loads(captured.out)
    assert captured.err == ""


def test_cli_format_text(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    corpus = make_corpus(
        tmp_path,
        {"doc.txt": "CLI text format should print the human-readable report header.\n" * 3},
    )
    code = main(["corpus", str(corpus), "--format", "text", "--no-progress"])
    captured = capsys.readouterr()
    assert code == 0
    assert "Samyak Corpus Intelligence Report" in captured.out
    assert captured.err == ""


def test_cli_file_not_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    not_dir = tmp_path / "file.txt"
    not_dir.write_text("this is a file not a corpus directory\n", encoding="utf-8")
    code = main(["corpus", str(not_dir), "--no-progress"])
    captured = capsys.readouterr()
    assert code == 2
    assert "error:" in captured.err
    assert "Traceback" not in captured.err


def test_cli_output_write_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    corpus = make_corpus(
        tmp_path,
        {"doc.txt": "Output path pointing at a directory should fail to write.\n" * 3},
    )
    code = main(
        [
            "corpus",
            str(corpus),
            "--format",
            "json",
            "--no-progress",
            "--output",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "error:" in captured.err
    assert "Traceback" not in captured.err
