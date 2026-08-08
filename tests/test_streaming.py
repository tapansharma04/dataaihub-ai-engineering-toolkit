"""Tests for streaming / bounded-memory architecture."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from helpers import make_corpus, write_text
from samyak import analyze_corpus
from samyak.corpus.config import AnalysisConfig
from samyak.corpus.pipeline import default_progress_callback
from samyak.corpus.report import render_json_report


def test_incremental_duplicate_detection(tmp_path: Path) -> None:
    body = "Shared exact body for duplicate detection across streaming loads.\n" * 5
    corpus = make_corpus(
        tmp_path,
        {
            "a.txt": body,
            "b.txt": body,
            "c.txt": "Unique document content that is long enough to not be tiny.\n" * 3,
        },
    )
    report = analyze_corpus(corpus)
    dup = next(f for f in report.findings if f.code == "EXACT_DUPLICATES")
    assert dup.evidence["affected_document_count"] == 2
    assert {"a.txt", "b.txt"} <= set(dup.affected_documents)


def test_load_failure_does_not_stop_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    write_text(corpus / "ok.txt", "A normal document with enough content to stay valid.\n" * 3)
    (corpus / "bad.pdf").write_bytes(b"%PDF-1.4\nnot a real pdf\n%%EOF\n")
    report = analyze_corpus(corpus)
    assert report.summary.analyzed_documents == 1
    assert report.summary.load_errors == 1
    assert any(f.code == "LOAD_ERRORS" for f in report.findings)


def test_affected_documents_are_bounded(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for i in range(80):
        write_text(corpus / f"empty_{i:03d}.txt", "")
    write_text(corpus / "ok.txt", "Enough text to remain a normal non-empty document here.\n")
    cfg = AnalysisConfig(max_affected_documents=10)
    report = analyze_corpus(corpus, config=cfg)
    empty = next(f for f in report.findings if f.code == "EMPTY_DOCUMENTS")
    assert empty.evidence["affected_document_count"] == 80
    assert empty.evidence["affected_documents_truncated"] is True
    assert len(empty.affected_documents) == 10


def test_results_deterministic(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {
            "one.txt": "Deterministic content alpha about policies and shipping.\n" * 4,
            "two.txt": "Deterministic content beta about refunds and returns.\n" * 4,
            "empty.txt": "",
        },
    )
    a = analyze_corpus(corpus)
    b = analyze_corpus(corpus)
    assert sorted(f.code for f in a.findings) == sorted(f.code for f in b.findings)
    assert render_json_report(a) == render_json_report(b)


def test_progress_goes_to_stderr_not_stdout(tmp_path: Path, capsys) -> None:
    corpus = make_corpus(
        tmp_path,
        {"doc.txt": "Progress reporting should not contaminate JSON stdout output.\n" * 3},
    )
    calls: list[tuple[int, int]] = []

    def cb(current: int, total: int) -> None:
        calls.append((current, total))
        default_progress_callback(current, total, every=1)

    report = analyze_corpus(corpus, progress_callback=cb)
    captured = capsys.readouterr()
    assert calls == [(1, 1)]
    assert "1 / 1" in captured.err
    assert captured.out == ""
    payload = json.loads(render_json_report(report))
    assert payload["product"] == "samyak"
    assert payload["capability"] == "corpus"
    assert "utility" not in payload


def test_cli_json_stdout_excludes_progress(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {"doc.txt": "CLI JSON mode must keep stdout machine-readable only.\n" * 3},
    )
    pkg_src = Path(__file__).resolve().parents[1] / "src"
    env = {**os.environ, "PYTHONPATH": str(pkg_src)}
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from samyak.cli import main; "
                f"raise SystemExit(main(['corpus', {str(corpus)!r}, '--format', 'json']))"
            ),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0
    json.loads(proc.stdout)
    assert "Analyzing" not in proc.stdout
