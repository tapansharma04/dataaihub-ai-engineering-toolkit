"""Samyak productization and CLI foundation tests."""

from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from helpers import make_corpus
from samyak import (
    AnalysisConfig,
    AnalysisReport,
    CorpusPathError,
    Finding,
    Severity,
    __version__,
    analyze_corpus,
)
from samyak.cli import main
from samyak.corpus.report import render_json_report, render_text_report

PUBLIC_API_NAMES = (
    "AnalysisConfig",
    "AnalysisReport",
    "CorpusPathError",
    "Finding",
    "Severity",
    "analyze_corpus",
    "__version__",
)


def test_package_version_constant() -> None:
    assert __version__ == "0.1.1"


def test_installed_package_metadata_matches_version() -> None:
    try:
        dist = importlib.metadata.metadata("samyak")
        installed = importlib.metadata.version("samyak")
    except importlib.metadata.PackageNotFoundError as exc:
        raise AssertionError(
            "Package metadata for 'samyak' was not found. "
            "Install the package before running tests: pip install -e '.[dev]'"
        ) from exc
    assert dist["Name"].lower() == "samyak"
    assert dist["Version"] == "0.1.1"
    assert installed == "0.1.1"
    assert installed == __version__


def test_public_api_exports() -> None:
    import samyak

    assert tuple(samyak.__all__) == PUBLIC_API_NAMES
    for name in PUBLIC_API_NAMES:
        assert hasattr(samyak, name)
    for internal in (
        "Document",
        "CorpusAccumulator",
        "PathBucket",
        "DiscoveredFile",
        "render_html_report",
        "render_report",
        "RunStore",
        "FileRunStore",
        "RunMetadata",
        "serve_viewer",
        "ViewerHandler",
        "compare_runs",
        "RunComparison",
        "FindingMatch",
    ):
        assert internal not in samyak.__all__
        assert not hasattr(samyak, internal)
    assert samyak.analyze_corpus is analyze_corpus
    assert samyak.AnalysisConfig is AnalysisConfig
    assert samyak.CorpusPathError is CorpusPathError
    assert samyak.AnalysisReport is AnalysisReport
    assert samyak.Finding is Finding
    assert samyak.Severity is Severity


def test_batch_analyzers_are_not_public() -> None:
    import samyak.corpus.analyzers as analyzers

    legacy = (
        "analyze_chunkability",
        "analyze_document_sizes",
        "analyze_empty_documents",
        "analyze_exact_duplicates",
        "analyze_html_documents",
        "analyze_pdf_documents",
        "analyze_text_quality",
    )
    public = getattr(analyzers, "__all__", ())
    for name in legacy:
        assert name not in public
        assert not hasattr(analyzers, name)


def test_report_product_capability_identity(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {"doc.txt": "Product identity should appear in machine-readable reports.\n" * 3},
    )
    report = analyze_corpus(corpus)
    payload = json.loads(render_json_report(report))
    assert payload["product"] == "samyak"
    assert payload["capability"] == "corpus"
    assert payload["version"] == "0.1.1"
    assert "utility" not in payload
    assert report.product == "samyak"
    assert report.capability == "corpus"
    assert report.version == "0.1.1"

    text = render_text_report(report)
    assert "Samyak Corpus Intelligence Report" in text
    assert "Product:                  samyak" in text
    assert "Capability:               corpus" in text
    assert "Version:                  0.1.1" in text


def test_cli_help() -> None:
    env = os.environ.copy()
    src = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "samyak", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert "corpus" in result.stdout.lower()
    assert "view" in result.stdout.lower()
    assert "Samyak" in result.stdout


def test_cli_version() -> None:
    env = os.environ.copy()
    src = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "samyak", "--version"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert "samyak" in result.stdout.lower()
    assert __version__ in result.stdout
    assert "0.1.1" in result.stdout


def test_cli_view_subcommand_help() -> None:
    env = os.environ.copy()
    src = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "samyak", "view", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert "--port" in result.stdout
    assert "--no-open" in result.stdout
    assert "127.0.0.1" in result.stdout


def test_cli_corpus_subcommand_help() -> None:
    env = os.environ.copy()
    src = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "samyak", "corpus", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert "--format" in result.stdout
    assert "--output" in result.stdout
    assert "--no-progress" in result.stdout
    assert "--save" in result.stdout
    assert "html" in result.stdout
    assert "json" in result.stdout
    assert "text" in result.stdout


def test_cli_requires_corpus_subcommand() -> None:
    code = main([])
    assert code == 2


def test_cli_corpus_invocation(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {"doc.txt": "Corpus subcommand integration document with enough content.\n"},
    )
    output = tmp_path / "out.json"
    code = main(["corpus", str(corpus), "--format", "json", "-o", str(output)])
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["product"] == "samyak"
    assert payload["capability"] == "corpus"
    assert payload["version"] == "0.1.1"
    assert "utility" not in payload


def test_public_report_models_are_immutable(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {
            "doc.txt": "Public reports should be immutable snapshots for callers.\n" * 3,
            "empty.txt": "",
        },
    )
    report = analyze_corpus(corpus)
    assert isinstance(report, AnalysisReport)
    assert isinstance(report.findings, tuple)
    assert report.findings
    with pytest.raises(FrozenInstanceError):
        report.product = "other"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        report.summary.corpus_root = "leaked"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        report.findings[0].title = "changed"  # type: ignore[misc]


def test_public_api_symbols_are_documented() -> None:
    assert analyze_corpus.__doc__
    assert "CorpusPathError" in (analyze_corpus.__doc__ or "")
    assert AnalysisConfig.__doc__
    assert AnalysisReport.__doc__
    assert Finding.__doc__
    assert Severity.__doc__
    assert CorpusPathError.__doc__
