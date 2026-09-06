"""Tests for corpus filesystem discovery access failures."""

from __future__ import annotations

from pathlib import Path

import pytest

from helpers import make_corpus, write_text
from samyak import AnalysisConfig, analyze_corpus
from samyak.corpus.discovery import (
    DiscoveryAccessError,
    DiscoveryResult,
    PathCheckResult,
    discover_files,
)


def test_inaccessible_file_is_recorded(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {"ok.txt": "Accessible document with enough content about policies.\n"},
    )

    def walk(root, followlinks=False, onerror=None):
        yield str(root), [], ["blocked.txt", "ok.txt"]

    def check_path(path: Path, root: Path) -> PathCheckResult:
        if path.name == "blocked.txt":
            return PathCheckResult(include=False, error_reason="inaccessible")
        if path.name == "ok.txt":
            return PathCheckResult(include=True)
        return PathCheckResult(include=False)

    result = discover_files(corpus, walk=walk, check_path=check_path)
    assert [item.relative_path for item in result.files] == ["ok.txt"]
    assert result.access_errors == (
        DiscoveryAccessError(relative_path="blocked.txt", reason="inaccessible"),
    )


def test_unreadable_directory_is_recorded(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {"ok.txt": "Accessible document with enough content about shipping.\n"},
    )

    def walk(root, followlinks=False, onerror=None):
        yield str(root), ["denied"], ["ok.txt"]
        if onerror is not None:
            err = OSError("listdir failed")
            err.filename = str(Path(root) / "denied")
            onerror(err)

    def check_path(path: Path, root: Path) -> PathCheckResult:
        return PathCheckResult(include=path.name == "ok.txt")

    result = discover_files(corpus, walk=walk, check_path=check_path)
    assert [item.relative_path for item in result.files] == ["ok.txt"]
    assert result.access_errors == (
        DiscoveryAccessError(relative_path="denied", reason="unreadable_directory"),
    )


def test_discovery_errors_are_deterministic(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {"ok.txt": "Deterministic discovery error ordering about refunds.\n"},
    )

    def walk(root, followlinks=False, onerror=None):
        yield str(root), [], ["z.txt", "a.txt", "ok.txt"]

    def check_path(path: Path, root: Path) -> PathCheckResult:
        if path.name in {"a.txt", "z.txt"}:
            return PathCheckResult(include=False, error_reason="inaccessible")
        return PathCheckResult(include=path.name == "ok.txt")

    first = discover_files(corpus, walk=walk, check_path=check_path)
    second = discover_files(corpus, walk=walk, check_path=check_path)
    assert first.access_errors == second.access_errors
    assert [item.relative_path for item in first.access_errors] == ["a.txt", "z.txt"]


def test_report_surfaces_discovery_errors(tmp_path: Path, monkeypatch) -> None:
    corpus = make_corpus(
        tmp_path,
        {"ok.txt": "Normal document with enough content about warehouse logistics.\n"},
    )

    def fake_discover(root: Path, **_kwargs) -> DiscoveryResult:
        real = discover_files(root)
        extra = DiscoveryAccessError(relative_path="blocked.txt", reason="inaccessible")
        return DiscoveryResult(files=list(real.files), access_errors=(extra,))

    monkeypatch.setattr("samyak.corpus.pipeline.discover_files", fake_discover)
    report = analyze_corpus(corpus)
    assert report.summary.discovery_errors == 1
    assert report.summary.discovery_error_paths == ("blocked.txt",)
    finding = next(f for f in report.findings if f.code == "DISCOVERY_ERRORS")
    assert finding.severity.value == "MEDIUM"
    assert "blocked.txt" in finding.affected_documents
    assert finding.evidence["sample"][0]["reason"] == "inaccessible"
    assert report.summary.analyzed_documents == 1
    assert report.summary.load_errors == 0
    assert report.summary.unsupported_files == 0


def test_discovery_unsupported_and_load_errors_are_distinct(tmp_path: Path, monkeypatch) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    write_text(corpus / "ok.txt", "Normal document with enough content about policies.\n" * 3)
    write_text(corpus / "skip.docx", "unsupported")
    (corpus / "bad.pdf").write_bytes(b"%PDF-1.4\nnot a real pdf file")

    def fake_discover(root: Path, **_kwargs) -> DiscoveryResult:
        real = discover_files(root)
        extra = DiscoveryAccessError(relative_path="hidden.txt", reason="inaccessible")
        return DiscoveryResult(files=list(real.files), access_errors=(extra,))

    monkeypatch.setattr("samyak.corpus.pipeline.discover_files", fake_discover)
    report = analyze_corpus(corpus, config=AnalysisConfig())
    codes = {f.code for f in report.findings}
    assert "DISCOVERY_ERRORS" in codes
    assert "UNSUPPORTED_FILES" in codes
    assert "LOAD_ERRORS" in codes
    summary = report.summary
    assert summary.discovery_errors == 1
    assert summary.unsupported_files == 1
    assert summary.load_errors == 1
    assert summary.analyzed_documents == 1
    assert summary.supported_files == 2
    assert summary.total_discovered_files == 3
    assert summary.supported_files + summary.unsupported_files == summary.total_discovered_files
    assert summary.analyzed_documents + summary.load_errors == summary.supported_files
    assert "hidden.txt" not in (summary.load_error_paths or ())
    discovery = next(f for f in report.findings if f.code == "DISCOVERY_ERRORS")
    load = next(f for f in report.findings if f.code == "LOAD_ERRORS")
    unsupported = next(f for f in report.findings if f.code == "UNSUPPORTED_FILES")
    assert discovery.affected_documents == ("hidden.txt",)
    assert "bad.pdf" in load.affected_documents
    assert "skip.docx" in unsupported.affected_documents


def test_broken_symlink_is_skipped_not_an_error(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {"ok.txt": "Accessible document with enough content about policies.\n"},
    )
    broken = corpus / "broken.txt"
    try:
        broken.symlink_to(corpus / "does-not-exist.txt")
    except OSError:
        pytest.skip("symlinks not supported in this environment")
    result = discover_files(corpus)
    rels = {item.relative_path for item in result.files}
    assert "ok.txt" in rels
    assert "broken.txt" not in rels
    assert result.access_errors == ()


def test_internal_file_symlink_is_included(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {"ok.txt": "Accessible document with enough content about shipping.\n"},
    )
    alias = corpus / "alias.txt"
    try:
        alias.symlink_to(corpus / "ok.txt")
    except OSError:
        pytest.skip("symlinks not supported in this environment")
    result = discover_files(corpus)
    rels = {item.relative_path for item in result.files}
    assert rels == {"alias.txt", "ok.txt"}
    assert result.access_errors == ()


def test_directory_symlink_to_parent_does_not_recurse(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {"ok.txt": "Accessible document with enough content about refunds.\n"},
    )
    loop = corpus / "loop"
    try:
        loop.symlink_to(corpus)
    except OSError:
        pytest.skip("symlinks not supported in this environment")
    result = discover_files(corpus)
    rels = [item.relative_path for item in result.files]
    assert rels == ["ok.txt"]
    assert result.access_errors == ()
    assert all(not path.startswith("loop/") for path in rels)


def test_external_directory_symlink_is_not_descended(tmp_path: Path) -> None:
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    write_text(outside_dir / "secret.txt", "Secret outside document that must not be ingested.\n")
    corpus = make_corpus(
        tmp_path,
        {"ok.txt": "Accessible document with enough content about warehouse logistics.\n"},
    )
    linked = corpus / "linked_dir"
    try:
        linked.symlink_to(outside_dir)
    except OSError:
        pytest.skip("symlinks not supported in this environment")
    result = discover_files(corpus)
    rels = {item.relative_path for item in result.files}
    assert rels == {"ok.txt"}
    assert "linked_dir/secret.txt" not in rels
    assert result.access_errors == ()
