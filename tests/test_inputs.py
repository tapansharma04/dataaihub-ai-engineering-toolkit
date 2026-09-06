"""Unusual but realistic document inputs should not abort a corpus."""

from __future__ import annotations

from pathlib import Path

from helpers import write_text
from samyak import analyze_corpus
from samyak.corpus.discovery import DiscoveredFile, DiscoveryResult, discover_files
from samyak.corpus.report import render_json_report


def test_zero_byte_and_empty_files_are_findings_not_failures(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "zero.txt").write_bytes(b"")
    write_text(corpus / "ok.txt", "A normal document with enough content about policies.\n" * 3)
    report = analyze_corpus(corpus)
    assert report.summary.analyzed_documents == 2
    assert report.summary.load_errors == 0
    assert any(f.code == "EMPTY_DOCUMENTS" for f in report.findings)


def test_binary_text_extension_does_not_abort_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "binary.txt").write_bytes(bytes(range(256)) + b"\x00\xff")
    write_text(corpus / "ok.txt", "A normal document with enough content about shipping.\n" * 3)
    report = analyze_corpus(corpus)
    assert report.summary.analyzed_documents == 2
    assert report.summary.load_errors == 0
    blob = render_json_report(report)
    assert str(tmp_path.resolve()) not in blob


def test_unicode_filename_and_content(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    name = "naïve_文档.txt"
    write_text(
        corpus / name,
        "Unicode content — café, naïve, 文档 — with enough text about refunds.\n" * 3,
    )
    write_text(
        corpus / "ok.txt",
        "ASCII document with enough content about warehouse logistics.\n" * 3,
    )
    report = analyze_corpus(corpus)
    assert report.summary.analyzed_documents == 2
    assert report.summary.load_errors == 0
    discovered = {item.relative_path for item in discover_files(corpus).files}
    assert name in discovered
    blob = render_json_report(report)
    assert str(tmp_path.resolve()) not in blob


def test_control_characters_do_not_abort_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    write_text(
        corpus / "controls.txt",
        "Policy text\x00with\x07control\x1bcharacters and enough content about returns.\n" * 3,
    )
    write_text(corpus / "ok.txt", "A normal document with enough content about onboarding.\n" * 3)
    report = analyze_corpus(corpus)
    assert report.summary.analyzed_documents == 2
    assert report.summary.load_errors == 0


def test_long_filename_and_nested_path(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    nested = corpus / "a" / "b" / "c"
    nested.mkdir(parents=True)
    long_name = ("n" * 120) + ".txt"
    write_text(
        nested / long_name,
        "Nested long-name document with enough content about billing.\n" * 3,
    )
    report = analyze_corpus(corpus)
    assert report.summary.analyzed_documents == 1
    relative = f"a/b/c/{long_name}"
    discovered = {item.relative_path for item in discover_files(corpus).files}
    assert relative in discovered


def test_truncated_pdf_does_not_abort_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "truncated.pdf").write_bytes(
        b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<< /Type /Catalog"
    )
    write_text(corpus / "ok.txt", "A normal document with enough content about invoices.\n" * 3)
    report = analyze_corpus(corpus)
    assert report.summary.analyzed_documents == 1
    assert report.summary.load_errors == 1
    assert any(f.code == "LOAD_ERRORS" for f in report.findings)
    load = next(f for f in report.findings if f.code == "LOAD_ERRORS")
    assert load.affected_documents == ("truncated.pdf",)


def test_missing_file_after_discovery_is_a_load_error(tmp_path: Path, monkeypatch) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    write_text(corpus / "ok.txt", "A normal document with enough content about policies.\n" * 3)

    def fake_discover(root: Path, **_kwargs) -> DiscoveryResult:
        real = discover_files(root)
        ghost = DiscoveredFile(
            absolute_path=root / "gone.txt",
            relative_path="gone.txt",
            extension=".txt",
            supported=True,
        )
        return DiscoveryResult(files=[*real.files, ghost], access_errors=real.access_errors)

    monkeypatch.setattr("samyak.corpus.pipeline.discover_files", fake_discover)
    report = analyze_corpus(corpus)
    assert report.summary.analyzed_documents == 1
    assert report.summary.load_errors == 1
    assert "gone.txt" in report.summary.load_error_paths
    assert report.summary.discovery_errors == 0
