"""HTML report presentation tests (no analysis-pipeline changes)."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

import pytest

from helpers import make_corpus, write_text
from samyak import __version__, analyze_corpus
from samyak.cli import main
from samyak.corpus.discovery import DiscoveryAccessError, DiscoveryResult, discover_files
from samyak.corpus.models import AnalysisReport, CorpusSummary, Finding, Severity
from samyak.corpus.report import render_html_report, render_json_report, render_text_report


class _HTMLStructureParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.tags: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)


def _parse(html: str) -> _HTMLStructureParser:
    parser = _HTMLStructureParser()
    parser.feed(html)
    parser.close()
    return parser


def test_html_contains_identity_and_is_standalone(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {"doc.txt": "HTML identity document with enough content about policies.\n" * 3},
    )
    report = analyze_corpus(corpus)
    html = render_html_report(report)
    assert html.startswith("<!DOCTYPE html>")
    assert "<html" in html
    assert "</html>" in html
    assert 'charset="utf-8"' in html
    assert "Samyak" in html
    assert report.version in html
    assert __version__ in html
    assert report.product in html
    assert report.capability in html
    assert "https://" not in html
    assert "http://" not in html
    assert "cdn" not in html.lower()
    parser = _parse(html)
    assert "html" in parser.tags
    assert "body" in parser.tags
    assert "style" in parser.tags


def test_html_summary_matches_analysis_report(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {
            "ok.txt": "Normal document with enough content about shipping and refunds.\n" * 3,
            "skip.docx": "unsupported",
        },
    )
    report = analyze_corpus(corpus)
    html = render_html_report(report)
    summary = report.summary
    assert str(summary.total_discovered_files) in html
    assert str(summary.supported_files) in html
    assert str(summary.unsupported_files) in html
    assert str(summary.analyzed_documents) in html
    assert str(summary.load_errors) in html
    assert str(summary.discovery_errors) in html
    assert summary.corpus_root in html
    assert ".docx" in html


def test_html_findings_match_analysis_report(tmp_path: Path) -> None:
    body = "Shared duplicate body for HTML finding checks about policies.\n" * 3
    corpus = make_corpus(
        tmp_path,
        {"a.txt": body, "b.txt": body, "empty.txt": ""},
    )
    report = analyze_corpus(corpus)
    html = render_html_report(report)
    assert "EXACT_DUPLICATES" in html
    assert "EMPTY_DOCUMENTS" in html
    for finding in report.findings:
        assert finding.code in html
        assert finding.title in html
        assert finding.message in html
        for path in finding.affected_documents:
            assert path in html
    for count in report.severity_counts().values():
        assert str(count) in html
    assert "HIGH" in html


def test_html_preserves_relative_paths_and_hides_absolute(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {
            "nested/dir/empty.txt": "",
            "ok.txt": "Nested inventory document with enough content about invoices.\n" * 3,
        },
    )
    report = analyze_corpus(corpus)
    html = render_html_report(report)
    blob = html.lower()
    assert "nested/dir/empty.txt" in html
    assert str(tmp_path) not in html
    assert str(tmp_path.resolve()) not in html
    assert "/users/" not in blob
    assert ".private" not in blob


def test_html_escapes_special_characters_in_filenames(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    name = 'ampersand_&_lt_<gt>_quote_".txt'
    write_text(corpus / name, "")
    write_text(corpus / "ok.txt", "A normal document with enough content about billing.\n" * 3)
    report = analyze_corpus(corpus)
    html = render_html_report(report)
    assert name not in html
    assert "ampersand_&amp;_lt_&lt;gt&gt;_quote_&quot;.txt" in html
    assert "<gt>" not in html
    empty = next(f for f in report.findings if f.code == "EMPTY_DOCUMENTS")
    assert name in empty.affected_documents


def test_html_empty_corpus(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    report = analyze_corpus(empty)
    html = render_html_report(report)
    assert report.findings == ()
    assert "No findings" in html
    assert ">0<" in html or "Count: 0" in html
    assert str(tmp_path) not in html


def test_html_load_errors(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "bad.pdf").write_bytes(b"%PDF-1.4\nnot a real pdf file")
    write_text(corpus / "ok.txt", "A normal document with enough content about policies.\n" * 3)
    report = analyze_corpus(corpus)
    html = render_html_report(report)
    assert report.summary.load_errors == 1
    assert "bad.pdf" in html
    assert "LOAD_ERRORS" in html
    assert str(tmp_path.resolve()) not in html


def test_html_discovery_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus = make_corpus(
        tmp_path,
        {"ok.txt": "Normal document with enough content about warehouse logistics.\n" * 3},
    )

    def fake_discover(root: Path, **_kwargs) -> DiscoveryResult:
        real = discover_files(root)
        extra = DiscoveryAccessError(relative_path="blocked.txt", reason="inaccessible")
        return DiscoveryResult(files=list(real.files), access_errors=(extra,))

    monkeypatch.setattr("samyak.corpus.pipeline.discover_files", fake_discover)
    report = analyze_corpus(corpus)
    html = render_html_report(report)
    assert "DISCOVERY_ERRORS" in html
    assert "blocked.txt" in html
    assert "inaccessible" in html
    assert report.summary.discovery_errors == 1


def test_html_duplicate_findings_are_deterministic(tmp_path: Path) -> None:
    body = "Duplicate HTML report body about returns and shipping policies.\n" * 3
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    write_text(corpus / "z.txt", body)
    write_text(corpus / "a.txt", body)
    first = render_html_report(analyze_corpus(corpus))
    second = render_html_report(analyze_corpus(corpus))
    assert first == second
    a_pos = first.index("a.txt")
    z_pos = first.index("z.txt")
    assert a_pos < z_pos


def test_cli_html_stdout_and_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    corpus = make_corpus(
        tmp_path,
        {"doc.txt": "CLI HTML output document with enough content for analysis.\n" * 3},
    )
    code = main(["corpus", str(corpus), "--format", "html", "--no-progress"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.startswith("<!DOCTYPE html>")
    assert "Samyak" in captured.out
    assert captured.err == ""

    output = tmp_path / "report.html"
    code = main(
        [
            "corpus",
            str(corpus),
            "--format",
            "html",
            "--no-progress",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    written = output.read_text(encoding="utf-8")
    assert written.startswith("<!DOCTYPE html>")
    assert written == captured.out


def test_cli_html_output_write_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    corpus = make_corpus(
        tmp_path,
        {"doc.txt": "HTML output path pointing at a directory should fail.\n" * 3},
    )
    code = main(
        [
            "corpus",
            str(corpus),
            "--format",
            "html",
            "--no-progress",
            "--output",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "error:" in captured.err
    assert "Traceback" not in captured.err


def test_cli_html_invalid_format_is_usage_error(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {"doc.txt": "Invalid format should be rejected by argparse.\n" * 3},
    )
    with pytest.raises(SystemExit) as exc:
        main(["corpus", str(corpus), "--format", "pdf"])
    assert exc.value.code == 2


def test_json_and_text_renderers_unchanged_by_html(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {
            "a.txt": "Compatibility document alpha about policies and shipping.\n" * 3,
            "b.txt": "Compatibility document alpha about policies and shipping.\n" * 3,
        },
    )
    report = analyze_corpus(corpus)
    text = render_text_report(report)
    payload = render_json_report(report)
    html = render_html_report(report)
    assert text.startswith("Samyak Corpus Intelligence Report")
    assert '"product": "samyak"' in payload
    assert html.startswith("<!DOCTYPE html>")
    assert text != html
    assert payload != html
    assert "Samyak Corpus Intelligence Report" in text
    assert "<!DOCTYPE html>" not in text
    assert "<!DOCTYPE html>" not in payload


def _summary() -> CorpusSummary:
    return CorpusSummary(
        corpus_root="docs",
        total_discovered_files=2,
        supported_files=2,
        unsupported_files=0,
        analyzed_documents=2,
        load_errors=0,
        total_characters=200,
        total_bytes=200,
        average_characters=100.0,
        median_characters=100.0,
        min_characters=80,
        max_characters=120,
    )


def _report_with_evidence(evidence: dict, *, affected: tuple[str, ...] = ()) -> AnalysisReport:
    return AnalysisReport(
        product="samyak",
        capability="corpus",
        version="0.1.0",
        summary=_summary(),
        findings=(
            Finding(
                code="EXACT_DUPLICATES",
                category="duplicates",
                severity=Severity.HIGH,
                title="Exact duplicate documents",
                message="2 documents belong to 1 exact-duplicate group(s).",
                why_it_matters="Duplicate content can produce redundant retrieval results.",
                recommendation="Review and remove duplicate documents.",
                evidence=evidence,
                affected_documents=affected,
            ),
        ),
        config={"small_document_chars": 100},
    )


def test_nested_duplicate_evidence_layout_and_escaping() -> None:
    long_path = "deep/" + "very_long_directory_name/" * 8 + 'file_&_lt_<gt>_quote_".txt'
    unbroken = "A" * 180
    html = render_html_report(
        _report_with_evidence(
            {
                "duplicate_group_count": 1,
                "affected_document_count": 2,
                "groups": [
                    {
                        "size": 2,
                        "representative_path": long_path,
                        "paths": [long_path, "normal_policy.txt"],
                        "note": unbroken,
                    }
                ],
                "count": 2,
                "flag": True,
                "unsafe": 'a & b <c> "quote"',
            },
            affected=(long_path, "normal_policy.txt"),
        )
    )
    assert "representative_path" in html
    assert "duplicate_group_count" in html
    assert long_path not in html
    assert "very_long_directory_name/" * 8 in html
    assert "file_&amp;_lt_&lt;gt&gt;_quote_&quot;.txt" in html
    assert unbroken in html
    assert "a &amp; b &lt;c&gt; &quot;quote&quot;" in html
    assert "<gt>" not in html
    assert html.count("<dt>count</dt>") == 1
    assert "true" in html
    style = html.split("<style>", 1)[1].split("</style>", 1)[0]
    assert "minmax(8rem,32%)" not in style
    assert "dl.nested>div{display:grid;grid-template-columns:minmax(0,1fr)" in style
    assert "dl.meta>div,dl.evidence>div{display:grid;min-width:0;" in style
    assert "grid-template-columns:minmax(0,32%) minmax(0,1fr)" in style
    assert "@media (max-width:40rem){dl.meta>div,dl.evidence>div{" in style
    assert "dt,dd{min-width:0;overflow-wrap:anywhere}" in style
    assert "https://" not in html


def test_simple_scalar_evidence_stays_definition_list() -> None:
    html = render_html_report(_report_with_evidence({"count": 2, "threshold": 0}))
    assert '<dl class="evidence">' in html
    assert "<dt>count</dt>" in html
    assert "<dd>2</dd>" in html
    assert "<dt>threshold</dt>" in html
    assert "<dd>0</dd>" in html
    assert '<dl class="nested">' not in html
    style = html.split("<style>", 1)[1].split("</style>", 1)[0]
    assert "dl.evidence>div{display:grid;min-width:0;" in style
    assert "grid-template-columns:minmax(0,32%) minmax(0,1fr)" in style
