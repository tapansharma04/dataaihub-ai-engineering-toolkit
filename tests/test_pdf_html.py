"""PDF and HTML loader/analyzer tests with small generated fixtures."""

from __future__ import annotations

from pathlib import Path

from helpers import write_text
from samyak import analyze_corpus
from samyak.corpus.discovery import DiscoveredFile
from samyak.corpus.loaders import load_document
from samyak.corpus.loaders.pdf import load_pdf_document


def _write_simple_pdf(path: Path, pages: list[str]) -> None:
    """Write a minimal multi-page PDF with extractable text (no external deps)."""
    objects: list[bytes] = []

    def add_obj(data: bytes) -> int:
        objects.append(data)
        return len(objects)

    font_id = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids: list[int] = []
    content_ids: list[int] = []
    for text in pages:
        safe = (
            text.replace("\\", "\\\\")
            .replace("(", "\\(")
            .replace(")", "\\)")
            .encode("latin-1", errors="replace")
            .decode("latin-1")
        )
        chunks = [safe[i : i + 60] for i in range(0, max(len(safe), 1), 60)] or [""]
        ops = ["BT /F1 12 Tf 50 750 Td"]
        for i, chunk in enumerate(chunks):
            if i:
                ops.append("0 -16 Td")
            ops.append(f"({chunk}) Tj")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1", errors="replace")
        content_id = add_obj(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        content_ids.append(content_id)
        page_id = add_obj(b"<< /Type /Page >>")
        page_ids.append(page_id)

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    pages_id = add_obj(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode())
    for i, pid in enumerate(page_ids):
        objects[pid - 1] = (
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_ids[i]} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        ).encode()
    catalog_id = add_obj(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode())

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{i} 0 obj\n".encode())
        out.extend(obj)
        out.extend(b"\nendobj\n")
    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode())
    out.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        ).encode()
    )
    path.write_bytes(bytes(out))


def test_pdf_normal_and_multipage(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    _write_simple_pdf(
        pdf_path,
        [
            "Page one discusses retrieval augmented generation basics.",
            "Page two covers chunking strategies and metadata hygiene.",
        ],
    )
    discovered = DiscoveredFile(pdf_path, "doc.pdf", ".pdf", True)
    doc = load_pdf_document(discovered)
    assert doc.metadata["page_count"] == 2
    assert len(doc.pages) == 2
    assert doc.char_count > 0


def test_pdf_text_poor_detection(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_simple_pdf(corpus / "scanned_like.pdf", ["", "", "", "", "x"])
    _write_simple_pdf(
        corpus / "ok.pdf",
        ["A sufficiently long page of born-digital PDF text about policies."] * 2,
    )
    report = analyze_corpus(corpus)
    assert report.summary.analyzed_documents == 2
    assert any(f.code == "PDF_OCR_LIKELY" for f in report.findings)


def test_pdf_malformed_does_not_abort_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "bad.pdf").write_bytes(b"%PDF-1.4\nnot a real pdf file")
    write_text(corpus / "ok.txt", "A normal text document with enough content for analysis.\n")
    report = analyze_corpus(corpus)
    assert report.summary.analyzed_documents >= 1
    assert report.summary.load_errors >= 1
    assert any(f.code == "LOAD_ERRORS" for f in report.findings)


def test_html_article_and_script_stripped(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    write_text(
        corpus / "article.html",
        """
        <html><head><title>Guide</title>
        <script>alert('x')</script><style>body{color:red}</style></head>
        <body>
        <nav>Home Docs API</nav>
        <h1>Installation</h1>
        <p>Install the package with pip and configure your corpus path carefully.</p>
        <pre><code>pip install example</code></pre>
        <footer>Copyright</footer>
        </body></html>
        """,
    )
    discovered = DiscoveredFile(corpus / "article.html", "article.html", ".html", True)
    doc = load_document(discovered)
    assert "alert" not in doc.text
    assert "Installation" in doc.text
    assert doc.metadata.get("format") == "html"
    report = analyze_corpus(corpus)
    assert report.summary.analyzed_documents == 1


def test_html_code_heavy_not_symbol_noise(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    code = "\n".join(f"x = {i} + {i} * {i}; // symboly" for i in range(40))
    write_text(
        corpus / "api.html",
        f"""
        <html><body>
        <h1>API</h1>
        <p>Reference examples for operators and punctuation-heavy code.</p>
        <pre><code>{code}</code></pre>
        </body></html>
        """,
    )
    report = analyze_corpus(corpus)
    assert not any(f.code == "HIGH_SYMBOL_RATIO" for f in report.findings)
    assert any(f.code == "HTML_CODE_HEAVY" for f in report.findings)


def test_html_boilerplate_heavy(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    nav = " ".join(["NavItem"] * 80)
    write_text(
        corpus / "chrome.html",
        f"""
        <html><body>
        <nav>{nav}</nav>
        <p>Short.</p>
        <footer>{nav}</footer>
        </body></html>
        """,
    )
    report = analyze_corpus(corpus)
    codes = {f.code for f in report.findings}
    assert "HTML_BOILERPLATE_DOMINATION" in codes or "HTML_LOW_MAIN_CONTENT_RATIO" in codes


def test_mixed_format_directory(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    write_text(corpus / "a.txt", "Text document with enough content about RAG readiness checks.\n")
    write_text(
        corpus / "b.md",
        "# Title\n\nMarkdown document with enough content about ingestion pipelines.\n",
    )
    _write_simple_pdf(
        corpus / "c.pdf",
        ["PDF document with enough extractable text about chunking quality."] * 2,
    )
    write_text(
        corpus / "d.html",
        (
            "<html><body><h1>HTML</h1>"
            "<p>Enough HTML article text about documentation corpora.</p>"
            "</body></html>"
        ),
    )
    write_text(corpus / "e.docx", "unsupported")
    report = analyze_corpus(corpus)
    assert report.summary.analyzed_documents == 4
    assert report.summary.unsupported_files == 1
    assert report.summary.analyzed_by_extension.get(".pdf") == 1
    assert report.summary.analyzed_by_extension.get(".html") == 1


def test_malformed_html_still_loads(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    write_text(corpus / "broken.html", "<html><body><p>Unclosed paragraph and <b>bold")
    report = analyze_corpus(corpus)
    assert report.summary.analyzed_documents == 1
    assert report.summary.load_errors == 0
