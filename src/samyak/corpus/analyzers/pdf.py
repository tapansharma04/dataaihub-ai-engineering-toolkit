"""PDF-specific readiness checks (text-layer only; no OCR)."""

from __future__ import annotations

from collections import Counter

from samyak.corpus.analyzers._helpers import sample_paths
from samyak.corpus.config import AnalysisConfig
from samyak.corpus.models import Document, Finding, Severity


def analyze_pdf_documents(
    documents: list[Document],
    config: AnalysisConfig,
) -> list[Finding]:
    """Run PDF-only heuristics on documents with format=pdf."""
    pdf_docs = [d for d in documents if d.metadata.get("format") == "pdf" or d.extension == ".pdf"]
    if not pdf_docs:
        return []

    findings: list[Finding] = []
    findings.extend(_ocr_likely(pdf_docs, config))
    findings.extend(_page_text_distribution(pdf_docs, config))
    findings.extend(_repeated_headers_footers(pdf_docs, config))
    findings.extend(_extraction_anomalies(pdf_docs, config))
    findings.extend(_complex_pdf(pdf_docs, config))
    findings.extend(_table_rich(pdf_docs, config))
    return findings


def _ocr_likely(docs: list[Document], config: AnalysisConfig) -> list[Finding]:
    affected: list[str] = []
    details: list[dict] = []
    for doc in docs:
        pages = doc.pages
        if len(pages) < config.pdf_min_pages_for_distribution and doc.char_count > 0:
            # Single/short docs: flag only if essentially no text but has images.
            if doc.char_count < config.pdf_empty_page_chars and doc.metadata.get(
                "image_page_count", 0
            ):
                affected.append(doc.path)
                details.append(
                    {
                        "path": doc.path,
                        "page_count": len(pages),
                        "image_page_count": doc.metadata.get("image_page_count"),
                    }
                )
            continue
        if not pages:
            if doc.char_count < config.pdf_empty_page_chars:
                affected.append(doc.path)
            continue
        poor = sum(1 for p in pages if p.char_count < config.pdf_empty_page_chars)
        ratio = poor / len(pages)
        if ratio >= config.pdf_text_poor_page_ratio:
            affected.append(doc.path)
            details.append(
                {
                    "path": doc.path,
                    "text_poor_pages": poor,
                    "page_count": len(pages),
                    "ratio": round(ratio, 3),
                    "image_page_count": doc.metadata.get("image_page_count", 0),
                }
            )
    if not affected:
        return []
    return [
        Finding(
            code="PDF_OCR_LIKELY",
            category="pdf",
            severity=Severity.HIGH,
            title="PDF pages likely require OCR",
            message=(
                f"{len(affected)} PDF(s) have little/no extractable text on a large share "
                "of pages (scanned/image-only signal)."
            ),
            why_it_matters=(
                "Without OCR or document-intelligence extraction, these PDFs contribute "
                "little usable text to a RAG index and may silently reduce coverage."
            ),
            recommendation=(
                "Run an OCR / document-intelligence step before indexing, or exclude "
                "scanned PDFs from text-only ingestion pipelines. This tool does not OCR."
            ),
            evidence={
                "count": len(affected),
                "threshold_page_ratio": config.pdf_text_poor_page_ratio,
                "sample": details[: config.max_sample_paths],
                "sample_paths": sample_paths(affected, config),
            },
            affected_documents=tuple(affected),
        )
    ]


def _page_text_distribution(docs: list[Document], config: AnalysisConfig) -> list[Finding]:
    affected: list[str] = []
    for doc in docs:
        pages = doc.pages
        if len(pages) < config.pdf_min_pages_for_distribution:
            continue
        empty = sum(1 for p in pages if p.char_count < config.pdf_empty_page_chars)
        partial = empty >= max(2, int(len(pages) * 0.25)) and empty < len(pages)
        # Partial empty pages (not full OCR-likely already covered at higher ratio)
        if partial and (empty / len(pages)) < config.pdf_text_poor_page_ratio:
            affected.append(doc.path)
    if not affected:
        return []
    return [
        Finding(
            code="PDF_UNEVEN_PAGE_TEXT",
            category="pdf",
            severity=Severity.MEDIUM,
            title="Uneven PDF page text distribution",
            message=(
                f"{len(affected)} PDF(s) contain multiple text-poor pages mixed with "
                "extractable pages."
            ),
            why_it_matters=(
                "Partial extraction gaps can create incomplete retrieval coverage and "
                "confusing citations when some pages index while others do not."
            ),
            recommendation=(
                "Inspect page-level extraction; consider OCR for text-poor pages or "
                "split hybrid born-digital / scanned sections."
            ),
            evidence={
                "count": len(affected),
                "sample_paths": sample_paths(affected, config),
            },
            affected_documents=tuple(affected),
        )
    ]


def _repeated_headers_footers(docs: list[Document], config: AnalysisConfig) -> list[Finding]:
    affected: list[str] = []
    evidence_rows: list[dict] = []
    for doc in docs:
        pages = doc.pages
        if len(pages) < max(4, config.pdf_min_pages_for_distribution):
            continue
        # Reconstruct per-page first/last non-empty lines from stored text blocks.
        page_texts = doc.text.split("\n\n")
        if len(page_texts) < len(pages):
            continue
        first_lines: list[str] = []
        last_lines: list[str] = []
        for block in page_texts[: len(pages)]:
            lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
            if not lines:
                continue
            if 8 <= len(lines[0]) <= 120:
                first_lines.append(lines[0])
            if 8 <= len(lines[-1]) <= 120:
                last_lines.append(lines[-1])
        candidates = []
        for label, values in (("header", first_lines), ("footer", last_lines)):
            if len(values) < 4:
                continue
            counts = Counter(values)
            line, count = counts.most_common(1)[0]
            ratio = count / len(pages)
            if ratio >= config.pdf_header_footer_page_ratio and count >= 4:
                candidates.append(
                    {
                        "role": label,
                        "line": line[:120],
                        "pages": count,
                        "ratio": round(ratio, 3),
                    }
                )
        if candidates:
            affected.append(doc.path)
            evidence_rows.append({"path": doc.path, "repeats": candidates})
    if not affected:
        return []
    return [
        Finding(
            code="PDF_REPEATED_HEADER_FOOTER",
            category="pdf",
            severity=Severity.MEDIUM,
            title="Repeated PDF header/footer lines",
            message=(
                f"{len(affected)} PDF(s) show lines repeated across many pages "
                "(likely headers/footers)."
            ),
            why_it_matters=(
                "Repeated headers/footers can contaminate chunks and retrieval results "
                "with boilerplate that dilutes semantic relevance."
            ),
            recommendation=(
                "Strip running headers/footers during parsing, or use structure-aware "
                "chunking that excludes page chrome."
            ),
            evidence={
                "count": len(affected),
                "sample": evidence_rows[: config.max_sample_paths],
                "sample_paths": sample_paths(affected, config),
            },
            affected_documents=tuple(affected),
        )
    ]


def _extraction_anomalies(docs: list[Document], config: AnalysisConfig) -> list[Finding]:
    affected: list[str] = []
    for doc in docs:
        if not doc.pages or doc.char_count < 200:
            continue
        # Extreme fragmentation relative to page count, or near-empty with many pages.
        avg = doc.char_count / max(len(doc.pages), 1)
        lines = [ln for ln in doc.text.split("\n") if ln.strip()]
        short = sum(1 for ln in lines if len(ln) < 8)
        if len(lines) >= 40 and (short / len(lines)) >= 0.8 and avg < 80:
            affected.append(doc.path)
    if not affected:
        return []
    return [
        Finding(
            code="PDF_EXTRACTION_ANOMALY",
            category="pdf",
            severity=Severity.LOW,
            title="Potential PDF extraction anomalies",
            message=(
                f"{len(affected)} PDF(s) show unusually fragmented/low-density extracted "
                "text that may indicate broken extraction."
            ),
            why_it_matters=(
                "Broken extraction yields noisy chunks. Equations and code can look "
                "fragmented—treat this as a review signal, not a definitive failure."
            ),
            recommendation=(
                "Spot-check extraction quality; try an alternate extractor or layout-aware "
                "pipeline for flagged files."
            ),
            evidence={
                "count": len(affected),
                "sample_paths": sample_paths(affected, config),
            },
            affected_documents=tuple(affected),
        )
    ]


def _complex_pdf(docs: list[Document], config: AnalysisConfig) -> list[Finding]:
    affected: list[str] = []
    details: list[dict] = []
    for doc in docs:
        page_count = len(doc.pages) or int(doc.metadata.get("page_count") or 0)
        large_text = doc.char_count >= config.large_document_chars
        if page_count >= config.pdf_high_page_count or large_text:
            affected.append(doc.path)
            densities = [p.char_count for p in doc.pages] if doc.pages else []
            details.append(
                {
                    "path": doc.path,
                    "page_count": page_count,
                    "char_count": doc.char_count,
                    "min_page_chars": min(densities) if densities else None,
                    "max_page_chars": max(densities) if densities else None,
                }
            )
    if not affected:
        return []
    return [
        Finding(
            code="PDF_LARGE_OR_COMPLEX",
            category="pdf",
            severity=Severity.INFO,
            title="Large or complex PDFs",
            message=(
                f"{len(affected)} PDF(s) have high page counts and/or very large extracted "
                "text volumes."
            ),
            why_it_matters=(
                "Large PDFs are not inherently bad, but often need deliberate parsing and "
                "chunking strategies for effective RAG."
            ),
            recommendation=(
                "Plan section-aware splitting, hierarchical retrieval, or page-window "
                "chunking for these documents."
            ),
            evidence={
                "count": len(affected),
                "page_threshold": config.pdf_high_page_count,
                "sample": details[: config.max_sample_paths],
                "sample_paths": sample_paths(affected, config),
            },
            affected_documents=tuple(affected),
        )
    ]


def _table_rich(docs: list[Document], config: AnalysisConfig) -> list[Finding]:
    """Conservative text heuristic for table-like layouts (not a table extractor)."""
    affected: list[str] = []
    for doc in docs:
        lines = [ln for ln in doc.text.split("\n") if ln.strip()]
        if len(lines) < config.pdf_table_like_min_lines:
            continue
        table_like = 0
        for ln in lines:
            # Multiple columns separated by 2+ spaces or pipes.
            if " | " in ln or len(ln.split()) >= 4 and "  " in ln:
                table_like += 1
        if (table_like / len(lines)) >= config.pdf_table_like_line_ratio:
            affected.append(doc.path)
    if not affected:
        return []
    return [
        Finding(
            code="PDF_TABLE_RICH",
            category="pdf",
            severity=Severity.INFO,
            title="Likely table-rich PDFs",
            message=(
                f"{len(affected)} PDF(s) show text patterns consistent with dense tabular "
                "content (heuristic)."
            ),
            why_it_matters=(
                "Naive text chunking can scramble tables. Table-aware parsing often "
                "preserves retrieval usefulness better."
            ),
            recommendation=(
                "Consider table-aware extraction for flagged files. This tool does not "
                "extract tables."
            ),
            evidence={
                "count": len(affected),
                "sample_paths": sample_paths(affected, config),
                "note": "Heuristic only; not a table detector engine.",
            },
            affected_documents=tuple(affected),
        )
    ]
