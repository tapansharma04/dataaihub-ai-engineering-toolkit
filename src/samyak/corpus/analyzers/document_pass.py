"""Single-document analysis pass used by the streaming pipeline."""

from __future__ import annotations

from collections import Counter

from samyak.corpus.analyzers.accumulator import CorpusAccumulator
from samyak.corpus.analyzers.text_stats import (
    collect_text_stats,
    excessive_whitespace,
    high_symbol_ratio,
    overly_fragmented,
    repeated_line_noise,
)
from samyak.corpus.config import AnalysisConfig
from samyak.corpus.models import Document


def process_document(doc: Document, config: AnalysisConfig, acc: CorpusAccumulator) -> None:
    """Analyze one document, record compact corpus state, then return.

    Callers should drop strong references to ``doc`` after this returns so the
    document text can be garbage-collected before the next load.
    """
    acc.record_document_stats(
        extension=doc.extension,
        char_count=doc.char_count,
        size_bytes=doc.size_bytes,
    )

    stats = collect_text_stats(doc.text, short_line_chars=config.short_line_chars)

    if stats.is_empty:
        acc.note("EMPTY_DOCUMENTS", doc.path)
    else:
        length = stats.meaningful_length
        if length < config.small_document_chars:
            acc.note("VERY_SMALL_DOCUMENTS", doc.path, extra=length)
        if length >= config.large_document_chars:
            acc.note("VERY_LARGE_DOCUMENTS", doc.path, extra=length)

        if stats.content_hash is not None:
            acc.note_hash(stats.content_hash, doc.path)

        if not doc.metadata.get("code_heavy") and high_symbol_ratio(
            stats, config.high_symbol_ratio
        ):
            acc.note("HIGH_SYMBOL_RATIO", doc.path)
        if excessive_whitespace(stats, config.excessive_whitespace_ratio):
            acc.note("EXCESSIVE_WHITESPACE", doc.path)
        if repeated_line_noise(
            stats,
            config.repeated_line_ratio,
            config.repeated_line_min_lines,
        ):
            acc.note("REPEATED_LINE_CONTENT", doc.path)

        if stats.max_paragraph_length >= config.long_paragraph_chars:
            acc.note("LONG_UNINTERRUPTED_BLOCKS", doc.path)
        if overly_fragmented(
            stats,
            config.short_line_ratio,
            config.short_line_min_lines,
        ):
            acc.note("OVERLY_FRAGMENTED_LINES", doc.path)

    if doc.format_family == "pdf" or doc.extension == ".pdf":
        _process_pdf(doc, config, acc, stats)
    elif doc.format_family == "html" or doc.extension in {".html", ".htm"}:
        _process_html(doc, config, acc, stats)


def _process_pdf(doc: Document, config: AnalysisConfig, acc: CorpusAccumulator, stats) -> None:
    pages = doc.pages
    page_count = len(pages) or int(doc.metadata.get("page_count") or 0)

    # OCR-likely / text-poor
    if len(pages) < config.pdf_min_pages_for_distribution and doc.char_count > 0:
        if doc.char_count < config.pdf_empty_page_chars and doc.metadata.get("image_page_count", 0):
            acc.note(
                "PDF_OCR_LIKELY",
                doc.path,
                extra={
                    "path": doc.path,
                    "page_count": len(pages),
                    "image_page_count": doc.metadata.get("image_page_count"),
                },
            )
    elif not pages:
        if doc.char_count < config.pdf_empty_page_chars:
            acc.note("PDF_OCR_LIKELY", doc.path)
    else:
        poor = sum(1 for p in pages if p.char_count < config.pdf_empty_page_chars)
        ratio = poor / len(pages)
        if ratio >= config.pdf_text_poor_page_ratio:
            acc.note(
                "PDF_OCR_LIKELY",
                doc.path,
                extra={
                    "path": doc.path,
                    "text_poor_pages": poor,
                    "page_count": len(pages),
                    "ratio": round(ratio, 3),
                    "image_page_count": doc.metadata.get("image_page_count", 0),
                },
            )
        elif poor >= max(2, int(len(pages) * 0.25)) and poor < len(pages):
            acc.note("PDF_UNEVEN_PAGE_TEXT", doc.path)

    # Repeated headers/footers — prefer compact metadata captured at load time.
    edge = doc.metadata.get("page_edge_lines")
    if isinstance(edge, dict) and page_count >= max(4, config.pdf_min_pages_for_distribution):
        candidates = []
        for label, key in (("header", "first"), ("footer", "last")):
            values = [v for v in edge.get(key, []) if isinstance(v, str)]
            if len(values) < 4:
                continue
            counts = Counter(values)
            line, count = counts.most_common(1)[0]
            ratio = count / page_count
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
            acc.note(
                "PDF_REPEATED_HEADER_FOOTER",
                doc.path,
                extra={"path": doc.path, "repeats": candidates},
            )
    elif page_count >= max(4, config.pdf_min_pages_for_distribution):
        # Fallback path for PDFs loaded without edge metadata: scan page blocks
        # without retaining a full secondary line list beyond this check.
        page_texts = doc.text.split("\n\n")
        if len(page_texts) >= len(pages):
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
                acc.note(
                    "PDF_REPEATED_HEADER_FOOTER",
                    doc.path,
                    extra={"path": doc.path, "repeats": candidates},
                )

    # Extraction anomalies
    if pages and doc.char_count >= 200:
        avg = doc.char_count / max(len(pages), 1)
        if (
            stats.non_empty_line_count >= 40
            and (stats.very_short_line_count / stats.non_empty_line_count) >= 0.8
            and avg < 80
        ):
            acc.note("PDF_EXTRACTION_ANOMALY", doc.path)

    # Large/complex
    large_text = doc.char_count >= config.large_document_chars
    if page_count >= config.pdf_high_page_count or large_text:
        densities = [p.char_count for p in pages] if pages else []
        acc.note(
            "PDF_LARGE_OR_COMPLEX",
            doc.path,
            extra={
                "path": doc.path,
                "page_count": page_count,
                "char_count": doc.char_count,
                "min_page_chars": min(densities) if densities else None,
                "max_page_chars": max(densities) if densities else None,
            },
        )

    # Table-rich heuristic
    if (
        stats.non_empty_line_count >= config.pdf_table_like_min_lines
        and (stats.table_like_line_count / stats.non_empty_line_count)
        >= config.pdf_table_like_line_ratio
    ):
        acc.note("PDF_TABLE_RICH", doc.path)


def _process_html(doc: Document, config: AnalysisConfig, acc: CorpusAccumulator, stats) -> None:
    acc.html_doc_count += 1
    content = max(int(doc.metadata.get("content_chars") or doc.char_count), 1)
    boilerplate = int(doc.metadata.get("boilerplate_chars") or 0)
    share = boilerplate / (content + boilerplate)
    if share >= config.html_boilerplate_token_ratio and boilerplate > 200:
        acc.note("HTML_BOILERPLATE_DOMINATION", doc.path)

    if doc.size_bytes >= config.html_low_content_min_raw_bytes:
        ratio = float(doc.metadata.get("raw_to_content_ratio") or 0.0)
        if ratio < config.html_low_content_ratio:
            acc.note("HTML_LOW_MAIN_CONTENT_RATIO", doc.path)

    if doc.char_count >= config.html_large_document_chars:
        acc.note("HTML_LARGE_DOCUMENT", doc.path)

    if doc.metadata.get("code_heavy"):
        acc.note("HTML_CODE_HEAVY", doc.path)

    if stats.first_non_empty_line:
        key = stats.first_non_empty_line[:160]
        if len(key) >= 12:
            acc.note_html_first_line(key, doc.path)
