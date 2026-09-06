"""Build Finding objects from accumulated corpus state."""

from __future__ import annotations

from samyak.corpus.analyzers._helpers import sample_paths
from samyak.corpus.analyzers.accumulator import CorpusAccumulator, PathBucket
from samyak.corpus.config import AnalysisConfig
from samyak.corpus.models import Finding, Severity


def build_findings(acc: CorpusAccumulator, config: AnalysisConfig) -> list[Finding]:
    """Materialize findings after all documents have been processed."""
    findings: list[Finding] = []
    n = acc.analyzed_documents

    findings.extend(_empty(acc, config, n))
    findings.extend(_sizes(acc, config, n))
    findings.extend(_duplicates(acc, config))
    findings.extend(_text_quality(acc, config))
    findings.extend(_chunkability(acc, config))
    findings.extend(_pdf(acc, config))
    findings.extend(_html(acc, config))
    return findings


def _affected(bucket: PathBucket) -> tuple[str, ...]:
    return tuple(bucket.paths)


def _empty(acc: CorpusAccumulator, config: AnalysisConfig, n: int) -> list[Finding]:
    bucket = acc.buckets.get("EMPTY_DOCUMENTS")
    if not bucket or bucket.count == 0:
        return []
    return [
        Finding(
            code="EMPTY_DOCUMENTS",
            category="empty_documents",
            severity=Severity.HIGH,
            title="Empty documents",
            message=(f"{bucket.count} of {n} documents contain no meaningful text."),
            why_it_matters=(
                "Empty documents usually should not be indexed. They waste embedding and "
                "storage cost and can pollute retrieval with contentless chunks."
            ),
            recommendation=(
                "Remove empty documents from the corpus before indexing, or fix the "
                "extraction/export process that produced them."
            ),
            evidence={
                "count": bucket.count,
                "sample_paths": sample_paths(bucket.paths, config),
                "affected_document_count": bucket.count,
                "affected_documents_truncated": bucket.count > len(bucket.paths),
            },
            affected_documents=_affected(bucket),
        )
    ]


def _sizes(acc: CorpusAccumulator, config: AnalysisConfig, n: int) -> list[Finding]:
    findings: list[Finding] = []
    small = acc.buckets.get("VERY_SMALL_DOCUMENTS")
    if small and small.count:
        findings.append(
            Finding(
                code="VERY_SMALL_DOCUMENTS",
                category="document_size",
                severity=Severity.MEDIUM,
                title="Very small / low-information documents",
                message=(
                    f"{small.count} of {n} documents have fewer than "
                    f"{config.small_document_chars} characters of meaningful text."
                ),
                why_it_matters=(
                    "Very short documents may not be useful as independent retrieval "
                    "units. They can also indicate truncated exports or incomplete files. "
                    "Not every small document is bad—short policies, FAQs, or titles may "
                    "be intentional."
                ),
                recommendation=(
                    "Review small documents: merge related stubs, restore truncated "
                    "exports, or keep them only when they are intentionally concise."
                ),
                evidence={
                    "threshold_chars": config.small_document_chars,
                    "count": small.count,
                    "sample_paths": sample_paths(small.paths, config),
                    "sample_lengths": list(small.extras[: config.max_sample_paths]),
                    "affected_document_count": small.count,
                    "affected_documents_truncated": small.count > len(small.paths),
                },
                affected_documents=_affected(small),
            )
        )
    large = acc.buckets.get("VERY_LARGE_DOCUMENTS")
    if large and large.count:
        findings.append(
            Finding(
                code="VERY_LARGE_DOCUMENTS",
                category="document_size",
                severity=Severity.MEDIUM,
                title="Very large documents",
                message=(
                    f"{large.count} of {n} documents have at least "
                    f"{config.large_document_chars} characters of meaningful text."
                ),
                why_it_matters=(
                    "Very large documents often need deliberate parsing and chunking. "
                    "Naive whole-document embedding can lose precision; naive fixed "
                    "chunking can split important context awkwardly."
                ),
                recommendation=(
                    "Review parsing and chunking strategy for these documents "
                    "(structure-aware splits, section boundaries, or hierarchical retrieval)."
                ),
                evidence={
                    "threshold_chars": config.large_document_chars,
                    "count": large.count,
                    "sample_paths": sample_paths(large.paths, config),
                    "sample_lengths": list(large.extras[: config.max_sample_paths]),
                    "affected_document_count": large.count,
                    "affected_documents_truncated": large.count > len(large.paths),
                },
                affected_documents=_affected(large),
            )
        )
    return findings


def _duplicates(acc: CorpusAccumulator, config: AnalysisConfig) -> list[Finding]:
    duplicate_groups = [sorted(paths) for paths in acc.hash_to_paths.values() if len(paths) > 1]
    duplicate_groups.sort(key=lambda g: (-len(g), g[0]))
    if not duplicate_groups:
        return []

    affected_count = len({path for group in duplicate_groups for path in group})
    # Retain bounded path examples across groups.
    retained: list[str] = []
    seen: set[str] = set()
    for group in duplicate_groups:
        for path in group:
            if path in seen:
                continue
            seen.add(path)
            if len(retained) < config.max_affected_documents:
                retained.append(path)
            if len(retained) >= config.max_affected_documents:
                break
        if len(retained) >= config.max_affected_documents:
            break
    retained = sorted(retained)

    group_summaries = [
        {
            "size": len(group),
            "representative_path": group[0],
            "paths": sample_paths(group, config),
        }
        for group in duplicate_groups[: config.max_sample_paths]
    ]
    return [
        Finding(
            code="EXACT_DUPLICATES",
            category="duplicates",
            severity=Severity.HIGH,
            title="Exact duplicate documents",
            message=(
                f"{affected_count} documents belong to {len(duplicate_groups)} "
                "exact-duplicate group(s)."
            ),
            why_it_matters=(
                "Duplicate content can produce redundant retrieval results, inflate "
                "embedding and storage cost, and waste context window space when multiple "
                "near-identical chunks are retrieved together."
            ),
            recommendation=(
                "Review and remove duplicate documents before indexing unless "
                "duplication is intentional (for example, mirrored sources)."
            ),
            evidence={
                "duplicate_group_count": len(duplicate_groups),
                "affected_document_count": affected_count,
                "groups": group_summaries,
                "affected_documents_truncated": affected_count > len(retained),
            },
            affected_documents=tuple(retained),
        )
    ]


def _text_quality(acc: CorpusAccumulator, config: AnalysisConfig) -> list[Finding]:
    findings: list[Finding] = []
    high = acc.buckets.get("HIGH_SYMBOL_RATIO")
    if high and high.count:
        findings.append(
            Finding(
                code="HIGH_SYMBOL_RATIO",
                category="text_quality",
                severity=Severity.LOW,
                title="Potential high symbol / non-alphanumeric ratio",
                message=(
                    f"{high.count} document(s) have a high ratio of non-alphanumeric "
                    f"characters (threshold {config.high_symbol_ratio:.0%})."
                ),
                why_it_matters=(
                    "Unusually symbol-heavy text can indicate extraction noise, corrupted "
                    "exports, or binary-like content. Legitimate code and markup can also "
                    "trigger this heuristic, so treat matches as candidates for review."
                ),
                recommendation=(
                    "Spot-check flagged files. If content is corrupted or non-textual, "
                    "exclude or re-extract before indexing."
                ),
                evidence={
                    "threshold": config.high_symbol_ratio,
                    "count": high.count,
                    "sample_paths": sample_paths(high.paths, config),
                    "affected_document_count": high.count,
                    "affected_documents_truncated": high.count > len(high.paths),
                },
                affected_documents=_affected(high),
            )
        )
    ws = acc.buckets.get("EXCESSIVE_WHITESPACE")
    if ws and ws.count:
        findings.append(
            Finding(
                code="EXCESSIVE_WHITESPACE",
                category="text_quality",
                severity=Severity.LOW,
                title="Potential excessive whitespace",
                message=(
                    f"{ws.count} document(s) appear dominated by whitespace "
                    f"(threshold {config.excessive_whitespace_ratio:.0%})."
                ),
                why_it_matters=(
                    "Whitespace-heavy documents may be sparsely extracted, padded, or "
                    "poorly converted. Indexing them can yield low-value chunks."
                ),
                recommendation=(
                    "Inspect extraction quality and normalize whitespace if the content "
                    "is otherwise useful."
                ),
                evidence={
                    "threshold": config.excessive_whitespace_ratio,
                    "count": ws.count,
                    "sample_paths": sample_paths(ws.paths, config),
                    "affected_document_count": ws.count,
                    "affected_documents_truncated": ws.count > len(ws.paths),
                },
                affected_documents=_affected(ws),
            )
        )
    repeated = acc.buckets.get("REPEATED_LINE_CONTENT")
    if repeated and repeated.count:
        findings.append(
            Finding(
                code="REPEATED_LINE_CONTENT",
                category="text_quality",
                severity=Severity.MEDIUM,
                title="Potential repeated-line / boilerplate noise",
                message=(
                    f"{repeated.count} document(s) show heavy line repetition "
                    f"(one line accounting for ≥{config.repeated_line_ratio:.0%} of "
                    "non-empty lines)."
                ),
                why_it_matters=(
                    "Repeated lines often indicate logging dumps, copy-paste corruption, "
                    "or boilerplate-dominated files that reduce retrieval usefulness."
                ),
                recommendation=(
                    "Review flagged documents and deduplicate or trim repetitive content "
                    "before indexing when the repetition is not intentional."
                ),
                evidence={
                    "threshold_ratio": config.repeated_line_ratio,
                    "min_lines": config.repeated_line_min_lines,
                    "count": repeated.count,
                    "sample_paths": sample_paths(repeated.paths, config),
                    "affected_document_count": repeated.count,
                    "affected_documents_truncated": repeated.count > len(repeated.paths),
                },
                affected_documents=_affected(repeated),
            )
        )
    return findings


def _chunkability(acc: CorpusAccumulator, config: AnalysisConfig) -> list[Finding]:
    findings: list[Finding] = []
    long_blocks = acc.buckets.get("LONG_UNINTERRUPTED_BLOCKS")
    if long_blocks and long_blocks.count:
        findings.append(
            Finding(
                code="LONG_UNINTERRUPTED_BLOCKS",
                category="chunkability",
                severity=Severity.MEDIUM,
                title="Long uninterrupted text blocks",
                message=(
                    f"{long_blocks.count} document(s) contain paragraph-like "
                    f"blocks of at least {config.long_paragraph_chars} characters "
                    "without a blank-line break."
                ),
                why_it_matters=(
                    "Very long uninterrupted blocks make structure-aware chunking harder. "
                    "Fixed-size chunking may split mid-argument and hurt retrieval precision. "
                    "This check does not choose an optimal chunk size."
                ),
                recommendation=(
                    "Consider structure-aware splitting (headings, sections) or smaller "
                    "semantic units before indexing these documents."
                ),
                evidence={
                    "threshold_chars": config.long_paragraph_chars,
                    "count": long_blocks.count,
                    "sample_paths": sample_paths(long_blocks.paths, config),
                    "affected_document_count": long_blocks.count,
                    "affected_documents_truncated": long_blocks.count > len(long_blocks.paths),
                },
                affected_documents=_affected(long_blocks),
            )
        )
    frag = acc.buckets.get("OVERLY_FRAGMENTED_LINES")
    if frag and frag.count:
        findings.append(
            Finding(
                code="OVERLY_FRAGMENTED_LINES",
                category="chunkability",
                severity=Severity.LOW,
                title="Documents dominated by very short lines",
                message=(
                    f"{frag.count} document(s) are dominated by very short lines "
                    f"(≥{config.short_line_ratio:.0%} of non-empty lines shorter than "
                    f"{config.short_line_chars} characters)."
                ),
                why_it_matters=(
                    "Highly fragmented text (menus, TOC dumps, broken extraction) can yield "
                    "weak standalone chunks and noisy retrieval results."
                ),
                recommendation=(
                    "Review whether these files should be reformatted, merged into richer "
                    "sections, or excluded from indexing."
                ),
                evidence={
                    "short_line_chars": config.short_line_chars,
                    "short_line_ratio": config.short_line_ratio,
                    "count": frag.count,
                    "sample_paths": sample_paths(frag.paths, config),
                    "affected_document_count": frag.count,
                    "affected_documents_truncated": frag.count > len(frag.paths),
                },
                affected_documents=_affected(frag),
            )
        )
    return findings


def _pdf(acc: CorpusAccumulator, config: AnalysisConfig) -> list[Finding]:
    findings: list[Finding] = []

    def add(
        code: str,
        severity: Severity,
        title: str,
        message: str,
        why: str,
        rec: str,
        *,
        extra_evidence: dict | None = None,
    ) -> None:
        bucket = acc.buckets.get(code)
        if not bucket or not bucket.count:
            return
        evidence = {
            "count": bucket.count,
            "sample_paths": sample_paths(bucket.paths, config),
            "affected_document_count": bucket.count,
            "affected_documents_truncated": bucket.count > len(bucket.paths),
        }
        if extra_evidence:
            evidence.update(extra_evidence)
        if bucket.extras:
            evidence["sample"] = list(bucket.extras[: config.max_sample_paths])
        findings.append(
            Finding(
                code=code,
                category="pdf",
                severity=severity,
                title=title,
                message=message.format(count=bucket.count),
                why_it_matters=why,
                recommendation=rec,
                evidence=evidence,
                affected_documents=_affected(bucket),
            )
        )

    add(
        "PDF_OCR_LIKELY",
        Severity.HIGH,
        "PDF pages likely require OCR",
        (
            "{count} PDF(s) have little/no extractable text on a large share "
            "of pages (scanned/image-only signal)."
        ),
        (
            "Without OCR or document-intelligence extraction, these PDFs contribute "
            "little usable text to a RAG index and may silently reduce coverage."
        ),
        (
            "Run an OCR / document-intelligence step before indexing, or exclude "
            "scanned PDFs from text-only ingestion pipelines. This tool does not OCR."
        ),
        extra_evidence={"threshold_page_ratio": config.pdf_text_poor_page_ratio},
    )
    add(
        "PDF_UNEVEN_PAGE_TEXT",
        Severity.MEDIUM,
        "Uneven PDF page text distribution",
        ("{count} PDF(s) contain multiple text-poor pages mixed with extractable pages."),
        (
            "Partial extraction gaps can create incomplete retrieval coverage and "
            "confusing citations when some pages index while others do not."
        ),
        (
            "Inspect page-level extraction; consider OCR for text-poor pages or "
            "split hybrid born-digital / scanned sections."
        ),
    )
    add(
        "PDF_REPEATED_HEADER_FOOTER",
        Severity.MEDIUM,
        "Repeated PDF header/footer lines",
        ("{count} PDF(s) show lines repeated across many pages (likely headers/footers)."),
        (
            "Repeated headers/footers can contaminate chunks and retrieval results "
            "with boilerplate that dilutes semantic relevance."
        ),
        (
            "Strip running headers/footers during parsing, or use structure-aware "
            "chunking that excludes page chrome."
        ),
    )
    add(
        "PDF_EXTRACTION_ANOMALY",
        Severity.LOW,
        "Potential PDF extraction anomalies",
        (
            "{count} PDF(s) show unusually fragmented/low-density extracted "
            "text that may indicate broken extraction."
        ),
        (
            "Broken extraction yields noisy chunks. Equations and code can look "
            "fragmented—treat this as a review signal, not a definitive failure."
        ),
        (
            "Spot-check extraction quality; try an alternate extractor or layout-aware "
            "pipeline for flagged files."
        ),
    )
    add(
        "PDF_LARGE_OR_COMPLEX",
        Severity.INFO,
        "Large or complex PDFs",
        ("{count} PDF(s) have high page counts and/or very large extracted text volumes."),
        (
            "Large PDFs are not inherently bad, but often need deliberate parsing and "
            "chunking strategies for effective RAG."
        ),
        (
            "Plan section-aware splitting, hierarchical retrieval, or page-window "
            "chunking for these documents."
        ),
        extra_evidence={"page_threshold": config.pdf_high_page_count},
    )
    add(
        "PDF_TABLE_RICH",
        Severity.INFO,
        "Likely table-rich PDFs",
        ("{count} PDF(s) show text patterns consistent with dense tabular content (heuristic)."),
        (
            "Naive text chunking can scramble tables. Table-aware parsing often "
            "preserves retrieval usefulness better."
        ),
        ("Consider table-aware extraction for flagged files. This tool does not extract tables."),
        extra_evidence={"note": "Heuristic only; not a table detector engine."},
    )
    return findings


def _html(acc: CorpusAccumulator, config: AnalysisConfig) -> list[Finding]:
    findings: list[Finding] = []

    def add(
        code: str,
        severity: Severity,
        title: str,
        message: str,
        why: str,
        rec: str,
        *,
        extra_evidence: dict | None = None,
    ) -> None:
        bucket = acc.buckets.get(code)
        if not bucket or not bucket.count:
            return
        evidence = {
            "count": bucket.count,
            "sample_paths": sample_paths(bucket.paths, config),
            "affected_document_count": bucket.count,
            "affected_documents_truncated": bucket.count > len(bucket.paths),
        }
        if extra_evidence:
            evidence.update(extra_evidence)
        findings.append(
            Finding(
                code=code,
                category="html",
                severity=severity,
                title=title,
                message=message.format(count=bucket.count),
                why_it_matters=why,
                recommendation=rec,
                evidence=evidence,
                affected_documents=_affected(bucket),
            )
        )

    add(
        "HTML_BOILERPLATE_DOMINATION",
        Severity.MEDIUM,
        "HTML boilerplate may dominate content",
        (
            "{count} HTML document(s) appear to have substantial "
            "nav/header/footer/aside text relative to main content."
        ),
        (
            "Navigation and chrome text can pollute chunks and retrieval with "
            "repeated site structure instead of article content."
        ),
        (
            "Extract main content more aggressively (readability/main region) before "
            "indexing. HTML structure varies—review flagged pages."
        ),
    )
    add(
        "HTML_LOW_MAIN_CONTENT_RATIO",
        Severity.MEDIUM,
        "Low HTML main-content ratio",
        ("{count} HTML document(s) yield little extracted text relative to raw file size."),
        (
            "Low content yield can indicate chrome-heavy pages, script-heavy shells, "
            "or weak main-content extraction."
        ),
        (
            "Inspect whether useful content is trapped in non-parsed structures; "
            "adjust extraction or exclude low-value pages."
        ),
        extra_evidence={"threshold": config.html_low_content_ratio},
    )
    add(
        "HTML_LARGE_DOCUMENT",
        Severity.INFO,
        "Very large HTML documents",
        (
            "{count} HTML document(s) exceed "
            f"{config.html_large_document_chars} extracted characters."
        ),
        (
            "Large documentation pages often need heading-aware chunking rather than "
            "naive fixed windows."
        ),
        "Chunk by headings/sections for these pages before embedding.",
        extra_evidence={"threshold_chars": config.html_large_document_chars},
    )
    add(
        "HTML_CODE_HEAVY",
        Severity.INFO,
        "Code-heavy technical HTML pages",
        "{count} HTML document(s) contain substantial <pre>/<code> content.",
        (
            "Code blocks are legitimate technical content. They should not be treated "
            "as symbol noise, but may need code-aware chunking."
        ),
        (
            "Keep code blocks; consider code-aware chunking and avoid generic "
            "symbol-noise filters on these pages."
        ),
    )

    # Cross-document repeated navigation from compact first-line fingerprints.
    if acc.html_doc_count >= 8:
        repeated = {
            k: v
            for k, v in acc.html_first_line_to_paths.items()
            if len(v) >= max(5, int(acc.html_doc_count * 0.15))
        }
        if repeated:
            affected_all = sorted({p for paths in repeated.values() for p in paths})
            retained = affected_all[: config.max_affected_documents]
            findings.append(
                Finding(
                    code="HTML_REPEATED_NAVIGATION",
                    category="html",
                    severity=Severity.LOW,
                    title="Repeated navigation-like text across HTML documents",
                    message=(
                        f"{len(affected_all)} HTML document(s) share repeated leading "
                        "lines across the corpus (possible shared navigation/chrome)."
                    ),
                    why_it_matters=(
                        "Shared nav text across pages increases duplicate-like retrieval noise."
                    ),
                    recommendation=(
                        "Strip site chrome consistently during HTML-to-text conversion."
                    ),
                    evidence={
                        "repeated_line_groups": len(repeated),
                        "sample_lines": [
                            {"line": k[:120], "count": len(v)}
                            for k, v in sorted(
                                repeated.items(),
                                key=lambda kv: (-len(kv[1]), kv[0]),
                            )[:5]
                        ],
                        "sample_paths": sample_paths(retained, config),
                        "affected_document_count": len(affected_all),
                        "affected_documents_truncated": len(affected_all) > len(retained),
                    },
                    affected_documents=tuple(retained),
                )
            )
    return findings
