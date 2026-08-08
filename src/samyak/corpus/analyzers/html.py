"""HTML-specific readiness checks (local documents only)."""

from __future__ import annotations

from samyak.corpus.analyzers._helpers import sample_paths
from samyak.corpus.config import AnalysisConfig
from samyak.corpus.models import Document, Finding, Severity


def analyze_html_documents(
    documents: list[Document],
    config: AnalysisConfig,
) -> list[Finding]:
    html_docs = [
        d
        for d in documents
        if d.metadata.get("format") == "html" or d.extension in {".html", ".htm"}
    ]
    if not html_docs:
        return []

    findings: list[Finding] = []
    findings.extend(_boilerplate_domination(html_docs, config))
    findings.extend(_low_main_content(html_docs, config))
    findings.extend(_large_html(html_docs, config))
    findings.extend(_code_heavy_info(html_docs, config))
    findings.extend(_cross_doc_repeated_boilerplate(html_docs, config))
    return findings


def _boilerplate_domination(docs: list[Document], config: AnalysisConfig) -> list[Finding]:
    affected: list[str] = []
    for doc in docs:
        content = max(int(doc.metadata.get("content_chars") or doc.char_count), 1)
        boilerplate = int(doc.metadata.get("boilerplate_chars") or 0)
        share = boilerplate / (content + boilerplate)
        if share >= config.html_boilerplate_token_ratio and boilerplate > 200:
            affected.append(doc.path)
    if not affected:
        return []
    return [
        Finding(
            code="HTML_BOILERPLATE_DOMINATION",
            category="html",
            severity=Severity.MEDIUM,
            title="HTML boilerplate may dominate content",
            message=(
                f"{len(affected)} HTML document(s) appear to have substantial "
                "nav/header/footer/aside text relative to main content."
            ),
            why_it_matters=(
                "Navigation and chrome text can pollute chunks and retrieval with "
                "repeated site structure instead of article content."
            ),
            recommendation=(
                "Extract main content more aggressively (readability/main region) before "
                "indexing. HTML structure varies—review flagged pages."
            ),
            evidence={
                "count": len(affected),
                "sample_paths": sample_paths(affected, config),
            },
            affected_documents=tuple(affected),
        )
    ]


def _low_main_content(docs: list[Document], config: AnalysisConfig) -> list[Finding]:
    affected: list[str] = []
    for doc in docs:
        if doc.size_bytes < config.html_low_content_min_raw_bytes:
            continue
        ratio = float(doc.metadata.get("raw_to_content_ratio") or 0.0)
        if ratio < config.html_low_content_ratio:
            affected.append(doc.path)
    if not affected:
        return []
    return [
        Finding(
            code="HTML_LOW_MAIN_CONTENT_RATIO",
            category="html",
            severity=Severity.MEDIUM,
            title="Low HTML main-content ratio",
            message=(
                f"{len(affected)} HTML document(s) yield little extracted text relative "
                "to raw file size."
            ),
            why_it_matters=(
                "Low content yield can indicate chrome-heavy pages, script-heavy shells, "
                "or weak main-content extraction."
            ),
            recommendation=(
                "Inspect whether useful content is trapped in non-parsed structures; "
                "adjust extraction or exclude low-value pages."
            ),
            evidence={
                "count": len(affected),
                "threshold": config.html_low_content_ratio,
                "sample_paths": sample_paths(affected, config),
            },
            affected_documents=tuple(affected),
        )
    ]


def _large_html(docs: list[Document], config: AnalysisConfig) -> list[Finding]:
    affected = [d.path for d in docs if d.char_count >= config.html_large_document_chars]
    if not affected:
        return []
    return [
        Finding(
            code="HTML_LARGE_DOCUMENT",
            category="html",
            severity=Severity.INFO,
            title="Very large HTML documents",
            message=(
                f"{len(affected)} HTML document(s) exceed "
                f"{config.html_large_document_chars} extracted characters."
            ),
            why_it_matters=(
                "Large documentation pages often need heading-aware chunking rather than "
                "naive fixed windows."
            ),
            recommendation=("Chunk by headings/sections for these pages before embedding."),
            evidence={
                "count": len(affected),
                "threshold_chars": config.html_large_document_chars,
                "sample_paths": sample_paths(affected, config),
            },
            affected_documents=tuple(affected),
        )
    ]


def _code_heavy_info(docs: list[Document], config: AnalysisConfig) -> list[Finding]:
    affected = [d.path for d in docs if d.metadata.get("code_heavy")]
    if not affected:
        return []
    return [
        Finding(
            code="HTML_CODE_HEAVY",
            category="html",
            severity=Severity.INFO,
            title="Code-heavy technical HTML pages",
            message=(f"{len(affected)} HTML document(s) contain substantial <pre>/<code> content."),
            why_it_matters=(
                "Code blocks are legitimate technical content. They should not be treated "
                "as symbol noise, but may need code-aware chunking."
            ),
            recommendation=(
                "Keep code blocks; consider code-aware chunking and avoid generic "
                "symbol-noise filters on these pages."
            ),
            evidence={
                "count": len(affected),
                "sample_paths": sample_paths(affected, config),
            },
            affected_documents=tuple(affected),
        )
    ]


def _cross_doc_repeated_boilerplate(
    docs: list[Document],
    config: AnalysisConfig,
) -> list[Finding]:
    """Detect identical leading navigation-like first lines across many HTML docs."""
    if len(docs) < 8:
        return []
    first_lines: dict[str, list[str]] = {}
    for doc in docs:
        lines = [ln.strip() for ln in doc.text.split("\n") if ln.strip()]
        if not lines:
            continue
        key = lines[0][:160]
        if len(key) < 12:
            continue
        first_lines.setdefault(key, []).append(doc.path)
    repeated = {k: v for k, v in first_lines.items() if len(v) >= max(5, int(len(docs) * 0.15))}
    if not repeated:
        return []
    affected = sorted({p for paths in repeated.values() for p in paths})
    return [
        Finding(
            code="HTML_REPEATED_NAVIGATION",
            category="html",
            severity=Severity.LOW,
            title="Repeated navigation-like text across HTML documents",
            message=(
                f"{len(affected)} HTML document(s) share repeated leading lines across "
                "the corpus (possible shared navigation/chrome)."
            ),
            why_it_matters=(
                "Shared nav text across pages increases duplicate-like retrieval noise."
            ),
            recommendation=("Strip site chrome consistently during HTML-to-text conversion."),
            evidence={
                "repeated_line_groups": len(repeated),
                "sample_lines": [
                    {"line": k[:120], "count": len(v)} for k, v in list(repeated.items())[:5]
                ],
                "sample_paths": sample_paths(affected, config),
            },
            affected_documents=tuple(affected),
        )
    ]
