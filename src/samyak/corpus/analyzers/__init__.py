"""Analyzer package — generic and format-specific checks."""

from samyak.corpus.analyzers.chunkability import analyze_chunkability
from samyak.corpus.analyzers.duplicates import analyze_exact_duplicates
from samyak.corpus.analyzers.empty import analyze_empty_documents
from samyak.corpus.analyzers.html import analyze_html_documents
from samyak.corpus.analyzers.pdf import analyze_pdf_documents
from samyak.corpus.analyzers.size import analyze_document_sizes
from samyak.corpus.analyzers.text_quality import analyze_text_quality

__all__ = [
    "analyze_chunkability",
    "analyze_document_sizes",
    "analyze_empty_documents",
    "analyze_exact_duplicates",
    "analyze_html_documents",
    "analyze_pdf_documents",
    "analyze_text_quality",
]
