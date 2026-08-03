"""RAG Data Readiness Analyzer — local corpus analysis before RAG indexing."""

from rag_readiness.__version__ import __version__
from rag_readiness.models import AnalysisReport, Document, Finding, Severity
from rag_readiness.pipeline import analyze_corpus

__all__ = [
    "AnalysisReport",
    "Document",
    "Finding",
    "Severity",
    "analyze_corpus",
    "__version__",
]
