"""Corpus Intelligence — analyze document collections before RAG/indexing use."""

from samyak.corpus.config import AnalysisConfig
from samyak.corpus.discovery import CorpusPathError
from samyak.corpus.models import AnalysisReport, Finding, Severity
from samyak.corpus.pipeline import analyze_corpus

__all__ = [
    "AnalysisConfig",
    "AnalysisReport",
    "CorpusPathError",
    "Finding",
    "Severity",
    "analyze_corpus",
]
