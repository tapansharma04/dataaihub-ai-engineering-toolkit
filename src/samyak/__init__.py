"""Samyak — AI engineering tooling for building reliable AI applications.

Public API: ``analyze_corpus``, ``AnalysisReport``, ``Finding``, ``Severity``,
``AnalysisConfig``, ``CorpusPathError``, and ``__version__``.
"""

from samyak.__version__ import __version__
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
    "__version__",
]
