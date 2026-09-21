"""Offline Google Gemini API documentation adapter.

Converts captured official Gemini API Markdown into canonical catalog records.
Does not fetch the network. Vertex AI is out of scope.
"""

from samyak.model.providers.google.adapter import catalog_from_google_sources
from samyak.model.providers.google.errors import (
    GoogleAdapterError,
    GoogleFetchError,
    GoogleParseError,
)
from samyak.model.providers.google.normalize import normalize_google_observations
from samyak.model.providers.google.parse import parse_google_sources
from samyak.model.providers.google.sources import (
    SOURCE_ID_DEPRECATIONS,
    SOURCE_ID_DERIVED_CONTEXT_WINDOW,
    SOURCE_ID_GEMINI_3,
    SOURCE_ID_MODEL_PAGE_PREFIX,
    SOURCE_ID_MODELS_INDEX,
    CapturedSource,
    captured_markdown,
    model_page_source_id,
    model_page_source_id_from_url,
)

__all__ = [
    "CapturedSource",
    "GoogleAdapterError",
    "GoogleFetchError",
    "GoogleParseError",
    "SOURCE_ID_DEPRECATIONS",
    "SOURCE_ID_DERIVED_CONTEXT_WINDOW",
    "SOURCE_ID_GEMINI_3",
    "SOURCE_ID_MODEL_PAGE_PREFIX",
    "SOURCE_ID_MODELS_INDEX",
    "captured_markdown",
    "catalog_from_google_sources",
    "model_page_source_id",
    "model_page_source_id_from_url",
    "normalize_google_observations",
    "parse_google_sources",
]
