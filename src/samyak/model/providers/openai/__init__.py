"""Offline OpenAI documentation adapter.

Converts captured official OpenAI Markdown into canonical catalog records.
Does not fetch the network.
"""

from samyak.model.providers.openai.adapter import catalog_from_openai_sources
from samyak.model.providers.openai.errors import (
    OpenAIAdapterError,
    OpenAIFetchError,
    OpenAIParseError,
)
from samyak.model.providers.openai.normalize import normalize_openai_observations
from samyak.model.providers.openai.parse import parse_openai_sources
from samyak.model.providers.openai.sources import (
    SOURCE_ID_DEPRECATIONS,
    SOURCE_ID_MODEL_PAGE_PREFIX,
    SOURCE_ID_MODELS_INDEX,
    CapturedSource,
    captured_markdown,
    model_page_source_id,
    model_page_source_id_from_url,
)

__all__ = [
    "CapturedSource",
    "OpenAIAdapterError",
    "OpenAIFetchError",
    "OpenAIParseError",
    "SOURCE_ID_DEPRECATIONS",
    "SOURCE_ID_MODEL_PAGE_PREFIX",
    "SOURCE_ID_MODELS_INDEX",
    "captured_markdown",
    "catalog_from_openai_sources",
    "model_page_source_id",
    "model_page_source_id_from_url",
    "normalize_openai_observations",
    "parse_openai_sources",
]
