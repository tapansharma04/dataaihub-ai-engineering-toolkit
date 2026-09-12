"""Offline Anthropic documentation adapter.

Converts captured official Anthropic Markdown into canonical catalog records.
Does not fetch the network.
"""

from samyak.model.providers.anthropic.adapter import catalog_from_anthropic_sources
from samyak.model.providers.anthropic.errors import (
    AnthropicAdapterError,
    AnthropicFetchError,
    AnthropicParseError,
)
from samyak.model.providers.anthropic.normalize import normalize_anthropic_observations
from samyak.model.providers.anthropic.parse import parse_anthropic_sources
from samyak.model.providers.anthropic.sources import (
    SOURCE_ID_DEPRECATIONS,
    SOURCE_ID_MODEL_PAGE_PREFIX,
    SOURCE_ID_MODELS_INDEX,
    CapturedSource,
    captured_markdown,
    model_page_source_id,
    model_page_source_id_from_url,
)

__all__ = [
    "AnthropicAdapterError",
    "AnthropicFetchError",
    "AnthropicParseError",
    "CapturedSource",
    "SOURCE_ID_DEPRECATIONS",
    "SOURCE_ID_MODEL_PAGE_PREFIX",
    "SOURCE_ID_MODELS_INDEX",
    "captured_markdown",
    "catalog_from_anthropic_sources",
    "model_page_source_id",
    "model_page_source_id_from_url",
    "normalize_anthropic_observations",
    "parse_anthropic_sources",
]
