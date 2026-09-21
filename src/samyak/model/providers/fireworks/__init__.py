"""Offline Fireworks AI documentation adapter.

Converts captured official Fireworks Markdown into canonical catalog records.
Does not fetch the network. Coverage is a documented public serving-ID subset,
not the complete Fireworks Model Library, Serverless inventory, or
fireworks.ai HTML catalogue.
"""

from samyak.model.providers.fireworks.adapter import catalog_from_fireworks_sources
from samyak.model.providers.fireworks.errors import (
    FireworksAdapterError,
    FireworksFetchError,
    FireworksParseError,
)
from samyak.model.providers.fireworks.normalize import normalize_fireworks_observations
from samyak.model.providers.fireworks.parse import parse_fireworks_sources
from samyak.model.providers.fireworks.sources import (
    SOURCE_ID_CHANGELOG,
    SOURCE_ID_EMBEDDINGS,
    SOURCE_ID_MODEL_PAGE_PREFIX,
    SOURCE_ID_SERVING_PATHS,
    SOURCE_ID_TEXT_MODELS,
    SOURCE_ID_TOOL_CALLING,
    SOURCE_ID_VISION_MODELS,
    CapturedSource,
    captured_markdown,
    model_page_source_id,
    model_page_source_id_from_url,
)

__all__ = [
    "CapturedSource",
    "FireworksAdapterError",
    "FireworksFetchError",
    "FireworksParseError",
    "SOURCE_ID_CHANGELOG",
    "SOURCE_ID_EMBEDDINGS",
    "SOURCE_ID_MODEL_PAGE_PREFIX",
    "SOURCE_ID_SERVING_PATHS",
    "SOURCE_ID_TEXT_MODELS",
    "SOURCE_ID_TOOL_CALLING",
    "SOURCE_ID_VISION_MODELS",
    "captured_markdown",
    "catalog_from_fireworks_sources",
    "model_page_source_id",
    "model_page_source_id_from_url",
    "normalize_fireworks_observations",
    "parse_fireworks_sources",
]
