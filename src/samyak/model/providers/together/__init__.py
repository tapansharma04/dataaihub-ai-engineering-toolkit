"""Offline Together AI documentation adapter.

Converts captured official Together Markdown into canonical catalog records.
Does not fetch the network. Scope is the public Serverless Models catalogue
plus allowlisted docs.together.ai serving-model quickstarts. Dedicated
endpoints, DCI, customer uploads, and www.together.ai/models are out of scope.
"""

from samyak.model.providers.together.adapter import catalog_from_together_sources
from samyak.model.providers.together.errors import (
    TogetherAdapterError,
    TogetherFetchError,
    TogetherParseError,
)
from samyak.model.providers.together.normalize import normalize_together_observations
from samyak.model.providers.together.parse import parse_together_sources
from samyak.model.providers.together.sources import (
    SOURCE_ID_CHANGELOG,
    SOURCE_ID_DEPRECATIONS,
    SOURCE_ID_MODEL_PAGE_PREFIX,
    SOURCE_ID_SERVERLESS,
    CapturedSource,
    captured_markdown,
    model_page_source_id,
    model_page_source_id_from_url,
)

__all__ = [
    "CapturedSource",
    "SOURCE_ID_CHANGELOG",
    "SOURCE_ID_DEPRECATIONS",
    "SOURCE_ID_MODEL_PAGE_PREFIX",
    "SOURCE_ID_SERVERLESS",
    "TogetherAdapterError",
    "TogetherFetchError",
    "TogetherParseError",
    "captured_markdown",
    "catalog_from_together_sources",
    "model_page_source_id",
    "model_page_source_id_from_url",
    "normalize_together_observations",
    "parse_together_sources",
]
