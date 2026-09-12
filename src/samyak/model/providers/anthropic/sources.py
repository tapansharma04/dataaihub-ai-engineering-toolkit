"""Captured Anthropic documentation sources. No network access."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import urlparse

from samyak.model.errors import CatalogValidationError
from samyak.model.facts import SourceKind, validate_timestamp
from samyak.model.providers.anthropic.errors import AnthropicParseError

PROVIDER_ID = "anthropic"

SOURCE_ID_MODELS_INDEX = "anthropic-docs-models-index"
SOURCE_ID_DEPRECATIONS = "anthropic-docs-deprecations"
SOURCE_ID_MODEL_PAGE_PREFIX = "anthropic-docs-model-page"

DOCS_HOST = "platform.claude.com"
DOCS_ORIGIN = f"https://{DOCS_HOST}"
DOCS_PATH_PREFIXES = ("/docs/en/models", "/docs/en/about-claude")
MODELS_INDEX_PATH = "/docs/en/models/overview.md"
DEPRECATIONS_PATH = "/docs/en/about-claude/model-deprecations.md"
MODELS_INDEX_URL = f"{DOCS_ORIGIN}{MODELS_INDEX_PATH}"
DEPRECATIONS_URL = f"{DOCS_ORIGIN}{DEPRECATIONS_PATH}"

_PAGE_IDENTITY = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_MODEL_PAGE_PATH = re.compile(r"^/docs/en/models/([^/]+)/overview\.md$")


class AnthropicSourceType(StrEnum):
    """Which class of official Anthropic document a captured body is.

    Source *type* is not source *identity*. The models overview and
    deprecations page each have one stable source_id. Every model Markdown
    document has its own source_id, derived from that document's path slug.
    """

    MODELS_INDEX = "models_index"
    MODEL_PAGE = "model_page"
    DEPRECATIONS = "deprecations"


_SOURCE_KINDS = {
    AnthropicSourceType.MODELS_INDEX: SourceKind.PROVIDER_DOCS,
    AnthropicSourceType.MODEL_PAGE: SourceKind.PROVIDER_DOCS,
    AnthropicSourceType.DEPRECATIONS: SourceKind.PROVIDER_DEPRECATIONS,
}


def model_page_url(page_identity: str) -> str:
    return f"{DOCS_ORIGIN}/docs/en/models/{page_identity}/overview.md"


def model_page_source_id(page_identity: str) -> str:
    """Stable source_id for one official model Markdown document.

    ``page_identity`` is the docs path slug, not a UUID and not a shared
    per-type token.
    """
    if not isinstance(page_identity, str) or not _PAGE_IDENTITY.fullmatch(page_identity):
        raise AnthropicParseError("Anthropic model page identity is not a documented page slug")
    return f"{SOURCE_ID_MODEL_PAGE_PREFIX}:{page_identity}"


def model_page_source_id_from_url(url: str) -> str:
    """Derive the model-page source_id from the canonical document URL."""
    if not isinstance(url, str) or not url.strip():
        raise AnthropicParseError("Anthropic model page URL is missing")
    parsed = urlparse(url.strip())
    match = _MODEL_PAGE_PATH.fullmatch(parsed.path or "")
    if match is None:
        raise AnthropicParseError("Anthropic model page URL is not a models document path")
    return model_page_source_id(match.group(1))


def content_hash_for_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def source_type_for_id(source_id: str) -> AnthropicSourceType:
    if source_id == SOURCE_ID_MODELS_INDEX:
        return AnthropicSourceType.MODELS_INDEX
    if source_id == SOURCE_ID_DEPRECATIONS:
        return AnthropicSourceType.DEPRECATIONS
    if _is_model_page_source_id(source_id):
        return AnthropicSourceType.MODEL_PAGE
    raise AnthropicParseError(
        f"Anthropic source_id {source_id!r} is not a known documentation source"
    )


def _is_model_page_source_id(source_id: str) -> bool:
    prefix = f"{SOURCE_ID_MODEL_PAGE_PREFIX}:"
    if not source_id.startswith(prefix):
        return False
    return _PAGE_IDENTITY.fullmatch(source_id[len(prefix) :]) is not None


@dataclass(frozen=True, slots=True)
class CapturedSource:
    """Official Anthropic documentation already captured. Bytes are not fetched here."""

    source_id: str
    source_url: str
    body: str
    retrieved_at: str
    media_type: str = "text/markdown"

    def __post_init__(self) -> None:
        source_type = source_type_for_id(self.source_id)
        if not isinstance(self.source_url, str) or not self.source_url.strip():
            raise AnthropicParseError("Anthropic source URL is missing")
        if source_type is AnthropicSourceType.MODEL_PAGE:
            expected = model_page_source_id_from_url(self.source_url)
            if expected != self.source_id:
                raise AnthropicParseError(
                    "Anthropic model page source_id does not match the document URL"
                )
        if not isinstance(self.body, str):
            raise AnthropicParseError("Anthropic documentation body must be text")
        try:
            validate_timestamp("retrieved_at", self.retrieved_at)
        except CatalogValidationError as exc:
            raise AnthropicParseError(
                "Anthropic source retrieved_at must be a timezone-aware timestamp"
            ) from exc
        if not isinstance(self.media_type, str) or not self.media_type.strip():
            raise AnthropicParseError("Anthropic source media type is missing")

    @property
    def source_type(self) -> AnthropicSourceType:
        return source_type_for_id(self.source_id)

    @property
    def source_kind(self) -> SourceKind:
        return _SOURCE_KINDS[self.source_type]

    @property
    def body_bytes(self) -> bytes:
        return self.body.encode("utf-8")

    @property
    def content_hash(self) -> str:
        return content_hash_for_bytes(self.body_bytes)


def captured_markdown(
    *,
    source_id: str,
    source_url: str,
    body: str,
    retrieved_at: str,
) -> CapturedSource:
    """Build a captured Markdown source with a stable source_id."""
    return CapturedSource(
        source_id=source_id,
        source_url=source_url,
        body=body,
        retrieved_at=retrieved_at,
        media_type="text/markdown",
    )


@dataclass(frozen=True, slots=True)
class AnthropicSourceDescriptor:
    """Official Anthropic document to retrieve. URLs live only in this module."""

    source_id: str
    source_type: AnthropicSourceType
    url: str
    required: bool

    @property
    def path(self) -> str:
        return urlparse(self.url).path


def models_index_descriptor() -> AnthropicSourceDescriptor:
    return AnthropicSourceDescriptor(
        source_id=SOURCE_ID_MODELS_INDEX,
        source_type=AnthropicSourceType.MODELS_INDEX,
        url=MODELS_INDEX_URL,
        required=True,
    )


def deprecations_descriptor() -> AnthropicSourceDescriptor:
    return AnthropicSourceDescriptor(
        source_id=SOURCE_ID_DEPRECATIONS,
        source_type=AnthropicSourceType.DEPRECATIONS,
        url=DEPRECATIONS_URL,
        required=True,
    )


def required_source_descriptors() -> tuple[AnthropicSourceDescriptor, ...]:
    return (models_index_descriptor(), deprecations_descriptor())


def model_page_descriptor(page_identity: str) -> AnthropicSourceDescriptor:
    url = model_page_url(page_identity)
    return AnthropicSourceDescriptor(
        source_id=model_page_source_id(page_identity),
        source_type=AnthropicSourceType.MODEL_PAGE,
        url=url,
        required=False,
    )


def expected_path_for_source(descriptor: AnthropicSourceDescriptor) -> str:
    if descriptor.source_type is AnthropicSourceType.MODELS_INDEX:
        return MODELS_INDEX_PATH
    if descriptor.source_type is AnthropicSourceType.DEPRECATIONS:
        return DEPRECATIONS_PATH
    parsed = urlparse(descriptor.url)
    return parsed.path


def page_identity_from_model_url(url: str) -> str:
    parsed = urlparse(url.strip())
    match = _MODEL_PAGE_PATH.fullmatch(parsed.path or "")
    if match is None:
        raise AnthropicParseError("Anthropic model page URL is not a models document path")
    return match.group(1)
