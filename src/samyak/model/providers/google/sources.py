"""Captured Google Gemini API documentation sources. No network access.

Serving source is the Gemini API / AI Studio documentation on ai.google.dev.
Vertex AI, Google Cloud resource paths, and credentialed Models API endpoints
are out of scope.

Official Markdown is retrieved as ``.md.txt``. The HTML ``.md`` URL is not a
documentation source for this adapter.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import urlparse

from samyak.model.errors import CatalogValidationError
from samyak.model.facts import SourceKind, validate_timestamp
from samyak.model.providers.google.errors import GoogleParseError

PROVIDER_ID = "google"

SOURCE_ID_MODELS_INDEX = "google-docs-models-index"
SOURCE_ID_DEPRECATIONS = "google-docs-deprecations"
SOURCE_ID_GEMINI_3 = "google-docs-gemini-3"
SOURCE_ID_MODEL_PAGE_PREFIX = "google-docs-model-page"
# Not a captured document. Combined context is derived from established
# max_input_tokens and max_output_tokens using Google's token-docs rule.
# Provenance cites TOKENS_URL with content_hash=None; numeric evidence stays
# on those sibling facts.
SOURCE_ID_DERIVED_CONTEXT_WINDOW = "google-derived-context-window"

DOCS_HOST = "ai.google.dev"
DOCS_ORIGIN = f"https://{DOCS_HOST}"
DOCS_PATH_PREFIX = "/gemini-api/docs"
MODELS_INDEX_PATH = "/gemini-api/docs/models.md.txt"
DEPRECATIONS_PATH = "/gemini-api/docs/deprecations.md.txt"
GEMINI_3_PATH = "/gemini-api/docs/gemini-3.md.txt"
TOKENS_PATH = "/gemini-api/docs/tokens.md.txt"
MODELS_INDEX_URL = f"{DOCS_ORIGIN}{MODELS_INDEX_PATH}"
DEPRECATIONS_URL = f"{DOCS_ORIGIN}{DEPRECATIONS_PATH}"
GEMINI_3_URL = f"{DOCS_ORIGIN}{GEMINI_3_PATH}"
TOKENS_URL = f"{DOCS_ORIGIN}{TOKENS_PATH}"

_PAGE_IDENTITY = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_MODEL_PAGE_PATH = re.compile(r"^/gemini-api/docs/models/([^/]+?)(?:\.md(?:\.txt)?)?$")


class GoogleSourceType(StrEnum):
    """Which class of official Gemini API document a captured body is.

    Source *type* is not source *identity*. The models index, deprecations
    page, and Gemini 3 guide each have one stable source_id. Every model
    Markdown document has its own source_id, derived from that document's
    path slug.
    """

    MODELS_INDEX = "models_index"
    MODEL_PAGE = "model_page"
    DEPRECATIONS = "deprecations"
    GEMINI_3 = "gemini_3"


_SOURCE_KINDS = {
    GoogleSourceType.MODELS_INDEX: SourceKind.PROVIDER_DOCS,
    GoogleSourceType.MODEL_PAGE: SourceKind.PROVIDER_DOCS,
    GoogleSourceType.DEPRECATIONS: SourceKind.PROVIDER_DEPRECATIONS,
    GoogleSourceType.GEMINI_3: SourceKind.PROVIDER_DOCS,
}


def model_page_url(page_identity: str) -> str:
    return f"{DOCS_ORIGIN}/gemini-api/docs/models/{page_identity}.md.txt"


def model_page_source_id(page_identity: str) -> str:
    """Stable source_id for one official Gemini API model Markdown document.

    ``page_identity`` is the docs path slug, not a UUID and not a shared
    per-type token. It is not assumed to equal the API model id.
    """
    if not isinstance(page_identity, str) or not _PAGE_IDENTITY.fullmatch(page_identity):
        raise GoogleParseError("Google model page identity is not a documented page slug")
    return f"{SOURCE_ID_MODEL_PAGE_PREFIX}:{page_identity}"


def model_page_source_id_from_url(url: str) -> str:
    """Derive the model-page source_id from the canonical document URL."""
    if not isinstance(url, str) or not url.strip():
        raise GoogleParseError("Google model page URL is missing")
    parsed = urlparse(url.strip())
    match = _MODEL_PAGE_PATH.fullmatch(parsed.path or "")
    if match is None:
        raise GoogleParseError("Google model page URL is not a models document path")
    return model_page_source_id(match.group(1))


def content_hash_for_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def source_type_for_id(source_id: str) -> GoogleSourceType:
    if source_id == SOURCE_ID_MODELS_INDEX:
        return GoogleSourceType.MODELS_INDEX
    if source_id == SOURCE_ID_DEPRECATIONS:
        return GoogleSourceType.DEPRECATIONS
    if source_id == SOURCE_ID_GEMINI_3:
        return GoogleSourceType.GEMINI_3
    if _is_model_page_source_id(source_id):
        return GoogleSourceType.MODEL_PAGE
    raise GoogleParseError(f"Google source_id {source_id!r} is not a known documentation source")


def _is_model_page_source_id(source_id: str) -> bool:
    prefix = f"{SOURCE_ID_MODEL_PAGE_PREFIX}:"
    if not source_id.startswith(prefix):
        return False
    return _PAGE_IDENTITY.fullmatch(source_id[len(prefix) :]) is not None


@dataclass(frozen=True, slots=True)
class CapturedSource:
    """Official Gemini API documentation already captured. Bytes are not fetched here."""

    source_id: str
    source_url: str
    body: str
    retrieved_at: str
    media_type: str = "text/markdown"

    def __post_init__(self) -> None:
        source_type = source_type_for_id(self.source_id)
        if not isinstance(self.source_url, str) or not self.source_url.strip():
            raise GoogleParseError("Google source URL is missing")
        if source_type is GoogleSourceType.MODEL_PAGE:
            expected = model_page_source_id_from_url(self.source_url)
            if expected != self.source_id:
                raise GoogleParseError(
                    "Google model page source_id does not match the document URL"
                )
        if source_type is GoogleSourceType.GEMINI_3:
            parsed = urlparse(self.source_url.strip())
            if parsed.path != GEMINI_3_PATH:
                raise GoogleParseError("Google Gemini 3 source_id does not match the document URL")
        if not isinstance(self.body, str):
            raise GoogleParseError("Google documentation body must be text")
        try:
            validate_timestamp("retrieved_at", self.retrieved_at)
        except CatalogValidationError as exc:
            raise GoogleParseError(
                "Google source retrieved_at must be a timezone-aware timestamp"
            ) from exc
        if not isinstance(self.media_type, str) or not self.media_type.strip():
            raise GoogleParseError("Google source media type is missing")

    @property
    def source_type(self) -> GoogleSourceType:
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
class GoogleSourceDescriptor:
    """Official Gemini API document to retrieve. URLs live only in this module."""

    source_id: str
    source_type: GoogleSourceType
    url: str
    required: bool

    @property
    def path(self) -> str:
        return urlparse(self.url).path


def models_index_descriptor() -> GoogleSourceDescriptor:
    return GoogleSourceDescriptor(
        source_id=SOURCE_ID_MODELS_INDEX,
        source_type=GoogleSourceType.MODELS_INDEX,
        url=MODELS_INDEX_URL,
        required=True,
    )


def deprecations_descriptor() -> GoogleSourceDescriptor:
    return GoogleSourceDescriptor(
        source_id=SOURCE_ID_DEPRECATIONS,
        source_type=GoogleSourceType.DEPRECATIONS,
        url=DEPRECATIONS_URL,
        required=True,
    )


def gemini_3_descriptor() -> GoogleSourceDescriptor:
    return GoogleSourceDescriptor(
        source_id=SOURCE_ID_GEMINI_3,
        source_type=GoogleSourceType.GEMINI_3,
        url=GEMINI_3_URL,
        required=True,
    )


def required_source_descriptors() -> tuple[GoogleSourceDescriptor, ...]:
    return (models_index_descriptor(), deprecations_descriptor(), gemini_3_descriptor())


def model_page_descriptor(page_identity: str) -> GoogleSourceDescriptor:
    url = model_page_url(page_identity)
    return GoogleSourceDescriptor(
        source_id=model_page_source_id(page_identity),
        source_type=GoogleSourceType.MODEL_PAGE,
        url=url,
        required=False,
    )


def expected_path_for_source(descriptor: GoogleSourceDescriptor) -> str:
    if descriptor.source_type is GoogleSourceType.MODELS_INDEX:
        return MODELS_INDEX_PATH
    if descriptor.source_type is GoogleSourceType.DEPRECATIONS:
        return DEPRECATIONS_PATH
    if descriptor.source_type is GoogleSourceType.GEMINI_3:
        return GEMINI_3_PATH
    parsed = urlparse(descriptor.url)
    return parsed.path


def page_identity_from_model_url(url: str) -> str:
    parsed = urlparse(url.strip())
    match = _MODEL_PAGE_PATH.fullmatch(parsed.path or "")
    if match is None:
        raise GoogleParseError("Google model page URL is not a models document path")
    return match.group(1)
