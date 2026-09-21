"""Captured Together documentation sources. No network access.

Serving source is Together AI documentation on docs.together.ai.
Scope is the public Serverless Models catalogue plus the official
per-model serving quickstarts listed below. Dedicated endpoints,
Dedicated Container Inference, customer uploads, the www.together.ai
model library, and the authenticated Together Models API are out of
scope. The public library mixes origin-model marketing specs with
Together serving IDs; token facts come from docs.together.ai.

Official Markdown is retrieved as ``.md``.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import urlparse

from samyak.model.errors import CatalogValidationError
from samyak.model.facts import SourceKind, validate_timestamp
from samyak.model.providers.together.errors import TogetherParseError

PROVIDER_ID = "together"

SOURCE_ID_SERVERLESS = "together-docs-serverless-models"
SOURCE_ID_DEPRECATIONS = "together-docs-deprecations"
SOURCE_ID_CHANGELOG = "together-docs-changelog"
SOURCE_ID_MODEL_PAGE_PREFIX = "together-docs-model-page"

DOCS_HOST = "docs.together.ai"
DOCS_ORIGIN = f"https://{DOCS_HOST}"
DOCS_PATH_PREFIX = "/docs"

SERVERLESS_PATH = "/docs/serverless/models.md"
DEPRECATIONS_PATH = "/docs/deprecations.md"
CHANGELOG_PATH = "/docs/changelog.md"

SERVERLESS_URL = f"{DOCS_ORIGIN}{SERVERLESS_PATH}"
DEPRECATIONS_URL = f"{DOCS_ORIGIN}{DEPRECATIONS_PATH}"
CHANGELOG_URL = f"{DOCS_ORIGIN}{CHANGELOG_PATH}"

# Allowlisted docs.together.ai serving-model quickstarts. Image/video
# quickstarts and www.together.ai/models pages are not in this set.
MODEL_PAGE_SLUGS = (
    "glm-5.2-quickstart",
    "kimi-k3-quickstart",
    "deepseek-v4-quickstart",
    "kimi-k2.6-quickstart",
    "gpt-oss",
)
_MODEL_PAGE_SLUGS = frozenset(MODEL_PAGE_SLUGS)
_PAGE_IDENTITY = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_MODEL_PAGE_PATH = re.compile(r"^/docs/([^/]+?)(?:\.md)?$")


class TogetherSourceType(StrEnum):
    """Which class of official Together document a captured body is."""

    SERVERLESS = "serverless"
    DEPRECATIONS = "deprecations"
    CHANGELOG = "changelog"
    MODEL_PAGE = "model_page"


_SOURCE_KINDS = {
    TogetherSourceType.SERVERLESS: SourceKind.PROVIDER_DOCS,
    TogetherSourceType.DEPRECATIONS: SourceKind.PROVIDER_DEPRECATIONS,
    TogetherSourceType.CHANGELOG: SourceKind.PROVIDER_CHANGELOG,
    TogetherSourceType.MODEL_PAGE: SourceKind.PROVIDER_DOCS,
}

_FIXED_SOURCE_TYPES = {
    SOURCE_ID_SERVERLESS: TogetherSourceType.SERVERLESS,
    SOURCE_ID_DEPRECATIONS: TogetherSourceType.DEPRECATIONS,
    SOURCE_ID_CHANGELOG: TogetherSourceType.CHANGELOG,
}


def model_page_url(page_identity: str) -> str:
    return f"{DOCS_ORIGIN}/docs/{page_identity}.md"


def model_page_source_id(page_identity: str) -> str:
    """Stable source_id for one allowlisted Together serving-model document."""
    if not isinstance(page_identity, str) or page_identity not in _MODEL_PAGE_SLUGS:
        raise TogetherParseError("Together model page identity is not an allowlisted docs slug")
    return f"{SOURCE_ID_MODEL_PAGE_PREFIX}:{page_identity}"


def model_page_source_id_from_url(url: str) -> str:
    """Derive the model-page source_id from the canonical document URL."""
    if not isinstance(url, str) or not url.strip():
        raise TogetherParseError("Together model page URL is missing")
    parsed = urlparse(url.strip())
    match = _MODEL_PAGE_PATH.fullmatch(parsed.path or "")
    if match is None or match.group(1) not in _MODEL_PAGE_SLUGS:
        raise TogetherParseError("Together model page URL is not an allowlisted docs path")
    return model_page_source_id(match.group(1))


def content_hash_for_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def source_type_for_id(source_id: str) -> TogetherSourceType:
    found = _FIXED_SOURCE_TYPES.get(source_id)
    if found is not None:
        return found
    if _is_model_page_source_id(source_id):
        return TogetherSourceType.MODEL_PAGE
    raise TogetherParseError(
        f"Together source_id {source_id!r} is not a known documentation source"
    )


def _is_model_page_source_id(source_id: str) -> bool:
    prefix = f"{SOURCE_ID_MODEL_PAGE_PREFIX}:"
    if not source_id.startswith(prefix):
        return False
    slug = source_id[len(prefix) :]
    return slug in _MODEL_PAGE_SLUGS and _PAGE_IDENTITY.fullmatch(slug) is not None


@dataclass(frozen=True, slots=True)
class CapturedSource:
    """Official Together documentation already captured. Bytes are not fetched here."""

    source_id: str
    source_url: str
    body: str
    retrieved_at: str
    media_type: str = "text/markdown"

    def __post_init__(self) -> None:
        source_type = source_type_for_id(self.source_id)
        if not isinstance(self.source_url, str) or not self.source_url.strip():
            raise TogetherParseError("Together source URL is missing")
        if source_type is TogetherSourceType.MODEL_PAGE:
            expected = model_page_source_id_from_url(self.source_url)
            if expected != self.source_id:
                raise TogetherParseError(
                    "Together model page source_id does not match the document URL"
                )
        if not isinstance(self.body, str):
            raise TogetherParseError("Together documentation body must be text")
        try:
            validate_timestamp("retrieved_at", self.retrieved_at)
        except CatalogValidationError as exc:
            raise TogetherParseError(
                "Together source retrieved_at must be a timezone-aware timestamp"
            ) from exc
        if not isinstance(self.media_type, str) or not self.media_type.strip():
            raise TogetherParseError("Together source media type is missing")

    @property
    def source_type(self) -> TogetherSourceType:
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
class TogetherSourceDescriptor:
    """Official Together document to retrieve. URLs live only in this module."""

    source_id: str
    source_type: TogetherSourceType
    url: str
    required: bool

    @property
    def path(self) -> str:
        return urlparse(self.url).path


def serverless_descriptor() -> TogetherSourceDescriptor:
    return TogetherSourceDescriptor(
        source_id=SOURCE_ID_SERVERLESS,
        source_type=TogetherSourceType.SERVERLESS,
        url=SERVERLESS_URL,
        required=True,
    )


def deprecations_descriptor() -> TogetherSourceDescriptor:
    return TogetherSourceDescriptor(
        source_id=SOURCE_ID_DEPRECATIONS,
        source_type=TogetherSourceType.DEPRECATIONS,
        url=DEPRECATIONS_URL,
        required=True,
    )


def changelog_descriptor() -> TogetherSourceDescriptor:
    return TogetherSourceDescriptor(
        source_id=SOURCE_ID_CHANGELOG,
        source_type=TogetherSourceType.CHANGELOG,
        url=CHANGELOG_URL,
        required=False,
    )


def required_source_descriptors() -> tuple[TogetherSourceDescriptor, ...]:
    return (serverless_descriptor(), deprecations_descriptor())


def optional_source_descriptors() -> tuple[TogetherSourceDescriptor, ...]:
    return (changelog_descriptor(),) + tuple(
        model_page_descriptor(slug) for slug in MODEL_PAGE_SLUGS
    )


def model_page_descriptor(page_identity: str) -> TogetherSourceDescriptor:
    url = model_page_url(page_identity)
    return TogetherSourceDescriptor(
        source_id=model_page_source_id(page_identity),
        source_type=TogetherSourceType.MODEL_PAGE,
        url=url,
        required=False,
    )


def expected_path_for_source(descriptor: TogetherSourceDescriptor) -> str:
    return urlparse(descriptor.url).path
