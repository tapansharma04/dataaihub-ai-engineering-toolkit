"""Captured Fireworks documentation sources. No network access.

Serving source is Fireworks AI documentation on docs.fireworks.ai.
Coverage is a documented public serving-ID subset from official Markdown,
not a complete Fireworks catalogue, Serverless inventory, or Model Library.

The public HTML library at fireworks.ai/models, the credentialed List
Models API, app.fireworks.ai, and customer deployment paths are out of
scope. Official Markdown is retrieved as ``.md``.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import urlparse

from samyak.model.errors import CatalogValidationError
from samyak.model.facts import SourceKind, validate_timestamp
from samyak.model.providers.fireworks.errors import FireworksParseError

PROVIDER_ID = "fireworks"

SOURCE_ID_EMBEDDINGS = "fireworks-docs-embeddings"
SOURCE_ID_SERVING_PATHS = "fireworks-docs-serving-paths"
SOURCE_ID_TEXT_MODELS = "fireworks-docs-text-models"
SOURCE_ID_VISION_MODELS = "fireworks-docs-vision-models"
SOURCE_ID_TOOL_CALLING = "fireworks-docs-tool-calling"
SOURCE_ID_CHANGELOG = "fireworks-docs-changelog"
SOURCE_ID_MODEL_PAGE_PREFIX = "fireworks-docs-model-page"

DOCS_HOST = "docs.fireworks.ai"
DOCS_ORIGIN = f"https://{DOCS_HOST}"
DOCS_PATH_PREFIXES = frozenset({"/guides", "/serverless", "/models", "/updates"})

EMBEDDINGS_PATH = "/guides/querying-embeddings-models.md"
SERVING_PATHS_PATH = "/serverless/serverless-modes.md"
TEXT_MODELS_PATH = "/guides/querying-text-models.md"
VISION_MODELS_PATH = "/guides/querying-vision-language-models.md"
TOOL_CALLING_PATH = "/guides/function-calling.md"
CHANGELOG_PATH = "/updates/changelog.md"
KIMI_K2_PATH = "/models/kimi-k2.md"

EMBEDDINGS_URL = f"{DOCS_ORIGIN}{EMBEDDINGS_PATH}"
SERVING_PATHS_URL = f"{DOCS_ORIGIN}{SERVING_PATHS_PATH}"
TEXT_MODELS_URL = f"{DOCS_ORIGIN}{TEXT_MODELS_PATH}"
VISION_MODELS_URL = f"{DOCS_ORIGIN}{VISION_MODELS_PATH}"
TOOL_CALLING_URL = f"{DOCS_ORIGIN}{TOOL_CALLING_PATH}"
CHANGELOG_URL = f"{DOCS_ORIGIN}{CHANGELOG_PATH}"
KIMI_K2_URL = f"{DOCS_ORIGIN}{KIMI_K2_PATH}"

_PAGE_IDENTITY = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_MODEL_PAGE_PATH = re.compile(r"^/models/([^/]+?)(?:\.md)?$")


class FireworksSourceType(StrEnum):
    """Which class of official Fireworks document a captured body is.

    Source *type* is not source *identity*. Each document has a stable
    source_id. Model Markdown documents use a path-slug source_id.
    """

    EMBEDDINGS = "embeddings"
    SERVING_PATHS = "serving_paths"
    TEXT_MODELS = "text_models"
    VISION_MODELS = "vision_models"
    TOOL_CALLING = "tool_calling"
    CHANGELOG = "changelog"
    MODEL_PAGE = "model_page"


_SOURCE_KINDS = {
    FireworksSourceType.EMBEDDINGS: SourceKind.PROVIDER_DOCS,
    FireworksSourceType.SERVING_PATHS: SourceKind.PROVIDER_DOCS,
    FireworksSourceType.TEXT_MODELS: SourceKind.PROVIDER_DOCS,
    FireworksSourceType.VISION_MODELS: SourceKind.PROVIDER_DOCS,
    FireworksSourceType.TOOL_CALLING: SourceKind.PROVIDER_DOCS,
    FireworksSourceType.CHANGELOG: SourceKind.PROVIDER_CHANGELOG,
    FireworksSourceType.MODEL_PAGE: SourceKind.PROVIDER_DOCS,
}

_FIXED_SOURCE_TYPES = {
    SOURCE_ID_EMBEDDINGS: FireworksSourceType.EMBEDDINGS,
    SOURCE_ID_SERVING_PATHS: FireworksSourceType.SERVING_PATHS,
    SOURCE_ID_TEXT_MODELS: FireworksSourceType.TEXT_MODELS,
    SOURCE_ID_VISION_MODELS: FireworksSourceType.VISION_MODELS,
    SOURCE_ID_TOOL_CALLING: FireworksSourceType.TOOL_CALLING,
    SOURCE_ID_CHANGELOG: FireworksSourceType.CHANGELOG,
}


def model_page_url(page_identity: str) -> str:
    return f"{DOCS_ORIGIN}/models/{page_identity}.md"


def model_page_source_id(page_identity: str) -> str:
    """Stable source_id for one official Fireworks model Markdown document."""
    if not isinstance(page_identity, str) or not _PAGE_IDENTITY.fullmatch(page_identity):
        raise FireworksParseError("Fireworks model page identity is not a documented page slug")
    return f"{SOURCE_ID_MODEL_PAGE_PREFIX}:{page_identity}"


def model_page_source_id_from_url(url: str) -> str:
    """Derive the model-page source_id from the canonical document URL."""
    if not isinstance(url, str) or not url.strip():
        raise FireworksParseError("Fireworks model page URL is missing")
    parsed = urlparse(url.strip())
    match = _MODEL_PAGE_PATH.fullmatch(parsed.path or "")
    if match is None:
        raise FireworksParseError("Fireworks model page URL is not a models document path")
    return model_page_source_id(match.group(1))


def content_hash_for_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def source_type_for_id(source_id: str) -> FireworksSourceType:
    found = _FIXED_SOURCE_TYPES.get(source_id)
    if found is not None:
        return found
    if _is_model_page_source_id(source_id):
        return FireworksSourceType.MODEL_PAGE
    raise FireworksParseError(
        f"Fireworks source_id {source_id!r} is not a known documentation source"
    )


def _is_model_page_source_id(source_id: str) -> bool:
    prefix = f"{SOURCE_ID_MODEL_PAGE_PREFIX}:"
    if not source_id.startswith(prefix):
        return False
    return _PAGE_IDENTITY.fullmatch(source_id[len(prefix) :]) is not None


@dataclass(frozen=True, slots=True)
class CapturedSource:
    """Official Fireworks documentation already captured. Bytes are not fetched here."""

    source_id: str
    source_url: str
    body: str
    retrieved_at: str
    media_type: str = "text/markdown"

    def __post_init__(self) -> None:
        source_type = source_type_for_id(self.source_id)
        if not isinstance(self.source_url, str) or not self.source_url.strip():
            raise FireworksParseError("Fireworks source URL is missing")
        if source_type is FireworksSourceType.MODEL_PAGE:
            expected = model_page_source_id_from_url(self.source_url)
            if expected != self.source_id:
                raise FireworksParseError(
                    "Fireworks model page source_id does not match the document URL"
                )
        if not isinstance(self.body, str):
            raise FireworksParseError("Fireworks documentation body must be text")
        try:
            validate_timestamp("retrieved_at", self.retrieved_at)
        except CatalogValidationError as exc:
            raise FireworksParseError(
                "Fireworks source retrieved_at must be a timezone-aware timestamp"
            ) from exc
        if not isinstance(self.media_type, str) or not self.media_type.strip():
            raise FireworksParseError("Fireworks source media type is missing")

    @property
    def source_type(self) -> FireworksSourceType:
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
class FireworksSourceDescriptor:
    """Official Fireworks document to retrieve. URLs live only in this module."""

    source_id: str
    source_type: FireworksSourceType
    url: str
    required: bool

    @property
    def path(self) -> str:
        return urlparse(self.url).path


def embeddings_descriptor() -> FireworksSourceDescriptor:
    return FireworksSourceDescriptor(
        source_id=SOURCE_ID_EMBEDDINGS,
        source_type=FireworksSourceType.EMBEDDINGS,
        url=EMBEDDINGS_URL,
        required=True,
    )


def serving_paths_descriptor() -> FireworksSourceDescriptor:
    return FireworksSourceDescriptor(
        source_id=SOURCE_ID_SERVING_PATHS,
        source_type=FireworksSourceType.SERVING_PATHS,
        url=SERVING_PATHS_URL,
        required=True,
    )


def changelog_descriptor() -> FireworksSourceDescriptor:
    return FireworksSourceDescriptor(
        source_id=SOURCE_ID_CHANGELOG,
        source_type=FireworksSourceType.CHANGELOG,
        url=CHANGELOG_URL,
        required=True,
    )


def required_source_descriptors() -> tuple[FireworksSourceDescriptor, ...]:
    return (embeddings_descriptor(), serving_paths_descriptor(), changelog_descriptor())


def optional_source_descriptors() -> tuple[FireworksSourceDescriptor, ...]:
    """Best-effort docs. Missing or unparsable pages omit those facts only."""
    return (
        FireworksSourceDescriptor(
            source_id=SOURCE_ID_TEXT_MODELS,
            source_type=FireworksSourceType.TEXT_MODELS,
            url=TEXT_MODELS_URL,
            required=False,
        ),
        FireworksSourceDescriptor(
            source_id=SOURCE_ID_VISION_MODELS,
            source_type=FireworksSourceType.VISION_MODELS,
            url=VISION_MODELS_URL,
            required=False,
        ),
        FireworksSourceDescriptor(
            source_id=SOURCE_ID_TOOL_CALLING,
            source_type=FireworksSourceType.TOOL_CALLING,
            url=TOOL_CALLING_URL,
            required=False,
        ),
        model_page_descriptor("kimi-k2"),
    )


def model_page_descriptor(page_identity: str) -> FireworksSourceDescriptor:
    url = model_page_url(page_identity)
    return FireworksSourceDescriptor(
        source_id=model_page_source_id(page_identity),
        source_type=FireworksSourceType.MODEL_PAGE,
        url=url,
        required=False,
    )


def expected_path_for_source(descriptor: FireworksSourceDescriptor) -> str:
    parsed = urlparse(descriptor.url)
    return parsed.path
