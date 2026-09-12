"""Capture official OpenAI documentation over an injected HTTP transport.

The parser does not import this module. Persistence and CLI update live elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlparse

from samyak.__version__ import __version__
from samyak.model.catalog import CatalogNotice, CatalogOverlay, ModelCatalog
from samyak.model.http import (
    Clock,
    HttpsTransport,
    HttpTransport,
    TransportError,
    TransportPolicy,
    user_agent,
)
from samyak.model.providers.openai.adapter import catalog_from_openai_sources
from samyak.model.providers.openai.errors import (
    OpenAIAdapterError,
    OpenAIFetchError,
    OpenAIParseError,
)
from samyak.model.providers.openai.parse import parse_openai_sources
from samyak.model.providers.openai.sources import (
    DOCS_HOST,
    DOCS_PATH_PREFIX,
    CapturedSource,
    OpenAISourceDescriptor,
    OpenAISourceType,
    captured_markdown,
    expected_path_for_source,
    model_page_descriptor,
    model_page_source_id_from_url,
    required_source_descriptors,
)

_NOTICE_PARTIAL = "PARTIAL_MODEL_PAGES"


@dataclass(frozen=True, slots=True)
class OpenAIRefreshResult:
    """In-memory refresh outcome. Does not write a catalog overlay.

    ``partial`` is True only when one or more best-effort model pages failed.
    Catalog notices are independent: an informational notice does not make
    the catalog partial.
    """

    ok: bool
    catalog: ModelCatalog | None
    error: str | None
    notices: tuple[CatalogNotice, ...]
    partial: bool


def openai_docs_policy() -> TransportPolicy:
    """HTTPS policy for official OpenAI developer documentation."""
    return TransportPolicy(
        allowed_hosts=frozenset({DOCS_HOST}),
        allowed_path_prefixes=frozenset({DOCS_PATH_PREFIX}),
    )


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def openai_https_transport(*, clock: Clock | None = None) -> HttpsTransport:
    """Production transport for official OpenAI documentation. No credentials."""
    return HttpsTransport(
        policy=openai_docs_policy(),
        clock=clock or utc_now_iso,
        user_agent_value=user_agent(__version__),
    )


def capture_openai_source(
    transport: HttpTransport,
    descriptor: OpenAISourceDescriptor,
) -> CapturedSource:
    """GET one official document and return a CapturedSource. No catalog write."""
    try:
        response = transport.get(descriptor.url)
    except TransportError as exc:
        raise OpenAIFetchError(str(exc)) from exc
    if response.status < 200 or response.status >= 300:
        raise OpenAIFetchError(
            f"OpenAI documentation {descriptor.source_id} returned HTTP {response.status}"
        )
    _assert_final_document(descriptor, response.final_url)
    try:
        body = response.body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OpenAIFetchError("OpenAI documentation is not UTF-8") from exc
    source_id = descriptor.source_id
    if descriptor.source_type is OpenAISourceType.MODEL_PAGE:
        source_id = model_page_source_id_from_url(response.final_url)
        if source_id != descriptor.source_id:
            raise OpenAIFetchError(
                "OpenAI model page redirected to a different documentation document"
            )
    return captured_markdown(
        source_id=source_id,
        source_url=response.final_url,
        body=body,
        retrieved_at=response.retrieved_at,
    )


def refresh_openai_catalog(
    transport: HttpTransport,
    *,
    generated_at: str,
    verified_at: str,
    overlay: CatalogOverlay = CatalogOverlay.USER_CACHE,
) -> OpenAIRefreshResult:
    """Fetch official sources and build a catalog in memory.

    Required sources (models index, deprecations) fail the operation.
    Individual model pages are best-effort: failures become catalog notices.
    This function does not write bundled or cache files.
    """
    captured: list[CapturedSource] = []
    try:
        for descriptor in required_source_descriptors():
            captured.append(capture_openai_source(transport, descriptor))
        parse_openai_sources(tuple(captured))
    except (OpenAIFetchError, OpenAIParseError, OpenAIAdapterError, TransportError) as exc:
        return OpenAIRefreshResult(
            ok=False, catalog=None, error=str(exc), notices=(), partial=False
        )

    index = next(
        source for source in captured if source.source_type is OpenAISourceType.MODELS_INDEX
    )
    page_ids = tuple(
        dict.fromkeys(
            observation.provider_model_id
            for observation in parse_openai_sources((index,))
            if observation.listed_on_index
        )
    )
    notices: list[CatalogNotice] = []
    for page_id in page_ids:
        descriptor = model_page_descriptor(page_id)
        try:
            page = capture_openai_source(transport, descriptor)
            parse_openai_sources((page,))
        except (OpenAIFetchError, TransportError):
            notices.append(_detail_failure_notice(descriptor, parsed=False))
            continue
        except (OpenAIParseError, OpenAIAdapterError):
            notices.append(_detail_failure_notice(descriptor, parsed=True))
            continue
        captured.append(page)

    partial = bool(notices)
    if partial:
        notices.insert(
            0,
            CatalogNotice(
                code=_NOTICE_PARTIAL,
                message=(
                    "One or more OpenAI model pages could not be retrieved; "
                    "facts from those pages are omitted"
                ),
            ),
        )

    try:
        catalog = catalog_from_openai_sources(
            captured,
            generated_at=generated_at,
            verified_at=verified_at,
            overlay=overlay,
            notices=tuple(notices),
        )
    except (OpenAIParseError, OpenAIAdapterError) as exc:
        return OpenAIRefreshResult(
            ok=False,
            catalog=None,
            error=str(exc),
            notices=tuple(notices),
            partial=False,
        )
    return OpenAIRefreshResult(
        ok=True,
        catalog=catalog,
        error=None,
        notices=tuple(notices),
        partial=partial,
    )


def _assert_final_document(descriptor: OpenAISourceDescriptor, final_url: str) -> None:
    parsed = urlparse(final_url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme.lower() != "https":
        raise OpenAIFetchError("OpenAI documentation URL must be HTTPS")
    if host != DOCS_HOST:
        raise OpenAIFetchError("OpenAI documentation redirected off the official host")
    if parsed.query or parsed.fragment or parsed.params:
        raise OpenAIFetchError("OpenAI documentation URL must not include a query or fragment")
    expected_path = expected_path_for_source(descriptor)
    if parsed.path != expected_path:
        raise OpenAIFetchError("OpenAI documentation redirected to a different official document")
    if descriptor.source_type is OpenAISourceType.MODEL_PAGE:
        derived = model_page_source_id_from_url(final_url)
        if derived != descriptor.source_id:
            raise OpenAIFetchError(
                "OpenAI model page redirected to a different documentation document"
            )


def _detail_failure_notice(descriptor: OpenAISourceDescriptor, *, parsed: bool) -> CatalogNotice:
    if parsed:
        return CatalogNotice(
            code=f"SOURCE_PARSE_FAILED:{descriptor.source_id}",
            message=f"OpenAI model page {descriptor.source_id} could not be parsed",
        )
    return CatalogNotice(
        code=f"SOURCE_FETCH_FAILED:{descriptor.source_id}",
        message=f"OpenAI model page {descriptor.source_id} could not be retrieved",
    )
