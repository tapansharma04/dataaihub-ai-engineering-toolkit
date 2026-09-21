"""Capture official Together documentation over an injected HTTP transport.

The parser does not import this module. Persistence and CLI update live elsewhere.
Dedicated endpoints, DCI, customer uploads, and the authenticated Together
Models API are not allowlisted.
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
from samyak.model.providers.together.adapter import catalog_from_together_sources
from samyak.model.providers.together.errors import (
    TogetherAdapterError,
    TogetherFetchError,
    TogetherParseError,
)
from samyak.model.providers.together.parse import parse_together_sources
from samyak.model.providers.together.sources import (
    DOCS_HOST,
    DOCS_PATH_PREFIX,
    CapturedSource,
    TogetherSourceDescriptor,
    TogetherSourceType,
    captured_markdown,
    expected_path_for_source,
    model_page_source_id_from_url,
    optional_source_descriptors,
    required_source_descriptors,
)

_NOTICE_PARTIAL = "PARTIAL_MODEL_PAGES:together"
_EMPTY_OPTIONAL = "did not list any serving model ids"


@dataclass(frozen=True, slots=True)
class TogetherRefreshResult:
    """In-memory refresh outcome. Does not write a catalog overlay.

    ``partial`` is True only when the optional changelog or an allowlisted
    serving-model quickstart could not be used. Serverless coverage itself
    is not a partial catalog.
    """

    ok: bool
    catalog: ModelCatalog | None
    error: str | None
    notices: tuple[CatalogNotice, ...]
    partial: bool


def together_docs_policy() -> TransportPolicy:
    """HTTPS policy for official Together documentation on docs.together.ai."""
    return TransportPolicy(
        allowed_hosts=frozenset({DOCS_HOST}),
        allowed_path_prefixes=frozenset({DOCS_PATH_PREFIX}),
    )


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def together_https_transport(*, clock: Clock | None = None) -> HttpsTransport:
    """Production transport for official Together documentation. No credentials."""
    return HttpsTransport(
        policy=together_docs_policy(),
        clock=clock or utc_now_iso,
        user_agent_value=user_agent(__version__),
    )


def capture_together_source(
    transport: HttpTransport,
    descriptor: TogetherSourceDescriptor,
) -> CapturedSource:
    """GET one official document and return a CapturedSource. No catalog write."""
    try:
        response = transport.get(descriptor.url)
    except TransportError as exc:
        raise TogetherFetchError(str(exc)) from exc
    if response.status < 200 or response.status >= 300:
        raise TogetherFetchError(
            f"Together documentation {descriptor.source_id} returned HTTP {response.status}"
        )
    _assert_final_document(descriptor, response.final_url)
    try:
        body = response.body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TogetherFetchError("Together documentation is not UTF-8") from exc
    source_id = descriptor.source_id
    if descriptor.source_type is TogetherSourceType.MODEL_PAGE:
        source_id = model_page_source_id_from_url(response.final_url)
        if source_id != descriptor.source_id:
            raise TogetherFetchError(
                "Together model page redirected to a different documentation document"
            )
    return captured_markdown(
        source_id=source_id,
        source_url=response.final_url,
        body=body,
        retrieved_at=response.retrieved_at,
    )


def refresh_together_catalog(
    transport: HttpTransport,
    *,
    generated_at: str,
    verified_at: str,
    overlay: CatalogOverlay = CatalogOverlay.USER_CACHE,
) -> TogetherRefreshResult:
    """Fetch official sources and build a catalog in memory.

    Required sources (serverless catalogue, deprecations) fail the operation.
    The changelog and allowlisted serving-model quickstarts are best-effort.
    Coverage is Together's public Serverless Models catalogue plus those
    docs.together.ai model pages, not dedicated endpoints, customer uploads,
    or www.together.ai/models.
    This function does not write bundled or cache files.
    """
    captured: list[CapturedSource] = []
    try:
        for descriptor in required_source_descriptors():
            captured.append(capture_together_source(transport, descriptor))
        parse_together_sources(tuple(captured))
    except (TogetherFetchError, TogetherParseError, TogetherAdapterError, TransportError) as exc:
        return TogetherRefreshResult(
            ok=False, catalog=None, error=str(exc), notices=(), partial=False
        )

    notices: list[CatalogNotice] = []
    optional_failures: list[CatalogNotice] = []
    for descriptor in optional_source_descriptors():
        try:
            page = capture_together_source(transport, descriptor)
        except (TogetherFetchError, TransportError):
            optional_failures.append(_detail_failure_notice(descriptor, parsed=False))
            continue
        try:
            parse_together_sources((page,))
        except (TogetherParseError, TogetherAdapterError) as exc:
            if _empty_optional_document(descriptor, exc):
                captured.append(page)
                continue
            optional_failures.append(_detail_failure_notice(descriptor, parsed=True))
            continue
        captured.append(page)

    partial = bool(optional_failures)
    if partial:
        notices.append(
            CatalogNotice(
                code=_NOTICE_PARTIAL,
                message=(
                    "One or more Together documentation pages could not be retrieved; "
                    "facts from those pages are omitted"
                ),
            )
        )
        notices.extend(optional_failures)

    try:
        catalog = catalog_from_together_sources(
            captured,
            generated_at=generated_at,
            verified_at=verified_at,
            overlay=overlay,
            notices=tuple(notices),
        )
    except (TogetherParseError, TogetherAdapterError) as exc:
        return TogetherRefreshResult(
            ok=False,
            catalog=None,
            error=str(exc),
            notices=tuple(notices),
            partial=False,
        )
    return TogetherRefreshResult(
        ok=True,
        catalog=catalog,
        error=None,
        notices=tuple(notices),
        partial=partial,
    )


def _empty_optional_document(descriptor: TogetherSourceDescriptor, exc: Exception) -> bool:
    if descriptor.source_type is TogetherSourceType.MODEL_PAGE:
        return False
    return _EMPTY_OPTIONAL in str(exc)


def _assert_final_document(descriptor: TogetherSourceDescriptor, final_url: str) -> None:
    parsed = urlparse(final_url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme.lower() != "https":
        raise TogetherFetchError("Together documentation URL must be HTTPS")
    if host != DOCS_HOST:
        raise TogetherFetchError("Together documentation redirected off the official host")
    if parsed.query or parsed.fragment or parsed.params:
        raise TogetherFetchError("Together documentation URL must not include a query or fragment")
    expected_path = expected_path_for_source(descriptor)
    if parsed.path != expected_path:
        raise TogetherFetchError(
            "Together documentation redirected to a different official document"
        )
    if descriptor.source_type is TogetherSourceType.MODEL_PAGE:
        derived = model_page_source_id_from_url(final_url)
        if derived != descriptor.source_id:
            raise TogetherFetchError(
                "Together model page redirected to a different documentation document"
            )


def _detail_failure_notice(descriptor: TogetherSourceDescriptor, *, parsed: bool) -> CatalogNotice:
    if parsed:
        return CatalogNotice(
            code=f"SOURCE_PARSE_FAILED:{descriptor.source_id}",
            message=f"Together documentation {descriptor.source_id} could not be parsed",
        )
    return CatalogNotice(
        code=f"SOURCE_FETCH_FAILED:{descriptor.source_id}",
        message=f"Together documentation {descriptor.source_id} could not be retrieved",
    )
