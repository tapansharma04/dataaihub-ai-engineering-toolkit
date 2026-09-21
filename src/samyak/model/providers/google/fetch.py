"""Capture official Google Gemini API documentation over an injected HTTP transport.

The parser does not import this module. Persistence and CLI update live elsewhere.
Vertex AI hosts and credentialed Models API URLs are not allowlisted.
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
from samyak.model.providers.google.adapter import catalog_from_google_sources
from samyak.model.providers.google.errors import (
    GoogleAdapterError,
    GoogleFetchError,
    GoogleParseError,
)
from samyak.model.providers.google.parse import discover_model_page_identities, parse_google_sources
from samyak.model.providers.google.sources import (
    DOCS_HOST,
    DOCS_PATH_PREFIX,
    CapturedSource,
    GoogleSourceDescriptor,
    GoogleSourceType,
    captured_markdown,
    expected_path_for_source,
    model_page_descriptor,
    model_page_source_id_from_url,
    required_source_descriptors,
)

_NOTICE_PARTIAL = "PARTIAL_MODEL_PAGES:google"


@dataclass(frozen=True, slots=True)
class GoogleRefreshResult:
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


def google_docs_policy() -> TransportPolicy:
    """HTTPS policy for official Gemini API documentation on ai.google.dev."""
    return TransportPolicy(
        allowed_hosts=frozenset({DOCS_HOST}),
        allowed_path_prefixes=frozenset({DOCS_PATH_PREFIX}),
    )


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def google_https_transport(*, clock: Clock | None = None) -> HttpsTransport:
    """Production transport for official Gemini API documentation. No credentials."""
    return HttpsTransport(
        policy=google_docs_policy(),
        clock=clock or utc_now_iso,
        user_agent_value=user_agent(__version__),
    )


def capture_google_source(
    transport: HttpTransport,
    descriptor: GoogleSourceDescriptor,
) -> CapturedSource:
    """GET one official document and return a CapturedSource. No catalog write."""
    try:
        response = transport.get(descriptor.url)
    except TransportError as exc:
        raise GoogleFetchError(str(exc)) from exc
    if response.status < 200 or response.status >= 300:
        raise GoogleFetchError(
            f"Google documentation {descriptor.source_id} returned HTTP {response.status}"
        )
    _assert_final_document(descriptor, response.final_url)
    try:
        body = response.body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GoogleFetchError("Google documentation is not UTF-8") from exc
    source_id = descriptor.source_id
    if descriptor.source_type is GoogleSourceType.MODEL_PAGE:
        source_id = model_page_source_id_from_url(response.final_url)
        if source_id != descriptor.source_id:
            raise GoogleFetchError(
                "Google model page redirected to a different documentation document"
            )
    return captured_markdown(
        source_id=source_id,
        source_url=response.final_url,
        body=body,
        retrieved_at=response.retrieved_at,
    )


def refresh_google_catalog(
    transport: HttpTransport,
    *,
    generated_at: str,
    verified_at: str,
    overlay: CatalogOverlay = CatalogOverlay.USER_CACHE,
) -> GoogleRefreshResult:
    """Fetch official sources and build a catalog in memory.

    Required sources (models index, deprecations, Gemini 3 guide) fail the
    operation. Individual model pages are best-effort: failures become catalog
    notices.
    This function does not write bundled or cache files.
    """
    captured: list[CapturedSource] = []
    try:
        for descriptor in required_source_descriptors():
            captured.append(capture_google_source(transport, descriptor))
        parse_google_sources(tuple(captured))
    except (GoogleFetchError, GoogleParseError, GoogleAdapterError, TransportError) as exc:
        return GoogleRefreshResult(
            ok=False, catalog=None, error=str(exc), notices=(), partial=False
        )

    overview = next(
        source for source in captured if source.source_type is GoogleSourceType.MODELS_INDEX
    )
    page_ids = discover_model_page_identities(overview.body)
    notices: list[CatalogNotice] = []
    for page_id in page_ids:
        descriptor = model_page_descriptor(page_id)
        try:
            page = capture_google_source(transport, descriptor)
            parse_google_sources((page,))
        except (GoogleFetchError, TransportError):
            notices.append(_detail_failure_notice(descriptor, parsed=False))
            continue
        except (GoogleParseError, GoogleAdapterError):
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
                    "One or more Google Gemini API model pages could not be retrieved; "
                    "facts from those pages are omitted"
                ),
            ),
        )

    try:
        catalog = catalog_from_google_sources(
            captured,
            generated_at=generated_at,
            verified_at=verified_at,
            overlay=overlay,
            notices=tuple(notices),
        )
    except (GoogleParseError, GoogleAdapterError) as exc:
        return GoogleRefreshResult(
            ok=False,
            catalog=None,
            error=str(exc),
            notices=tuple(notices),
            partial=False,
        )
    return GoogleRefreshResult(
        ok=True,
        catalog=catalog,
        error=None,
        notices=tuple(notices),
        partial=partial,
    )


def _assert_final_document(descriptor: GoogleSourceDescriptor, final_url: str) -> None:
    parsed = urlparse(final_url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme.lower() != "https":
        raise GoogleFetchError("Google documentation URL must be HTTPS")
    if host != DOCS_HOST:
        raise GoogleFetchError("Google documentation redirected off the official host")
    if parsed.query or parsed.fragment or parsed.params:
        raise GoogleFetchError("Google documentation URL must not include a query or fragment")
    expected_path = expected_path_for_source(descriptor)
    if parsed.path != expected_path:
        raise GoogleFetchError("Google documentation redirected to a different official document")
    if descriptor.source_type is GoogleSourceType.MODEL_PAGE:
        derived = model_page_source_id_from_url(final_url)
        if derived != descriptor.source_id:
            raise GoogleFetchError(
                "Google model page redirected to a different documentation document"
            )


def _detail_failure_notice(descriptor: GoogleSourceDescriptor, *, parsed: bool) -> CatalogNotice:
    if parsed:
        return CatalogNotice(
            code=f"SOURCE_PARSE_FAILED:{descriptor.source_id}",
            message=f"Google model page {descriptor.source_id} could not be parsed",
        )
    return CatalogNotice(
        code=f"SOURCE_FETCH_FAILED:{descriptor.source_id}",
        message=f"Google model page {descriptor.source_id} could not be retrieved",
    )
