"""Capture official Fireworks documentation over an injected HTTP transport.

The parser does not import this module. Persistence and CLI update live elsewhere.
fireworks.ai/models, the credentialed List Models API, app.fireworks.ai, and
customer deployments are not allowlisted.
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
from samyak.model.providers.fireworks.adapter import catalog_from_fireworks_sources
from samyak.model.providers.fireworks.errors import (
    FireworksAdapterError,
    FireworksFetchError,
    FireworksParseError,
)
from samyak.model.providers.fireworks.parse import parse_fireworks_sources
from samyak.model.providers.fireworks.sources import (
    DOCS_HOST,
    DOCS_PATH_PREFIXES,
    CapturedSource,
    FireworksSourceDescriptor,
    FireworksSourceType,
    captured_markdown,
    expected_path_for_source,
    model_page_source_id_from_url,
    optional_source_descriptors,
    required_source_descriptors,
)

_NOTICE_PARTIAL = "PARTIAL_MODEL_PAGES:fireworks"
_NOTICE_SUBSET = "DOCUMENTED_SUBSET:fireworks"
_EMPTY_OPTIONAL = "did not list any serving model ids"


@dataclass(frozen=True, slots=True)
class FireworksRefreshResult:
    """In-memory refresh outcome. Does not write a catalog overlay.

    ``partial`` is True only when one or more best-effort optional documents
    failed. The documented-subset notice is informational and does not make
    the catalog partial.
    """

    ok: bool
    catalog: ModelCatalog | None
    error: str | None
    notices: tuple[CatalogNotice, ...]
    partial: bool


def fireworks_docs_policy() -> TransportPolicy:
    """HTTPS policy for official Fireworks documentation on docs.fireworks.ai."""
    return TransportPolicy(
        allowed_hosts=frozenset({DOCS_HOST}),
        allowed_path_prefixes=DOCS_PATH_PREFIXES,
    )


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def fireworks_https_transport(*, clock: Clock | None = None) -> HttpsTransport:
    """Production transport for official Fireworks documentation. No credentials."""
    return HttpsTransport(
        policy=fireworks_docs_policy(),
        clock=clock or utc_now_iso,
        user_agent_value=user_agent(__version__),
    )


def capture_fireworks_source(
    transport: HttpTransport,
    descriptor: FireworksSourceDescriptor,
) -> CapturedSource:
    """GET one official document and return a CapturedSource. No catalog write."""
    try:
        response = transport.get(descriptor.url)
    except TransportError as exc:
        raise FireworksFetchError(str(exc)) from exc
    if response.status < 200 or response.status >= 300:
        raise FireworksFetchError(
            f"Fireworks documentation {descriptor.source_id} returned HTTP {response.status}"
        )
    _assert_final_document(descriptor, response.final_url)
    try:
        body = response.body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FireworksFetchError("Fireworks documentation is not UTF-8") from exc
    source_id = descriptor.source_id
    if descriptor.source_type is FireworksSourceType.MODEL_PAGE:
        source_id = model_page_source_id_from_url(response.final_url)
        if source_id != descriptor.source_id:
            raise FireworksFetchError(
                "Fireworks model page redirected to a different documentation document"
            )
    return captured_markdown(
        source_id=source_id,
        source_url=response.final_url,
        body=body,
        retrieved_at=response.retrieved_at,
    )


def refresh_fireworks_catalog(
    transport: HttpTransport,
    *,
    generated_at: str,
    verified_at: str,
    overlay: CatalogOverlay = CatalogOverlay.USER_CACHE,
) -> FireworksRefreshResult:
    """Fetch official sources and build a catalog in memory.

    Required sources fail the operation. Optional guides and model pages are
    best-effort and omitted when unavailable. Coverage is a documented public
    serving-ID subset from docs.fireworks.ai Markdown, not the complete
    Fireworks Model Library or Serverless inventory. This function does not
    write bundled or cache files.
    """
    captured: list[CapturedSource] = []
    try:
        for descriptor in required_source_descriptors():
            captured.append(capture_fireworks_source(transport, descriptor))
        parse_fireworks_sources(tuple(captured))
    except (FireworksFetchError, FireworksParseError, FireworksAdapterError, TransportError) as exc:
        return FireworksRefreshResult(
            ok=False, catalog=None, error=str(exc), notices=(), partial=False
        )

    notices: list[CatalogNotice] = [_subset_notice()]
    optional_failures: list[CatalogNotice] = []
    for descriptor in optional_source_descriptors():
        try:
            page = capture_fireworks_source(transport, descriptor)
        except (FireworksFetchError, TransportError):
            optional_failures.append(_detail_failure_notice(descriptor, parsed=False))
            continue
        try:
            parse_fireworks_sources((page,))
        except (FireworksParseError, FireworksAdapterError) as exc:
            if _empty_optional_guide(descriptor, exc):
                captured.append(page)
                continue
            optional_failures.append(_detail_failure_notice(descriptor, parsed=True))
            continue
        captured.append(page)

    partial = bool(optional_failures)
    if partial:
        notices.insert(
            0,
            CatalogNotice(
                code=_NOTICE_PARTIAL,
                message=(
                    "One or more Fireworks documentation pages could not be retrieved; "
                    "facts from those pages are omitted"
                ),
            ),
        )
    notices.extend(optional_failures)

    try:
        catalog = catalog_from_fireworks_sources(
            captured,
            generated_at=generated_at,
            verified_at=verified_at,
            overlay=overlay,
            notices=tuple(notices),
        )
    except (FireworksParseError, FireworksAdapterError) as exc:
        return FireworksRefreshResult(
            ok=False,
            catalog=None,
            error=str(exc),
            notices=tuple(notices),
            partial=False,
        )
    return FireworksRefreshResult(
        ok=True,
        catalog=catalog,
        error=None,
        notices=tuple(notices),
        partial=partial,
    )


def _subset_notice() -> CatalogNotice:
    return CatalogNotice(
        code=_NOTICE_SUBSET,
        message=(
            "Fireworks coverage is the documented public serving IDs in official "
            "Markdown, not the complete Fireworks Model Library"
        ),
    )


def _empty_optional_guide(descriptor: FireworksSourceDescriptor, exc: Exception) -> bool:
    if descriptor.source_type is FireworksSourceType.MODEL_PAGE:
        return False
    return _EMPTY_OPTIONAL in str(exc)


def _assert_final_document(descriptor: FireworksSourceDescriptor, final_url: str) -> None:
    parsed = urlparse(final_url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme.lower() != "https":
        raise FireworksFetchError("Fireworks documentation URL must be HTTPS")
    if host != DOCS_HOST:
        raise FireworksFetchError("Fireworks documentation redirected off the official host")
    if parsed.query or parsed.fragment or parsed.params:
        raise FireworksFetchError(
            "Fireworks documentation URL must not include a query or fragment"
        )
    expected_path = expected_path_for_source(descriptor)
    if parsed.path != expected_path:
        raise FireworksFetchError(
            "Fireworks documentation redirected to a different official document"
        )
    if descriptor.source_type is FireworksSourceType.MODEL_PAGE:
        derived = model_page_source_id_from_url(final_url)
        if derived != descriptor.source_id:
            raise FireworksFetchError(
                "Fireworks model page redirected to a different documentation document"
            )


def _detail_failure_notice(descriptor: FireworksSourceDescriptor, *, parsed: bool) -> CatalogNotice:
    if parsed:
        return CatalogNotice(
            code=f"SOURCE_PARSE_FAILED:{descriptor.source_id}",
            message=f"Fireworks documentation {descriptor.source_id} could not be parsed",
        )
    return CatalogNotice(
        code=f"SOURCE_FETCH_FAILED:{descriptor.source_id}",
        message=f"Fireworks documentation {descriptor.source_id} could not be retrieved",
    )
