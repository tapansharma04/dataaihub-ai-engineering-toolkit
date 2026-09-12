"""Commit a provider Model Intelligence refresh to the local overlay.

Internal orchestration. Not part of the public Samyak API.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from samyak.model.catalog import (
    CatalogFreshness,
    CatalogNotice,
    CatalogOverlay,
    ModelCatalog,
    build_catalog,
)
from samyak.model.errors import (
    CatalogSchemaError,
    CatalogStoreError,
    ModelCatalogError,
)
from samyak.model.facts import FactStatus
from samyak.model.http import Clock, HttpTransport
from samyak.model.providers.anthropic.fetch import (
    anthropic_https_transport,
    refresh_anthropic_catalog,
)
from samyak.model.providers.openai.fetch import (
    openai_https_transport,
    refresh_openai_catalog,
    utc_now_iso,
)
from samyak.model.records import RECORD_FACT_NAMES, ModelRecord
from samyak.model.store import FileCatalogStore

PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
SUPPORTED_PROVIDERS = (PROVIDER_ANTHROPIC, PROVIDER_OPENAI)


@dataclass(frozen=True, slots=True)
class CatalogUpdateResult:
    """Outcome of a local overlay update. Does not imply a public API.

    ``partial`` is copied from the refresh result: True only when best-effort
    model pages failed for the provider being updated. Notices alone do not
    mark the catalog partial.
    """

    provider_id: str
    ok: bool
    committed: bool
    partial: bool
    catalog: ModelCatalog | None
    catalog_path: Path
    previous_existed: bool
    error: str | None
    notices: tuple[CatalogNotice, ...]

    @property
    def provider_model_count(self) -> int:
        """How many records belong to the provider this command updated."""
        if self.catalog is None:
            return 0
        return sum(1 for item in self.catalog.models if item.provider_id == self.provider_id)

    @property
    def catalog_model_count(self) -> int:
        """How many records the mixed overlay contains after this command."""
        if self.catalog is None:
            return 0
        return len(self.catalog.models)


OpenAIUpdateResult = CatalogUpdateResult


def update_openai_catalog(
    *,
    transport: HttpTransport | None = None,
    store: FileCatalogStore | None = None,
    clock: Clock | None = None,
) -> CatalogUpdateResult:
    """Fetch official OpenAI docs and merge them into the local overlay on success."""
    return update_provider_catalog(
        PROVIDER_OPENAI,
        transport=transport,
        store=store,
        clock=clock,
    )


def update_anthropic_catalog(
    *,
    transport: HttpTransport | None = None,
    store: FileCatalogStore | None = None,
    clock: Clock | None = None,
) -> CatalogUpdateResult:
    """Fetch official Anthropic docs and merge them into the local overlay on success."""
    return update_provider_catalog(
        PROVIDER_ANTHROPIC,
        transport=transport,
        store=store,
        clock=clock,
    )


def update_provider_catalog(
    provider_id: str,
    *,
    transport: HttpTransport | None = None,
    store: FileCatalogStore | None = None,
    clock: Clock | None = None,
) -> CatalogUpdateResult:
    """Replace one provider's records in the overlay. Other providers are kept.

    Required source failures do not write. An existing live catalog that
    cannot be loaded is not treated as missing: the file is left unchanged.
    Best-effort model-page failures still commit a catalog that includes
    notices. Timestamps are taken from ``clock`` (or now) and passed into
    refresh; parsers do not invent them.
    """
    store = store or FileCatalogStore()
    clock_fn = clock or utc_now_iso
    stamp = clock_fn()
    previous_existed = store.exists()
    previous, load_error = _load_previous_if_present(store, previous_existed)
    if load_error is not None:
        return _failed(provider_id, store, previous_existed, load_error)
    if transport is None:
        transport = _default_transport(provider_id, clock_fn)
    try:
        refresh = _refresh_provider(
            provider_id,
            transport,
            generated_at=stamp,
            verified_at=stamp,
        )
    except ModelCatalogError as exc:
        return _failed(provider_id, store, previous_existed, str(exc))
    if not refresh.ok or refresh.catalog is None:
        return _failed(
            provider_id,
            store,
            previous_existed,
            refresh.error or f"{provider_id} catalog refresh failed",
        )
    merged = merge_provider_refresh(previous, refresh.catalog, provider_id=provider_id)
    try:
        store.save(merged)
    except CatalogStoreError as exc:
        return _failed(provider_id, store, previous_existed, str(exc), notices=refresh.notices)
    except OSError:
        return _failed(
            provider_id,
            store,
            previous_existed,
            "model catalog could not be written",
            notices=refresh.notices,
        )
    return CatalogUpdateResult(
        provider_id=provider_id,
        ok=True,
        committed=True,
        partial=refresh.partial,
        catalog=merged,
        catalog_path=store.catalog_path,
        previous_existed=previous_existed,
        error=None,
        notices=merged.notices,
    )


def merge_provider_refresh(
    previous: ModelCatalog | None,
    incoming: ModelCatalog,
    *,
    provider_id: str,
) -> ModelCatalog:
    """Replace ``provider_id`` records and keep every other provider's records."""
    if previous is None:
        return incoming
    kept_models = tuple(item for item in previous.models if item.provider_id != provider_id)
    models = kept_models + incoming.models
    kept_notices = tuple(
        notice for notice in previous.notices if _notice_provider(notice.code) != provider_id
    )
    incoming_ids = {item.source_id for item in incoming.freshness.source_retrieved_at}
    kept_sources = tuple(
        item
        for item in previous.freshness.source_retrieved_at
        if _source_provider(item.source_id) != provider_id and item.source_id not in incoming_ids
    )
    return build_catalog(
        models=models,
        generated_at=incoming.generated_at,
        freshness=CatalogFreshness(
            status=incoming.freshness.status,
            generated_at=incoming.generated_at,
            overlay=incoming.freshness.overlay,
            oldest_verified_at=_oldest_verified_at(models),
            source_retrieved_at=kept_sources + incoming.freshness.source_retrieved_at,
            stale_after=incoming.freshness.stale_after,
        ),
        notices=kept_notices + incoming.notices,
        version=incoming.version,
    )


def _refresh_provider(
    provider_id: str,
    transport: HttpTransport,
    *,
    generated_at: str,
    verified_at: str,
):
    if provider_id == PROVIDER_OPENAI:
        return refresh_openai_catalog(
            transport,
            generated_at=generated_at,
            verified_at=verified_at,
            overlay=CatalogOverlay.USER_CACHE,
        )
    if provider_id == PROVIDER_ANTHROPIC:
        return refresh_anthropic_catalog(
            transport,
            generated_at=generated_at,
            verified_at=verified_at,
            overlay=CatalogOverlay.USER_CACHE,
        )
    raise ModelCatalogError(f"unknown provider {provider_id!r}")


def _default_transport(provider_id: str, clock: Clock) -> HttpTransport:
    if provider_id == PROVIDER_OPENAI:
        return openai_https_transport(clock=clock)
    if provider_id == PROVIDER_ANTHROPIC:
        return anthropic_https_transport(clock=clock)
    raise ModelCatalogError(f"unknown provider {provider_id!r}")


def _load_previous_if_present(
    store: FileCatalogStore, previous_existed: bool
) -> tuple[ModelCatalog | None, str | None]:
    """Load the live overlay, or fail closed if it exists but cannot be read.

    A missing catalog is ``(None, None)``. A present unreadable catalog is
    ``(None, error)`` so callers must not write a replacement overlay.
    """
    if not previous_existed:
        return None, None
    try:
        return store.load(), None
    except CatalogSchemaError as exc:
        return None, str(exc)
    except CatalogStoreError as exc:
        return None, str(exc)


def _notice_provider(code: str) -> str | None:
    if code == "PARTIAL_MODEL_PAGES" or code.startswith("PARTIAL_MODEL_PAGES:openai"):
        return PROVIDER_OPENAI
    if code.startswith("PARTIAL_MODEL_PAGES:anthropic"):
        return PROVIDER_ANTHROPIC
    if "openai-docs-" in code:
        return PROVIDER_OPENAI
    if "anthropic-docs-" in code:
        return PROVIDER_ANTHROPIC
    return None


def _source_provider(source_id: str) -> str | None:
    if source_id.startswith("openai-docs-"):
        return PROVIDER_OPENAI
    if source_id.startswith("anthropic-docs-"):
        return PROVIDER_ANTHROPIC
    return None


def _oldest_verified_at(records: Sequence[ModelRecord]) -> str | None:
    times: list[str] = []
    for record in records:
        for name in RECORD_FACT_NAMES:
            fact = getattr(record, name)
            if fact.status is FactStatus.KNOWN and fact.provenance is not None:
                stamp = fact.provenance.verified_at
                if stamp is not None:
                    times.append(stamp)
            elif fact.status is FactStatus.CONFLICT:
                for claim in fact.claims:
                    stamp = claim.provenance.verified_at
                    if stamp is not None:
                        times.append(stamp)
    return min(times) if times else None


def _failed(
    provider_id: str,
    store: FileCatalogStore,
    previous_existed: bool,
    error: str,
    notices: tuple[CatalogNotice, ...] = (),
) -> CatalogUpdateResult:
    return CatalogUpdateResult(
        provider_id=provider_id,
        ok=False,
        committed=False,
        partial=False,
        catalog=None,
        catalog_path=store.catalog_path,
        previous_existed=previous_existed,
        error=error,
        notices=notices,
    )
