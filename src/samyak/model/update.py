"""Commit a provider Model Intelligence refresh to the local overlay.

Internal orchestration. Not part of the public Samyak API.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
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
    CatalogValidationError,
    HistorySchemaError,
    ModelCatalogError,
)
from samyak.model.facts import FactStatus
from samyak.model.history import history_events_from_changes
from samyak.model.http import Clock, HttpTransport
from samyak.model.lifecycle import LifecycleChange, diff_provider_lifecycle
from samyak.model.providers.anthropic.fetch import (
    anthropic_https_transport,
    refresh_anthropic_catalog,
)
from samyak.model.providers.fireworks.fetch import (
    fireworks_https_transport,
    refresh_fireworks_catalog,
)
from samyak.model.providers.google.fetch import (
    google_https_transport,
    refresh_google_catalog,
)
from samyak.model.providers.openai.fetch import (
    openai_https_transport,
    refresh_openai_catalog,
    utc_now_iso,
)
from samyak.model.providers.together.fetch import (
    refresh_together_catalog,
    together_https_transport,
)
from samyak.model.records import (
    RECORD_FACT_NAMES,
    EvidenceDisposition,
    ModelRecord,
    derived_observed_sources,
)
from samyak.model.source_archive import (
    DEFAULT_ARCHIVE_KEEP,
    SourceArchive,
    close_source_archive,
    open_source_archive,
)
from samyak.model.store import FileCatalogStore
from samyak.model.transaction import CatalogHistoryTransaction, CatalogHistoryTransactionError

PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_GOOGLE = "google"
PROVIDER_FIREWORKS = "fireworks"
PROVIDER_TOGETHER = "together"
SUPPORTED_PROVIDERS = (
    PROVIDER_ANTHROPIC,
    PROVIDER_FIREWORKS,
    PROVIDER_GOOGLE,
    PROVIDER_OPENAI,
    PROVIDER_TOGETHER,
)
_SOURCE_ID_PREFIX = "-docs-"
_PARTIAL_NOTICE_PREFIX = "PARTIAL_MODEL_PAGES"
_FAILED_SOURCE_PREFIXES = ("SOURCE_FETCH_FAILED:", "SOURCE_PARSE_FAILED:")
HISTORY_WRITE_FAILED_CODE = "HISTORY_WRITE_FAILED"
HISTORY_WRITE_FAILED_SUMMARY = "lifecycle history could not be written."
HISTORY_WRITE_FAILED_DETAIL = (
    "The model catalog was updated, but lifecycle changes were not persisted to history."
)


def is_history_write_failure_code(code: str) -> bool:
    """True for a durable marker that history for one catalog update was not saved."""
    return code == HISTORY_WRITE_FAILED_CODE or code.startswith(f"{HISTORY_WRITE_FAILED_CODE}:")


def history_write_failure_code(provider_id: str, observed_at: str) -> str:
    """Identify the provider update whose lifecycle events were not appended.

    The timestamp may contain colons. The provider is the single field between
    the prefix and the first following colon.
    """
    return f"{HISTORY_WRITE_FAILED_CODE}:{provider_id}:{observed_at}"


@dataclass(frozen=True, slots=True)
class CatalogUpdateResult:
    """Outcome of a local overlay update. Does not imply a public API.

    ``partial`` is copied from the refresh result: True only when best-effort
    model pages failed for the provider being updated. Notices alone do not
    mark the catalog partial.

    ``lifecycle_changes`` is an in-memory diff of this provider's previous
    overlay against the incoming refresh. It is not catalog schema and is not
    written to ``catalog.json``.

    ``notices`` includes a ``HISTORY_WRITE_FAILED:<provider>:<observed_at>``
    marker when history persistence fails after a successful catalog commit.
    The marker is stored on the catalog. A later history append, including
    another provider in the same ``update all``, does not remove it. The
    omitted events are not retained and are not reconstructed.
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
    lifecycle_changes: tuple[LifecycleChange, ...] = ()

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
UpdateAllProgress = Callable[
    [str, str, int, int, CatalogUpdateResult | None],
    None,
]


class _ProgressHttpTransport:
    """Wrap a transport and report each GET so the CLI can show live progress."""

    def __init__(self, inner: HttpTransport, on_request: Callable[[], None]) -> None:
        self._inner = inner
        self._on_request = on_request

    def get(self, url: str):
        self._on_request()
        return self._inner.get(url)


def update_openai_catalog(
    *,
    transport: HttpTransport | None = None,
    store: FileCatalogStore | None = None,
    clock: Clock | None = None,
    archive: SourceArchive | None = None,
    archive_sources: bool = False,
    archive_keep: int = DEFAULT_ARCHIVE_KEEP,
    finalize_archive: bool = True,
) -> CatalogUpdateResult:
    """Fetch official OpenAI docs and merge them into the local overlay on success."""
    return update_provider_catalog(
        PROVIDER_OPENAI,
        transport=transport,
        store=store,
        clock=clock,
        archive=archive,
        archive_sources=archive_sources,
        archive_keep=archive_keep,
        finalize_archive=finalize_archive,
    )


def update_anthropic_catalog(
    *,
    transport: HttpTransport | None = None,
    store: FileCatalogStore | None = None,
    clock: Clock | None = None,
    archive: SourceArchive | None = None,
    archive_sources: bool = False,
    archive_keep: int = DEFAULT_ARCHIVE_KEEP,
    finalize_archive: bool = True,
) -> CatalogUpdateResult:
    """Fetch official Anthropic docs and merge them into the local overlay on success."""
    return update_provider_catalog(
        PROVIDER_ANTHROPIC,
        transport=transport,
        store=store,
        clock=clock,
        archive=archive,
        archive_sources=archive_sources,
        archive_keep=archive_keep,
        finalize_archive=finalize_archive,
    )


def update_google_catalog(
    *,
    transport: HttpTransport | None = None,
    store: FileCatalogStore | None = None,
    clock: Clock | None = None,
    archive: SourceArchive | None = None,
    archive_sources: bool = False,
    archive_keep: int = DEFAULT_ARCHIVE_KEEP,
    finalize_archive: bool = True,
) -> CatalogUpdateResult:
    """Fetch official Gemini API docs and merge them into the local overlay on success."""
    return update_provider_catalog(
        PROVIDER_GOOGLE,
        transport=transport,
        store=store,
        clock=clock,
        archive=archive,
        archive_sources=archive_sources,
        archive_keep=archive_keep,
        finalize_archive=finalize_archive,
    )


def update_fireworks_catalog(
    *,
    transport: HttpTransport | None = None,
    store: FileCatalogStore | None = None,
    clock: Clock | None = None,
    archive: SourceArchive | None = None,
    archive_sources: bool = False,
    archive_keep: int = DEFAULT_ARCHIVE_KEEP,
    finalize_archive: bool = True,
) -> CatalogUpdateResult:
    """Fetch official Fireworks docs and merge them into the local overlay on success."""
    return update_provider_catalog(
        PROVIDER_FIREWORKS,
        transport=transport,
        store=store,
        clock=clock,
        archive=archive,
        archive_sources=archive_sources,
        archive_keep=archive_keep,
        finalize_archive=finalize_archive,
    )


def update_together_catalog(
    *,
    transport: HttpTransport | None = None,
    store: FileCatalogStore | None = None,
    clock: Clock | None = None,
    archive: SourceArchive | None = None,
    archive_sources: bool = False,
    archive_keep: int = DEFAULT_ARCHIVE_KEEP,
    finalize_archive: bool = True,
) -> CatalogUpdateResult:
    """Fetch official Together docs and merge them into the local overlay on success."""
    return update_provider_catalog(
        PROVIDER_TOGETHER,
        transport=transport,
        store=store,
        clock=clock,
        archive=archive,
        archive_sources=archive_sources,
        archive_keep=archive_keep,
        finalize_archive=finalize_archive,
    )


def update_all_catalogs(
    *,
    store: FileCatalogStore | None = None,
    clock: Clock | None = None,
    progress: UpdateAllProgress | None = None,
    archive_sources: bool = False,
    archive_keep: int = DEFAULT_ARCHIVE_KEEP,
) -> tuple[CatalogUpdateResult, ...]:
    """Refresh every supported provider sequentially using the same update path.

    Provider order is ``SUPPORTED_PROVIDERS``. One provider's failure does not
    skip the remaining providers and does not write an empty replacement for
    that provider. This function does not persist a separate all-update record.
    ``progress`` is a CLI hook only and does not change catalog or history
    semantics. ``archive_sources`` writes a local diagnostic generation under
    the model cache and does not change catalog bytes, history, or publication.
    One command shares one refresh generation across providers. A provider
    failure leaves the other providers' archived sources in that generation.

    The clock is read once before the first provider. That stamp is
    ``generated_at``, the ``verified_at`` of facts fetched in this run, and
    the ``observed_at`` of history events from this run. Source
    ``retrieved_at`` still comes from the clock at each GET. A provider that
    does not commit keeps its existing facts and provenance. Records retained
    after an optional-source failure are not re-stamped. ``freshness.status``
    remains the adapter's as-of label and does not claim those older facts
    were verified at this stamp.
    """
    store = store or FileCatalogStore()
    clock_fn = clock or utc_now_iso
    shared_stamp = clock_fn()
    archive = (
        open_source_archive(store.models_dir, refresh_stamp=shared_stamp, keep=archive_keep)
        if archive_sources
        else None
    )
    total = len(SUPPORTED_PROVIDERS)
    results: list[CatalogUpdateResult] = []
    try:
        for index, provider_id in enumerate(SUPPORTED_PROVIDERS, start=1):
            if progress is not None:
                progress("start", provider_id, index, total, None)
            transport = None
            if progress is not None:
                report = progress

                def on_request(
                    event_provider: str = provider_id,
                    event_index: int = index,
                    emit: UpdateAllProgress = report,
                ) -> None:
                    emit("request", event_provider, event_index, total, None)

                transport = _ProgressHttpTransport(
                    _default_transport(provider_id, clock_fn),
                    on_request,
                )
            try:
                result = update_provider_catalog(
                    provider_id,
                    transport=transport,
                    store=store,
                    clock=clock_fn,
                    stamp=shared_stamp,
                    archive=archive,
                    finalize_archive=False,
                )
            except OSError as exc:
                result = _failed(provider_id, store, store.exists(), str(exc))
            results.append(result)
            if progress is not None:
                progress("finish", provider_id, index, total, result)
        return tuple(results)
    finally:
        close_source_archive(archive)


def update_provider_catalog(
    provider_id: str,
    *,
    transport: HttpTransport | None = None,
    store: FileCatalogStore | None = None,
    clock: Clock | None = None,
    stamp: str | None = None,
    archive: SourceArchive | None = None,
    archive_sources: bool = False,
    archive_keep: int = DEFAULT_ARCHIVE_KEEP,
    finalize_archive: bool = True,
) -> CatalogUpdateResult:
    """Replace one provider's records in the overlay. Other providers are kept.

    Required source failures do not write. An existing live catalog that
    cannot be loaded is not treated as missing: the file is left unchanged.
    Best-effort model-page failures still commit a catalog that includes
    notices. ``stamp``, when set, is the verification time for facts fetched
    in this refresh, for catalog ``generated_at``, and for history
    ``observed_at``. A single-provider command omits it and reads ``clock``
    once instead. Source retrieval still uses ``clock`` at each GET. Records
    retained after an optional-source failure keep their existing provenance.
    ``archive`` or ``archive_sources`` records captured documents outside the
    catalog. Archive errors do not roll back a commit and do not add a notice.
    """
    store = store or FileCatalogStore()
    clock_fn = clock or utc_now_iso
    resolved_stamp = clock_fn() if stamp is None else stamp
    created_archive = False
    active_archive = archive
    try:
        if active_archive is None and archive_sources:
            active_archive = open_source_archive(
                store.models_dir,
                refresh_stamp=resolved_stamp,
                keep=archive_keep,
            )
            created_archive = True
        return _commit_provider_catalog(
            provider_id,
            transport=transport,
            store=store,
            clock_fn=clock_fn,
            resolved_stamp=resolved_stamp,
            diagnostics=active_archive,
        )
    finally:
        if created_archive and finalize_archive:
            close_source_archive(active_archive)


def _commit_provider_catalog(
    provider_id: str,
    *,
    transport: HttpTransport | None,
    store: FileCatalogStore,
    clock_fn: Clock,
    resolved_stamp: str,
    diagnostics: SourceArchive | None,
) -> CatalogUpdateResult:
    try:
        with CatalogHistoryTransaction(store.root) as transaction:
            previous, history = transaction.load_pair()
            previous_existed = previous is not None
            if transport is None:
                transport = _default_transport(provider_id, clock_fn)
            try:
                refresh = _refresh_provider(
                    provider_id,
                    transport,
                    generated_at=resolved_stamp,
                    verified_at=resolved_stamp,
                    diagnostics=diagnostics,
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
            incoming = _retain_records_for_failed_sources(
                previous,
                refresh.catalog,
                provider_id=provider_id,
                notices=refresh.notices,
            )
            if provider_id == PROVIDER_FIREWORKS:
                incoming = _retain_absent_fireworks_records(
                    previous,
                    incoming,
                    capture_complete=not refresh.partial,
                )
            lifecycle_changes = diff_provider_lifecycle(
                previous,
                incoming,
                provider_id=provider_id,
            )
            merged = merge_provider_refresh(previous, incoming, provider_id=provider_id)
            try:
                events = history_events_from_changes(
                    lifecycle_changes,
                    provider_id=provider_id,
                    observed_at=resolved_stamp,
                )
            except (CatalogValidationError, HistorySchemaError) as exc:
                return _failed(
                    provider_id, store, previous_existed, str(exc), notices=refresh.notices
                )
            transaction.commit(
                merged,
                type(history)(events=history.events + events),
            )
    except CatalogHistoryTransactionError as exc:
        return _failed(provider_id, store, store.exists(), str(exc))
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
        lifecycle_changes=lifecycle_changes,
    )


def _catalog_with_notices(
    catalog: ModelCatalog, notices: tuple[CatalogNotice, ...]
) -> ModelCatalog:
    if catalog.notices == notices:
        return catalog
    return build_catalog(
        models=catalog.models,
        generated_at=catalog.generated_at,
        freshness=catalog.freshness,
        notices=notices,
        version=catalog.version,
    )


def _save_catalog_notices(store: FileCatalogStore, catalog: ModelCatalog) -> ModelCatalog | None:
    """Persist a notice-only catalog rewrite. Model records are already committed."""
    try:
        store.save(catalog)
    except (CatalogStoreError, OSError):
        return None
    return catalog


def _record_history_write_failure(
    store: FileCatalogStore,
    catalog: ModelCatalog,
    *,
    provider_id: str,
    observed_at: str,
) -> tuple[ModelCatalog, tuple[CatalogNotice, ...]]:
    """Persist the failed update. Do not drop markers left by other updates."""
    code = history_write_failure_code(provider_id, observed_at)
    notices = catalog.notices
    if not any(notice.code == code for notice in notices):
        notices = notices + (
            CatalogNotice(
                code=code,
                message=(
                    f"{HISTORY_WRITE_FAILED_SUMMARY} {HISTORY_WRITE_FAILED_DETAIL} "
                    f"The unresolved update is {provider_id} at {observed_at}."
                ),
            ),
        )
    updated = _catalog_with_notices(catalog, notices)
    saved = _save_catalog_notices(store, updated)
    if saved is None:
        return catalog, notices
    return saved, saved.notices


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


def _failed_optional_source_ids(notices: tuple[CatalogNotice, ...]) -> frozenset[str]:
    """Source identities Samyak could not inspect on this refresh."""
    failed: set[str] = set()
    for notice in notices:
        for prefix in _FAILED_SOURCE_PREFIXES:
            if notice.code.startswith(prefix):
                source_id = notice.code.removeprefix(prefix)
                if source_id:
                    failed.add(source_id)
                break
    return frozenset(failed)


def _retain_records_for_failed_sources(
    previous: ModelCatalog | None,
    incoming: ModelCatalog,
    *,
    provider_id: str,
    notices: tuple[CatalogNotice, ...],
) -> ModelCatalog:
    """Keep last-known records when their evidence source could not be inspected.

    A failed optional source is not evidence that a model left the documented
    catalog. Retention is keyed to the sources that established documented
    presence, not to every fact provenance entry. Disappearances whose
    presence sources remain healthy stay removals. Preserved records keep
    their existing facts and provenance; they are not re-verified.
    """
    if previous is None:
        return incoming
    failed = _failed_optional_source_ids(notices)
    if not failed:
        return incoming
    incoming_ids = {item.samyak_id for item in incoming.models if item.provider_id == provider_id}
    retained = tuple(
        record
        for record in previous.models
        if record.provider_id == provider_id
        and record.samyak_id not in incoming_ids
        and _presence_depends_on_failed_sources(record, failed)
    )
    if not retained:
        return incoming
    return build_catalog(
        models=incoming.models + retained,
        generated_at=incoming.generated_at,
        freshness=incoming.freshness,
        notices=incoming.notices,
        version=incoming.version,
    )


def _retain_absent_fireworks_records(
    previous: ModelCatalog | None,
    incoming: ModelCatalog,
    *,
    capture_complete: bool,
) -> ModelCatalog:
    """Retain Fireworks identities omitted by a later capture.

    The documented Fireworks subset has no authoritative deletion feed. A
    complete capture can therefore establish only that a former identity was
    not observed, never that it was retired or replaced. A partial capture is
    weaker still and leaves the previous disposition unchanged.
    """
    if previous is None:
        return incoming
    incoming_ids = {
        item.samyak_id for item in incoming.models if item.provider_id == PROVIDER_FIREWORKS
    }
    retained: list[ModelRecord] = []
    for record in previous.models:
        if record.provider_id != PROVIDER_FIREWORKS or record.samyak_id in incoming_ids:
            continue
        retained.append(
            replace(
                record,
                evidence_disposition=(
                    EvidenceDisposition.NOT_OBSERVED_IN_CURRENT_CAPTURE
                    if capture_complete
                    else record.evidence_disposition
                ),
            )
        )
    if not retained:
        return incoming
    return build_catalog(
        models=incoming.models + tuple(retained),
        generated_at=incoming.generated_at,
        freshness=incoming.freshness,
        notices=incoming.notices,
        version=incoming.version,
    )


def _presence_depends_on_failed_sources(record: ModelRecord, failed: frozenset[str]) -> bool:
    """True when continued presence cannot be confirmed without a failed source."""
    presence = _presence_source_ids(record)
    return bool(presence) and presence <= failed


def _presence_source_ids(record: ModelRecord) -> frozenset[str]:
    """Sources that established documented catalog membership for ``record``.

    Optional model pages contribute facts but do not, by themselves, keep a
    model present when a healthy inventory source omitted it. When the only
    observed sources are optional pages, those pages *are* the presence
    evidence.
    """
    observed = derived_observed_sources(record)
    inventory = frozenset(
        source_id for source_id in observed if not _is_optional_model_page_source(source_id)
    )
    return inventory if inventory else frozenset(observed)


def _is_optional_model_page_source(source_id: str) -> bool:
    return "-docs-model-page:" in source_id


def _refresh_provider(
    provider_id: str,
    transport: HttpTransport,
    *,
    generated_at: str,
    verified_at: str,
    diagnostics: SourceArchive | None = None,
):
    if provider_id == PROVIDER_OPENAI:
        return refresh_openai_catalog(
            transport,
            generated_at=generated_at,
            verified_at=verified_at,
            overlay=CatalogOverlay.USER_CACHE,
            diagnostics=diagnostics,
        )
    if provider_id == PROVIDER_ANTHROPIC:
        return refresh_anthropic_catalog(
            transport,
            generated_at=generated_at,
            verified_at=verified_at,
            overlay=CatalogOverlay.USER_CACHE,
            diagnostics=diagnostics,
        )
    if provider_id == PROVIDER_GOOGLE:
        return refresh_google_catalog(
            transport,
            generated_at=generated_at,
            verified_at=verified_at,
            overlay=CatalogOverlay.USER_CACHE,
            diagnostics=diagnostics,
        )
    if provider_id == PROVIDER_FIREWORKS:
        return refresh_fireworks_catalog(
            transport,
            generated_at=generated_at,
            verified_at=verified_at,
            overlay=CatalogOverlay.USER_CACHE,
            diagnostics=diagnostics,
        )
    if provider_id == PROVIDER_TOGETHER:
        return refresh_together_catalog(
            transport,
            generated_at=generated_at,
            verified_at=verified_at,
            overlay=CatalogOverlay.USER_CACHE,
            diagnostics=diagnostics,
        )
    raise ModelCatalogError(f"unknown provider {provider_id!r}")


def _default_transport(provider_id: str, clock: Clock) -> HttpTransport:
    if provider_id == PROVIDER_OPENAI:
        return openai_https_transport(clock=clock)
    if provider_id == PROVIDER_ANTHROPIC:
        return anthropic_https_transport(clock=clock)
    if provider_id == PROVIDER_GOOGLE:
        return google_https_transport(clock=clock)
    if provider_id == PROVIDER_FIREWORKS:
        return fireworks_https_transport(clock=clock)
    if provider_id == PROVIDER_TOGETHER:
        return together_https_transport(clock=clock)
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
    """Return the serving source that owns a catalog notice, if determinable.

    OpenAI's original unprefixed ``PARTIAL_MODEL_PAGES`` code is preserved.
    Newer notices use ``{CODE}:{provider_id}``. Source-prefixed fetch/parse
    codes are attributed from ``{provider_id}-docs-``.

    History-write markers name a provider, but they are not refresh notices.
    Replacing that provider's records must not drop them.
    """
    if is_history_write_failure_code(code):
        return None
    if code == _PARTIAL_NOTICE_PREFIX:
        return PROVIDER_OPENAI
    if ":" in code:
        slug = code.split(":", 1)[1].split(":", 1)[0]
        if slug in SUPPORTED_PROVIDERS:
            return slug
    return _source_provider(code)


def _source_provider(source_id: str) -> str | None:
    for provider_id in SUPPORTED_PROVIDERS:
        token = f"{provider_id}{_SOURCE_ID_PREFIX}"
        if source_id.startswith(token) or token in source_id:
            return provider_id
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
