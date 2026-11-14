"""Read-only index over the local Model Intelligence overlay.

Does not refresh, write, or fetch. Overlay uniqueness is catalog evidence, not
source-established provider identity.
"""

from __future__ import annotations

from dataclasses import dataclass

from samyak.app.models import (
    NOTICE_CATALOG_ABSENT,
    NOTICE_CATALOG_SCHEMA,
    NOTICE_CATALOG_STALE,
    NOTICE_CATALOG_UNREADABLE,
    AppNotice,
    CatalogContext,
    MatchKind,
)
from samyak.app.observations import Observation, ObservationKind
from samyak.model.catalog import FreshnessStatus, ModelCatalog
from samyak.model.errors import CatalogNotFoundError, CatalogSchemaError, CatalogStoreError
from samyak.model.records import ModelRecord
from samyak.model.store import FileCatalogStore


@dataclass(frozen=True, slots=True)
class Resolution:
    """Result of resolving one model_reference against a CatalogView."""

    match_kind: MatchKind
    provider_model_id: str
    source_provider_id: str | None = None
    record: ModelRecord | None = None
    candidates: tuple[ModelRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class CatalogView:
    """In-memory lookup over ``catalog.models``. Built once per analysis."""

    catalog: ModelCatalog
    _by_samyak_id: dict[str, ModelRecord]
    _by_provider_model: dict[tuple[str, str], ModelRecord]
    _by_model_id: dict[str, tuple[ModelRecord, ...]]
    providers: frozenset[str]

    @classmethod
    def from_catalog(cls, catalog: ModelCatalog) -> CatalogView:
        by_samyak_id: dict[str, ModelRecord] = {}
        by_provider_model: dict[tuple[str, str], ModelRecord] = {}
        grouped: dict[str, list[ModelRecord]] = {}
        providers: set[str] = set()
        for record in catalog.models:
            by_samyak_id[record.samyak_id] = record
            by_provider_model[(record.provider_id, record.provider_model_id)] = record
            grouped.setdefault(record.provider_model_id, []).append(record)
            providers.add(record.provider_id)
        by_model_id = {
            model_id: tuple(sorted(rows, key=lambda item: item.samyak_id))
            for model_id, rows in grouped.items()
        }
        return cls(
            catalog=catalog,
            _by_samyak_id=by_samyak_id,
            _by_provider_model=by_provider_model,
            _by_model_id=by_model_id,
            providers=frozenset(providers),
        )

    def freshness_dict(self) -> dict[str, str]:
        freshness = self.catalog.freshness
        return {
            "status": freshness.status.value,
            "generated_at": freshness.generated_at,
            "overlay": freshness.overlay.value,
        }

    def resolve(self, observation: Observation) -> Resolution | None:
        """Resolve a model_reference. Other observation kinds return None."""
        if observation.kind is not ObservationKind.MODEL_REFERENCE:
            return None
        model_id = observation.location.detected_value
        if not model_id:
            return None
        provider_id = observation.provider_id
        if provider_id is not None:
            if provider_id not in self.providers:
                return Resolution(
                    match_kind=MatchKind.PROVIDER_NOT_IN_CATALOG,
                    provider_model_id=model_id,
                    source_provider_id=provider_id,
                )
            record = self._by_provider_model.get((provider_id, model_id))
            if record is None:
                return Resolution(
                    match_kind=MatchKind.NOT_IN_CATALOG,
                    provider_model_id=model_id,
                    source_provider_id=provider_id,
                )
            return Resolution(
                match_kind=MatchKind.EXACT_IDENTITY,
                provider_model_id=model_id,
                source_provider_id=provider_id,
                record=record,
            )
        matches = self._by_model_id.get(model_id, ())
        if not matches:
            return Resolution(
                match_kind=MatchKind.NOT_IN_CATALOG,
                provider_model_id=model_id,
            )
        if len(matches) == 1:
            return Resolution(
                match_kind=MatchKind.MODEL_ID_UNIQUE,
                provider_model_id=model_id,
                candidates=matches,
            )
        return Resolution(
            match_kind=MatchKind.MODEL_ID_AMBIGUOUS,
            provider_model_id=model_id,
            candidates=matches,
        )


def load_catalog_view(
    store: FileCatalogStore | None = None,
) -> tuple[CatalogView | None, CatalogContext, tuple[AppNotice, ...]]:
    """Load the overlay read-only. Missing or unreadable catalogs are notices."""
    catalog_store = store if store is not None else FileCatalogStore()
    try:
        catalog = catalog_store.load()
    except CatalogNotFoundError:
        return (
            None,
            CatalogContext(present=False),
            (
                AppNotice(
                    code=NOTICE_CATALOG_ABSENT,
                    message=(
                        "No local Model Intelligence catalog overlay was found. "
                        "Lifecycle rules were not applied. Run `samyak model update` "
                        "to create one; this inspection does not refresh the catalog."
                    ),
                ),
            ),
        )
    except CatalogSchemaError:
        return (
            None,
            CatalogContext(present=False),
            (
                AppNotice(
                    code=NOTICE_CATALOG_SCHEMA,
                    message=(
                        "The local Model Intelligence catalog overlay uses an "
                        "unsupported schema. Lifecycle rules were not applied."
                    ),
                ),
            ),
        )
    except CatalogStoreError:
        return (
            None,
            CatalogContext(present=False),
            (
                AppNotice(
                    code=NOTICE_CATALOG_UNREADABLE,
                    message=(
                        "The local Model Intelligence catalog overlay could not "
                        "be read. Lifecycle rules were not applied."
                    ),
                ),
            ),
        )
    view = CatalogView.from_catalog(catalog)
    notices: list[AppNotice] = []
    if catalog.freshness.status is FreshnessStatus.STALE:
        notices.append(
            AppNotice(
                code=NOTICE_CATALOG_STALE,
                message=(
                    "The local Model Intelligence catalog overlay is labeled "
                    "stale. Lifecycle facts were used as stored."
                ),
            )
        )
    context = CatalogContext(
        present=True,
        freshness_status=catalog.freshness.status.value,
        generated_at=catalog.freshness.generated_at,
        overlay=catalog.freshness.overlay.value,
        providers=tuple(item.id for item in catalog.providers),
    )
    return view, context, tuple(notices)
