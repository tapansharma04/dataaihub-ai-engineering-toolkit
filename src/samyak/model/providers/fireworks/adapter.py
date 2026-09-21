"""Assemble a catalog from captured Fireworks documentation. No network access."""

from __future__ import annotations

from collections.abc import Sequence

from samyak.model.catalog import (
    CatalogFreshness,
    CatalogNotice,
    CatalogOverlay,
    FreshnessStatus,
    ModelCatalog,
    SourceRetrieval,
    build_catalog,
)
from samyak.model.facts import FactStatus
from samyak.model.providers.fireworks.normalize import normalize_fireworks_observations
from samyak.model.providers.fireworks.parse import parse_fireworks_sources
from samyak.model.providers.fireworks.sources import CapturedSource
from samyak.model.records import RECORD_FACT_NAMES, ModelRecord


def catalog_from_fireworks_sources(
    sources: Sequence[CapturedSource],
    *,
    generated_at: str,
    verified_at: str,
    overlay: CatalogOverlay,
    notices: Sequence[CatalogNotice] = (),
) -> ModelCatalog:
    """Parse, normalize, and validate captured Fireworks documentation.

    ``overlay`` is catalog metadata only. This function does not write a bundled
    snapshot or a cache file. Coverage is the documented public serving IDs in
    ``sources``, not the complete Fireworks Model Library.
    """
    observations = parse_fireworks_sources(sources)
    records = normalize_fireworks_observations(observations, verified_at=verified_at)
    retrieved: list[SourceRetrieval] = []
    seen: set[str] = set()
    for source in sources:
        if source.source_id in seen:
            continue
        seen.add(source.source_id)
        retrieved.append(
            SourceRetrieval(source_id=source.source_id, retrieved_at=source.retrieved_at)
        )
    return build_catalog(
        models=records,
        generated_at=generated_at,
        freshness=CatalogFreshness(
            status=FreshnessStatus.AS_OF,
            generated_at=generated_at,
            overlay=overlay,
            oldest_verified_at=_oldest_verified_at(records),
            source_retrieved_at=tuple(retrieved),
        ),
        notices=tuple(notices),
    )


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
