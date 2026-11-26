"""Rules for retaining verified specifications no longer published by a provider."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Final

from samyak.model.catalog import ModelCatalog, build_catalog
from samyak.model.facts import Fact, FactStatus
from samyak.model.records import EvidenceDisposition, ModelRecord

# These are the only facts that may be retained as historical specifications.
# Keep this allowlist shared with export freshness validation.
HISTORICAL_SPECIFICATION_FACT_NAMES: Final[tuple[str, ...]] = (
    "context_window",
    "max_input_tokens",
    "max_output_tokens",
    "input_modalities",
    "output_modalities",
    "tool_calling",
    "structured_output",
)
HISTORICAL_SPECIFICATION_PROVIDER_ID: Final = "google"


def reconcile_historical_specifications(
    previous: ModelCatalog | None,
    incoming: ModelCatalog,
    *,
    provider_id: str,
) -> ModelCatalog:
    """Retain exact prior specification facts for eligible absent documentation.

    The caller opts a provider into this behavior. A current fact is replaced
    only when it is explicitly ``not_verified``; an unknown, known, or
    conflicting current observation is evidence and therefore wins.
    """
    if previous is None:
        return incoming
    previous_by_id = {
        record.samyak_id: record for record in previous.models if record.provider_id == provider_id
    }
    reconciled: list[ModelRecord] = []
    changed = False
    for record in incoming.models:
        prior = previous_by_id.get(record.samyak_id)
        if (
            record.provider_id != provider_id
            or record.evidence_disposition is not EvidenceDisposition.HISTORICAL_REFERENCE_ONLY
            or prior is None
        ):
            reconciled.append(record)
            continue
        retained: dict[str, Fact[object]] = {}
        for name in HISTORICAL_SPECIFICATION_FACT_NAMES:
            current = getattr(record, name)
            historical = getattr(prior, name)
            if current.status is FactStatus.NOT_VERIFIED and historical.status in {
                FactStatus.KNOWN,
                FactStatus.CONFLICT,
            }:
                retained[name] = historical
        if retained:
            record = replace(record, **retained)
            changed = True
        reconciled.append(record)
    if not changed:
        return incoming
    return build_catalog(
        models=tuple(reconciled),
        generated_at=incoming.generated_at,
        freshness=incoming.freshness,
        notices=incoming.notices,
        version=incoming.version,
    )


def historical_specification_is_publishable(
    record: ModelRecord,
    field_name: str,
    fact: Fact[object],
    *,
    generated_at: str,
) -> bool:
    """Whether old specification provenance is eligible for export freshness.

    This narrow Google-only exception applies solely to the shared historical
    specification allowlist. It is deliberately fact-scoped: lifecycle and
    operational evidence still must belong to the current snapshot. Catalog
    and provenance schema validation require timezone-aware timestamps before
    this comparison; incompatible, missing, or future timestamps fail closed.
    """
    if (
        record.provider_id != HISTORICAL_SPECIFICATION_PROVIDER_ID
        or record.evidence_disposition is not EvidenceDisposition.HISTORICAL_REFERENCE_ONLY
        or field_name not in HISTORICAL_SPECIFICATION_FACT_NAMES
        or fact.status not in {FactStatus.KNOWN, FactStatus.CONFLICT}
    ):
        return False
    try:
        snapshot = datetime.fromisoformat(generated_at)
        timestamps = _verified_timestamps(fact)
        return bool(timestamps) and all(
            timestamp is not None and datetime.fromisoformat(timestamp) <= snapshot
            for timestamp in timestamps
        )
    except (TypeError, ValueError):
        return False


def _verified_timestamps(fact: Fact[object]) -> tuple[str | None, ...]:
    if fact.status is FactStatus.CONFLICT:
        return tuple(claim.provenance.verified_at for claim in fact.claims)
    return () if fact.provenance is None else (fact.provenance.verified_at,)
