"""Deterministic context-window audit over a ModelCatalog. No network access.

This helper classifies catalog facts. It does not infer missing token counts.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from samyak.model.catalog import ModelCatalog
from samyak.model.facts import FactStatus
from samyak.model.records import ModelRecord

KNOWN = "known"
UNKNOWN_NO_EVIDENCE = "unknown_source_does_not_establish_context"
UNKNOWN_INPUT_OUTPUT_ONLY = "unknown_input_output_stated_not_context"
NOT_VERIFIED = "not_verified_source_coverage"
CONFLICT = "conflict"
OTHER = "other"


@dataclass(frozen=True, slots=True)
class ContextWindowRow:
    samyak_id: str
    provider_id: str
    provider_model_id: str
    status: FactStatus
    tokens: int | None
    kind: str | None
    classification: str
    provenance_source_id: str | None


@dataclass(frozen=True, slots=True)
class ProviderContextWindowAudit:
    provider_id: str
    total: int
    known: int
    unknown: int
    not_verified: int
    conflict: int
    other: int
    rows: tuple[ContextWindowRow, ...]

    @property
    def non_known(self) -> tuple[ContextWindowRow, ...]:
        return tuple(row for row in self.rows if row.status is not FactStatus.KNOWN)


def classify_context_window(record: ModelRecord) -> str:
    """Explain why a context-window fact is known or not. Never fills a value."""
    fact = record.context_window
    if fact.status is FactStatus.KNOWN:
        return KNOWN
    if fact.status is FactStatus.CONFLICT:
        return CONFLICT
    if fact.status is FactStatus.NOT_VERIFIED:
        return NOT_VERIFIED
    if fact.status is FactStatus.UNKNOWN:
        if (
            record.max_input_tokens.status is FactStatus.KNOWN
            or record.max_output_tokens.status is FactStatus.KNOWN
        ):
            return UNKNOWN_INPUT_OUTPUT_ONLY
        return UNKNOWN_NO_EVIDENCE
    return OTHER


def audit_context_windows(
    catalog: ModelCatalog, *, provider_id: str | None = None
) -> tuple[ProviderContextWindowAudit, ...]:
    grouped: dict[str, list[ModelRecord]] = {}
    for record in catalog.models:
        if provider_id is not None and record.provider_id != provider_id:
            continue
        grouped.setdefault(record.provider_id, []).append(record)
    audits: list[ProviderContextWindowAudit] = []
    for pid in sorted(grouped):
        rows = tuple(_row(item) for item in grouped[pid])
        counts = Counter(row.status for row in rows)
        counted = {
            FactStatus.KNOWN,
            FactStatus.UNKNOWN,
            FactStatus.NOT_VERIFIED,
            FactStatus.CONFLICT,
        }
        audits.append(
            ProviderContextWindowAudit(
                provider_id=pid,
                total=len(rows),
                known=counts[FactStatus.KNOWN],
                unknown=counts[FactStatus.UNKNOWN],
                not_verified=counts[FactStatus.NOT_VERIFIED],
                conflict=counts[FactStatus.CONFLICT],
                other=sum(count for status, count in counts.items() if status not in counted),
                rows=rows,
            )
        )
    return tuple(audits)


def _row(record: ModelRecord) -> ContextWindowRow:
    fact = record.context_window
    tokens: int | None = None
    kind: str | None = None
    source_id: str | None = None
    if fact.status is FactStatus.KNOWN and fact.value is not None:
        tokens = fact.value.tokens
        kind = fact.value.kind.value
        if fact.provenance is not None:
            source_id = fact.provenance.source_id
    elif fact.status is FactStatus.CONFLICT and fact.claims:
        source_id = ",".join(sorted({claim.provenance.source_id for claim in fact.claims}))
    return ContextWindowRow(
        samyak_id=record.samyak_id,
        provider_id=record.provider_id,
        provider_model_id=record.provider_model_id,
        status=fact.status,
        tokens=tokens,
        kind=kind,
        classification=classify_context_window(record),
        provenance_source_id=source_id,
    )
