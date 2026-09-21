"""Update-time lifecycle diffs over existing ModelRecord facts.

The catalog remains the source of truth for current documented state. This
module compares one provider's previous overlay records with the incoming
refresh and returns in-memory change events. Events are not persisted, are
not catalog schema, and are not a second source of truth.

Provenance-only differences are ignored. Unknown and not_verified are both
unestablished facts for this comparison; they are not collapsed into a
lifecycle state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from samyak.model.catalog import ModelCatalog
from samyak.model.facts import Fact, FactStatus, LifecycleState
from samyak.model.identity import ModelIdentity
from samyak.model.records import ModelRecord

_UNESTABLISHED = frozenset({FactStatus.UNKNOWN, FactStatus.NOT_VERIFIED})


class LifecycleChangeType(StrEnum):
    """Documented catalog events discovered during one provider update."""

    ADDED = "added"
    REMOVED = "removed"
    LIFECYCLE_CHANGED = "lifecycle_changed"
    DEPRECATION_DATE_CHANGED = "deprecation_date_changed"
    RETIREMENT_DATE_CHANGED = "retirement_date_changed"
    REPLACEMENT_CHANGED = "replacement_changed"


_CHANGE_TYPE_ORDER = {
    LifecycleChangeType.ADDED: 0,
    LifecycleChangeType.REMOVED: 1,
    LifecycleChangeType.LIFECYCLE_CHANGED: 2,
    LifecycleChangeType.DEPRECATION_DATE_CHANGED: 3,
    LifecycleChangeType.RETIREMENT_DATE_CHANGED: 4,
    LifecycleChangeType.REPLACEMENT_CHANGED: 5,
}


@dataclass(frozen=True, slots=True)
class LifecycleChange:
    """One update-time lifecycle fact change for a single ``samyak_id``.

    ``previous`` is absent for added models. ``current`` is absent for models
    removed from the documented catalog. The records are snapshots of current
    documented state, not a persisted event log.
    """

    change_type: LifecycleChangeType
    samyak_id: str
    previous: ModelRecord | None = None
    current: ModelRecord | None = None


def diff_provider_lifecycle(
    previous: ModelCatalog | None,
    incoming: ModelCatalog,
    *,
    provider_id: str,
) -> tuple[LifecycleChange, ...]:
    """Compare one provider's previous overlay with its incoming refresh.

    No previous overlay, and a first refresh of this provider into an existing
    mixed overlay, produce no events. Other providers in ``previous`` are
    ignored: disappearing from another provider's subset is not a removal.
    """
    if previous is None:
        return ()
    previous_records = _records_for_provider(previous, provider_id)
    incoming_records = _records_for_provider(incoming, provider_id)
    if not previous_records:
        return ()

    changes: list[LifecycleChange] = []
    previous_ids = set(previous_records)
    incoming_ids = set(incoming_records)
    for samyak_id in incoming_ids - previous_ids:
        changes.append(
            LifecycleChange(
                change_type=LifecycleChangeType.ADDED,
                samyak_id=samyak_id,
                current=incoming_records[samyak_id],
            )
        )
    for samyak_id in previous_ids - incoming_ids:
        changes.append(
            LifecycleChange(
                change_type=LifecycleChangeType.REMOVED,
                samyak_id=samyak_id,
                previous=previous_records[samyak_id],
            )
        )
    for samyak_id in previous_ids & incoming_ids:
        before = previous_records[samyak_id]
        after = incoming_records[samyak_id]
        changes.extend(_record_fact_changes(samyak_id, before, after))
    return tuple(sorted(changes, key=_change_sort_key))


def facts_semantically_equal(left: Fact, right: Fact) -> bool:
    """Return whether two facts encode the same documented meaning.

    Status and established values (including competing conflict claims) are
    compared. Provenance, timestamps, and unknown-versus-not_verified are not
    treated as lifecycle intelligence events.
    """
    return _semantic_key(left) == _semantic_key(right)


def _records_for_provider(catalog: ModelCatalog, provider_id: str) -> dict[str, ModelRecord]:
    return {
        record.samyak_id: record for record in catalog.models if record.provider_id == provider_id
    }


def _record_fact_changes(
    samyak_id: str, previous: ModelRecord, current: ModelRecord
) -> tuple[LifecycleChange, ...]:
    comparisons = (
        (LifecycleChangeType.LIFECYCLE_CHANGED, previous.lifecycle, current.lifecycle),
        (
            LifecycleChangeType.DEPRECATION_DATE_CHANGED,
            previous.deprecated_at,
            current.deprecated_at,
        ),
        (
            LifecycleChangeType.RETIREMENT_DATE_CHANGED,
            previous.retirement_at,
            current.retirement_at,
        ),
        (
            LifecycleChangeType.REPLACEMENT_CHANGED,
            previous.replacement,
            current.replacement,
        ),
    )
    return tuple(
        LifecycleChange(
            change_type=change_type,
            samyak_id=samyak_id,
            previous=previous,
            current=current,
        )
        for change_type, before, after in comparisons
        if not facts_semantically_equal(before, after)
    )


def _semantic_key(fact: Fact) -> tuple[object, ...]:
    if fact.status in _UNESTABLISHED:
        return ("unestablished",)
    if fact.status is FactStatus.KNOWN:
        return ("known", _value_key(fact.value))
    if fact.status is FactStatus.CONFLICT:
        return ("conflict", frozenset(_value_key(claim.value) for claim in fact.claims))
    return (fact.status.value,)


def _value_key(value: object) -> object:
    if isinstance(value, tuple) and value and isinstance(value[0], ModelIdentity):
        return tuple(item.samyak_id for item in value)
    if isinstance(value, LifecycleState):
        return value.value
    return value


def _change_sort_key(change: LifecycleChange) -> tuple[str, int]:
    return (change.samyak_id, _CHANGE_TYPE_ORDER[change.change_type])


def is_unestablished(fact: Fact) -> bool:
    """Return True when the fact has no established documented value."""
    return fact.status in _UNESTABLISHED
