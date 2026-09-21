"""Documented Model Intelligence lifecycle history.

Current ``ModelRecord`` values remain the source of truth for present state.
This module is an observation log of previous update-time lifecycle diffs.
It is not catalog schema, not a second catalog, and not a reconstruction of
current lifecycle state.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from samyak.model.catalog import CAPABILITY_NAME, PRODUCT_NAME
from samyak.model.errors import CatalogDecodeError, CatalogValidationError, HistorySchemaError
from samyak.model.facts import Fact, FactStatus, LifecycleState, validate_timestamp
from samyak.model.identity import ModelIdentity, parse_samyak_id, validate_provider_id
from samyak.model.lifecycle import LifecycleChange, LifecycleChangeType
from samyak.model.records import ModelRecord

HISTORY_SCHEMA_VERSION = 1
SUPPORTED_HISTORY_SCHEMA_VERSIONS = frozenset({HISTORY_SCHEMA_VERSION})

_HISTORY_FIELDS = frozenset(
    {
        "product",
        "capability",
        "lifecycle_history_schema_version",
        "events",
    }
)
_EVENT_FIELDS = frozenset(
    {
        "provider_id",
        "samyak_id",
        "observed_at",
        "change_type",
        "previous_lifecycle",
        "current_lifecycle",
        "previous_deprecated_at",
        "current_deprecated_at",
        "previous_retirement_at",
        "current_retirement_at",
        "previous_replacement",
        "current_replacement",
    }
)
_SNAPSHOT_FIELDS = frozenset({"status", "value", "claims"})


@dataclass(frozen=True, slots=True)
class LifecycleFactSnapshot:
    """A persisted lifecycle fact as observed at update time.

    Competing conflict values are kept on ``claims``. Provenance is omitted:
    history explains what changed, not source ranking.
    """

    status: FactStatus
    value: object | None = None
    claims: tuple[object, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "value": self.value,
            "claims": list(self.claims),
        }


@dataclass(frozen=True, slots=True)
class LifecycleHistoryEvent:
    """One documented lifecycle change observed during a catalog update."""

    provider_id: str
    samyak_id: str
    observed_at: str
    change_type: LifecycleChangeType
    previous_lifecycle: LifecycleFactSnapshot | None = None
    current_lifecycle: LifecycleFactSnapshot | None = None
    previous_deprecated_at: LifecycleFactSnapshot | None = None
    current_deprecated_at: LifecycleFactSnapshot | None = None
    previous_retirement_at: LifecycleFactSnapshot | None = None
    current_retirement_at: LifecycleFactSnapshot | None = None
    previous_replacement: LifecycleFactSnapshot | None = None
    current_replacement: LifecycleFactSnapshot | None = None

    def __post_init__(self) -> None:
        validate_provider_id(self.provider_id)
        identity = parse_samyak_id(self.samyak_id)
        if identity.provider_id != self.provider_id:
            raise CatalogValidationError("lifecycle history samyak_id does not match provider_id")
        validate_timestamp("observed_at", self.observed_at)
        if not isinstance(self.change_type, LifecycleChangeType):
            raise CatalogValidationError("lifecycle history change_type is not a known change type")

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "samyak_id": self.samyak_id,
            "observed_at": self.observed_at,
            "change_type": self.change_type.value,
            "previous_lifecycle": _snapshot_to_json(self.previous_lifecycle),
            "current_lifecycle": _snapshot_to_json(self.current_lifecycle),
            "previous_deprecated_at": _snapshot_to_json(self.previous_deprecated_at),
            "current_deprecated_at": _snapshot_to_json(self.current_deprecated_at),
            "previous_retirement_at": _snapshot_to_json(self.previous_retirement_at),
            "current_retirement_at": _snapshot_to_json(self.current_retirement_at),
            "previous_replacement": _snapshot_to_json(self.previous_replacement),
            "current_replacement": _snapshot_to_json(self.current_replacement),
        }


@dataclass(frozen=True, slots=True)
class LifecycleHistory:
    """Versioned log of documented lifecycle changes. Not current catalog state."""

    events: tuple[LifecycleHistoryEvent, ...] = ()
    product: str = PRODUCT_NAME
    capability: str = CAPABILITY_NAME
    lifecycle_history_schema_version: int = HISTORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.product != PRODUCT_NAME:
            raise CatalogValidationError("product must be 'samyak'")
        if self.capability != CAPABILITY_NAME:
            raise CatalogValidationError("capability must be 'model'")
        if self.lifecycle_history_schema_version not in SUPPORTED_HISTORY_SCHEMA_VERSIONS:
            raise HistorySchemaError(self.lifecycle_history_schema_version)
        if not isinstance(self.events, tuple):
            raise CatalogValidationError("lifecycle history events must be a tuple")
        if any(not isinstance(item, LifecycleHistoryEvent) for item in self.events):
            raise CatalogValidationError("lifecycle history events are invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "product": self.product,
            "capability": self.capability,
            "lifecycle_history_schema_version": self.lifecycle_history_schema_version,
            "events": [item.to_dict() for item in self.events],
        }


def history_events_from_changes(
    changes: Sequence[LifecycleChange],
    *,
    provider_id: str,
    observed_at: str,
) -> tuple[LifecycleHistoryEvent, ...]:
    """Project in-memory update diffs into persistable history events."""
    return tuple(
        _event_from_change(change, provider_id=provider_id, observed_at=observed_at)
        for change in changes
    )


def filter_history_events(
    events: Sequence[LifecycleHistoryEvent],
    *,
    provider_id: str | None = None,
    samyak_id: str | None = None,
) -> tuple[LifecycleHistoryEvent, ...]:
    """Return events matching optional provider and model identity filters."""
    selected = tuple(events)
    if provider_id is not None:
        validate_provider_id(provider_id)
        selected = tuple(item for item in selected if item.provider_id == provider_id)
    if samyak_id is not None:
        parse_samyak_id(samyak_id)
        selected = tuple(item for item in selected if item.samyak_id == samyak_id)
    return selected


def history_from_dict(payload: object) -> LifecycleHistory:
    """Reconstruct history. Fail closed; do not repair."""
    if not isinstance(payload, dict):
        raise CatalogDecodeError("lifecycle history JSON must be an object")
    extra = set(payload) - _HISTORY_FIELDS
    if extra:
        raise CatalogDecodeError(f"lifecycle history has unexpected fields: {sorted(extra)}")
    schema_version = payload.get("lifecycle_history_schema_version")
    if schema_version not in SUPPORTED_HISTORY_SCHEMA_VERSIONS:
        raise HistorySchemaError(schema_version)
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise CatalogDecodeError("lifecycle history events must be a list")
    events = tuple(_event_from_dict(item) for item in raw_events)
    return LifecycleHistory(
        product=_require_str("product", payload.get("product")),
        capability=_require_str("capability", payload.get("capability")),
        lifecycle_history_schema_version=schema_version,
        events=events,
    )


def render_lifecycle_history(events: Sequence[LifecycleHistoryEvent]) -> str:
    """Render documented history for the CLI. Evidence only; no predictions."""
    if not events:
        return "No documented lifecycle history.\n"
    lines = [
        "Model lifecycle history",
        "------------------------------",
        "",
    ]
    for index, event in enumerate(events):
        if index:
            lines.append("")
        lines.append(_calendar_date(event.observed_at))
        lines.append(f"  {event.samyak_id}")
        lines.extend(f"  {line}" for line in _event_detail_lines(event))
    return "\n".join(lines) + "\n"


def _event_from_change(
    change: LifecycleChange,
    *,
    provider_id: str,
    observed_at: str,
) -> LifecycleHistoryEvent:
    previous = change.previous
    current = change.current
    kwargs: dict[str, LifecycleFactSnapshot | None] = {
        "previous_lifecycle": None,
        "current_lifecycle": None,
        "previous_deprecated_at": None,
        "current_deprecated_at": None,
        "previous_retirement_at": None,
        "current_retirement_at": None,
        "previous_replacement": None,
        "current_replacement": None,
    }
    if change.change_type is LifecycleChangeType.ADDED:
        kwargs.update(_record_snapshots(current, prefix="current"))
    elif change.change_type is LifecycleChangeType.REMOVED:
        kwargs.update(_record_snapshots(previous, prefix="previous"))
    elif change.change_type is LifecycleChangeType.LIFECYCLE_CHANGED:
        kwargs["previous_lifecycle"] = snapshot_from_fact(
            None if previous is None else previous.lifecycle
        )
        kwargs["current_lifecycle"] = snapshot_from_fact(
            None if current is None else current.lifecycle
        )
        kwargs["current_retirement_at"] = snapshot_from_fact(
            None if current is None else current.retirement_at
        )
        kwargs["current_replacement"] = snapshot_from_fact(
            None if current is None else current.replacement
        )
    elif change.change_type is LifecycleChangeType.DEPRECATION_DATE_CHANGED:
        kwargs["previous_deprecated_at"] = snapshot_from_fact(
            None if previous is None else previous.deprecated_at
        )
        kwargs["current_deprecated_at"] = snapshot_from_fact(
            None if current is None else current.deprecated_at
        )
    elif change.change_type is LifecycleChangeType.RETIREMENT_DATE_CHANGED:
        kwargs["previous_retirement_at"] = snapshot_from_fact(
            None if previous is None else previous.retirement_at
        )
        kwargs["current_retirement_at"] = snapshot_from_fact(
            None if current is None else current.retirement_at
        )
    else:
        kwargs["previous_replacement"] = snapshot_from_fact(
            None if previous is None else previous.replacement
        )
        kwargs["current_replacement"] = snapshot_from_fact(
            None if current is None else current.replacement
        )
    return LifecycleHistoryEvent(
        provider_id=provider_id,
        samyak_id=change.samyak_id,
        observed_at=observed_at,
        change_type=change.change_type,
        **kwargs,
    )


def _record_snapshots(
    record: ModelRecord | None, *, prefix: str
) -> dict[str, LifecycleFactSnapshot | None]:
    if record is None:
        return {
            f"{prefix}_lifecycle": None,
            f"{prefix}_deprecated_at": None,
            f"{prefix}_retirement_at": None,
            f"{prefix}_replacement": None,
        }
    return {
        f"{prefix}_lifecycle": snapshot_from_fact(record.lifecycle),
        f"{prefix}_deprecated_at": snapshot_from_fact(record.deprecated_at),
        f"{prefix}_retirement_at": snapshot_from_fact(record.retirement_at),
        f"{prefix}_replacement": snapshot_from_fact(record.replacement),
    }


def snapshot_from_fact(fact: Fact | None) -> LifecycleFactSnapshot | None:
    if fact is None:
        return None
    if fact.status is FactStatus.KNOWN:
        return LifecycleFactSnapshot(status=FactStatus.KNOWN, value=_encode_value(fact.value))
    if fact.status is FactStatus.CONFLICT:
        return LifecycleFactSnapshot(
            status=FactStatus.CONFLICT,
            claims=tuple(_encode_value(claim.value) for claim in fact.claims),
        )
    return LifecycleFactSnapshot(status=fact.status)


def _encode_value(value: object) -> object:
    if isinstance(value, LifecycleState):
        return value.value
    if isinstance(value, ModelIdentity):
        return value.samyak_id
    if isinstance(value, tuple) and value and isinstance(value[0], ModelIdentity):
        return [item.samyak_id for item in value]
    return value


def _event_from_dict(payload: object) -> LifecycleHistoryEvent:
    if not isinstance(payload, dict):
        raise CatalogDecodeError("lifecycle history event must be an object")
    extra = set(payload) - _EVENT_FIELDS
    if extra:
        raise CatalogDecodeError(f"lifecycle history event has unexpected fields: {sorted(extra)}")
    change_type = payload.get("change_type")
    try:
        parsed_type = LifecycleChangeType(change_type)
    except ValueError as exc:
        raise CatalogDecodeError(
            "lifecycle history change_type is not a known change type"
        ) from exc
    return LifecycleHistoryEvent(
        provider_id=_require_str("provider_id", payload.get("provider_id")),
        samyak_id=_require_str("samyak_id", payload.get("samyak_id")),
        observed_at=_require_str("observed_at", payload.get("observed_at")),
        change_type=parsed_type,
        previous_lifecycle=_snapshot_from_json(payload.get("previous_lifecycle")),
        current_lifecycle=_snapshot_from_json(payload.get("current_lifecycle")),
        previous_deprecated_at=_snapshot_from_json(payload.get("previous_deprecated_at")),
        current_deprecated_at=_snapshot_from_json(payload.get("current_deprecated_at")),
        previous_retirement_at=_snapshot_from_json(payload.get("previous_retirement_at")),
        current_retirement_at=_snapshot_from_json(payload.get("current_retirement_at")),
        previous_replacement=_snapshot_from_json(payload.get("previous_replacement")),
        current_replacement=_snapshot_from_json(payload.get("current_replacement")),
    )


def _snapshot_from_json(payload: object) -> LifecycleFactSnapshot | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise CatalogDecodeError("lifecycle fact snapshot must be an object")
    extra = set(payload) - _SNAPSHOT_FIELDS
    if extra:
        raise CatalogDecodeError(f"lifecycle fact snapshot has unexpected fields: {sorted(extra)}")
    status_raw = payload.get("status")
    try:
        status = FactStatus(status_raw)
    except ValueError as exc:
        raise CatalogDecodeError("lifecycle fact snapshot status is not a known status") from exc
    claims = payload.get("claims")
    if not isinstance(claims, list):
        raise CatalogDecodeError("lifecycle fact snapshot claims must be a list")
    return LifecycleFactSnapshot(status=status, value=payload.get("value"), claims=tuple(claims))


def _snapshot_to_json(snapshot: LifecycleFactSnapshot | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    return snapshot.to_dict()


def _event_detail_lines(event: LifecycleHistoryEvent) -> list[str]:
    heading = _event_heading(event)
    lines = [heading]
    if event.change_type is LifecycleChangeType.LIFECYCLE_CHANGED:
        previous = _snapshot_label(event.previous_lifecycle)
        current = _snapshot_label(event.current_lifecycle)
        lines.append(f"{previous} → {current}")
        retirement = _format_snapshot(event.current_retirement_at)
        if retirement is not None:
            lines.append(f"Retirement: {retirement}")
        replacement = _format_snapshot(event.current_replacement)
        if replacement is not None:
            lines.append(f"Replacement: {replacement}")
        return lines
    if event.change_type is LifecycleChangeType.DEPRECATION_DATE_CHANGED:
        lines.append(
            "Deprecation: "
            f"{_format_snapshot(event.previous_deprecated_at) or 'not established'} → "
            f"{_format_snapshot(event.current_deprecated_at) or 'not established'}"
        )
        return lines
    if event.change_type is LifecycleChangeType.RETIREMENT_DATE_CHANGED:
        lines.append(
            "Retirement: "
            f"{_format_snapshot(event.previous_retirement_at) or 'not established'} → "
            f"{_format_snapshot(event.current_retirement_at) or 'not established'}"
        )
        return lines
    if event.change_type is LifecycleChangeType.REPLACEMENT_CHANGED:
        lines.append(
            "Replacement: "
            f"{_format_snapshot(event.previous_replacement) or 'not established'} → "
            f"{_format_snapshot(event.current_replacement) or 'not established'}"
        )
        return lines
    if event.change_type is LifecycleChangeType.REMOVED:
        return lines
    retirement = _format_snapshot(event.current_retirement_at)
    if retirement is not None:
        lines.append(f"Retirement: {retirement}")
    replacement = _format_snapshot(event.current_replacement)
    if replacement is not None:
        lines.append(f"Replacement: {replacement}")
    return lines


def _event_heading(event: LifecycleHistoryEvent) -> str:
    if event.change_type is LifecycleChangeType.ADDED:
        return "ADDED"
    if event.change_type is LifecycleChangeType.REMOVED:
        return "REMOVED FROM DOCUMENTED CATALOG"
    if event.change_type is LifecycleChangeType.DEPRECATION_DATE_CHANGED:
        return "DEPRECATION DATE CHANGED"
    if event.change_type is LifecycleChangeType.RETIREMENT_DATE_CHANGED:
        return "RETIREMENT DATE CHANGED"
    if event.change_type is LifecycleChangeType.REPLACEMENT_CHANGED:
        return "REPLACEMENT CHANGED"
    label = _snapshot_label(event.current_lifecycle)
    if label in {"DEPRECATED", "RETIRED", "LEGACY", "ACTIVE", "CONFLICT"}:
        return label
    return "LIFECYCLE CHANGED"


def _snapshot_label(snapshot: LifecycleFactSnapshot | None) -> str:
    if snapshot is None or snapshot.status in {FactStatus.UNKNOWN, FactStatus.NOT_VERIFIED}:
        return "NOT ESTABLISHED"
    if snapshot.status is FactStatus.CONFLICT:
        return "CONFLICT"
    if snapshot.status is FactStatus.NOT_APPLICABLE:
        return "NOT APPLICABLE"
    if snapshot.status is FactStatus.KNOWN and isinstance(snapshot.value, str):
        try:
            return LifecycleState(snapshot.value).value.upper()
        except ValueError:
            return snapshot.value.upper()
    return "LIFECYCLE CHANGED"


def _format_snapshot(snapshot: LifecycleFactSnapshot | None) -> str | None:
    if snapshot is None:
        return None
    if snapshot.status in {FactStatus.UNKNOWN, FactStatus.NOT_VERIFIED}:
        return None
    if snapshot.status is FactStatus.KNOWN:
        return _format_snapshot_value(snapshot.value)
    if snapshot.status is FactStatus.CONFLICT:
        values = ", ".join(sorted(_format_snapshot_value(item) for item in snapshot.claims))
        return f"conflict ({values})"
    return snapshot.status.value


def _format_snapshot_value(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _calendar_date(observed_at: str) -> str:
    return observed_at[:10]


def _require_str(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise CatalogDecodeError(f"{name} must be a non-empty string")
    return value
