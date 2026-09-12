"""Normalize Anthropic observations into canonical ModelRecord values."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from samyak.model.errors import CatalogValidationError
from samyak.model.facts import (
    AvailabilityScope,
    Confidence,
    ContextWindow,
    ContextWindowKind,
    Fact,
    FactClaim,
    FactStatus,
    LifecycleState,
    Modality,
    Provenance,
    validate_timestamp,
)
from samyak.model.identity import IdentityKind, ModelIdentity
from samyak.model.providers.anthropic.errors import AnthropicAdapterError
from samyak.model.providers.anthropic.observations import (
    AnthropicLifecycle,
    AnthropicModelObservation,
)
from samyak.model.providers.anthropic.sources import PROVIDER_ID, AnthropicSourceType
from samyak.model.records import ModelRecord

_SOURCE_RANK = {
    AnthropicSourceType.DEPRECATIONS: 3,
    AnthropicSourceType.MODEL_PAGE: 2,
    AnthropicSourceType.MODELS_INDEX: 1,
}

_LIFECYCLE_FIELDS = frozenset({"lifecycle", "deprecated_at", "retirement_at", "replacement"})
_CAPABILITY_FIELDS = frozenset(
    {
        "context_window",
        "max_input_tokens",
        "max_output_tokens",
        "input_modalities",
        "output_modalities",
        "tool_calling",
        "structured_output",
        "api_access",
        "availability_scope",
        "regions",
    }
)
_MODALITY = {
    "text": Modality.TEXT,
    "image": Modality.IMAGE,
    "audio": Modality.AUDIO,
    "video": Modality.VIDEO,
}
_SCOPE = {
    "global": AvailabilityScope.GLOBAL,
    "regional": AvailabilityScope.REGIONAL,
    "account": AvailabilityScope.ACCOUNT,
}
_LIFECYCLE = {
    AnthropicLifecycle.ACTIVE: LifecycleState.ACTIVE,
    AnthropicLifecycle.LEGACY: LifecycleState.LEGACY,
    AnthropicLifecycle.DEPRECATED: LifecycleState.DEPRECATED,
    AnthropicLifecycle.RETIRED: LifecycleState.RETIRED,
}


def normalize_anthropic_observations(
    observations: Sequence[AnthropicModelObservation],
    *,
    verified_at: str,
) -> tuple[ModelRecord, ...]:
    """Map Anthropic observations to canonical records. Does not invent facts."""
    if not observations:
        raise AnthropicAdapterError("Anthropic observations are missing")
    try:
        validate_timestamp("verified_at", verified_at)
    except CatalogValidationError as exc:
        raise AnthropicAdapterError(
            "verified_at must be a timezone-aware ISO-8601 timestamp"
        ) from exc

    grouped: dict[str, list[AnthropicModelObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.provider_model_id].append(observation)

    resolves_to = {
        model_id: _merged_resolves_to(group, verified_at) for model_id, group in grouped.items()
    }
    aliases_of: dict[str, list[str]] = defaultdict(list)
    for alias_id, target in resolves_to.items():
        if target is not None:
            aliases_of[target].append(alias_id)

    records: list[ModelRecord] = []
    try:
        for model_id, group in grouped.items():
            records.append(
                _record_for(
                    model_id,
                    group,
                    verified_at=verified_at,
                    resolves_to=resolves_to.get(model_id),
                    alias_ids=tuple(aliases_of.get(model_id, ())),
                )
            )
    except CatalogValidationError as exc:
        raise AnthropicAdapterError("Anthropic observations could not be normalized") from exc
    return tuple(records)


def _record_for(
    model_id: str,
    group: Sequence[AnthropicModelObservation],
    *,
    verified_at: str,
    resolves_to: str | None,
    alias_ids: tuple[str, ...],
) -> ModelRecord:
    identity = ModelIdentity(provider_id=PROVIDER_ID, provider_model_id=model_id)
    identity_kind = IdentityKind.ALIAS if resolves_to is not None else IdentityKind.CANONICAL
    page_present = any(item.source.source_type is AnthropicSourceType.MODEL_PAGE for item in group)
    deprecations_present = any(
        item.source.source_type is AnthropicSourceType.DEPRECATIONS for item in group
    )
    fallback = _fallback_observation(group, page_present=page_present)
    lifecycle_group = _observations_for_lifecycle_fields(group)

    display_name = _merge(group, lambda item: item.display_name, verified_at, fallback)
    family = _merge(
        group,
        lambda item: item.family,
        verified_at,
        fallback,
        evaluated=page_present,
    )
    lifecycle = _lifecycle_fact(lifecycle_group, verified_at, fallback)
    deprecated_at = _merge(
        lifecycle_group,
        lambda item: item.deprecated_at,
        verified_at,
        fallback,
        field="deprecated_at",
    )
    retirement_at = _merge(
        lifecycle_group,
        lambda item: item.retirement_at,
        verified_at,
        fallback,
        field="retirement_at",
    )
    _reject_contradictory_dates(model_id, deprecated_at, retirement_at)
    replacement = _merge(
        lifecycle_group,
        lambda item: _replacement_identities(item.replacements, identity),
        verified_at,
        fallback,
        field="replacement",
    )
    context_window = _merge(
        group,
        lambda item: _context_window(item.context_window_tokens),
        verified_at,
        fallback,
        field="context_window",
        evaluated=page_present,
    )
    max_input = _merge(
        group,
        lambda item: item.max_input_tokens,
        verified_at,
        fallback,
        field="max_input_tokens",
        evaluated=page_present,
    )
    max_output = _merge(
        group,
        lambda item: item.max_output_tokens,
        verified_at,
        fallback,
        field="max_output_tokens",
        evaluated=page_present,
    )
    input_modalities = _merge(
        group,
        lambda item: _modalities(item.input_modalities),
        verified_at,
        fallback,
        field="input_modalities",
        evaluated=page_present,
    )
    output_modalities = _merge(
        group,
        lambda item: _modalities(item.output_modalities),
        verified_at,
        fallback,
        field="output_modalities",
        evaluated=page_present,
    )
    tool_calling = _merge(
        group,
        lambda item: item.tool_calling,
        verified_at,
        fallback,
        field="tool_calling",
        evaluated=page_present,
    )
    structured_output = _merge(
        group,
        lambda item: item.structured_output,
        verified_at,
        fallback,
        field="structured_output",
        evaluated=page_present,
    )
    api_access = _merge(
        group,
        lambda item: item.api_access,
        verified_at,
        fallback,
        field="api_access",
        evaluated=page_present,
    )
    availability_scope = _merge(
        group,
        lambda item: _scope(item.availability_scope),
        verified_at,
        fallback,
        field="availability_scope",
        evaluated=page_present,
    )
    regions = _merge(
        group,
        lambda item: item.regions,
        verified_at,
        fallback,
        field="regions",
        evaluated=page_present,
    )

    aliases: Fact[tuple[str, ...]]
    if identity_kind is IdentityKind.ALIAS:
        aliases = Fact(status=FactStatus.NOT_VERIFIED)
    else:
        stated_aliases = _merge(group, lambda item: item.aliases, verified_at, fallback)
        combined: list[str] = []
        if stated_aliases.status is FactStatus.CONFLICT:
            aliases = stated_aliases
        else:
            if stated_aliases.status is FactStatus.KNOWN and stated_aliases.value:
                combined.extend(stated_aliases.value)
            combined.extend(alias_ids)
            unique = tuple(dict.fromkeys(combined))
            if unique:
                source = (
                    _alias_source(group, unique[0])
                    if stated_aliases.status is not FactStatus.KNOWN
                    else group[0]
                )
                provenance = (
                    stated_aliases.provenance
                    if stated_aliases.status is FactStatus.KNOWN and stated_aliases.provenance
                    else _provenance(source, verified_at)
                )
                aliases = Fact(status=FactStatus.KNOWN, value=unique, provenance=provenance)
            elif page_present:
                aliases = Fact(
                    status=FactStatus.UNKNOWN,
                    provenance=_provenance(fallback, verified_at),
                )
            else:
                aliases = Fact(status=FactStatus.NOT_VERIFIED)

    resolves_fact: Fact[ModelIdentity]
    if identity_kind is IdentityKind.ALIAS and resolves_to is not None:
        source = next(
            (item for item in group if item.resolves_to == resolves_to),
            group[0],
        )
        resolves_fact = Fact(
            status=FactStatus.KNOWN,
            value=ModelIdentity(provider_id=PROVIDER_ID, provider_model_id=resolves_to),
            provenance=_provenance(source, verified_at),
        )
    else:
        resolves_fact = Fact(status=FactStatus.NOT_VERIFIED)

    if not deprecations_present:
        if deprecated_at.status is FactStatus.UNKNOWN:
            deprecated_at = Fact(status=FactStatus.NOT_VERIFIED)
        if retirement_at.status is FactStatus.UNKNOWN:
            retirement_at = Fact(status=FactStatus.NOT_VERIFIED)
        if replacement.status is FactStatus.UNKNOWN:
            replacement = Fact(status=FactStatus.NOT_VERIFIED)

    return ModelRecord(
        identity=identity,
        identity_kind=identity_kind,
        display_name=display_name,
        aliases=aliases,
        resolves_to=resolves_fact,
        family=family,
        lifecycle=lifecycle,
        deprecated_at=deprecated_at,
        retirement_at=retirement_at,
        replacement=replacement,
        context_window=context_window,
        max_input_tokens=max_input,
        max_output_tokens=max_output,
        input_modalities=input_modalities,
        output_modalities=output_modalities,
        tool_calling=tool_calling,
        structured_output=structured_output,
        api_access=api_access,
        availability_scope=availability_scope,
        regions=regions,
    )


@dataclass(frozen=True, slots=True)
class _DeprecationEvent:
    observation: AnthropicModelObservation

    @property
    def listing(self) -> str | None:
        return self.observation.deprecation_listing

    @property
    def identity(self) -> tuple[object, ...]:
        item = self.observation
        return (
            item.deprecation_listing,
            item.lifecycle,
            item.deprecated_at,
            item.retirement_at,
            item.replacements,
        )


def _observations_for_lifecycle_fields(
    group: Sequence[AnthropicModelObservation],
) -> tuple[AnthropicModelObservation, ...]:
    deprecation = [
        item for item in group if item.source.source_type is AnthropicSourceType.DEPRECATIONS
    ]
    others = [
        item for item in group if item.source.source_type is not AnthropicSourceType.DEPRECATIONS
    ]
    if len(deprecation) <= 1:
        return tuple(group)
    current = _current_deprecation_observations(deprecation)
    return tuple(others) + current


def _current_deprecation_observations(
    observations: Sequence[AnthropicModelObservation],
) -> tuple[AnthropicModelObservation, ...]:
    """Select the current Anthropic deprecation event(s) for one model id.

    Anthropic's status table is the current event, including Active, Legacy,
    Deprecated, and Retired rows. History sections are past announcements.
    Current/upcoming evidence wins even after a retirement date has elapsed.
    """
    events = tuple(_DeprecationEvent(item) for item in observations)
    current = _unique_events(event for event in events if event.listing in {"current", "upcoming"})
    past = _unique_events(event for event in events if event.listing == "past")
    unlabeled = _unique_events(
        event for event in events if event.listing not in {"current", "upcoming", "past"}
    )
    if unlabeled:
        return tuple(event.observation for event in _unique_events(events))
    if current:
        return tuple(event.observation for event in current)
    if not past:
        return tuple(event.observation for event in _unique_events(events))
    latest = _unique_latest_past(past)
    if latest is None:
        return tuple(event.observation for event in past)
    return (latest.observation,)


def _unique_events(events: Iterable[_DeprecationEvent]) -> tuple[_DeprecationEvent, ...]:
    seen: set[tuple[object, ...]] = set()
    unique: list[_DeprecationEvent] = []
    for event in events:
        if event.identity in seen:
            continue
        seen.add(event.identity)
        unique.append(event)
    return tuple(unique)


def _unique_latest_past(events: Sequence[_DeprecationEvent]) -> _DeprecationEvent | None:
    announced = [event.observation.deprecated_at for event in events]
    if announced and all(value is not None for value in announced):
        latest_announced = max(value for value in announced if value is not None)
        winners = _unique_events(
            event for event in events if event.observation.deprecated_at == latest_announced
        )
        if len(winners) == 1:
            return winners[0]
        return None
    if any(value is not None for value in announced):
        return None
    retired = [event.observation.retirement_at for event in events]
    if not retired or not all(value is not None for value in retired):
        return None
    latest_retired = max(value for value in retired if value is not None)
    winners = _unique_events(
        event for event in events if event.observation.retirement_at == latest_retired
    )
    if len(winners) == 1:
        return winners[0]
    return None


def _merged_resolves_to(group: Sequence[AnthropicModelObservation], verified_at: str) -> str | None:
    fact = _merge(group, lambda item: item.resolves_to, verified_at, group[0])
    if fact.status is FactStatus.KNOWN and isinstance(fact.value, str):
        return fact.value
    return None


def _lifecycle_fact(
    group: Sequence[AnthropicModelObservation],
    verified_at: str,
    fallback: AnthropicModelObservation,
) -> Fact[LifecycleState]:
    stated = _merge(
        group,
        lambda item: _lifecycle_value(item, verified_at),
        verified_at,
        fallback,
    )
    if stated.status is not FactStatus.UNKNOWN:
        return stated
    if any(item.listed_on_index for item in group):
        index = next(item for item in group if item.listed_on_index)
        return Fact(
            status=FactStatus.KNOWN,
            value=LifecycleState.ACTIVE,
            provenance=_provenance(index, verified_at, confidence=Confidence.MEDIUM),
        )
    return Fact(status=FactStatus.UNKNOWN, provenance=_provenance(fallback, verified_at))


def _reject_contradictory_dates(
    model_id: str, deprecated_at: Fact[Any], retirement_at: Fact[Any]
) -> None:
    if deprecated_at.status is not FactStatus.KNOWN:
        return
    if retirement_at.status is not FactStatus.KNOWN:
        return
    if not isinstance(deprecated_at.value, str) or not isinstance(retirement_at.value, str):
        return
    if retirement_at.value < deprecated_at.value:
        raise AnthropicAdapterError(
            f"Anthropic documentation for {model_id} has retirement_at earlier than deprecated_at"
        )


def _lifecycle_value(item: AnthropicModelObservation, verified_at: str) -> LifecycleState | None:
    if item.lifecycle is None:
        return None
    if item.lifecycle is AnthropicLifecycle.RETIRED:
        return LifecycleState.RETIRED
    if item.retirement_at is not None and _date_on_or_before(item.retirement_at, verified_at):
        return LifecycleState.RETIRED
    return _LIFECYCLE[item.lifecycle]


def _date_on_or_before(iso_day: str, verified_at: str) -> bool:
    try:
        retirement = date.fromisoformat(iso_day)
        verified = datetime.fromisoformat(verified_at).date()
    except ValueError:
        return False
    return retirement <= verified


def _merge(
    group: Sequence[AnthropicModelObservation],
    extract: Callable[[AnthropicModelObservation], Any],
    verified_at: str,
    fallback: AnthropicModelObservation,
    *,
    field: str | None = None,
    evaluated: bool | None = None,
) -> Fact[Any]:
    stated: list[tuple[AnthropicModelObservation, Any]] = []
    for item in group:
        value = extract(item)
        if value is not None:
            stated.append((item, value))
    if not stated:
        if evaluated is False or (evaluated is None and field in _CAPABILITY_FIELDS):
            return Fact(status=FactStatus.NOT_VERIFIED)
        if field in _LIFECYCLE_FIELDS and not any(
            item.source.source_type is AnthropicSourceType.DEPRECATIONS
            or item.lifecycle is not None
            or item.deprecated_at is not None
            or item.retirement_at is not None
            or item.replacements is not None
            for item in group
        ):
            return Fact(status=FactStatus.NOT_VERIFIED)
        return Fact(status=FactStatus.UNKNOWN, provenance=_provenance(fallback, verified_at))

    top_rank = max(_SOURCE_RANK[item.source.source_type] for item, _value in stated)
    top = [
        (item, value) for item, value in stated if _SOURCE_RANK[item.source.source_type] == top_rank
    ]
    distinct: list[tuple[AnthropicModelObservation, Any]] = []
    seen: list[Any] = []
    for item, value in top:
        if value not in seen:
            seen.append(value)
            distinct.append((item, value))
    if len(distinct) == 1:
        item, value = distinct[0]
        return Fact(status=FactStatus.KNOWN, value=value, provenance=_provenance(item, verified_at))
    return Fact(
        status=FactStatus.CONFLICT,
        claims=tuple(
            FactClaim(value=value, provenance=_provenance(item, verified_at))
            for item, value in distinct
        ),
    )


def _fallback_observation(
    group: Sequence[AnthropicModelObservation], *, page_present: bool
) -> AnthropicModelObservation:
    if page_present:
        return next(
            item for item in group if item.source.source_type is AnthropicSourceType.MODEL_PAGE
        )
    ranked = sorted(group, key=lambda item: _SOURCE_RANK[item.source.source_type], reverse=True)
    return ranked[0]


def _alias_source(
    group: Sequence[AnthropicModelObservation], alias_id: str
) -> AnthropicModelObservation:
    for item in group:
        if item.aliases and alias_id in item.aliases:
            return item
        if item.provider_model_id == alias_id:
            return item
    return group[0]


def _provenance(
    item: AnthropicModelObservation,
    verified_at: str,
    *,
    confidence: Confidence | None = None,
) -> Provenance:
    if confidence is None:
        confidence = (
            Confidence.MEDIUM
            if item.source.source_type is AnthropicSourceType.MODELS_INDEX
            else Confidence.HIGH
        )
    return Provenance(
        provider=item.source.provider,
        source_id=item.source.source_id,
        source_kind=item.source.source_kind,
        source_url=item.source.source_url,
        content_hash=item.source.content_hash,
        retrieved_at=item.source.retrieved_at,
        observed_at=None,
        verified_at=verified_at,
        confidence=confidence,
    )


def _context_window(tokens: int | None) -> ContextWindow | None:
    if tokens is None:
        return None
    return ContextWindow(tokens=tokens, kind=ContextWindowKind.UNKNOWN)


def _modalities(values: tuple[str, ...] | None) -> tuple[Modality, ...] | None:
    if values is None:
        return None
    mapped: list[Modality] = []
    for item in values:
        modality = _MODALITY.get(item)
        if modality is None:
            return None
        if modality not in mapped:
            mapped.append(modality)
    return tuple(mapped) or None


def _scope(value: str | None) -> AvailabilityScope | None:
    if value is None:
        return None
    return _SCOPE.get(value)


def _replacement_identities(
    replacements: tuple[str, ...] | None, current: ModelIdentity
) -> tuple[ModelIdentity, ...] | None:
    if replacements is None:
        return None
    identities = tuple(
        ModelIdentity(provider_id=PROVIDER_ID, provider_model_id=item)
        for item in replacements
        if item != current.provider_model_id
    )
    return identities or None
