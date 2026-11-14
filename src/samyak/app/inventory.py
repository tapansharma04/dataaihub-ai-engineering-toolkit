"""Build the application model inventory from bound observations and AI operations.

Catalog lookup runs only for bound ``model_reference`` rows. Unresolved and
dynamic operations are never treated as model ids.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from samyak.app.catalog_view import CatalogView, Resolution
from samyak.app.models import (
    AiOperationKind,
    AppInventory,
    ApplicationIdentityState,
    CatalogEvidence,
    InventoryOperation,
    MatchKind,
    ModelDependency,
    UnboundReason,
    UnresolvedOperation,
    dependency_sort_key,
    inventory_operation_sort_key,
    unresolved_operation_sort_key,
)
from samyak.app.observations import Observation, ObservationKind
from samyak.model.facts import FactStatus, LifecycleState
from samyak.model.records import ModelRecord


@dataclass
class _Group:
    state: ApplicationIdentityState
    resolution: Resolution | None
    observations: list[Observation] = field(default_factory=list)


def build_inventory(
    observations: tuple[Observation, ...],
    view: CatalogView | None,
) -> AppInventory:
    """Group bound model uses and list unresolved/dynamic operations."""
    groups: dict[tuple[object, ...], _Group] = {}
    unresolved: list[UnresolvedOperation] = []
    freshness = None if view is None else view.freshness_dict()

    for observation in observations:
        if observation.kind is ObservationKind.AI_OPERATION:
            unresolved.append(_unresolved_operation(observation))
            continue
        if observation.kind is not ObservationKind.MODEL_REFERENCE:
            continue
        resolution = None if view is None else view.resolve(observation)
        key, state = _dependency_key(observation, resolution)
        group = groups.get(key)
        if group is None:
            groups[key] = _Group(state=state, resolution=resolution, observations=[observation])
        else:
            group.observations.append(observation)

    dependencies = [_dependency_from_group(group, freshness) for group in groups.values()]
    return AppInventory(
        dependencies=tuple(sorted(dependencies, key=dependency_sort_key)),
        unresolved_operations=tuple(sorted(unresolved, key=unresolved_operation_sort_key)),
    )


def _unresolved_operation(observation: Observation) -> UnresolvedOperation:
    location = observation.location
    identity = observation.identity_state or ApplicationIdentityState.UNRESOLVED
    if identity not in {
        ApplicationIdentityState.UNRESOLVED,
        ApplicationIdentityState.DYNAMIC,
    }:
        identity = ApplicationIdentityState.UNRESOLVED
    reason = observation.unbound_reason
    if reason is None:
        reason = UnboundReason.UNRESOLVED_EXPRESSION
    return UnresolvedOperation(
        application_identity=identity,
        unbound_reason=reason,
        expression=observation.expression or "",
        path=location.path,
        evidence_type=location.evidence_type,
        line_start=location.line_start,
        line_end=location.line_end,
        slot=location.config_key,
        symbol=location.symbol,
        operation_kind=observation.operation_kind or AiOperationKind.MODEL_INFERENCE,
    )


def _dependency_key(
    observation: Observation,
    resolution: Resolution | None,
) -> tuple[tuple[object, ...], ApplicationIdentityState]:
    state = _application_identity(observation, resolution)
    detected = observation.location.detected_value
    if state is ApplicationIdentityState.ESTABLISHED:
        if resolution is not None and resolution.record is not None:
            return (state.value, resolution.record.samyak_id), state
        provider = observation.provider_id or ""
        return (state.value, provider, detected), state
    return (state.value, detected), state


def _application_identity(
    observation: Observation,
    resolution: Resolution | None,
) -> ApplicationIdentityState:
    if observation.provider_id is not None:
        return ApplicationIdentityState.ESTABLISHED
    if resolution is None:
        return ApplicationIdentityState.BOUND
    kind = resolution.match_kind
    if kind is MatchKind.MODEL_ID_UNIQUE:
        return ApplicationIdentityState.CANDIDATE
    if kind is MatchKind.MODEL_ID_AMBIGUOUS:
        return ApplicationIdentityState.AMBIGUOUS
    return ApplicationIdentityState.BOUND


def _dependency_from_group(
    group: _Group,
    freshness: dict[str, str] | None,
) -> ModelDependency:
    first = group.observations[0]
    detected = first.location.detected_value
    resolution = group.resolution
    operations = tuple(
        sorted(
            (_bound_operation(item) for item in group.observations),
            key=inventory_operation_sort_key,
        )
    )
    provider_id = first.provider_id
    samyak_id: str | None = None
    match_kind = None if resolution is None else resolution.match_kind
    if (
        group.state is ApplicationIdentityState.ESTABLISHED
        and resolution is not None
        and resolution.record is not None
    ):
        samyak_id = resolution.record.samyak_id
    return ModelDependency(
        application_identity=group.state,
        detected_model_id=detected,
        operations=operations,
        provider_id=provider_id,
        samyak_id=samyak_id,
        match_kind=match_kind,
        catalog_evidence=_catalog_citation(resolution, freshness),
    )


def _bound_operation(observation: Observation) -> InventoryOperation:
    location = observation.location
    return InventoryOperation(
        path=location.path,
        evidence_type=location.evidence_type,
        detected_model_id=location.detected_value,
        line_start=location.line_start,
        line_end=location.line_end,
        slot=location.config_key,
        symbol=location.symbol,
        expression=observation.expression,
        resolution_method=observation.resolution_method,
        resolution_source=observation.resolution_source,
    )


def _catalog_citation(
    resolution: Resolution | None,
    freshness: dict[str, str] | None,
) -> CatalogEvidence | None:
    if resolution is None:
        return None
    fresh = None if freshness is None else dict(freshness)
    kind = resolution.match_kind
    detected = resolution.provider_model_id
    if kind is MatchKind.EXACT_IDENTITY and resolution.record is not None:
        record = resolution.record
        lifecycle_value = _known_lifecycle_value(record)
        return CatalogEvidence(
            match_kind=kind,
            provider_model_id=detected,
            samyak_id=record.samyak_id,
            provider_id=resolution.source_provider_id,
            identity_kind=record.identity_kind.value,
            lifecycle_status=record.lifecycle.status.value,
            lifecycle_value=lifecycle_value,
            catalog_freshness=fresh,
            fact_confidence=_fact_confidence(record),
            source_id=_lifecycle_source_id(record),
            source_url=_lifecycle_source_url(record),
        )
    if kind is MatchKind.MODEL_ID_UNIQUE:
        candidate_ids = tuple(item.samyak_id for item in resolution.candidates)
        return CatalogEvidence(
            match_kind=kind,
            provider_model_id=detected,
            catalog_freshness=fresh,
            candidate_samyak_ids=candidate_ids,
        )
    if kind is MatchKind.MODEL_ID_AMBIGUOUS:
        candidate_ids = tuple(item.samyak_id for item in resolution.candidates)
        return CatalogEvidence(
            match_kind=kind,
            provider_model_id=detected,
            catalog_freshness=fresh,
            candidate_samyak_ids=candidate_ids,
        )
    return CatalogEvidence(
        match_kind=kind,
        provider_model_id=detected,
        provider_id=resolution.source_provider_id,
        catalog_freshness=fresh,
    )


def _known_lifecycle_value(record: ModelRecord) -> str | None:
    fact = record.lifecycle
    if fact.status is not FactStatus.KNOWN or fact.value is None:
        return None
    value = fact.value
    if isinstance(value, LifecycleState):
        return value.value
    return str(value)


def _fact_confidence(record: ModelRecord) -> str | None:
    provenance = record.lifecycle.provenance
    if provenance is None or provenance.confidence is None:
        return None
    return provenance.confidence.value


def _lifecycle_source_id(record: ModelRecord) -> str | None:
    provenance = record.lifecycle.provenance
    if provenance is None:
        return None
    return provenance.source_id


def _lifecycle_source_url(record: ModelRecord) -> str | None:
    provenance = record.lifecycle.provenance
    if provenance is None:
        return None
    return provenance.source_url
