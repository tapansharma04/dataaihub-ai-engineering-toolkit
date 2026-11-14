"""Tiny Application Readiness observation IR.

Not a program analysis framework. Bound model references stay distinct from
unresolved/dynamic AI operations so catalog lookup never sees an invented id.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from samyak.app.models import (
    AiOperationKind,
    ApplicationIdentityState,
    ResolutionMethod,
    SourceLocation,
    UnboundReason,
)


class ObservationKind(StrEnum):
    """Closed observation kinds. ``model_reference`` is bound literals only."""

    MODEL_REFERENCE = "model_reference"
    PROVIDER_SDK_IMPORT = "provider_sdk_import"
    AI_OPERATION = "ai_operation"


@dataclass(frozen=True, slots=True)
class Observation:
    """One extracted fact from source or structured config.

    ``provider_id`` is source evidence only (same-file SDK import). It is never
    filled from a model catalog. ``unbound_reason`` / ``expression`` apply only
    to ``ai_operation`` rows; they are not model ids. ``operation_kind`` is set
    only on AI operations (currently ``model_inference``).
    """

    kind: ObservationKind
    location: SourceLocation
    provider_id: str | None = None
    identity_state: ApplicationIdentityState | None = None
    unbound_reason: UnboundReason | None = None
    expression: str | None = None
    operation_kind: AiOperationKind | None = None
    resolution_method: ResolutionMethod | None = None
    resolution_source: SourceLocation | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": self.kind.value,
            "provider_id": self.provider_id,
            "location": self.location.to_dict(),
        }
        if self.identity_state is not None:
            payload["identity_state"] = self.identity_state.value
        if self.unbound_reason is not None:
            payload["unbound_reason"] = self.unbound_reason.value
        if self.expression is not None:
            payload["expression"] = self.expression
        if self.operation_kind is not None:
            payload["operation_kind"] = self.operation_kind.value
        if self.resolution_method is not None:
            payload["resolution_method"] = self.resolution_method.value
        if self.resolution_source is not None:
            payload["resolution_source"] = self.resolution_source.to_dict()
        return payload


def observation_sort_key(item: Observation) -> tuple[object, ...]:
    """Deterministic order: path, then line when present, then identity."""
    location = item.location
    return (
        location.path,
        location.line_start is None,
        location.line_start if location.line_start is not None else 0,
        location.line_end if location.line_end is not None else 0,
        item.kind.value,
        location.config_key or "",
        location.symbol or "",
        location.detected_value,
        item.provider_id or "",
        location.evidence_type.value,
        item.identity_state.value if item.identity_state is not None else "",
        item.unbound_reason.value if item.unbound_reason is not None else "",
        item.expression or "",
        item.operation_kind.value if item.operation_kind is not None else "",
        item.resolution_method.value if item.resolution_method is not None else "",
        "" if item.resolution_source is None else item.resolution_source.path,
        0
        if item.resolution_source is None or item.resolution_source.line_start is None
        else item.resolution_source.line_start,
    )


def sort_observations(
    items: list[Observation] | tuple[Observation, ...],
) -> tuple[Observation, ...]:
    return tuple(sorted(items, key=observation_sort_key))
