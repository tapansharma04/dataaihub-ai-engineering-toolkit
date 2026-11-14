"""TOML extractor (stdlib ``tomllib`` only).

Line numbers are omitted: ``tomllib`` does not provide a location map.
"""

from __future__ import annotations

import tomllib
from typing import Any

from samyak.app.models import EvidenceType, SourceLocation
from samyak.app.observations import Observation, ObservationKind, sort_observations

MODEL_SLOT_KEYS = frozenset({"model", "model_id", "model_name"})


def extract_toml_observations(relative_path: str, source: bytes) -> tuple[Observation, ...]:
    """Parse TOML bytes and return model-slot observations without invented lines."""
    payload = tomllib.loads(source.decode("utf-8-sig"))
    found: list[Observation] = []
    _walk(relative_path, payload, "", found)
    return sort_observations(found)


def _walk(path: str, value: object, prefix: str, found: list[Observation]) -> None:
    if isinstance(value, dict):
        mapping: dict[Any, Any] = value
        for key, child in mapping.items():
            if not isinstance(key, str):
                continue
            child_path = key if not prefix else f"{prefix}.{key}"
            if key in MODEL_SLOT_KEYS and isinstance(child, str):
                found.append(
                    Observation(
                        kind=ObservationKind.MODEL_REFERENCE,
                        location=SourceLocation(
                            path=path,
                            evidence_type=EvidenceType.TOML_VALUE,
                            detected_value=child,
                            config_key=child_path,
                        ),
                    )
                )
                continue
            _walk(path, child, child_path, found)
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            _walk(path, child, child_path, found)
