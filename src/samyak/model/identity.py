"""Canonical model identity.

Identity is ``(provider_id, provider_model_id)``. ``samyak_id`` is derived and
stable. Aliases are separate records; they do not replace canonical identity.

Provider ids are validated slugs, not free-form HTML text. Adapters currently
exist for ``openai`` and ``anthropic``; identity remains an open slug so a
closed provider enum is not required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from samyak.model.errors import CatalogValidationError

PROVIDER_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


class IdentityKind(StrEnum):
    """Whether this record is a concrete model/snapshot or a routing alias."""

    CANONICAL = "canonical"
    ALIAS = "alias"


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Stable identity for one provider-published model id."""

    provider_id: str
    provider_model_id: str

    def __post_init__(self) -> None:
        validate_provider_id(self.provider_id)
        validate_provider_model_id(self.provider_model_id)

    @property
    def samyak_id(self) -> str:
        return samyak_id(self.provider_id, self.provider_model_id)

    def to_dict(self) -> dict[str, str]:
        return {
            "provider_id": self.provider_id,
            "provider_model_id": self.provider_model_id,
            "samyak_id": self.samyak_id,
        }


def samyak_id(provider_id: str, provider_model_id: str) -> str:
    """Return the deterministic Samyak lookup key."""
    return f"{provider_id}:{provider_model_id}"


def parse_samyak_id(value: str) -> ModelIdentity:
    """Parse ``provider:model`` splitting on the first colon only."""
    if not isinstance(value, str) or not value:
        raise CatalogValidationError("samyak_id must be a non-empty string")
    if ":" not in value:
        raise CatalogValidationError("samyak_id must be 'provider_id:provider_model_id'")
    provider_id, provider_model_id = value.split(":", 1)
    return ModelIdentity(provider_id=provider_id, provider_model_id=provider_model_id)


def validate_provider_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise CatalogValidationError("provider_id must be a non-empty string")
    if not PROVIDER_ID_PATTERN.fullmatch(value):
        raise CatalogValidationError(
            "provider_id must be a lowercase slug matching [a-z][a-z0-9_]*"
        )
    return value


def validate_provider_model_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise CatalogValidationError("provider_model_id must be a non-empty string")
    if value != value.strip():
        raise CatalogValidationError(
            "provider_model_id must not have leading or trailing whitespace"
        )
    if _CONTROL_CHARS.search(value):
        raise CatalogValidationError("provider_model_id must not contain control characters")
    return value
