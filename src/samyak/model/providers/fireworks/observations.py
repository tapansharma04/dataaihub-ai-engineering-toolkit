"""Fireworks-native observations. These are not canonical ModelRecord values."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from samyak.model.facts import SourceKind
from samyak.model.identity import IdentityKind
from samyak.model.providers.fireworks.errors import FireworksParseError
from samyak.model.providers.fireworks.sources import PROVIDER_ID, FireworksSourceType


class FireworksLifecycle(StrEnum):
    """Lifecycle as stated for a Fireworks serving offering.

    Fireworks "legacy" BERT/sentence-transformers labels are catalogue
    categories, not this enum. Unstated lifecycle is ``None``.
    """

    ACTIVE = "active"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


@dataclass(frozen=True, slots=True)
class FireworksSourceRef:
    """Provenance fields copied from a captured source. No generated UUIDs."""

    source_id: str
    source_kind: SourceKind
    source_url: str
    content_hash: str
    retrieved_at: str
    source_type: FireworksSourceType

    @property
    def provider(self) -> str:
        return PROVIDER_ID


@dataclass(frozen=True, slots=True)
class FireworksModelObservation:
    """Facts extracted from one official Fireworks document for one model id.

    ``None`` on an optional field means the source did not state it. Boolean
    ``False`` is an explicit documented negative, never a missing field.

    Customer deployment paths and origin-provider identities are not
    represented here. Display names are never used as API identifiers.
    """

    provider_model_id: str
    source: FireworksSourceRef
    identity_kind: IdentityKind = IdentityKind.CANONICAL
    display_name: str | None = None
    documented_serverless: bool = False
    aliases: tuple[str, ...] | None = None
    resolves_to: str | None = None
    family: str | None = None
    lifecycle: FireworksLifecycle | None = None
    deprecation_listing: str | None = None
    deprecated_at: str | None = None
    retirement_at: str | None = None
    replacements: tuple[str, ...] | None = None
    context_window_tokens: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    input_modalities: tuple[str, ...] | None = None
    output_modalities: tuple[str, ...] | None = None
    tool_calling: bool | None = None
    structured_output: bool | None = None
    api_access: bool | None = None
    availability_scope: str | None = None
    regions: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.provider_model_id, str) or not self.provider_model_id.strip():
            raise FireworksParseError("Fireworks observation is missing a model identifier")
        if self.provider_model_id != self.provider_model_id.strip():
            raise FireworksParseError("Fireworks model identifier has surrounding whitespace")
        if self.resolves_to is not None and self.identity_kind is not IdentityKind.ALIAS:
            raise FireworksParseError(
                "Fireworks alias observation must declare identity_kind=alias "
                "when resolves_to is set"
            )
        if (
            self.deprecated_at is not None
            and self.retirement_at is not None
            and self.retirement_at < self.deprecated_at
        ):
            raise FireworksParseError(
                f"Fireworks documentation for {self.provider_model_id} has "
                "retirement_at earlier than deprecated_at"
            )
