"""Google Gemini API-native observations. These are not canonical ModelRecord values."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from samyak.model.facts import SourceKind
from samyak.model.identity import IdentityKind
from samyak.model.providers.google.errors import GoogleParseError
from samyak.model.providers.google.sources import PROVIDER_ID, GoogleSourceType


class GoogleLifecycle(StrEnum):
    """Lifecycle as stated for the Gemini API offering.

    Google's Stable / Preview / Latest / Experimental labels are version
    channels, not lifecycle states. Unstated lifecycle is ``None`` on the
    observation. Marketing words such as "legacy" are not this enum.
    """

    ACTIVE = "active"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


@dataclass(frozen=True, slots=True)
class GoogleSourceRef:
    """Provenance fields copied from a captured source. No generated UUIDs."""

    source_id: str
    source_kind: SourceKind
    source_url: str
    content_hash: str
    retrieved_at: str
    source_type: GoogleSourceType

    @property
    def provider(self) -> str:
        return PROVIDER_ID


@dataclass(frozen=True, slots=True)
class GoogleModelObservation:
    """Facts extracted from one official Gemini API document for one model id.

    ``None`` on an optional field means the source did not state it. Boolean
    ``False`` is an explicit documented negative, never a missing field.

    Vertex / Google Cloud identifiers are not represented here. Deprecations
    table shutdown dates are earliest-possible commitments and must not appear
    as ``retirement_at``.
    """

    provider_model_id: str
    source: GoogleSourceRef
    identity_kind: IdentityKind = IdentityKind.CANONICAL
    display_name: str | None = None
    listed_on_index: bool = False
    aliases: tuple[str, ...] | None = None
    resolves_to: str | None = None
    family: str | None = None
    lifecycle: GoogleLifecycle | None = None
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
            raise GoogleParseError("Google observation is missing a model identifier")
        if self.provider_model_id != self.provider_model_id.strip():
            raise GoogleParseError("Google model identifier has surrounding whitespace")
        if self.resolves_to is not None and self.identity_kind is not IdentityKind.ALIAS:
            raise GoogleParseError(
                "Google alias observation must declare identity_kind=alias when resolves_to is set"
            )
        if (
            self.deprecated_at is not None
            and self.retirement_at is not None
            and self.retirement_at < self.deprecated_at
        ):
            raise GoogleParseError(
                f"Google documentation for {self.provider_model_id} has "
                "retirement_at earlier than deprecated_at"
            )
