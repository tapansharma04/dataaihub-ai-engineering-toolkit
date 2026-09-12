"""Anthropic-native observations. These are not canonical ModelRecord values."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from samyak.model.facts import SourceKind
from samyak.model.identity import IdentityKind
from samyak.model.providers.anthropic.errors import AnthropicParseError
from samyak.model.providers.anthropic.sources import PROVIDER_ID, AnthropicSourceType


class AnthropicLifecycle(StrEnum):
    """Lifecycle as stated by Anthropic documentation.

    Anthropic documents Active, Legacy, Deprecated, and Retired as distinct
    current states. Unstated lifecycle is ``None`` on the observation, not a
    lifecycle value. Overview lineup prose that groups "legacy models" is not
    this state.
    """

    ACTIVE = "active"
    LEGACY = "legacy"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


@dataclass(frozen=True, slots=True)
class AnthropicSourceRef:
    """Provenance fields copied from a captured source. No generated UUIDs."""

    source_id: str
    source_kind: SourceKind
    source_url: str
    content_hash: str
    retrieved_at: str
    source_type: AnthropicSourceType

    @property
    def provider(self) -> str:
        return PROVIDER_ID


@dataclass(frozen=True, slots=True)
class AnthropicModelObservation:
    """Facts extracted from one official Anthropic document for one model id.

    ``None`` on an optional field means the source did not state it. Boolean
    ``False`` is an explicit documented negative, never a missing field.

    Partner-platform identifiers (Bedrock, Google Cloud, Foundry) are not
    represented here. ``tool_calling`` maps from Anthropic's documented
    "tool use" wording. ``structured_output`` is not inferred from JSON,
    JSON mode, or tool use.
    """

    provider_model_id: str
    source: AnthropicSourceRef
    identity_kind: IdentityKind = IdentityKind.CANONICAL
    display_name: str | None = None
    listed_on_index: bool = False
    aliases: tuple[str, ...] | None = None
    resolves_to: str | None = None
    family: str | None = None
    lifecycle: AnthropicLifecycle | None = None
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
            raise AnthropicParseError("Anthropic observation is missing a model identifier")
        if self.provider_model_id != self.provider_model_id.strip():
            raise AnthropicParseError("Anthropic model identifier has surrounding whitespace")
        if self.resolves_to is not None and self.identity_kind is not IdentityKind.ALIAS:
            raise AnthropicParseError(
                "Anthropic alias observation must declare identity_kind=alias "
                "when resolves_to is set"
            )
        if (
            self.deprecated_at is not None
            and self.retirement_at is not None
            and self.retirement_at < self.deprecated_at
        ):
            raise AnthropicParseError(
                f"Anthropic documentation for {self.provider_model_id} has "
                "retirement_at earlier than deprecated_at"
            )
