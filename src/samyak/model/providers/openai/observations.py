"""OpenAI-native observations. These are not canonical ModelRecord values."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from samyak.model.facts import SourceKind
from samyak.model.identity import IdentityKind
from samyak.model.providers.openai.errors import OpenAIParseError
from samyak.model.providers.openai.sources import PROVIDER_ID, OpenAISourceType


class OpenAILifecycle(StrEnum):
    """Lifecycle as stated by OpenAI documentation.

    Unstated lifecycle is ``None`` on the observation, not a lifecycle value.
    There is no legacy state.
    """

    ACTIVE = "active"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


@dataclass(frozen=True, slots=True)
class OpenAISourceRef:
    """Provenance fields copied from a captured source. No generated UUIDs."""

    source_id: str
    source_kind: SourceKind
    source_url: str
    content_hash: str
    retrieved_at: str
    source_type: OpenAISourceType

    @property
    def provider(self) -> str:
        return PROVIDER_ID


@dataclass(frozen=True, slots=True)
class OpenAIModelObservation:
    """Facts extracted from one official OpenAI document for one model id.

    ``None`` on an optional field means the source did not state it. Boolean
    ``False`` is an explicit documented negative, never a missing field.

    OpenAI "Supported features" lists are positive-only; omission is not false.
    The Endpoints table's Support column is an explicit Supported / Not
    supported flag and is the only current negative representation used for
    ``api_access``.
    """

    provider_model_id: str
    source: OpenAISourceRef
    identity_kind: IdentityKind = IdentityKind.CANONICAL
    display_name: str | None = None
    listed_on_index: bool = False
    aliases: tuple[str, ...] | None = None
    resolves_to: str | None = None
    family: str | None = None
    lifecycle: OpenAILifecycle | None = None
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
            raise OpenAIParseError("OpenAI observation is missing a model identifier")
        if self.provider_model_id != self.provider_model_id.strip():
            raise OpenAIParseError("OpenAI model identifier has surrounding whitespace")
        if self.resolves_to is not None and self.identity_kind is not IdentityKind.ALIAS:
            raise OpenAIParseError(
                "OpenAI alias observation must declare identity_kind=alias when resolves_to is set"
            )
        if (
            self.deprecated_at is not None
            and self.retirement_at is not None
            and self.retirement_at < self.deprecated_at
        ):
            raise OpenAIParseError(
                f"OpenAI documentation for {self.provider_model_id} has "
                "retirement_at earlier than deprecated_at"
            )
