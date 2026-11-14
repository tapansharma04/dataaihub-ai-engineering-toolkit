"""Read-only comparison of archived observations before source-rank selection.

This module does not select catalog facts, write the catalog, or modify the
archive. Catalog reproducibility uses each provider's production normalizer
with that provider's original normalization timestamp, not the catalog's
global generated_at. Source disagreements are classified separately after
that provider's lifecycle filter. Identical repeated evidence is collapsed.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from samyak.__version__ import __version__
from samyak.model.catalog import ModelCatalog
from samyak.model.errors import CatalogValidationError, ModelCatalogError
from samyak.model.facts import Confidence, Fact, FactStatus, LifecycleState, validate_timestamp
from samyak.model.providers.anthropic.errors import AnthropicParseError
from samyak.model.providers.anthropic.normalize import (
    _SOURCE_RANK as _ANTHROPIC_RANK,
)
from samyak.model.providers.anthropic.normalize import (
    _lifecycle_fact as _anthropic_lifecycle_fact,
)
from samyak.model.providers.anthropic.normalize import (
    _lifecycle_value as _anthropic_lifecycle_value,
)
from samyak.model.providers.anthropic.normalize import (
    _observations_for_lifecycle_fields as _anthropic_lifecycle_group,
)
from samyak.model.providers.anthropic.normalize import (
    normalize_anthropic_observations,
)
from samyak.model.providers.anthropic.parse import parse_anthropic_sources
from samyak.model.providers.anthropic.sources import (
    AnthropicSourceType,
)
from samyak.model.providers.anthropic.sources import (
    captured_markdown as capture_anthropic,
)
from samyak.model.providers.anthropic.sources import (
    source_type_for_id as anthropic_source_type,
)
from samyak.model.providers.fireworks.errors import FireworksParseError
from samyak.model.providers.fireworks.normalize import (
    _SOURCE_RANK as _FIREWORKS_RANK,
)
from samyak.model.providers.fireworks.normalize import (
    _lifecycle_fact as _fireworks_lifecycle_fact,
)
from samyak.model.providers.fireworks.normalize import (
    _lifecycle_value as _fireworks_lifecycle_value,
)
from samyak.model.providers.fireworks.normalize import (
    _observations_for_lifecycle_fields as _fireworks_lifecycle_group,
)
from samyak.model.providers.fireworks.normalize import (
    normalize_fireworks_observations,
)
from samyak.model.providers.fireworks.parse import parse_fireworks_source
from samyak.model.providers.fireworks.sources import (
    FireworksSourceType,
)
from samyak.model.providers.fireworks.sources import (
    captured_markdown as capture_fireworks,
)
from samyak.model.providers.fireworks.sources import (
    source_type_for_id as fireworks_source_type,
)
from samyak.model.providers.google.errors import GoogleParseError
from samyak.model.providers.google.normalize import (
    _SOURCE_RANK as _GOOGLE_RANK,
)
from samyak.model.providers.google.normalize import (
    _lifecycle_fact as _google_lifecycle_fact,
)
from samyak.model.providers.google.normalize import (
    _lifecycle_value as _google_lifecycle_value,
)
from samyak.model.providers.google.normalize import (
    _observations_for_lifecycle_fields as _google_lifecycle_group,
)
from samyak.model.providers.google.normalize import (
    normalize_google_observations,
)
from samyak.model.providers.google.parse import parse_google_sources
from samyak.model.providers.google.sources import (
    GoogleSourceType,
)
from samyak.model.providers.google.sources import (
    captured_markdown as capture_google,
)
from samyak.model.providers.google.sources import (
    source_type_for_id as google_source_type,
)
from samyak.model.providers.openai.errors import OpenAIParseError
from samyak.model.providers.openai.normalize import (
    _SOURCE_RANK as _OPENAI_RANK,
)
from samyak.model.providers.openai.normalize import (
    _lifecycle_fact as _openai_lifecycle_fact,
)
from samyak.model.providers.openai.normalize import (
    _lifecycle_value as _openai_lifecycle_value,
)
from samyak.model.providers.openai.normalize import (
    _observations_for_lifecycle_fields as _openai_lifecycle_group,
)
from samyak.model.providers.openai.normalize import (
    normalize_openai_observations,
)
from samyak.model.providers.openai.parse import parse_openai_sources
from samyak.model.providers.openai.sources import (
    OpenAISourceType,
)
from samyak.model.providers.openai.sources import (
    captured_markdown as capture_openai,
)
from samyak.model.providers.openai.sources import (
    source_type_for_id as openai_source_type,
)
from samyak.model.providers.together.errors import TogetherParseError
from samyak.model.providers.together.normalize import (
    _SOURCE_RANK as _TOGETHER_RANK,
)
from samyak.model.providers.together.normalize import (
    _lifecycle_fact as _together_lifecycle_fact,
)
from samyak.model.providers.together.normalize import (
    _lifecycle_value as _together_lifecycle_value,
)
from samyak.model.providers.together.normalize import (
    _observations_for_lifecycle_fields as _together_lifecycle_group,
)
from samyak.model.providers.together.normalize import (
    _observations_for_retirement_dates as _together_retirement_group,
)
from samyak.model.providers.together.normalize import (
    normalize_together_observations,
)
from samyak.model.providers.together.parse import parse_together_sources
from samyak.model.providers.together.sources import (
    TogetherSourceType,
)
from samyak.model.providers.together.sources import (
    captured_markdown as capture_together,
)
from samyak.model.providers.together.sources import (
    source_type_for_id as together_source_type,
)
from samyak.model.records import RECORD_FACT_NAMES, EvidenceDisposition
from samyak.model.source_archive import blob_path, load_manifest, refresh_id_from_stamp
from samyak.model.update import SUPPORTED_PROVIDERS

_PARSE_ERRORS = (
    AnthropicParseError,
    FireworksParseError,
    GoogleParseError,
    OpenAIParseError,
    TogetherParseError,
    UnicodeError,
    ValueError,
    TypeError,
    KeyError,
)

_KIND_ORDER = (
    "supporting",
    "override",
    "lower_rank_disagreement",
    "winning_rank_conflict",
    "excluded",
    "derived",
    "incomparable",
    "not_comparable",
    "mismatch",
)
_POLICY_FACTS = frozenset({"lifecycle", "deprecated_at", "replacement", "retirement_at"})
_REFRESH_ID_STAMP = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})T(?P<h>\d{2})-(?P<m>\d{2})-(?P<s>\d{2})"
    r"(?P<frac>\.\d+)?Z$"
)

# Observation attribute, catalog fact name. Internal flags such as
# documented_serverless are not catalog facts and are not compared.
_OBSERVATION_FACTS = (
    ("display_name", "display_name"),
    ("aliases", "aliases"),
    ("resolves_to", "resolves_to"),
    ("family", "family"),
    ("lifecycle", "lifecycle"),
    ("deprecated_at", "deprecated_at"),
    ("retirement_at", "retirement_at"),
    ("replacements", "replacement"),
    ("context_window_tokens", "context_window"),
    ("max_input_tokens", "max_input_tokens"),
    ("max_output_tokens", "max_output_tokens"),
    ("input_modalities", "input_modalities"),
    ("output_modalities", "output_modalities"),
    ("tool_calling", "tool_calling"),
    ("structured_output", "structured_output"),
    ("api_access", "api_access"),
    ("availability_scope", "availability_scope"),
    ("regions", "regions"),
)


@dataclass(frozen=True, slots=True)
class _ProviderTools:
    captured: Callable[..., Any]
    parse_one: Callable[[Any], tuple[Any, ...]]
    source_type_for_id: Callable[[str], Any]
    ranks: Mapping[Any, int]
    medium_types: frozenset[Any]
    normalize: Callable[..., tuple[Any, ...]]
    lifecycle_group: Callable[[Sequence[Any]], tuple[Any, ...]]
    lifecycle_value: Callable[..., Any]
    lifecycle_fact: Callable[..., Fact[Any]]


_PROVIDERS: dict[str, _ProviderTools] = {
    "anthropic": _ProviderTools(
        captured=capture_anthropic,
        parse_one=lambda source: parse_anthropic_sources((source,)),
        source_type_for_id=anthropic_source_type,
        ranks=_ANTHROPIC_RANK,
        medium_types=frozenset({AnthropicSourceType.MODELS_INDEX}),
        normalize=normalize_anthropic_observations,
        lifecycle_group=_anthropic_lifecycle_group,
        lifecycle_value=_anthropic_lifecycle_value,
        lifecycle_fact=_anthropic_lifecycle_fact,
    ),
    "fireworks": _ProviderTools(
        captured=capture_fireworks,
        parse_one=parse_fireworks_source,
        source_type_for_id=fireworks_source_type,
        ranks=_FIREWORKS_RANK,
        medium_types=frozenset(
            {FireworksSourceType.TEXT_MODELS, FireworksSourceType.VISION_MODELS}
        ),
        normalize=normalize_fireworks_observations,
        lifecycle_group=_fireworks_lifecycle_group,
        lifecycle_value=_fireworks_lifecycle_value,
        lifecycle_fact=_fireworks_lifecycle_fact,
    ),
    "google": _ProviderTools(
        captured=capture_google,
        parse_one=lambda source: parse_google_sources((source,)),
        source_type_for_id=google_source_type,
        ranks=_GOOGLE_RANK,
        medium_types=frozenset({GoogleSourceType.MODELS_INDEX}),
        normalize=normalize_google_observations,
        lifecycle_group=_google_lifecycle_group,
        lifecycle_value=_google_lifecycle_value,
        lifecycle_fact=_google_lifecycle_fact,
    ),
    "openai": _ProviderTools(
        captured=capture_openai,
        parse_one=lambda source: parse_openai_sources((source,)),
        source_type_for_id=openai_source_type,
        ranks=_OPENAI_RANK,
        medium_types=frozenset({OpenAISourceType.MODELS_INDEX}),
        normalize=normalize_openai_observations,
        lifecycle_group=_openai_lifecycle_group,
        lifecycle_value=_openai_lifecycle_value,
        lifecycle_fact=_openai_lifecycle_fact,
    ),
    "together": _ProviderTools(
        captured=capture_together,
        parse_one=lambda source: parse_together_sources((source,)),
        source_type_for_id=together_source_type,
        ranks=_TOGETHER_RANK,
        medium_types=frozenset({TogetherSourceType.CHANGELOG}),
        normalize=normalize_together_observations,
        lifecycle_group=_together_lifecycle_group,
        lifecycle_value=_together_lifecycle_value,
        lifecycle_fact=_together_lifecycle_fact,
    ),
}


@dataclass(frozen=True, slots=True)
class StatedObservation:
    """One stated observation value, before rank filtering."""

    provider: str
    samyak_id: str
    fact_name: str
    value: Any
    value_type: str
    source_id: str
    rank: int
    generation: str
    retrieved_at: str
    content_hash: str
    confidence: str


@dataclass(frozen=True, slots=True)
class DiagnosticClaim:
    value: Any
    source_id: str
    source_rank: int
    generation: str
    retrieved_at: str
    content_hash: str
    confidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": _json_ready(self.value),
            "source_id": self.source_id,
            "source_rank": self.source_rank,
            "generation": self.generation,
            "retrieved_at": self.retrieved_at,
            "content_hash": self.content_hash,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class Finding:
    provider: str
    samyak_id: str
    fact_name: str
    kind: str
    comparability_status: str
    explanation: str
    selected_value: Any = None
    selected_source_id: str | None = None
    selected_source_rank: int | None = None
    selected_generation: str | None = None
    selected_retrieved_at: str | None = None
    selected_content_hash: str | None = None
    selected_confidence: str | None = None
    other_value: Any = None
    other_source_id: str | None = None
    other_source_rank: int | None = None
    other_generation: str | None = None
    other_retrieved_at: str | None = None
    other_content_hash: str | None = None
    other_confidence: str | None = None
    claims: tuple[DiagnosticClaim, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "samyak_id": self.samyak_id,
            "fact_name": self.fact_name,
            "kind": self.kind,
            "selected_value": _json_ready(self.selected_value),
            "selected_source_id": self.selected_source_id,
            "selected_source_rank": self.selected_source_rank,
            "selected_generation": self.selected_generation,
            "other_value": _json_ready(self.other_value),
            "other_source_id": self.other_source_id,
            "other_source_rank": self.other_source_rank,
            "other_generation": self.other_generation,
            "selected_retrieved_at": self.selected_retrieved_at,
            "other_retrieved_at": self.other_retrieved_at,
            "selected_content_hash": self.selected_content_hash,
            "other_content_hash": self.other_content_hash,
            "selected_confidence": self.selected_confidence,
            "other_confidence": self.other_confidence,
            "claims": [item.to_dict() for item in self.claims],
            "comparability_status": self.comparability_status,
            "explanation": self.explanation,
        }


@dataclass(frozen=True, slots=True)
class Limitation:
    code: str
    message: str
    provider: str | None = None
    generation: str | None = None
    source_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "provider": self.provider,
            "generation": self.generation,
            "source_id": self.source_id,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class GenerationSummary:
    provider: str
    refresh_id: str
    source_count: int
    retrieved_at: tuple[str, ...]
    normalization_timestamp: str | None = None
    normalization_timestamp_status: str | None = None
    normalization_timestamp_method: str | None = None
    normalization_timestamp_evidence_count: int | None = None
    catalog_provider_verified_at: str | None = None
    distinct_normalization_timestamps: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "refresh_id": self.refresh_id,
            "source_count": self.source_count,
            "retrieved_at": list(self.retrieved_at),
            "normalization_timestamp": self.normalization_timestamp,
            "normalization_timestamp_status": self.normalization_timestamp_status,
            "normalization_timestamp_method": self.normalization_timestamp_method,
            "normalization_timestamp_evidence_count": (self.normalization_timestamp_evidence_count),
            "catalog_provider_verified_at": self.catalog_provider_verified_at,
            "distinct_normalization_timestamps": list(self.distinct_normalization_timestamps),
        }


@dataclass(frozen=True, slots=True)
class TimestampResolution:
    """How the diagnostic resolved a provider's original normalization clock."""

    provider: str
    generation: str
    status: str
    timestamp: str | None = None
    catalog_provider_verified_at: str | None = None
    method: str | None = None
    evidence_count: int = 0
    distinct_timestamps: tuple[str, ...] = ()
    message: str | None = None


@dataclass(frozen=True, slots=True)
class CatalogCheck:
    provider: str
    samyak_id: str
    fact_name: str
    status: str
    generation: str
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "samyak_id": self.samyak_id,
            "fact_name": self.fact_name,
            "status": self.status,
            "generation": self.generation,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ReconstructionSummary:
    """What the selected archived sources reconstruct, before any merge.

    The diagnostic deliberately does not run a provider refresh merge.  In
    particular, source absence cannot establish retained records here.
    """

    record_count: int
    current_documented_count: int
    historical_reference_only_count: int
    not_observed_in_current_capture_count: int
    legacy_uncertified_count: int
    added_reconstructed_ids: tuple[str, ...]
    unreconstructed_providers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": "direct_archive_reconstruction",
            "record_count": self.record_count,
            "evidence_disposition_counts": {
                "current_documented": self.current_documented_count,
                "historical_reference_only": self.historical_reference_only_count,
                "not_observed_in_current_capture": self.not_observed_in_current_capture_count,
                "legacy_uncertified": self.legacy_uncertified_count,
            },
            "added_reconstructed_ids": list(self.added_reconstructed_ids),
            "unreconstructed_providers": list(self.unreconstructed_providers),
            "merge_evaluated": False,
            "retained_absence_ids": [],
            "source_absence_decisions": "not_evaluated_no_merge_was_run",
        }


@dataclass(frozen=True, slots=True)
class OverrideReport:
    parser_version: str
    catalog_generated_at: str | None
    generations: tuple[GenerationSummary, ...]
    limitations: tuple[Limitation, ...]
    findings: tuple[Finding, ...]
    catalog_checks: tuple[CatalogCheck, ...]
    reconstruction: ReconstructionSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "product": "samyak",
            "capability": "model_override_diagnostic",
            "parser_version": self.parser_version,
            "catalog_generated_at": self.catalog_generated_at,
            "generations": [item.to_dict() for item in self.generations],
            "limitations": [item.to_dict() for item in self.limitations],
            "findings": [item.to_dict() for item in self.findings],
            "catalog_checks": [item.to_dict() for item in self.catalog_checks],
            "catalog_check_summary": _catalog_check_summary(self),
            "reconstruction": self.reconstruction.to_dict(),
        }


def classify_statements(
    statements: Sequence[StatedObservation],
    *,
    comparability_status: str = "archive_only",
) -> tuple[Finding, ...]:
    """Classify stated observations. Unstated fields are not statements."""
    grouped: dict[tuple[str, str, str], list[StatedObservation]] = defaultdict(list)
    for item in statements:
        grouped[(item.provider, item.samyak_id, item.fact_name)].append(item)
    findings: list[Finding] = []
    for group in grouped.values():
        findings.extend(_classify_group(group, comparability_status=comparability_status))
    return tuple(sorted(findings, key=_finding_sort_key))


def diagnose_overrides(
    *,
    archive_root: Path,
    generations: Sequence[tuple[str, str]],
    catalog: ModelCatalog | None = None,
    model_id: str | None = None,
    fact_name: str | None = None,
    catalog_unavailable: bool = False,
) -> OverrideReport:
    """Read one archive generation per provider and compare observations.

    ``generations`` is ``(provider_id, refresh_id)`` pairs. A provider that is
    absent is not evaluated. The newest generation is never chosen implicitly.
    """
    limitations: list[Limitation] = []
    summaries: list[GenerationSummary] = []
    statements: list[StatedObservation] = []
    observations: dict[str, list[Any]] = defaultdict(list)
    loaded_hashes: dict[str, set[str]] = defaultdict(set)
    hash_generation: dict[tuple[str, str], str] = {}
    requested = {provider for provider, _refresh_id in generations}
    refresh_ids = [refresh_id for _provider, refresh_id in generations]

    for provider in SUPPORTED_PROVIDERS:
        if provider not in requested:
            limitations.append(
                Limitation(
                    code="provider_not_evaluated",
                    provider=provider,
                    message=(
                        f"No archive generation was selected for {provider}. "
                        "The newest generation was not used."
                    ),
                )
            )

    if catalog is None and catalog_unavailable:
        limitations.append(
            Limitation(
                code="catalog_unavailable",
                message="No local model catalog was found. Findings are archive observations only.",
            )
        )

    if len(set(refresh_ids)) > 1:
        listed = ", ".join(f"{provider}={refresh_id}" for provider, refresh_id in generations)
        limitations.append(
            Limitation(
                code="generations_differ",
                message=("Selected generations are not one contemporaneous snapshot: " + listed),
            )
        )

    for provider, refresh_id in generations:
        summary, provider_statements, provider_limitations, hashes, parsed = _load_generation(
            archive_root,
            provider,
            refresh_id,
            model_id=model_id,
            fact_name=fact_name,
        )
        summaries.append(summary)
        statements.extend(provider_statements)
        observations[provider].extend(parsed)
        limitations.extend(provider_limitations)
        loaded_hashes[provider].update(hashes)
        for digest in hashes:
            hash_generation[(provider, digest)] = refresh_id

    resolutions: dict[str, TimestampResolution] = {}
    lifecycle_clocks: dict[str, str | None] = {}
    for provider, refresh_id in generations:
        if catalog is None:
            resolution = TimestampResolution(
                provider=provider,
                generation=refresh_id,
                status="unavailable",
                method=None,
                message="No catalog was loaded; archive observations are not catalog-checked.",
            )
            # No catalog clock: keep structural filters, do not invent promotion time.
            lifecycle_clocks[provider] = None
        else:
            resolution = _resolve_provider_normalization_timestamp(
                catalog,
                provider,
                refresh_id,
            )
            # Archive reconstruction is governed by the selected archive's
            # precise timestamp.  Catalog provenance is assessed per fact in
            # _apply_catalog; a mixed catalog clock is therefore diagnostic
            # information, not a reason to skip archive normalization.
            lifecycle_clocks[provider] = _stamp_from_refresh_id(refresh_id)
            if resolution.status != "resolved":
                limitations.append(_timestamp_limitation(resolution))
        resolutions[provider] = resolution

    summaries = [
        replace(
            summary,
            normalization_timestamp=resolutions[summary.provider].timestamp
            if resolutions[summary.provider].status == "resolved"
            else None,
            normalization_timestamp_status=resolutions[summary.provider].status,
            normalization_timestamp_method=resolutions[summary.provider].method,
            normalization_timestamp_evidence_count=resolutions[summary.provider].evidence_count,
            catalog_provider_verified_at=resolutions[summary.provider].catalog_provider_verified_at,
            distinct_normalization_timestamps=resolutions[summary.provider].distinct_timestamps,
        )
        for summary in summaries
    ]

    lifecycle_findings: list[Finding] = []
    if fact_name is None or fact_name in _POLICY_FACTS:
        statements, lifecycle_findings = _apply_lifecycle_policy(
            statements,
            observations,
            generations,
            verified_at_by_provider=lifecycle_clocks,
            model_id=model_id,
            fact_name=fact_name,
        )
    findings = [*classify_statements(statements), *lifecycle_findings]
    checks: list[CatalogCheck] = []
    if catalog is None:
        findings = [_with_status(item, "archive_only") for item in findings]
    else:
        findings, checks, reproducibility_limits = _apply_catalog(
            findings,
            observations,
            catalog,
            loaded_hashes,
            hash_generation,
            generations,
            resolutions,
            model_id=model_id,
            fact_name=fact_name,
        )
        limitations.extend(reproducibility_limits)
    findings = _collapse_repeated(findings)
    reconstruction = _reconstruct_archive_records(
        observations,
        generations,
        lifecycle_clocks,
        catalog=catalog,
        model_id=model_id,
    )
    return OverrideReport(
        parser_version=__version__,
        catalog_generated_at=(None if catalog is None else catalog.freshness.generated_at),
        generations=tuple(summaries),
        limitations=tuple(limitations),
        findings=tuple(sorted(findings, key=_finding_sort_key)),
        catalog_checks=tuple(sorted(checks, key=_check_sort_key)),
        reconstruction=reconstruction,
    )


def render_override_report(report: OverrideReport, *, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n"
    return _render_text(report)


def _reconstruct_archive_records(
    observations: Mapping[str, Sequence[Any]],
    generations: Sequence[tuple[str, str]],
    lifecycle_clocks: Mapping[str, str | None],
    *,
    catalog: ModelCatalog | None,
    model_id: str | None,
) -> ReconstructionSummary:
    """Normalize archive observations only; never merge them into the catalog."""
    records = []
    unreconstructed: list[str] = []
    for provider, _refresh_id in generations:
        items = observations.get(provider, ())
        verified_at = lifecycle_clocks.get(provider)
        if not items or verified_at is None:
            unreconstructed.append(provider)
            continue
        try:
            normalized = _PROVIDERS[provider].normalize(tuple(items), verified_at=verified_at)
        except ModelCatalogError:
            unreconstructed.append(provider)
            continue
        records.extend(
            record for record in normalized if model_id is None or record.samyak_id == model_id
        )
    dispositions = Counter(record.evidence_disposition for record in records)
    catalog_ids = set() if catalog is None else {record.samyak_id for record in catalog.models}
    return ReconstructionSummary(
        record_count=len(records),
        current_documented_count=dispositions[EvidenceDisposition.CURRENT_DOCUMENTED],
        historical_reference_only_count=dispositions[EvidenceDisposition.HISTORICAL_REFERENCE_ONLY],
        not_observed_in_current_capture_count=(
            dispositions[EvidenceDisposition.NOT_OBSERVED_IN_CURRENT_CAPTURE]
        ),
        legacy_uncertified_count=dispositions[None],
        added_reconstructed_ids=tuple(
            sorted(record.samyak_id for record in records if record.samyak_id not in catalog_ids)
        ),
        unreconstructed_providers=tuple(sorted(unreconstructed)),
    )


def _catalog_check_summary(report: OverrideReport) -> dict[str, Any]:
    historical = [
        item for item in report.generations if item.normalization_timestamp_status == "historical"
    ]
    statuses = Counter(item.status for item in report.catalog_checks)
    reasons = Counter(item.reason for item in report.catalog_checks if item.reason is not None)
    # The denominator is deliberately every catalog fact for which this run
    # attempted a selected-generation comparison.  It is not the provider's
    # full catalog, because facts without usable provenance are excluded.
    summary: dict[str, Any] = {
        "count": len(report.catalog_checks),
        "reproducibility_coverage_denominator": len(report.catalog_checks),
        "matching_count": statuses["comparable"],
        "mismatching_count": statuses["mismatching"],
        "not_comparable_count": statuses["not_comparable"],
        "not_comparable_reasons": dict(sorted(reasons.items())),
        "selected_generation_reproducibility": (
            "not_evaluated"
            if not report.catalog_checks
            else "partial"
            if statuses["not_comparable"]
            else "complete"
        ),
        "provider_wide_reproducibility": "not_claimed",
    }
    if not report.catalog_checks and historical:
        summary["reason"] = (
            "0 catalog checks: selected archive generation differs from the generation "
            "used to produce the current catalog. This is not a successful reproducibility "
            "comparison and does not mean there are zero differences."
        )
        summary["selected_generations"] = [
            {"provider": item.provider, "refresh_id": item.refresh_id} for item in historical
        ]
        summary["catalog_provenance_generations"] = [
            {
                "provider": item.provider,
                "verified_at": item.catalog_provider_verified_at,
                "refresh_id": (
                    None
                    if item.catalog_provider_verified_at is None
                    else refresh_id_from_stamp(item.catalog_provider_verified_at)
                ),
            }
            for item in historical
        ]
    return summary


def _load_generation(
    archive_root: Path,
    provider: str,
    refresh_id: str,
    *,
    model_id: str | None,
    fact_name: str | None,
) -> tuple[GenerationSummary, list[StatedObservation], list[Limitation], set[str], list[Any]]:
    limitations: list[Limitation] = []
    statements: list[StatedObservation] = []
    parsed: list[Any] = []
    hashes: set[str] = set()
    manifest = load_manifest(archive_root, refresh_id)
    if manifest is None:
        limitations.append(
            Limitation(
                code="missing_manifest",
                provider=provider,
                generation=refresh_id,
                message=f"Completed archive generation {refresh_id} was not found.",
            )
        )
        return (
            GenerationSummary(provider, refresh_id, 0, ()),
            statements,
            limitations,
            hashes,
            parsed,
        )
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        limitations.append(
            Limitation(
                code="missing_manifest",
                provider=provider,
                generation=refresh_id,
                message=f"Archive generation {refresh_id} has no source list.",
            )
        )
        return (
            GenerationSummary(provider, refresh_id, 0, ()),
            statements,
            limitations,
            hashes,
            parsed,
        )
    matched = [
        item for item in sources if isinstance(item, dict) and item.get("provider_id") == provider
    ]
    if not matched:
        limitations.append(
            Limitation(
                code="missing_provider_source",
                provider=provider,
                generation=refresh_id,
                message=(
                    f"Generation {refresh_id} has no {provider} sources. "
                    "A generation is not the whole catalog."
                ),
            )
        )
        return (
            GenerationSummary(provider, refresh_id, 0, ()),
            statements,
            limitations,
            hashes,
            parsed,
        )
    retrieved: list[str] = []
    tools = _PROVIDERS[provider]
    for item in matched:
        loaded = _read_archived_source(archive_root, provider, refresh_id, item, tools)
        if isinstance(loaded, Limitation):
            limitations.append(loaded)
            continue
        source, content_hash, retrieved_at = loaded
        retrieved.append(retrieved_at)
        try:
            observations = tools.parse_one(source)
        except _PARSE_ERRORS as exc:
            limitations.append(
                Limitation(
                    code="parse_failed",
                    provider=provider,
                    generation=refresh_id,
                    source_id=source.source_id,
                    message=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        hashes.add(content_hash)
        parsed.extend(observations)
        statements.extend(
            _statements_from_observations(
                provider,
                refresh_id,
                observations,
                tools,
                model_id=model_id,
                fact_name=fact_name,
            )
        )
    return (
        GenerationSummary(provider, refresh_id, len(matched), tuple(sorted(set(retrieved)))),
        statements,
        limitations,
        hashes,
        parsed,
    )


def _read_archived_source(
    archive_root: Path,
    provider: str,
    refresh_id: str,
    item: Mapping[str, Any],
    tools: _ProviderTools,
) -> tuple[Any, str, str] | Limitation:
    source_id = item.get("source_id")
    source_url = item.get("source_url")
    content_hash = item.get("content_hash")
    retrieved_at = item.get("retrieved_at")
    byte_length = item.get("byte_length")
    if not isinstance(source_id, str) or not source_id.startswith(f"{provider}-docs-"):
        return Limitation(
            code="invalid_attribution",
            provider=provider,
            generation=refresh_id,
            source_id=source_id if isinstance(source_id, str) else None,
            message="Archived source id does not belong to the recorded provider.",
        )
    if not isinstance(content_hash, str) or not content_hash.startswith("sha256:"):
        return Limitation(
            code="hash_mismatch",
            provider=provider,
            generation=refresh_id,
            source_id=source_id,
            message="Archived source is missing a sha256 content hash.",
        )
    if not isinstance(retrieved_at, str) or not isinstance(source_url, str):
        return Limitation(
            code="invalid_attribution",
            provider=provider,
            generation=refresh_id,
            source_id=source_id,
            message="Archived source is missing a URL or retrieval timestamp.",
        )
    path = blob_path(archive_root, content_hash)
    if not path.is_file():
        return Limitation(
            code="missing_blob",
            provider=provider,
            generation=refresh_id,
            source_id=source_id,
            message=f"Archived blob for {source_id} is missing.",
        )
    payload = path.read_bytes()
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    if digest != content_hash:
        return Limitation(
            code="hash_mismatch",
            provider=provider,
            generation=refresh_id,
            source_id=source_id,
            message="Archived blob SHA-256 does not match the manifest.",
        )
    if (
        not isinstance(byte_length, int)
        or isinstance(byte_length, bool)
        or byte_length != len(payload)
    ):
        return Limitation(
            code="byte_length_mismatch",
            provider=provider,
            generation=refresh_id,
            source_id=source_id,
            message="Archived blob length does not match the manifest.",
        )
    try:
        tools.source_type_for_id(source_id)
        text = payload.decode("utf-8")
        source = tools.captured(
            source_id=source_id,
            source_url=source_url,
            body=text,
            retrieved_at=retrieved_at,
        )
    except _PARSE_ERRORS as exc:
        return Limitation(
            code="invalid_source",
            provider=provider,
            generation=refresh_id,
            source_id=source_id,
            message=f"{type(exc).__name__}: {exc}",
        )
    return source, content_hash, retrieved_at


def _statements_from_observations(
    provider: str,
    refresh_id: str,
    observations: Sequence[Any],
    tools: _ProviderTools,
    *,
    model_id: str | None,
    fact_name: str | None,
) -> list[StatedObservation]:
    stated: list[StatedObservation] = []
    for observation in observations:
        samyak_id = f"{provider}:{observation.provider_model_id}"
        if model_id is not None and samyak_id != model_id:
            continue
        source = observation.source
        source_type = tools.source_type_for_id(source.source_id)
        rank = int(tools.ranks[source_type])
        confidence = "medium" if source_type in tools.medium_types else "high"
        for attribute, fact in _OBSERVATION_FACTS:
            if fact_name is not None and fact != fact_name:
                continue
            raw = getattr(observation, attribute)
            if raw is None:
                continue
            value, value_type = _canonicalize(raw)
            stated.append(
                StatedObservation(
                    provider=provider,
                    samyak_id=samyak_id,
                    fact_name=fact,
                    value=value,
                    value_type=value_type,
                    source_id=source.source_id,
                    rank=rank,
                    generation=refresh_id,
                    retrieved_at=source.retrieved_at,
                    content_hash=source.content_hash,
                    confidence=confidence,
                )
            )
    return stated


def _classify_group(
    items: Sequence[StatedObservation],
    *,
    comparability_status: str,
) -> list[Finding]:
    types = {item.value_type for item in items}
    if len(types) > 1:
        return [
            _incomparable(items, comparability_status=comparability_status),
        ]
    top = max(item.rank for item in items)
    winners = [item for item in items if item.rank == top]
    distinct = _distinct(winners)
    findings: list[Finding] = []
    if len(distinct) > 1:
        findings.append(
            _conflict_finding(
                distinct,
                kind="winning_rank_conflict",
                comparability_status=comparability_status,
                explanation=(
                    f"{len(distinct)} distinct values at rank {top}. No single value is selected."
                ),
            )
        )
    else:
        selected = distinct[0]
        for item in items:
            if item is selected:
                continue
            if item.value == selected.value:
                findings.append(
                    _pair_finding(
                        selected,
                        item,
                        kind="supporting",
                        comparability_status=comparability_status,
                        explanation=(
                            "This source states the selected value and is not "
                            "the selected provenance."
                        ),
                    )
                )
            elif item.rank < top:
                findings.append(
                    _pair_finding(
                        selected,
                        item,
                        kind="override",
                        comparability_status=comparability_status,
                        explanation=(
                            f"Rank {item.rank} states a different value from "
                            f"selected rank {top}. The lower value is not a claim."
                        ),
                    )
                )
    lowers = [item for item in items if item.rank < top]
    lower_distinct = _distinct(lowers)
    if len(lower_distinct) > 1:
        findings.append(
            _conflict_finding(
                lower_distinct,
                kind="lower_rank_disagreement",
                comparability_status=comparability_status,
                explanation=(
                    "Lower-ranked sources state more than one value. "
                    "Those values are not catalog claims."
                ),
                selected=distinct[0] if len(distinct) == 1 else None,
            )
        )
    return findings


def _distinct(items: Sequence[StatedObservation]) -> list[StatedObservation]:
    found: list[StatedObservation] = []
    seen: list[Any] = []
    for item in items:
        if item.value not in seen:
            seen.append(item.value)
            found.append(item)
    return found


def _pair_finding(
    selected: StatedObservation,
    other: StatedObservation,
    *,
    kind: str,
    comparability_status: str,
    explanation: str,
) -> Finding:
    return Finding(
        provider=selected.provider,
        samyak_id=selected.samyak_id,
        fact_name=selected.fact_name,
        kind=kind,
        comparability_status=comparability_status,
        explanation=explanation,
        selected_value=selected.value,
        selected_source_id=selected.source_id,
        selected_source_rank=selected.rank,
        selected_generation=selected.generation,
        selected_retrieved_at=selected.retrieved_at,
        selected_content_hash=selected.content_hash,
        selected_confidence=selected.confidence,
        other_value=other.value,
        other_source_id=other.source_id,
        other_source_rank=other.rank,
        other_generation=other.generation,
        other_retrieved_at=other.retrieved_at,
        other_content_hash=other.content_hash,
        other_confidence=other.confidence,
    )


def _conflict_finding(
    claims: Sequence[StatedObservation],
    *,
    kind: str,
    comparability_status: str,
    explanation: str,
    selected: StatedObservation | None = None,
) -> Finding:
    first = claims[0]
    return Finding(
        provider=first.provider,
        samyak_id=first.samyak_id,
        fact_name=first.fact_name,
        kind=kind,
        comparability_status=comparability_status,
        explanation=explanation,
        selected_value=None if selected is None else selected.value,
        selected_source_id=None if selected is None else selected.source_id,
        selected_source_rank=None if selected is None else selected.rank,
        selected_generation=first.generation if selected is None else selected.generation,
        selected_retrieved_at=None if selected is None else selected.retrieved_at,
        selected_content_hash=None if selected is None else selected.content_hash,
        selected_confidence=None if selected is None else selected.confidence,
        claims=tuple(_claim(item) for item in sorted(claims, key=lambda item: item.source_id)),
    )


def _incomparable(
    items: Sequence[StatedObservation],
    *,
    comparability_status: str,
) -> Finding:
    first = items[0]
    listed = ", ".join(sorted({item.value_type for item in items}))
    return Finding(
        provider=first.provider,
        samyak_id=first.samyak_id,
        fact_name=first.fact_name,
        kind="incomparable",
        comparability_status=comparability_status,
        explanation=f"Values have incompatible types ({listed}) and were not compared.",
        selected_generation=first.generation,
        claims=tuple(_claim(item) for item in sorted(items, key=lambda item: item.source_id)),
    )


def _claim(item: StatedObservation) -> DiagnosticClaim:
    return DiagnosticClaim(
        value=item.value,
        source_id=item.source_id,
        source_rank=item.rank,
        generation=item.generation,
        retrieved_at=item.retrieved_at,
        content_hash=item.content_hash,
        confidence=item.confidence,
    )


def _apply_lifecycle_policy(
    statements: Sequence[StatedObservation],
    observations: Mapping[str, Sequence[Any]],
    generations: Sequence[tuple[str, str]],
    *,
    verified_at_by_provider: Mapping[str, str | None],
    model_id: str | None,
    fact_name: str | None,
) -> tuple[list[StatedObservation], list[Finding]]:
    """Classify lifecycle-related facts with each provider's production filter."""
    active = {provider for provider, items in observations.items() if items}
    rebuilding = {name for name in _POLICY_FACTS if fact_name is None or name == fact_name}
    kept = [
        item
        for item in statements
        if item.provider not in active or item.fact_name not in rebuilding
    ]
    added: list[StatedObservation] = []
    extras: list[Finding] = []
    generation_of = dict(generations)
    for provider, items in observations.items():
        if not items:
            continue
        verified_at = verified_at_by_provider.get(provider)
        tools = _PROVIDERS[provider]
        grouped: dict[str, list[Any]] = defaultdict(list)
        for observation in items:
            grouped[str(observation.provider_model_id)].append(observation)
        for provider_model_id, group in grouped.items():
            samyak_id = f"{provider}:{provider_model_id}"
            if model_id is not None and samyak_id != model_id:
                continue
            lifecycle_group = tuple(tools.lifecycle_group(group))
            retirement_group = _retirement_observations(provider, group, lifecycle_group)
            fact: Fact[Any] | None = None
            if verified_at is not None and lifecycle_group:
                fallback = max(group, key=lambda item: _observation_rank(tools, item))
                fact = tools.lifecycle_fact(lifecycle_group, verified_at, fallback)
            if "lifecycle" in rebuilding:
                if _is_derived_active(fact, lifecycle_group, tools, verified_at):
                    assert fact is not None and fact.provenance is not None
                    extras.append(
                        _derived_finding(
                            provider,
                            samyak_id,
                            generation_of[provider],
                            fact,
                            tools,
                        )
                    )
                else:
                    promoted = _promoted_lifecycle(lifecycle_group, tools, verified_at)
                    if fact is not None and fact.status is FactStatus.KNOWN:
                        selected = _catalog_known_value(fact)
                        for observation, value in promoted:
                            if value != selected:
                                extras.append(
                                    _excluded_finding(
                                        provider,
                                        samyak_id,
                                        "lifecycle",
                                        generation_of[provider],
                                        observation,
                                        tools,
                                        other_value=value,
                                        selected=fact,
                                    )
                                )
                        promoted = [
                            (observation, value)
                            for observation, value in promoted
                            if value == selected
                        ]
                    for observation, value in promoted:
                        added.append(
                            _statement_for(
                                provider,
                                generation_of[provider],
                                observation,
                                tools,
                                "lifecycle",
                                value,
                            )
                        )
            if "deprecated_at" in rebuilding:
                added.extend(
                    _attribute_statements(
                        provider,
                        generation_of[provider],
                        lifecycle_group,
                        tools,
                        attribute="deprecated_at",
                        catalog_fact="deprecated_at",
                    )
                )
            if "replacement" in rebuilding:
                added.extend(
                    _attribute_statements(
                        provider,
                        generation_of[provider],
                        lifecycle_group,
                        tools,
                        attribute="replacements",
                        catalog_fact="replacement",
                    )
                )
            if "retirement_at" in rebuilding:
                added.extend(
                    _attribute_statements(
                        provider,
                        generation_of[provider],
                        retirement_group,
                        tools,
                        attribute="retirement_at",
                        catalog_fact="retirement_at",
                    )
                )
            if fact_name in {None, "lifecycle"}:
                kept_ids = {id(item) for item in lifecycle_group}
                for observation in group:
                    if id(observation) in kept_ids or observation.lifecycle is None:
                        continue
                    selected = None
                    if fact is not None and fact.status is FactStatus.KNOWN:
                        selected = fact
                    extras.append(
                        _excluded_finding(
                            provider,
                            samyak_id,
                            "lifecycle",
                            generation_of[provider],
                            observation,
                            tools,
                            other_value=observation.lifecycle.value,
                            selected=selected,
                        )
                    )
    return kept + added, extras


def _retirement_observations(
    provider: str,
    group: Sequence[Any],
    lifecycle_group: Sequence[Any],
) -> tuple[Any, ...]:
    if provider == "together":
        return _together_retirement_group(group, lifecycle_group)
    return tuple(lifecycle_group)


def _promoted_lifecycle(
    group: Sequence[Any],
    tools: _ProviderTools,
    verified_at: str | None,
) -> list[tuple[Any, str]]:
    promoted: list[tuple[Any, str]] = []
    for observation in group:
        if verified_at is None:
            if observation.lifecycle is None:
                continue
            promoted.append((observation, observation.lifecycle.value))
            continue
        value = tools.lifecycle_value(observation, verified_at)
        if value is not None:
            promoted.append((observation, value.value))
    return promoted


def _is_derived_active(
    fact: Fact[Any] | None,
    group: Sequence[Any],
    tools: _ProviderTools,
    verified_at: str | None,
) -> bool:
    if fact is None or verified_at is None or fact.provenance is None:
        return False
    if fact.status is not FactStatus.KNOWN or fact.value is not LifecycleState.ACTIVE:
        return False
    if fact.provenance.confidence is not Confidence.MEDIUM:
        return False
    return all(tools.lifecycle_value(item, verified_at) is None for item in group)


def _attribute_statements(
    provider: str,
    generation: str,
    group: Sequence[Any],
    tools: _ProviderTools,
    *,
    attribute: str,
    catalog_fact: str,
) -> list[StatedObservation]:
    stated: list[StatedObservation] = []
    for observation in group:
        raw = getattr(observation, attribute)
        if raw is None:
            continue
        stated.append(_statement_for(provider, generation, observation, tools, catalog_fact, raw))
    return stated


def _statement_for(
    provider: str,
    generation: str,
    observation: Any,
    tools: _ProviderTools,
    fact_name: str,
    value: Any,
) -> StatedObservation:
    source = observation.source
    source_type = tools.source_type_for_id(source.source_id)
    rank = int(tools.ranks[source_type])
    confidence = "medium" if source_type in tools.medium_types else "high"
    canonical, value_type = _canonicalize(value)
    return StatedObservation(
        provider=provider,
        samyak_id=f"{provider}:{observation.provider_model_id}",
        fact_name=fact_name,
        value=canonical,
        value_type=value_type,
        source_id=source.source_id,
        rank=rank,
        generation=generation,
        retrieved_at=source.retrieved_at,
        content_hash=source.content_hash,
        confidence=confidence,
    )


def _observation_rank(tools: _ProviderTools, observation: Any) -> int:
    return int(tools.ranks[tools.source_type_for_id(observation.source.source_id)])


def _derived_finding(
    provider: str,
    samyak_id: str,
    generation: str,
    fact: Fact[Any],
    tools: _ProviderTools,
) -> Finding:
    provenance = fact.provenance
    assert provenance is not None
    return Finding(
        provider=provider,
        samyak_id=samyak_id,
        fact_name="lifecycle",
        kind="derived",
        comparability_status="comparable",
        explanation=(
            "Lifecycle active is derived from documented membership. No source stated a lifecycle."
        ),
        selected_value=_catalog_known_value(fact),
        selected_source_id=provenance.source_id,
        selected_source_rank=_rank_for_source(tools, provenance.source_id),
        selected_generation=generation,
        selected_retrieved_at=provenance.retrieved_at,
        selected_content_hash=provenance.content_hash,
        selected_confidence=provenance.confidence.value,
    )


def _excluded_finding(
    provider: str,
    samyak_id: str,
    fact_name: str,
    generation: str,
    observation: Any,
    tools: _ProviderTools,
    *,
    other_value: Any,
    selected: Fact[Any] | None,
) -> Finding:
    source = observation.source
    source_type = tools.source_type_for_id(source.source_id)
    selected_value = None
    selected_source_id = None
    selected_rank = None
    selected_hash = None
    selected_retrieved = None
    selected_confidence = None
    if selected is not None and selected.provenance is not None:
        selected_value = _catalog_known_value(selected)
        selected_source_id = selected.provenance.source_id
        selected_rank = _rank_for_source(tools, selected.provenance.source_id)
        selected_hash = selected.provenance.content_hash
        selected_retrieved = selected.provenance.retrieved_at
        selected_confidence = selected.provenance.confidence.value
    canonical, _value_type = _canonicalize(other_value)
    return Finding(
        provider=provider,
        samyak_id=samyak_id,
        fact_name=fact_name,
        kind="excluded",
        comparability_status="comparable",
        explanation=(
            "Production excludes this observation before lifecycle selection. "
            "It is not the selected value and not a current conflict."
        ),
        selected_value=selected_value,
        selected_source_id=selected_source_id,
        selected_source_rank=selected_rank,
        selected_generation=generation if selected_source_id is not None else None,
        selected_retrieved_at=selected_retrieved,
        selected_content_hash=selected_hash,
        selected_confidence=selected_confidence,
        other_value=canonical,
        other_source_id=source.source_id,
        other_source_rank=int(tools.ranks[source_type]),
        other_generation=generation,
        other_retrieved_at=source.retrieved_at,
        other_content_hash=source.content_hash,
        other_confidence="medium" if source_type in tools.medium_types else "high",
    )


def _stamp_from_refresh_id(refresh_id: str) -> str | None:
    """Invert a standard archive generation id into a verification timestamp.

    Collision suffixes such as ``-2`` are not inverted. Retrieval times are not
    used. Returns None when the id is not a precise stamp encoding.
    """
    match = _REFRESH_ID_STAMP.fullmatch(refresh_id)
    if match is None:
        return None
    frac = match.group("frac") or ""
    stamp = (
        f"{match.group('date')}T{match.group('h')}:{match.group('m')}:"
        f"{match.group('s')}{frac}+00:00"
    )
    try:
        validate_timestamp("refresh_id", stamp)
    except CatalogValidationError:
        return None
    if refresh_id_from_stamp(stamp) != refresh_id:
        return None
    return stamp


def _collect_provider_verified_at(
    catalog: ModelCatalog,
    provider: str,
) -> tuple[Counter[str], tuple[str, ...]]:
    """Collect provenance.verified_at values for one provider.

    Sufficiency: any known, unknown, or conflict fact/claim with a valid
    timezone-aware verified_at is eligible. not_verified facts contribute
    nothing. Invalid timestamps are returned separately and never guessed away.
    """
    counts: Counter[str] = Counter()
    invalid: list[str] = []
    for record in catalog.models:
        if record.provider_id != provider:
            continue
        for name in RECORD_FACT_NAMES:
            fact = getattr(record, name)
            for stamp in _fact_verified_at_values(fact):
                try:
                    validate_timestamp(f"{provider}.{name}.verified_at", stamp)
                except CatalogValidationError:
                    invalid.append(stamp)
                    continue
                counts[stamp] += 1
    return counts, tuple(invalid)


def _fact_verified_at_values(fact: Fact[Any]) -> tuple[str, ...]:
    values: list[str] = []
    if fact.provenance is not None and fact.provenance.verified_at is not None:
        values.append(fact.provenance.verified_at)
    for claim in fact.claims:
        if claim.provenance.verified_at is not None:
            values.append(claim.provenance.verified_at)
    return tuple(values)


def _resolve_provider_normalization_timestamp(
    catalog: ModelCatalog,
    provider: str,
    refresh_id: str,
) -> TimestampResolution:
    """Resolve the original normalization timestamp for one provider generation.

    Current catalog reproducibility requires a single catalog provenance
    timestamp whose archive generation id equals the selected refresh id.
    Matching content hashes alone do not establish generation identity.
    """
    counts, invalid = _collect_provider_verified_at(catalog, provider)
    method = "catalog_fact_provenance.verified_at"
    if invalid:
        return TimestampResolution(
            provider=provider,
            generation=refresh_id,
            status="invalid",
            method=method,
            evidence_count=len(invalid) + sum(counts.values()),
            distinct_timestamps=tuple(sorted(set(invalid) | set(counts))),
            message=(
                f"{provider} catalog facts include verified_at values that are not "
                "timezone-aware ISO-8601 timestamps. No timestamp was substituted."
            ),
        )
    if not counts:
        return TimestampResolution(
            provider=provider,
            generation=refresh_id,
            status="missing",
            method=method,
            evidence_count=0,
            message=(
                f"No {provider} catalog fact carries a usable provenance.verified_at. "
                "The global catalog generated_at was not substituted."
            ),
        )
    distinct = tuple(sorted(counts))
    if len(distinct) > 1:
        return TimestampResolution(
            provider=provider,
            generation=refresh_id,
            status="conflict",
            method=method,
            evidence_count=sum(counts.values()),
            distinct_timestamps=distinct,
            message=(
                f"{provider} catalog facts identify multiple normalization timestamps: "
                + ", ".join(f"{stamp} ({counts[stamp]})" for stamp in distinct)
            ),
        )
    stamp = distinct[0]
    evidence = counts[stamp]
    if refresh_id_from_stamp(stamp) != refresh_id:
        return TimestampResolution(
            provider=provider,
            generation=refresh_id,
            status="historical",
            timestamp=None,
            catalog_provider_verified_at=stamp,
            method=method,
            evidence_count=evidence,
            distinct_timestamps=(stamp,),
            message=(
                f"{provider} catalog facts were verified at {stamp}, which is not "
                f"archive generation {refresh_id}. Current catalog reproducibility "
                "is not evaluated for this selection."
            ),
        )
    return TimestampResolution(
        provider=provider,
        generation=refresh_id,
        status="resolved",
        timestamp=stamp,
        catalog_provider_verified_at=stamp,
        method=method,
        evidence_count=evidence,
        distinct_timestamps=(stamp,),
    )


def _timestamp_limitation(resolution: TimestampResolution) -> Limitation:
    code = {
        "missing": "normalization_timestamp_unavailable",
        "conflict": "normalization_timestamp_conflict",
        "invalid": "normalization_timestamp_invalid",
        "historical": "historical_generation",
    }.get(resolution.status, "normalization_timestamp_unavailable")
    return Limitation(
        code=code,
        provider=resolution.provider,
        generation=resolution.generation,
        message=resolution.message
        or f"Normalization timestamp could not be resolved ({resolution.status}).",
    )


def _apply_catalog(
    findings: list[Finding],
    observations: Mapping[str, Sequence[Any]],
    catalog: ModelCatalog,
    loaded_hashes: Mapping[str, set[str]],
    hash_generation: Mapping[tuple[str, str], str],
    generations: Sequence[tuple[str, str]],
    resolutions: Mapping[str, TimestampResolution],
    *,
    model_id: str | None,
    fact_name: str | None,
) -> tuple[list[Finding], list[CatalogCheck], list[Limitation]]:
    """Compare production normalizer output with stored catalog facts."""
    del hash_generation
    generation_of = dict(generations)
    records = {item.samyak_id: item for item in catalog.models}
    checks: list[CatalogCheck] = []
    mismatches: list[Finding] = []
    replaced: set[tuple[str, str, str]] = set()
    limitations: list[Limitation] = []
    produced: set[str] = set()
    normalized_providers: set[str] = set()
    for provider, items in observations.items():
        if not items:
            continue
        generation = generation_of[provider]
        # A selected archive id is the only clock used for reconstruction.
        # Provider-wide catalog clocks may be mixed after safe retention, but
        # that must not prevent a newer fact from being checked on its own.
        verified_at = _stamp_from_refresh_id(generation)
        if verified_at is None:
            limitations.append(
                Limitation(
                    code="normalization_timestamp_unavailable",
                    provider=provider,
                    generation=generation,
                    message="The selected archive id cannot be converted to a precise timestamp.",
                )
            )
            continue
        tools = _PROVIDERS[provider]
        try:
            normalized = tools.normalize(tuple(items), verified_at=verified_at)
        except ModelCatalogError as exc:
            limitations.append(
                Limitation(
                    code="normalize_failed",
                    provider=provider,
                    generation=generation_of.get(provider),
                    message=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        normalized_providers.add(provider)
        hashes = loaded_hashes[provider]
        for record in normalized:
            produced.add(record.samyak_id)
            if model_id is not None and record.samyak_id != model_id:
                continue
            catalog_record = records.get(record.samyak_id)
            if catalog_record is None or catalog_record.provider_id != provider:
                continue
            for name in RECORD_FACT_NAMES:
                if fact_name is not None and name != fact_name:
                    continue
                compared = _compare_normalized_fact(
                    provider,
                    record.samyak_id,
                    name,
                    generation_of[provider],
                    getattr(record, name),
                    getattr(catalog_record, name),
                    hashes,
                    tools,
                )
                if compared is None:
                    continue
                status, mismatch = compared
                checks.append(
                    CatalogCheck(
                        provider=provider,
                        samyak_id=record.samyak_id,
                        fact_name=name,
                        status=status,
                        generation=generation_of[provider],
                        reason=(
                            _fact_generation_reason(
                                getattr(catalog_record, name),
                                generation_of[provider],
                                hashes,
                            )
                            if status == "not_comparable"
                            else None
                        ),
                    )
                )
                if mismatch is not None:
                    replaced.add((provider, record.samyak_id, name))
                    mismatches.append(mismatch)
    # Account for every relevant stored fact that was not compared above.
    # This includes retained identities absent from the selected archive and
    # facts whose own provenance is older or ambiguous.  It intentionally
    # works at fact granularity: one retained value cannot veto a fresh value
    # on the same record.
    seen = {(item.provider, item.samyak_id, item.fact_name) for item in checks}
    for record in catalog.models:
        provider = record.provider_id
        if provider not in generation_of:
            continue
        if model_id is not None and record.samyak_id != model_id:
            continue
        generation = generation_of[provider]
        for name in RECORD_FACT_NAMES:
            if fact_name is not None and name != fact_name:
                continue
            key = (provider, record.samyak_id, name)
            if key in seen:
                continue
            fact = getattr(record, name)
            reason = _fact_generation_reason(fact, generation, loaded_hashes.get(provider, set()))
            if reason is None:
                reason = "required_archive_evidence_absent"
            checks.append(CatalogCheck(*key, "not_comparable", generation, reason))
            mismatches.append(_not_comparable_fact_finding(record, name, generation, reason))
            replaced.add(key)
    kept: list[Finding] = []
    for item in findings:
        key = (item.provider, item.samyak_id, item.fact_name)
        if key in replaced:
            continue
        kept.append(_with_status(item, _status_for(item, checks)))
    return kept + mismatches, checks, limitations


def _blocked_timestamp_findings(
    catalog: ModelCatalog,
    providers: set[str],
    loaded_hashes: Mapping[str, set[str]],
    generation_of: Mapping[str, str],
    *,
    model_id: str | None,
    fact_name: str | None,
) -> list[Finding]:
    """Mark current catalog facts not_comparable when the timestamp is unusable."""
    findings: list[Finding] = []
    for record in catalog.models:
        provider = record.provider_id
        if provider not in providers:
            continue
        if model_id is not None and record.samyak_id != model_id:
            continue
        hashes = loaded_hashes.get(provider, set())
        generation = generation_of.get(provider)
        if generation is None:
            continue
        for name in RECORD_FACT_NAMES:
            if fact_name is not None and name != fact_name:
                continue
            fact = getattr(record, name)
            if fact.status is FactStatus.NOT_VERIFIED:
                continue
            if not _fact_tied(fact, hashes):
                continue
            findings.append(
                Finding(
                    provider=provider,
                    samyak_id=record.samyak_id,
                    fact_name=name,
                    kind="not_comparable",
                    comparability_status="not_comparable",
                    explanation=(
                        "The provider normalization timestamp could not be resolved "
                        "from catalog provenance, so this fact was not compared with "
                        "the production normalizer. The global catalog generated_at "
                        "was not used."
                    ),
                    selected_generation=generation,
                    other_value=(
                        _catalog_known_value(fact) if fact.status is FactStatus.KNOWN else None
                    ),
                    other_source_id=(
                        None if fact.provenance is None else fact.provenance.source_id
                    ),
                    other_generation=generation,
                    other_retrieved_at=(
                        None if fact.provenance is None else fact.provenance.retrieved_at
                    ),
                    other_content_hash=(
                        None if fact.provenance is None else fact.provenance.content_hash
                    ),
                )
            )
    return findings


def _unreproduced_catalog_facts(
    catalog: ModelCatalog,
    produced: set[str],
    normalized_providers: set[str],
    loaded_hashes: Mapping[str, set[str]],
    generation_of: Mapping[str, str],
    checks: Sequence[CatalogCheck],
    *,
    model_id: str | None,
    fact_name: str | None,
) -> tuple[list[Finding], list[CatalogCheck]]:
    """Report a stored fact whose model the normalizer did not emit."""
    already = {(item.provider, item.samyak_id, item.fact_name) for item in checks}
    findings: list[Finding] = []
    extra_checks: list[CatalogCheck] = []
    for record in catalog.models:
        provider = record.provider_id
        if provider not in normalized_providers or record.samyak_id in produced:
            continue
        if model_id is not None and record.samyak_id != model_id:
            continue
        hashes = loaded_hashes[provider]
        generation = generation_of.get(provider)
        for name in RECORD_FACT_NAMES:
            if fact_name is not None and name != fact_name:
                continue
            key = (provider, record.samyak_id, name)
            if key in already:
                continue
            fact = getattr(record, name)
            if fact.status is not FactStatus.KNOWN or fact.provenance is None:
                continue
            if fact.provenance.content_hash not in hashes or generation is None:
                continue
            findings.append(
                Finding(
                    provider=provider,
                    samyak_id=record.samyak_id,
                    fact_name=name,
                    kind="not_comparable",
                    comparability_status="not_comparable",
                    explanation=(
                        "The current normalizer did not reproduce the catalog observation "
                        "from this archived document. This is not a source conflict."
                    ),
                    selected_generation=generation,
                    other_value=_catalog_known_value(fact),
                    other_source_id=fact.provenance.source_id,
                    other_generation=generation,
                    other_retrieved_at=fact.provenance.retrieved_at,
                    other_content_hash=fact.provenance.content_hash,
                )
            )
            extra_checks.append(
                CatalogCheck(
                    provider=provider,
                    samyak_id=record.samyak_id,
                    fact_name=name,
                    status="not_comparable",
                    generation=generation,
                )
            )
    return findings, extra_checks


def _compare_normalized_fact(
    provider: str,
    samyak_id: str,
    fact_name: str,
    generation: str,
    normalized: Fact[Any],
    catalog_fact: Fact[Any],
    hashes: set[str],
    tools: _ProviderTools,
) -> tuple[str, Finding | None] | None:
    if catalog_fact.status is FactStatus.NOT_VERIFIED:
        return None
    reason = _fact_generation_reason(catalog_fact, generation, hashes)
    if reason is not None:
        return "not_comparable", _not_comparable_fact_finding(
            None,
            fact_name,
            generation,
            reason,
            provider=provider,
            samyak_id=samyak_id,
            fact=catalog_fact,
        )
    if _fact_signature(normalized) == _fact_signature(catalog_fact):
        return "comparable", None
    return "mismatching", _mismatch_finding(
        provider,
        samyak_id,
        fact_name,
        generation,
        normalized,
        catalog_fact,
        tools,
    )


def _fact_tied(fact: Fact[Any], hashes: set[str]) -> bool:
    digests = _fact_hashes(fact)
    return bool(digests) and all(item in hashes for item in digests)


def _fact_generation_reason(
    fact: Fact[Any],
    selected_generation: str,
    hashes: set[str],
) -> str | None:
    """Return why a stored fact cannot be checked against this generation.

    Provenance is fact-level (or claim-level for conflicts).  Content hashes
    establish that the required archived bytes are available; verified_at
    establishes generation identity.  Neither provider, URL, nor a reused hash
    is allowed to stand in for the latter.
    """
    if fact.status is FactStatus.NOT_VERIFIED:
        return "not_verified"
    if fact.status in {FactStatus.UNKNOWN, FactStatus.NOT_APPLICABLE}:
        return "unknown_global_fact"
    stamps = _fact_verified_at_values(fact)
    if not stamps:
        return "missing_generation_metadata"
    valid: set[str] = set()
    for stamp in stamps:
        try:
            validate_timestamp("fact.provenance.verified_at", stamp)
        except CatalogValidationError:
            return "ambiguous_generation_provenance"
        valid.add(stamp)
    if len(valid) != 1:
        return "ambiguous_generation_provenance"
    stamp = next(iter(valid))
    if refresh_id_from_stamp(stamp) != selected_generation:
        return "retained_older_generation"
    if not _fact_tied(fact, hashes):
        return "required_archive_evidence_absent"
    return None


def _not_comparable_fact_finding(
    record: Any | None,
    fact_name: str,
    generation: str,
    reason: str,
    *,
    provider: str | None = None,
    samyak_id: str | None = None,
    fact: Fact[Any] | None = None,
) -> Finding:
    if record is not None:
        provider = record.provider_id
        samyak_id = record.samyak_id
        fact = getattr(record, fact_name)
    assert provider is not None and samyak_id is not None and fact is not None
    provenance = fact.provenance
    messages = {
        "retained_older_generation": (
            "This fact is retained from a different archive generation and was not "
            "compared with the selected archive."
        ),
        "ambiguous_generation_provenance": (
            "This fact has conflicting or invalid provenance timestamps, so its "
            "generation cannot be established."
        ),
        "missing_generation_metadata": (
            "This fact has no usable provenance generation metadata; generation was not inferred."
        ),
        "required_archive_evidence_absent": (
            "The selected archive does not contain all evidence required by this fact."
        ),
        "not_verified": "This fact is not verified and has no evidence to compare.",
        "unknown_global_fact": (
            "This global fact is unknown or not applicable; an absent reconstructed value "
            "is not a mismatch."
        ),
    }
    return Finding(
        provider=provider,
        samyak_id=samyak_id,
        fact_name=fact_name,
        kind="not_comparable",
        comparability_status="not_comparable",
        explanation=messages[reason],
        selected_generation=generation,
        other_value=_catalog_known_value(fact) if fact.status is FactStatus.KNOWN else None,
        other_source_id=None if provenance is None else provenance.source_id,
        other_retrieved_at=None if provenance is None else provenance.retrieved_at,
        other_content_hash=None if provenance is None else provenance.content_hash,
    )


def _fact_hashes(fact: Fact[Any]) -> tuple[str, ...]:
    digests: list[str] = []
    if fact.provenance is not None and fact.provenance.content_hash:
        digests.append(fact.provenance.content_hash)
    for claim in fact.claims:
        if claim.provenance.content_hash:
            digests.append(claim.provenance.content_hash)
    return tuple(digests)


def _fact_signature(fact: Fact[Any]) -> tuple[Any, ...]:
    if fact.status is FactStatus.KNOWN:
        provenance = fact.provenance
        return (
            "known",
            _catalog_known_value(fact),
            None if provenance is None else provenance.source_id,
            None if provenance is None else provenance.content_hash,
        )
    if fact.status is FactStatus.CONFLICT:
        pairs = tuple(
            sorted(
                (
                    _catalog_claim_value(claim.value),
                    claim.provenance.source_id,
                    claim.provenance.content_hash,
                )
                for claim in fact.claims
            )
        )
        return ("conflict", pairs)
    if fact.status is FactStatus.UNKNOWN:
        provenance = fact.provenance
        return (
            "unknown",
            None if provenance is None else provenance.source_id,
            None if provenance is None else provenance.content_hash,
        )
    if fact.status is FactStatus.NOT_VERIFIED:
        return ("not_verified",)
    if fact.status is FactStatus.NOT_APPLICABLE:
        return ("not_applicable",)
    return ("other", fact.status.value)


def _mismatch_finding(
    provider: str,
    samyak_id: str,
    fact_name: str,
    generation: str,
    normalized: Fact[Any],
    catalog_fact: Fact[Any],
    tools: _ProviderTools,
) -> Finding:
    selected_value = None
    selected_source_id = None
    selected_rank = None
    selected_retrieved = None
    selected_hash = None
    selected_confidence = None
    if normalized.status is FactStatus.KNOWN and normalized.provenance is not None:
        selected_value = _catalog_known_value(normalized)
        selected_source_id = normalized.provenance.source_id
        selected_rank = _rank_for_source(tools, normalized.provenance.source_id)
        selected_retrieved = normalized.provenance.retrieved_at
        selected_hash = normalized.provenance.content_hash
        selected_confidence = normalized.provenance.confidence.value
    other_value = None
    other_source_id = None
    other_retrieved = None
    other_hash = None
    if catalog_fact.status is FactStatus.KNOWN and catalog_fact.provenance is not None:
        other_value = _catalog_known_value(catalog_fact)
        other_source_id = catalog_fact.provenance.source_id
        other_retrieved = catalog_fact.provenance.retrieved_at
        other_hash = catalog_fact.provenance.content_hash
    return Finding(
        provider=provider,
        samyak_id=samyak_id,
        fact_name=fact_name,
        kind="mismatch",
        comparability_status="comparable",
        explanation=(
            "The current normalizer does not reproduce the catalog fact that cites "
            "this archived document. This is not a source conflict."
        ),
        selected_value=selected_value,
        selected_source_id=selected_source_id,
        selected_source_rank=selected_rank,
        selected_generation=generation,
        selected_retrieved_at=selected_retrieved,
        selected_content_hash=selected_hash,
        selected_confidence=selected_confidence,
        other_value=other_value,
        other_source_id=other_source_id,
        other_generation=generation if other_source_id is not None else None,
        other_retrieved_at=other_retrieved,
        other_content_hash=other_hash,
    )


def _rank_for_source(tools: _ProviderTools, source_id: str) -> int | None:
    try:
        source_type = tools.source_type_for_id(source_id)
    except ModelCatalogError:
        return None
    rank = tools.ranks.get(source_type)
    return None if rank is None else int(rank)


def _collapse_repeated(findings: Sequence[Finding]) -> list[Finding]:
    """Keep the first identical evidence row and record how many were repeated."""
    counts: dict[str, int] = defaultdict(int)
    first: dict[str, Finding] = {}
    order: list[str] = []
    for finding in findings:
        key = json.dumps(
            {
                "provider": finding.provider,
                "samyak_id": finding.samyak_id,
                "fact_name": finding.fact_name,
                "kind": finding.kind,
                "selected_source_id": finding.selected_source_id,
                "other_source_id": finding.other_source_id,
                "selected_value": _json_ready(finding.selected_value),
                "other_value": _json_ready(finding.other_value),
                "selected_source_rank": finding.selected_source_rank,
                "other_source_rank": finding.other_source_rank,
                "selected_content_hash": finding.selected_content_hash,
                "other_content_hash": finding.other_content_hash,
                "selected_generation": finding.selected_generation,
                "other_generation": finding.other_generation,
                "claims": [item.to_dict() for item in finding.claims],
            },
            sort_keys=True,
            default=str,
        )
        counts[key] += 1
        if key not in first:
            first[key] = finding
            order.append(key)
    collapsed: list[Finding] = []
    for key in order:
        finding = first[key]
        count = counts[key]
        if count > 1:
            finding = replace(
                finding,
                explanation=f"{finding.explanation} Repeated {count} times.",
            )
        collapsed.append(finding)
    return collapsed


def _catalog_claim_value(value: Any) -> Any:
    converted, _value_type = _canonicalize(value)
    if hasattr(value, "tokens") and hasattr(value, "kind"):
        return value.tokens
    if hasattr(value, "provider_model_id") and not isinstance(value, str):
        return value.provider_model_id
    if isinstance(value, tuple) and value and hasattr(value[0], "provider_model_id"):
        return tuple(item.provider_model_id for item in value)
    if isinstance(value, tuple) and value and hasattr(value[0], "value"):
        return tuple(item.value for item in value)
    return converted


def _catalog_known_value(fact: Fact[Any]) -> Any:
    return _catalog_claim_value(fact.value)


def _status_for(finding: Finding, checks: Sequence[CatalogCheck]) -> str:
    for check in checks:
        if (
            check.provider == finding.provider
            and check.samyak_id == finding.samyak_id
            and check.fact_name == finding.fact_name
        ):
            return check.status
    return "historical"


def _with_status(finding: Finding, status: str) -> Finding:
    if finding.comparability_status == status:
        return finding
    return Finding(
        provider=finding.provider,
        samyak_id=finding.samyak_id,
        fact_name=finding.fact_name,
        kind=finding.kind,
        comparability_status=status,
        explanation=finding.explanation,
        selected_value=finding.selected_value,
        selected_source_id=finding.selected_source_id,
        selected_source_rank=finding.selected_source_rank,
        selected_generation=finding.selected_generation,
        selected_retrieved_at=finding.selected_retrieved_at,
        selected_content_hash=finding.selected_content_hash,
        selected_confidence=finding.selected_confidence,
        other_value=finding.other_value,
        other_source_id=finding.other_source_id,
        other_source_rank=finding.other_source_rank,
        other_generation=finding.other_generation,
        other_retrieved_at=finding.other_retrieved_at,
        other_content_hash=finding.other_content_hash,
        other_confidence=finding.other_confidence,
        claims=finding.claims,
    )


def _canonicalize(value: Any) -> tuple[Any, str]:
    if isinstance(value, bool):
        return value, "bool"
    if isinstance(value, StrEnum):
        return value.value, "str"
    if isinstance(value, int):
        return value, "int"
    if isinstance(value, str):
        return value, "str"
    if isinstance(value, tuple):
        converted = tuple(_canonicalize(item)[0] for item in value)
        return converted, "sequence"
    if hasattr(value, "tokens") and hasattr(value, "kind"):
        return value.tokens, "int"
    return value, type(value).__name__


def _value_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, str):
        return "str"
    if isinstance(value, tuple):
        return "sequence"
    return type(value).__name__


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def _finding_sort_key(finding: Finding) -> tuple[Any, ...]:
    return (
        finding.provider,
        finding.samyak_id,
        finding.fact_name,
        _KIND_ORDER.index(finding.kind),
        finding.selected_source_id or "",
        finding.other_source_id or "",
    )


def _check_sort_key(check: CatalogCheck) -> tuple[str, str, str]:
    return (check.provider, check.samyak_id, check.fact_name)


def _render_text(report: OverrideReport) -> str:
    lines = [
        "Source override diagnostic",
        f"Parser: {report.parser_version}",
        f"Catalog generated_at: {report.catalog_generated_at or 'not loaded'}",
        "",
        "Generations",
    ]
    if not report.generations:
        lines.append("  none selected")
    for item in report.generations:
        stamps = ", ".join(item.retrieved_at) if item.retrieved_at else "no retrieved sources"
        lines.append(f"  {item.provider}  {item.refresh_id}  sources {item.source_count}  {stamps}")
        if item.normalization_timestamp_status is not None:
            clock = item.normalization_timestamp or "unresolved"
            lines.append(
                f"    normalization_timestamp {item.normalization_timestamp_status}: {clock}"
            )
    lines.extend(["", "Limitations"])
    if not report.limitations:
        lines.append("  none")
    for item in report.limitations:
        where = item.provider or "catalog"
        generation = f" {item.generation}" if item.generation else ""
        lines.append(f"  {item.code}: {where}{generation}: {item.message}")
    counts: dict[str, int] = defaultdict(int)
    for finding in report.findings:
        counts[finding.kind] += 1
    lines.extend(["", f"Findings: {len(report.findings)}"])
    for kind in _KIND_ORDER:
        if counts[kind]:
            lines.append(f"  {kind}: {counts[kind]}")
    for finding in report.findings:
        lines.append(
            f"{finding.provider} {finding.samyak_id} {finding.fact_name} "
            f"{finding.kind} [{finding.comparability_status}]"
        )
        if finding.selected_source_id is not None:
            lines.append(
                f"  selected {finding.selected_value!r} {finding.selected_source_id} "
                f"rank {finding.selected_source_rank} {finding.selected_generation}"
            )
        if finding.other_source_id is not None:
            lines.append(
                f"  other {finding.other_value!r} {finding.other_source_id} "
                f"rank {finding.other_source_rank} {finding.other_generation}"
            )
        for claim in finding.claims:
            lines.append(
                f"  claim {claim.value!r} {claim.source_id} rank {claim.source_rank} "
                f"{claim.generation}"
            )
        lines.append(f"  {finding.explanation}")
    if report.catalog_checks:
        summary = _catalog_check_summary(report)
        lines.extend(
            [
                "",
                "Catalog checks",
                (
                    "  coverage: "
                    f"{summary['reproducibility_coverage_denominator']} fact checks; "
                    f"matching {summary['matching_count']}; "
                    f"mismatching {summary['mismatching_count']}; "
                    f"not comparable {summary['not_comparable_count']}"
                ),
                (
                    "  selected-generation reproducibility: "
                    f"{summary['selected_generation_reproducibility']}"
                ),
                "  provider-wide reproducibility: not claimed",
            ]
        )
        for check in report.catalog_checks:
            reason = "" if check.reason is None else f" ({check.reason})"
            lines.append(
                f"  {check.status}: {check.samyak_id} {check.fact_name} {check.generation}{reason}"
            )
    else:
        check_summary = _catalog_check_summary(report)
        reason = check_summary.get("reason")
        if isinstance(reason, str):
            lines.extend(["", reason])
            for item in check_summary["selected_generations"]:
                lines.append(f"  selected: {item['provider']} {item['refresh_id']}")
            for item in check_summary["catalog_provenance_generations"]:
                generation = item["refresh_id"] or item["verified_at"] or "unavailable"
                lines.append(f"  catalog provenance: {item['provider']} {generation}")
    reconstruction = report.reconstruction
    lines.extend(
        [
            "",
            "Direct archive reconstruction (not a merged catalog)",
            f"  records: {reconstruction.record_count}",
            f"  current_documented: {reconstruction.current_documented_count}",
            f"  historical_reference_only: {reconstruction.historical_reference_only_count}",
            (
                "  not_observed_in_current_capture: "
                f"{reconstruction.not_observed_in_current_capture_count}"
            ),
            f"  legacy_uncertified: {reconstruction.legacy_uncertified_count}",
            f"  added reconstructed identities: {len(reconstruction.added_reconstructed_ids)}",
            "  retained absence/source-absence decisions: not evaluated (no merge was run)",
        ]
    )
    if reconstruction.unreconstructed_providers:
        lines.append(
            "  unreconstructed providers: " + ", ".join(reconstruction.unreconstructed_providers)
        )
    lines.append("")
    return "\n".join(lines)
