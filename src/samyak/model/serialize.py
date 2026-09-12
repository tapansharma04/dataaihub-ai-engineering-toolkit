"""JSON serialization for Model Intelligence catalogs.

Mirrors corpus report reconstruction: explicit dict shape, fail closed, no
silent repair. Not part of the public Samyak API in this milestone.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from samyak.model.catalog import (
    SUPPORTED_CATALOG_SCHEMA_VERSIONS,
    CatalogFreshness,
    CatalogNotice,
    CatalogOverlay,
    FreshnessStatus,
    ModelCatalog,
    SourceRetrieval,
    build_catalog,
)
from samyak.model.errors import CatalogDecodeError, CatalogSchemaError, CatalogValidationError
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
    SourceKind,
    identity_from_mapping,
)
from samyak.model.identity import IdentityKind, ModelIdentity
from samyak.model.records import ModelRecord, replacement_identities_from_json

_CATALOG_FIELDS = frozenset(
    {
        "product",
        "capability",
        "version",
        "catalog_schema_version",
        "generated_at",
        "freshness",
        "providers",
        "models",
        "notices",
    }
)
_FRESHNESS_FIELDS = frozenset(
    {
        "status",
        "generated_at",
        "overlay",
        "oldest_verified_at",
        "source_retrieved_at",
        "stale_after",
    }
)
_FACT_FIELDS = frozenset({"status", "value", "provenance", "claims"})
_CLAIM_FIELDS = frozenset({"value", "provenance"})
_PROVENANCE_FIELDS = frozenset(
    {
        "provider",
        "source_id",
        "source_kind",
        "source_url",
        "content_hash",
        "retrieved_at",
        "observed_at",
        "verified_at",
        "confidence",
    }
)
_RECORD_FIELDS = frozenset(
    {
        "samyak_id",
        "provider_id",
        "provider_model_id",
        "identity_kind",
        "display_name",
        "aliases",
        "resolves_to",
        "family",
        "lifecycle",
        "deprecated_at",
        "retirement_at",
        "replacement",
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
        "observed_sources",
    }
)


def catalog_to_json(catalog: ModelCatalog) -> str:
    """Serialize a catalog to stable, pretty-printed JSON."""
    return json.dumps(catalog.to_dict(), indent=2, sort_keys=False, ensure_ascii=False) + "\n"


def catalog_from_json(text: str) -> ModelCatalog:
    """Deserialize catalog JSON. Rejects non-objects."""
    if not isinstance(text, str):
        raise CatalogDecodeError("catalog JSON must be a string")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CatalogDecodeError("catalog JSON is malformed") from exc
    return catalog_from_dict(payload)


def catalog_from_dict(data: object) -> ModelCatalog:
    """Rebuild a catalog from ``ModelCatalog.to_dict()`` output."""
    mapping = _require_object("catalog", data, _CATALOG_FIELDS)
    schema_version = mapping.get("catalog_schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise CatalogDecodeError("catalog_schema_version must be an integer")
    if schema_version not in SUPPORTED_CATALOG_SCHEMA_VERSIONS:
        raise CatalogSchemaError(schema_version)
    models_raw = mapping.get("models")
    if not isinstance(models_raw, list):
        raise CatalogDecodeError("models must be an array")
    notices_raw = mapping.get("notices")
    if not isinstance(notices_raw, list):
        raise CatalogDecodeError("notices must be an array")
    try:
        models = tuple(_record_from_dict(item, index) for index, item in enumerate(models_raw))
        freshness = _freshness_from_dict(mapping.get("freshness"))
        notices = tuple(_notice_from_dict(item, index) for index, item in enumerate(notices_raw))
        catalog = build_catalog(
            models=models,
            generated_at=_require_str("generated_at", mapping.get("generated_at")),
            freshness=freshness,
            version=_require_str("version", mapping.get("version")),
            notices=notices,
        )
    except CatalogValidationError as exc:
        raise CatalogDecodeError(str(exc)) from exc
    if catalog.product != _require_str("product", mapping.get("product")):
        raise CatalogDecodeError("product must be 'samyak'")
    if catalog.capability != _require_str("capability", mapping.get("capability")):
        raise CatalogDecodeError("capability must be 'model'")
    providers_raw = mapping.get("providers")
    if not isinstance(providers_raw, list):
        raise CatalogDecodeError("providers must be an array")
    if [item.to_dict() for item in catalog.providers] != providers_raw:
        raise CatalogDecodeError("providers must match models")
    return catalog


def _freshness_from_dict(data: object) -> CatalogFreshness:
    mapping = _require_object("freshness", data, _FRESHNESS_FIELDS)
    retrieved_raw = mapping.get("source_retrieved_at")
    if not isinstance(retrieved_raw, list):
        raise CatalogDecodeError("freshness.source_retrieved_at must be an array")
    retrieved: list[SourceRetrieval] = []
    for index, item in enumerate(retrieved_raw):
        row = _require_object(
            f"freshness.source_retrieved_at[{index}]",
            item,
            frozenset({"source_id", "retrieved_at"}),
        )
        retrieved.append(
            SourceRetrieval(
                source_id=_require_str(
                    f"freshness.source_retrieved_at[{index}].source_id",
                    row.get("source_id"),
                ),
                retrieved_at=_require_str(
                    f"freshness.source_retrieved_at[{index}].retrieved_at",
                    row.get("retrieved_at"),
                ),
            )
        )
    try:
        status = FreshnessStatus(_require_str("freshness.status", mapping.get("status")))
        overlay = CatalogOverlay(_require_str("freshness.overlay", mapping.get("overlay")))
    except ValueError as exc:
        raise CatalogDecodeError("freshness has an unknown status or overlay") from exc
    return CatalogFreshness(
        status=status,
        generated_at=_require_str("freshness.generated_at", mapping.get("generated_at")),
        overlay=overlay,
        oldest_verified_at=_optional_str(
            "freshness.oldest_verified_at", mapping.get("oldest_verified_at")
        ),
        source_retrieved_at=tuple(retrieved),
        stale_after=_optional_str("freshness.stale_after", mapping.get("stale_after")),
    )


def _notice_from_dict(data: object, index: int) -> CatalogNotice:
    mapping = _require_object(f"notices[{index}]", data, frozenset({"code", "message"}))
    return CatalogNotice(
        code=_require_str(f"notices[{index}].code", mapping.get("code")),
        message=_require_str(f"notices[{index}].message", mapping.get("message")),
    )


def _record_from_dict(data: object, index: int) -> ModelRecord:
    prefix = f"models[{index}]"
    mapping = _require_object(prefix, data, _RECORD_FIELDS)
    identity = ModelIdentity(
        provider_id=_require_str(f"{prefix}.provider_id", mapping.get("provider_id")),
        provider_model_id=_require_str(
            f"{prefix}.provider_model_id", mapping.get("provider_model_id")
        ),
    )
    raw_id = _require_str(f"{prefix}.samyak_id", mapping.get("samyak_id"))
    if raw_id != identity.samyak_id:
        raise CatalogDecodeError(f"{prefix}.samyak_id does not match provider_id/provider_model_id")
    try:
        kind = IdentityKind(_require_str(f"{prefix}.identity_kind", mapping.get("identity_kind")))
    except ValueError as exc:
        raise CatalogDecodeError(f"{prefix}.identity_kind is not a known identity kind") from exc
    sources = mapping.get("observed_sources")
    if not isinstance(sources, list):
        raise CatalogDecodeError(f"{prefix}.observed_sources must be an array")
    observed = tuple(
        _require_str(f"{prefix}.observed_sources[{i}]", item) for i, item in enumerate(sources)
    )
    try:
        record = ModelRecord(
            identity=identity,
            identity_kind=kind,
            display_name=_fact(f"{prefix}.display_name", mapping.get("display_name"), _as_str),
            aliases=_fact(f"{prefix}.aliases", mapping.get("aliases"), _as_str_tuple),
            resolves_to=_fact(f"{prefix}.resolves_to", mapping.get("resolves_to"), _as_identity),
            family=_fact(f"{prefix}.family", mapping.get("family"), _as_str),
            lifecycle=_fact(f"{prefix}.lifecycle", mapping.get("lifecycle"), _as_lifecycle),
            deprecated_at=_fact(f"{prefix}.deprecated_at", mapping.get("deprecated_at"), _as_str),
            retirement_at=_fact(f"{prefix}.retirement_at", mapping.get("retirement_at"), _as_str),
            replacement=_fact(f"{prefix}.replacement", mapping.get("replacement"), _as_replacement),
            context_window=_fact(
                f"{prefix}.context_window", mapping.get("context_window"), _as_context_window
            ),
            max_input_tokens=_fact(
                f"{prefix}.max_input_tokens", mapping.get("max_input_tokens"), _as_positive_int
            ),
            max_output_tokens=_fact(
                f"{prefix}.max_output_tokens", mapping.get("max_output_tokens"), _as_positive_int
            ),
            input_modalities=_fact(
                f"{prefix}.input_modalities", mapping.get("input_modalities"), _as_modalities
            ),
            output_modalities=_fact(
                f"{prefix}.output_modalities", mapping.get("output_modalities"), _as_modalities
            ),
            tool_calling=_fact(f"{prefix}.tool_calling", mapping.get("tool_calling"), _as_bool),
            structured_output=_fact(
                f"{prefix}.structured_output", mapping.get("structured_output"), _as_bool
            ),
            api_access=_fact(f"{prefix}.api_access", mapping.get("api_access"), _as_bool),
            availability_scope=_fact(
                f"{prefix}.availability_scope", mapping.get("availability_scope"), _as_scope
            ),
            regions=_fact(f"{prefix}.regions", mapping.get("regions"), _as_str_tuple),
        )
    except CatalogValidationError as exc:
        raise CatalogDecodeError(f"{prefix}: {exc}") from exc
    if observed != record.observed_sources:
        raise CatalogDecodeError(
            f"{prefix}.observed_sources must match fact-level provenance source ids"
        )
    return record


def _fact(name: str, data: object, decode_value: Callable[[str, object], Any]) -> Fact[Any]:
    mapping = _require_object(name, data, _FACT_FIELDS)
    try:
        status = FactStatus(_require_str(f"{name}.status", mapping.get("status")))
    except ValueError as exc:
        raise CatalogDecodeError(f"{name}.status is not a known fact status") from exc
    provenance = _provenance(f"{name}.provenance", mapping.get("provenance"))
    claims_raw = mapping.get("claims")
    if not isinstance(claims_raw, list):
        raise CatalogDecodeError(f"{name}.claims must be an array")
    claims = tuple(
        _claim(f"{name}.claims[{index}]", item, decode_value)
        for index, item in enumerate(claims_raw)
    )
    raw_value = mapping.get("value")
    value = None if status is not FactStatus.KNOWN else decode_value(f"{name}.value", raw_value)
    if status is not FactStatus.KNOWN and raw_value is not None:
        raise CatalogDecodeError(f"{name}.value must be null when status is {status.value}")
    try:
        return Fact(status=status, value=value, provenance=provenance, claims=claims)
    except CatalogValidationError as exc:
        raise CatalogDecodeError(f"{name}: {exc}") from exc


def _claim(name: str, data: object, decode_value: Callable[[str, object], Any]) -> FactClaim[Any]:
    mapping = _require_object(name, data, _CLAIM_FIELDS)
    provenance = _require_provenance(f"{name}.provenance", mapping.get("provenance"))
    value = decode_value(f"{name}.value", mapping.get("value"))
    try:
        return FactClaim(value=value, provenance=provenance)
    except CatalogValidationError as exc:
        raise CatalogDecodeError(f"{name}: {exc}") from exc


def _provenance(name: str, data: object) -> Provenance | None:
    if data is None:
        return None
    return _require_provenance(name, data)


def _require_provenance(name: str, data: object) -> Provenance:
    mapping = _require_object(name, data, _PROVENANCE_FIELDS)
    try:
        kind = SourceKind(_require_str(f"{name}.source_kind", mapping.get("source_kind")))
    except ValueError as exc:
        raise CatalogDecodeError(f"{name}.source_kind is not a known source kind") from exc
    confidence_raw = mapping.get("confidence")
    if confidence_raw is None:
        confidence: Confidence | None = None
    else:
        try:
            confidence = Confidence(_require_str(f"{name}.confidence", confidence_raw))
        except ValueError as exc:
            raise CatalogDecodeError(f"{name}.confidence is not a known confidence") from exc
    try:
        return Provenance(
            provider=_require_str(f"{name}.provider", mapping.get("provider")),
            source_id=_require_str(f"{name}.source_id", mapping.get("source_id")),
            source_kind=kind,
            source_url=_optional_str(f"{name}.source_url", mapping.get("source_url")),
            content_hash=_optional_str(f"{name}.content_hash", mapping.get("content_hash")),
            retrieved_at=_optional_str(f"{name}.retrieved_at", mapping.get("retrieved_at")),
            observed_at=_optional_str(f"{name}.observed_at", mapping.get("observed_at")),
            verified_at=_optional_str(f"{name}.verified_at", mapping.get("verified_at")),
            confidence=confidence,
        )
    except CatalogValidationError as exc:
        raise CatalogDecodeError(f"{name}: {exc}") from exc


def _as_str(name: str, value: object) -> str:
    return _require_str(name, value)


def _as_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise CatalogDecodeError(f"{name} must be a boolean")
    return value


def _as_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CatalogDecodeError(f"{name} must be an integer")
    if value <= 0:
        raise CatalogDecodeError(f"{name} must be > 0")
    return value


def _as_str_tuple(name: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise CatalogDecodeError(f"{name} must be an array of strings")
    return tuple(_require_str(f"{name}[{index}]", item) for index, item in enumerate(value))


def _as_lifecycle(name: str, value: object) -> LifecycleState:
    raw = _require_str(name, value)
    try:
        return LifecycleState(raw)
    except ValueError as exc:
        raise CatalogDecodeError(f"{name} is not a known lifecycle state") from exc


def _as_scope(name: str, value: object) -> AvailabilityScope:
    raw = _require_str(name, value)
    try:
        return AvailabilityScope(raw)
    except ValueError as exc:
        raise CatalogDecodeError(f"{name} is not a known availability scope") from exc


def _as_modalities(name: str, value: object) -> tuple[Modality, ...]:
    if not isinstance(value, list):
        raise CatalogDecodeError(f"{name} must be an array")
    result: list[Modality] = []
    for index, item in enumerate(value):
        raw = _require_str(f"{name}[{index}]", item)
        try:
            result.append(Modality(raw))
        except ValueError as exc:
            raise CatalogDecodeError(f"{name}[{index}] is not a known modality") from exc
    return tuple(result)


def _as_identity(name: str, value: object) -> ModelIdentity:
    try:
        return identity_from_mapping(name, value)
    except CatalogValidationError as exc:
        raise CatalogDecodeError(str(exc)) from exc


def _as_replacement(name: str, value: object) -> tuple[ModelIdentity, ...]:
    try:
        return replacement_identities_from_json(name, value)
    except CatalogValidationError as exc:
        raise CatalogDecodeError(str(exc)) from exc


def _as_context_window(name: str, value: object) -> ContextWindow:
    mapping = _require_object(name, value, frozenset({"tokens", "kind"}))
    tokens = _as_positive_int(f"{name}.tokens", mapping.get("tokens"))
    try:
        kind = ContextWindowKind(_require_str(f"{name}.kind", mapping.get("kind")))
    except ValueError as exc:
        raise CatalogDecodeError(f"{name}.kind is not a known context window kind") from exc
    return ContextWindow(tokens=tokens, kind=kind)


def _require_object(name: str, data: object, allowed: frozenset[str]) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise CatalogDecodeError(f"{name} must be an object")
    extra = set(data) - allowed
    if extra:
        raise CatalogDecodeError(f"{name} has unexpected fields: {sorted(extra)}")
    missing = allowed - set(data)
    if missing:
        raise CatalogDecodeError(f"{name} is missing fields: {sorted(missing)}")
    return data


def _require_str(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise CatalogDecodeError(f"{name} must be a string")
    return value


def _optional_str(name: str, value: object) -> str | None:
    if value is None:
        return None
    return _require_str(name, value)
