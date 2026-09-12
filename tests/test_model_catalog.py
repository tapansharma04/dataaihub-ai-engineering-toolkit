"""Deterministic tests for the Model Intelligence catalog domain."""

from __future__ import annotations

import copy
import json
from dataclasses import FrozenInstanceError

import pytest

from samyak.model.catalog import (
    CATALOG_SCHEMA_VERSION,
    CatalogFreshness,
    CatalogNotice,
    CatalogOverlay,
    FreshnessStatus,
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
    unverified_fact,
)
from samyak.model.identity import IdentityKind, ModelIdentity, parse_samyak_id
from samyak.model.records import ModelRecord
from samyak.model.serialize import catalog_from_dict, catalog_from_json, catalog_to_json
from samyak.model.store import CatalogSource

GENERATED_AT = "2026-09-09T12:00:00+00:00"


def _provenance(**overrides: object) -> Provenance:
    payload: dict[str, object] = {
        "provider": "openai",
        "source_id": "openai-docs-model-page",
        "source_kind": SourceKind.PROVIDER_DOCS,
        "source_url": "https://example.invalid/models/gpt-test.md",
        "content_hash": "sha256:abc",
        "retrieved_at": "2026-09-09T11:59:00+00:00",
        "observed_at": None,
        "verified_at": "2026-09-09T12:00:00+00:00",
        "confidence": Confidence.HIGH,
    }
    payload.update(overrides)
    return Provenance(**payload)  # type: ignore[arg-type]


def _known(value: object, **overrides: object) -> Fact:
    return Fact(status=FactStatus.KNOWN, value=value, provenance=_provenance(**overrides))


def _unknown(**overrides: object) -> Fact:
    return Fact(status=FactStatus.UNKNOWN, provenance=_provenance(**overrides))


def _conflict(*values: object) -> Fact:
    claims = tuple(
        FactClaim(value=value, provenance=_provenance(source_id=f"source-{index}"))
        for index, value in enumerate(values)
    )
    return Fact(status=FactStatus.CONFLICT, claims=claims)


def _freshness() -> CatalogFreshness:
    return CatalogFreshness(
        status=FreshnessStatus.AS_OF,
        generated_at=GENERATED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )


def _model(
    provider_model_id: str = "gpt-test",
    *,
    provider_id: str = "openai",
    identity_kind: IdentityKind = IdentityKind.CANONICAL,
    **fields: object,
) -> ModelRecord:
    return ModelRecord(
        identity=ModelIdentity(provider_id=provider_id, provider_model_id=provider_model_id),
        identity_kind=identity_kind,
        **fields,  # type: ignore[arg-type]
    )


def test_samyak_id_is_stable_and_splits_on_first_colon() -> None:
    identity = ModelIdentity(provider_id="openai", provider_model_id="gpt-5.6:preview")
    assert identity.samyak_id == "openai:gpt-5.6:preview"
    parsed = parse_samyak_id(identity.samyak_id)
    assert parsed == identity


def test_alias_does_not_replace_canonical_identity() -> None:
    canonical = _model("gpt-5.6-sol")
    alias = _model(
        "gpt-5.6",
        identity_kind=IdentityKind.ALIAS,
        resolves_to=_known(canonical.identity),
    )
    catalog = build_catalog(
        models=(alias, canonical),
        generated_at=GENERATED_AT,
        freshness=_freshness(),
    )
    assert catalog.models[0].samyak_id == "openai:gpt-5.6"
    assert catalog.models[1].samyak_id == "openai:gpt-5.6-sol"
    assert catalog.models[0].identity_kind is IdentityKind.ALIAS
    assert catalog.models[1].identity_kind is IdentityKind.CANONICAL
    assert catalog.models[0].identity != catalog.models[1].identity
    assert catalog.models[0].resolves_to.value == canonical.identity


def test_invalid_provider_and_model_ids() -> None:
    assert ModelIdentity(provider_id="anthropic", provider_model_id="claude").samyak_id == (
        "anthropic:claude"
    )
    with pytest.raises(CatalogValidationError, match="provider_id"):
        ModelIdentity(provider_id="", provider_model_id="gpt")
    with pytest.raises(CatalogValidationError, match="slug"):
        ModelIdentity(provider_id="OpenAI", provider_model_id="gpt")
    with pytest.raises(CatalogValidationError, match="provider_model_id"):
        ModelIdentity(provider_id="openai", provider_model_id="")
    with pytest.raises(CatalogValidationError, match="whitespace"):
        ModelIdentity(provider_id="openai", provider_model_id=" gpt")
    with pytest.raises(CatalogValidationError, match="samyak_id"):
        parse_samyak_id("gpt-test")


def test_valid_model_record_defaults_are_not_verified() -> None:
    record = _model()
    assert record.samyak_id == "openai:gpt-test"
    assert record.tool_calling.status is FactStatus.NOT_VERIFIED
    assert record.tool_calling.value is None
    assert record.lifecycle.status is FactStatus.NOT_VERIFIED
    assert record.deprecated_at.status is FactStatus.NOT_VERIFIED


def test_lifecycle_states_and_optional_dates() -> None:
    record = _model(
        lifecycle=_known(LifecycleState.DEPRECATED, source_id="openai-docs-deprecations"),
        deprecated_at=_known("2026-06-11"),
        retirement_at=_known("2026-12-11"),
        replacement=_known((ModelIdentity(provider_id="openai", provider_model_id="gpt-5.6-sol"),)),
    )
    assert record.lifecycle.value is LifecycleState.DEPRECATED
    assert record.deprecated_at.value == "2026-06-11"
    assert record.retirement_at.value == "2026-12-11"
    unknown = _model(lifecycle=_unknown())
    assert unknown.lifecycle.status is FactStatus.UNKNOWN
    assert unknown.lifecycle.value is None
    assert not hasattr(LifecycleState, "UNKNOWN")
    unverified = _model(lifecycle=unverified_fact())
    assert unverified.lifecycle.status is FactStatus.NOT_VERIFIED


def test_invalid_lifecycle_and_dates() -> None:
    with pytest.raises(CatalogValidationError, match="lifecycle"):
        _model(lifecycle=_known("deprecated"))  # type: ignore[arg-type]
    with pytest.raises(CatalogValidationError, match="ISO calendar date"):
        _model(deprecated_at=_known("09/09/2026"))
    with pytest.raises(CatalogValidationError, match="ISO calendar date"):
        _model(retirement_at=_known("2026-13-01"))
    with pytest.raises(CatalogValidationError, match="timezone"):
        Provenance(
            provider="openai",
            source_id="src",
            source_kind=SourceKind.PROVIDER_DOCS,
            retrieved_at="2026-09-09T12:00:00",
        )


def test_every_fact_status() -> None:
    p1 = _provenance(source_id="source-a")
    conflict = _conflict("family-a", "family-b")
    statuses = {
        FactStatus.KNOWN: Fact(status=FactStatus.KNOWN, value=True, provenance=p1),
        FactStatus.UNKNOWN: Fact(status=FactStatus.UNKNOWN, provenance=p1),
        FactStatus.NOT_APPLICABLE: Fact(status=FactStatus.NOT_APPLICABLE, provenance=p1),
        FactStatus.NOT_VERIFIED: unverified_fact(),
        FactStatus.CONFLICT: conflict,
    }
    assert set(statuses) == set(FactStatus)
    record = _model(
        tool_calling=statuses[FactStatus.KNOWN],
        structured_output=statuses[FactStatus.UNKNOWN],
        api_access=statuses[FactStatus.NOT_APPLICABLE],
        display_name=statuses[FactStatus.NOT_VERIFIED],
        family=conflict,
    )
    assert record.tool_calling.is_known_true()
    assert not record.structured_output.is_known_false()
    assert record.structured_output.value is None
    assert record.api_access.status is FactStatus.NOT_APPLICABLE
    assert record.family.status is FactStatus.CONFLICT
    assert record.family.value is None
    assert [claim.value for claim in record.family.claims] == ["family-a", "family-b"]


def test_unknown_is_not_false() -> None:
    unknown = _unknown()
    known_false = _known(False)
    assert unknown != known_false
    assert unknown.value is not False
    assert unknown.value is None
    assert not unknown.is_known_false()
    assert not unknown.is_known_true()
    assert known_false.is_known_false()
    with pytest.raises(TypeError, match="unknown is not false"):
        bool(unknown)
    with pytest.raises(CatalogValidationError, match="unknown is not false"):
        Fact(status=FactStatus.UNKNOWN, value=False, provenance=_provenance())
    record = _model(tool_calling=unknown)
    payload = record.to_dict()
    assert payload["tool_calling"]["status"] == "unknown"
    assert payload["tool_calling"]["value"] is None
    assert payload["tool_calling"]["value"] is not False


def test_provenance_is_preserved_and_not_invented() -> None:
    provenance = _provenance(
        observed_at=None, retrieved_at=None, verified_at=None, content_hash=None
    )
    record = _model(tool_calling=Fact(status=FactStatus.KNOWN, value=True, provenance=provenance))
    encoded = record.to_dict()["tool_calling"]["provenance"]
    assert encoded["source_id"] == "openai-docs-model-page"
    assert encoded["source_url"] == "https://example.invalid/models/gpt-test.md"
    assert encoded["retrieved_at"] is None
    assert encoded["observed_at"] is None
    assert encoded["verified_at"] is None
    assert encoded["content_hash"] is None
    assert record.tool_calling.provenance is not None
    assert record.tool_calling.provenance.retrieved_at is None


def test_duplicate_identity_rejected() -> None:
    with pytest.raises(CatalogValidationError, match="duplicate canonical identity"):
        build_catalog(
            models=(_model("gpt-test"), _model("gpt-test")),
            generated_at=GENERATED_AT,
            freshness=_freshness(),
        )


def test_catalog_round_trip_and_deterministic_json() -> None:
    canonical = _model(
        "gpt-5.6-sol",
        display_name=_known("GPT-5.6 Sol"),
        lifecycle=_known(LifecycleState.ACTIVE),
        context_window=_known(ContextWindow(tokens=1_050_000, kind=ContextWindowKind.COMBINED)),
        input_modalities=_known((Modality.TEXT, Modality.IMAGE)),
        output_modalities=_known((Modality.TEXT,)),
        tool_calling=_known(True),
        structured_output=_known(True),
        api_access=_known(True),
        availability_scope=_known(AvailabilityScope.GLOBAL),
    )
    alias = _model(
        "gpt-5.6",
        identity_kind=IdentityKind.ALIAS,
        resolves_to=_known(canonical.identity),
        lifecycle=_known(LifecycleState.ACTIVE),
    )
    catalog = build_catalog(
        models=(canonical, alias),
        generated_at=GENERATED_AT,
        freshness=_freshness(),
        notices=(CatalogNotice(code="TEST_NOTICE", message="fixture catalog"),),
    )
    payload = catalog.to_dict()
    assert payload["product"] == "samyak"
    assert payload["capability"] == "model"
    assert payload["catalog_schema_version"] == CATALOG_SCHEMA_VERSION
    assert "findings" not in payload
    assert "utility" not in payload
    rebuilt = catalog_from_dict(payload)
    assert rebuilt.to_dict() == payload
    assert catalog_from_json(catalog_to_json(catalog)).to_dict() == payload
    reversed_models = build_catalog(
        models=(alias, canonical),
        generated_at=GENERATED_AT,
        freshness=_freshness(),
        notices=(CatalogNotice(code="TEST_NOTICE", message="fixture catalog"),),
    )
    assert catalog_to_json(reversed_models) == catalog_to_json(catalog)
    assert json.loads(catalog_to_json(catalog)) == payload


def test_malformed_catalog_and_schema_rejected() -> None:
    catalog = build_catalog(
        models=(_model(),),
        generated_at=GENERATED_AT,
        freshness=_freshness(),
    )
    with pytest.raises(CatalogDecodeError, match="must be an object"):
        catalog_from_dict([])
    with pytest.raises(CatalogDecodeError, match="malformed"):
        catalog_from_json("{")
    payload = catalog.to_dict()
    payload["findings"] = []
    with pytest.raises(CatalogDecodeError, match="unexpected fields"):
        catalog_from_dict(payload)
    payload = catalog.to_dict()
    payload["catalog_schema_version"] = 99
    with pytest.raises(CatalogSchemaError, match="unsupported catalog_schema_version"):
        catalog_from_dict(payload)
    payload = catalog.to_dict()
    payload["capability"] = "corpus"
    with pytest.raises(CatalogDecodeError, match="capability"):
        catalog_from_dict(payload)


def test_malformed_fact_values_rejected() -> None:
    catalog = build_catalog(
        models=(_model(tool_calling=_known(True)),),
        generated_at=GENERATED_AT,
        freshness=_freshness(),
    )
    payload = catalog.to_dict()
    payload["models"][0]["tool_calling"]["value"] = 1
    with pytest.raises(CatalogDecodeError, match="boolean"):
        catalog_from_dict(payload)
    payload = catalog.to_dict()
    payload["models"][0]["lifecycle"]["status"] = "known"
    payload["models"][0]["lifecycle"]["value"] = "sunset"
    payload["models"][0]["lifecycle"]["provenance"] = _provenance().to_dict()
    payload["models"][0]["lifecycle"]["claims"] = []
    with pytest.raises(CatalogDecodeError, match="lifecycle"):
        catalog_from_dict(payload)


def test_replacement_references() -> None:
    successor = ModelIdentity(provider_id="openai", provider_model_id="gpt-5.6-sol")
    current = _model(
        "gpt-5",
        lifecycle=_known(LifecycleState.DEPRECATED),
        replacement=_known((successor,)),
    )
    catalog = build_catalog(
        models=(current,),
        generated_at=GENERATED_AT,
        freshness=_freshness(),
    )
    assert catalog.models[0].replacement.value == (successor,)
    with pytest.raises(CatalogValidationError, match="replacement must not include"):
        _model(
            "gpt-5",
            replacement=_known((ModelIdentity(provider_id="openai", provider_model_id="gpt-5"),)),
        )
    missing = ModelIdentity(provider_id="openai", provider_model_id="missing-target")
    with pytest.raises(CatalogValidationError, match="resolves_to"):
        build_catalog(
            models=(
                _model(
                    "gpt-alias",
                    identity_kind=IdentityKind.ALIAS,
                    resolves_to=_known(missing),
                ),
            ),
            generated_at=GENERATED_AT,
            freshness=_freshness(),
        )


def test_catalog_source_protocol_is_not_a_filesystem() -> None:
    snapshot = build_catalog(
        models=(_model(),),
        generated_at=GENERATED_AT,
        freshness=_freshness(),
    )

    class MemorySource:
        def load(self) -> object:
            return snapshot

    source: CatalogSource = MemorySource()
    loaded = source.load()
    assert loaded.models[0].samyak_id == "openai:gpt-test"
    assert loaded.freshness.overlay is CatalogOverlay.BUNDLED


def test_records_and_catalogs_are_immutable() -> None:
    record = _model()
    catalog = build_catalog(
        models=(record,),
        generated_at=GENERATED_AT,
        freshness=_freshness(),
    )
    with pytest.raises(FrozenInstanceError):
        record.identity_kind = IdentityKind.ALIAS  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        catalog.product = "other"  # type: ignore[misc]


def test_public_root_api_does_not_export_model_types() -> None:
    import samyak

    for name in (
        "ModelCatalog",
        "ModelRecord",
        "Fact",
        "CatalogSource",
        "build_catalog",
        "catalog_from_dict",
    ):
        assert name not in samyak.__all__
        assert not hasattr(samyak, name)


def test_deserialized_unknown_stays_null_not_false() -> None:
    catalog = build_catalog(
        models=(_model(tool_calling=_unknown()),),
        generated_at=GENERATED_AT,
        freshness=_freshness(),
    )
    payload = copy.deepcopy(catalog.to_dict())
    rebuilt = catalog_from_dict(payload)
    fact = rebuilt.models[0].tool_calling
    assert fact.status is FactStatus.UNKNOWN
    assert fact.value is None
    assert not fact.is_known_false()
    with pytest.raises(TypeError):
        bool(fact)


def test_conflict_preserves_competing_values_and_round_trips() -> None:
    conflict = _conflict(True, False)
    catalog = build_catalog(
        models=(_model(tool_calling=conflict),),
        generated_at=GENERATED_AT,
        freshness=_freshness(),
    )
    payload = catalog.to_dict()
    encoded = payload["models"][0]["tool_calling"]
    assert encoded["status"] == "conflict"
    assert encoded["value"] is None
    assert encoded["provenance"] is None
    assert [claim["value"] for claim in encoded["claims"]] == [True, False]
    assert encoded["claims"][0]["provenance"]["source_id"] == "source-0"
    assert encoded["claims"][1]["provenance"]["source_id"] == "source-1"
    rebuilt = catalog_from_json(catalog_to_json(catalog))
    assert rebuilt.to_dict() == payload
    claims = rebuilt.models[0].tool_calling.claims
    assert claims[0].value is True
    assert claims[1].value is False
    assert rebuilt.models[0].tool_calling.value is None


def test_conflict_without_distinct_values_is_rejected() -> None:
    with pytest.raises(CatalogValidationError, match="distinct competing values"):
        Fact(
            status=FactStatus.CONFLICT,
            claims=(
                FactClaim(value=True, provenance=_provenance(source_id="a")),
                FactClaim(value=True, provenance=_provenance(source_id="b")),
            ),
        )
    with pytest.raises(CatalogValidationError, match="at least two claims"):
        Fact(
            status=FactStatus.CONFLICT,
            claims=(FactClaim(value=True, provenance=_provenance(source_id="a")),),
        )
    with pytest.raises(CatalogValidationError, match="must have a value"):
        FactClaim(value=None, provenance=_provenance())
    catalog = build_catalog(
        models=(_model(tool_calling=_conflict(True, False)),),
        generated_at=GENERATED_AT,
        freshness=_freshness(),
    )
    payload = catalog.to_dict()
    payload["models"][0]["tool_calling"]["claims"] = [
        {"provenance": _provenance(source_id="only-source").to_dict()},
        {"provenance": _provenance(source_id="other-source").to_dict()},
    ]
    with pytest.raises(CatalogDecodeError, match="missing fields"):
        catalog_from_dict(payload)


def test_structured_conflict_values_round_trip() -> None:
    conflict = Fact(
        status=FactStatus.CONFLICT,
        claims=(
            FactClaim(
                value=ContextWindow(tokens=128_000, kind=ContextWindowKind.INPUT),
                provenance=_provenance(source_id="docs-a"),
            ),
            FactClaim(
                value=ContextWindow(tokens=200_000, kind=ContextWindowKind.COMBINED),
                provenance=_provenance(source_id="docs-b"),
            ),
        ),
    )
    catalog = build_catalog(
        models=(_model(context_window=conflict),),
        generated_at=GENERATED_AT,
        freshness=_freshness(),
    )
    payload = catalog.to_dict()
    encoded = payload["models"][0]["context_window"]
    assert [claim["value"] for claim in encoded["claims"]] == [
        {"tokens": 128000, "kind": "input"},
        {"tokens": 200000, "kind": "combined"},
    ]
    rebuilt = catalog_from_json(catalog_to_json(catalog))
    assert rebuilt.to_dict() == payload
    claims = rebuilt.models[0].context_window.claims
    assert claims[0].value == ContextWindow(tokens=128_000, kind=ContextWindowKind.INPUT)
    assert claims[1].value == ContextWindow(tokens=200_000, kind=ContextWindowKind.COMBINED)


def test_stale_after_must_be_timezone_aware_timestamp() -> None:
    freshness = CatalogFreshness(
        status=FreshnessStatus.STALE,
        generated_at=GENERATED_AT,
        overlay=CatalogOverlay.BUNDLED,
        stale_after="2026-10-01T00:00:00+00:00",
    )
    catalog = build_catalog(
        models=(_model(),),
        generated_at=GENERATED_AT,
        freshness=freshness,
    )
    assert catalog.freshness.stale_after == "2026-10-01T00:00:00+00:00"
    rebuilt = catalog_from_dict(catalog.to_dict())
    assert rebuilt.freshness.stale_after == "2026-10-01T00:00:00+00:00"
    with pytest.raises(CatalogValidationError, match="timezone"):
        CatalogFreshness(
            status=FreshnessStatus.AS_OF,
            generated_at=GENERATED_AT,
            overlay=CatalogOverlay.BUNDLED,
            stale_after="2026-10-01T00:00:00",
        )
    with pytest.raises(CatalogValidationError, match="ISO-8601"):
        CatalogFreshness(
            status=FreshnessStatus.AS_OF,
            generated_at=GENERATED_AT,
            overlay=CatalogOverlay.BUNDLED,
            stale_after="P7D",
        )
    payload = catalog.to_dict()
    payload["freshness"]["stale_after"] = "7d"
    with pytest.raises(CatalogDecodeError, match="ISO-8601"):
        catalog_from_dict(payload)


def test_retirement_must_not_precede_deprecation() -> None:
    with pytest.raises(CatalogValidationError, match="retirement_at must not be earlier"):
        _model(
            deprecated_at=_known("2026-12-11"),
            retirement_at=_known("2026-06-11"),
        )
    same_day = _model(
        deprecated_at=_known("2026-12-11"),
        retirement_at=_known("2026-12-11"),
    )
    assert same_day.retirement_at.value == "2026-12-11"
    retirement_only = _model(retirement_at=_known("2026-06-11"))
    assert retirement_only.deprecated_at.status is FactStatus.NOT_VERIFIED
    conflicted_deprecation = _model(
        deprecated_at=_conflict("2026-12-11", "2026-06-11"),
        retirement_at=_known("2026-01-01"),
    )
    assert conflicted_deprecation.retirement_at.value == "2026-01-01"
    unknown_deprecation = _model(
        deprecated_at=_unknown(),
        retirement_at=_known("2026-01-01"),
    )
    assert unknown_deprecation.deprecated_at.status is FactStatus.UNKNOWN


def test_observed_sources_are_derived_from_fact_provenance() -> None:
    record = _model(
        display_name=_known("Named", source_id="openai-docs-models-index"),
        tool_calling=_known(True, source_id="openai-docs-model-page"),
        family=_conflict("a", "b"),
    )
    assert record.observed_sources == (
        "openai-docs-models-index",
        "source-0",
        "source-1",
        "openai-docs-model-page",
    )
    catalog = build_catalog(
        models=(record,),
        generated_at=GENERATED_AT,
        freshness=_freshness(),
    )
    payload = catalog.to_dict()
    assert payload["models"][0]["observed_sources"] == list(record.observed_sources)
    payload["models"][0]["observed_sources"] = ["invented-source"]
    with pytest.raises(CatalogDecodeError, match="observed_sources must match"):
        catalog_from_dict(payload)
    empty = _model()
    assert empty.observed_sources == ()


def test_lifecycle_unknown_is_a_fact_status_not_a_state() -> None:
    catalog = build_catalog(
        models=(_model(lifecycle=_unknown()),),
        generated_at=GENERATED_AT,
        freshness=_freshness(),
    )
    payload = catalog.to_dict()
    payload["models"][0]["lifecycle"]["status"] = "known"
    payload["models"][0]["lifecycle"]["value"] = "unknown"
    payload["models"][0]["lifecycle"]["provenance"] = _provenance().to_dict()
    payload["models"][0]["lifecycle"]["claims"] = []
    with pytest.raises(CatalogDecodeError, match="lifecycle"):
        catalog_from_dict(payload)
    payload = catalog.to_dict()
    payload["models"][0]["lifecycle"]["status"] = "known"
    payload["models"][0]["lifecycle"]["value"] = "sunset"
    payload["models"][0]["lifecycle"]["provenance"] = _provenance().to_dict()
    payload["models"][0]["lifecycle"]["claims"] = []
    with pytest.raises(CatalogDecodeError, match="lifecycle"):
        catalog_from_dict(payload)
    assert LifecycleState.LEGACY.value == "legacy"


def test_lifecycle_legacy_round_trips() -> None:
    catalog = build_catalog(
        models=(_model(lifecycle=_known(LifecycleState.LEGACY)),),
        generated_at=GENERATED_AT,
        freshness=_freshness(),
    )
    restored = catalog_from_dict(catalog.to_dict())
    assert restored.models[0].lifecycle.value is LifecycleState.LEGACY
