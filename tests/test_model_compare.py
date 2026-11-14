"""Contract tests for evidence-safe model comparison projections."""

from __future__ import annotations

import json

import pytest

from samyak.model.compare import ModelComparisonError, compare_models, comparison_to_dict
from samyak.model.facts import (
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
)
from samyak.model.identity import ModelIdentity
from samyak.model.records import EvidenceDisposition, ModelRecord


def _provenance(source_id: str = "docs") -> Provenance:
    return Provenance(
        provider="openai",
        source_id=source_id,
        source_kind=SourceKind.PROVIDER_DOCS,
        verified_at="2026-11-01T00:00:00+00:00",
        confidence=Confidence.HIGH,
    )


def _known(value: object, source_id: str = "docs") -> Fact:
    return Fact(FactStatus.KNOWN, value, _provenance(source_id))


def _record(
    model_id: str,
    *,
    disposition: EvidenceDisposition | None = EvidenceDisposition.CURRENT_DOCUMENTED,
    **facts: object,
) -> ModelRecord:
    defaults: dict[str, object] = {
        "display_name": _known(model_id.title()),
        "lifecycle": _known(LifecycleState.ACTIVE),
        "api_access": _known(True),
        "tool_calling": _known(True),
        "structured_output": _known(True),
        "input_modalities": _known((Modality.TEXT,)),
        "output_modalities": _known((Modality.TEXT,)),
        "context_window": _known(ContextWindow(100, ContextWindowKind.INPUT)),
    }
    defaults.update(facts)
    return ModelRecord(
        identity=ModelIdentity("openai", model_id),
        evidence_disposition=disposition,
        **defaults,  # type: ignore[arg-type]
    )


def test_comparison_requires_two_or_three_unique_known_ids_and_preserves_order() -> None:
    records = (
        _record("one", aliases=_known(("one-alias",))),
        _record("two"),
        _record("three"),
        _record("four"),
    )
    result = compare_models(records, ("openai:three", "openai:one", "openai:two"))
    assert [item.samyak_id for item in result.models] == [
        "openai:three",
        "openai:one",
        "openai:two",
    ]
    for ids, message in (
        (("openai:one",), "exactly two"),
        (("openai:one", "openai:two", "openai:three", "openai:four"), "exactly two"),
        (("openai:one", "openai:one"), "unique"),
        (("openai:one", "openai:missing"), "unknown"),
        (("openai:one-alias", "openai:two"), "unknown"),
    ):
        with pytest.raises(ModelComparisonError, match=message):
            compare_models(records, ids)


def test_documentation_requires_explicit_opt_in_and_warns_on_mixed_levels() -> None:
    current = _record("current")
    legacy = _record("legacy", disposition=None)
    historical = _record("historical", disposition=EvidenceDisposition.HISTORICAL_REFERENCE_ONLY)
    with pytest.raises(ModelComparisonError, match="legacy_uncertified"):
        compare_models((current, legacy), (current.samyak_id, legacy.samyak_id))
    legacy_result = compare_models(
        (current, legacy), (current.samyak_id, legacy.samyak_id), include_legacy=True
    )
    assert legacy_result.warnings == (
        "comparison includes multiple documentation certification levels: "
        "current_documented, legacy_uncertified",
    )
    historical_result = compare_models(
        (current, historical),
        (current.samyak_id, historical.samyak_id),
        documentation=(EvidenceDisposition.HISTORICAL_REFERENCE_ONLY,),
    )
    assert historical_result.models[1].documentation_disposition is (
        EvidenceDisposition.HISTORICAL_REFERENCE_ONLY
    )


def test_facts_preserve_false_unknown_unverified_conflicts_and_missing_provenance() -> None:
    conflict = Fact(
        FactStatus.CONFLICT,
        claims=(
            FactClaim(True, _provenance("true-docs")),
            FactClaim(False, _provenance("false-docs")),
        ),
    )
    first = _record(
        "first",
        api_access=_known(False, "api-false"),
        tool_calling=Fact(FactStatus.UNKNOWN, provenance=_provenance("tool-unknown")),
        structured_output=Fact(FactStatus.NOT_VERIFIED),
    )
    second = _record("second", tool_calling=conflict)
    result = compare_models((first, second), (first.samyak_id, second.samyak_id))
    first_result, second_result = result.models
    assert first_result.api_access.value is False
    assert first_result.tool_calling.status is FactStatus.UNKNOWN
    assert first_result.tool_calling.value is None
    assert first_result.structured_output.status is FactStatus.NOT_VERIFIED
    assert first_result.structured_output.source_ids == ()
    assert second_result.tool_calling.status is FactStatus.CONFLICT
    assert [(item.value, item.source_id) for item in second_result.tool_calling.claims] == [
        (True, "true-docs"),
        (False, "false-docs"),
    ]
    assert (
        next(item for item in result.dimensions if item.name == "tool_calling").comparable is False
    )


def test_context_comparability_never_converts_combined_or_unknown_kind() -> None:
    input_context = _record(
        "input", context_window=_known(ContextWindow(100, ContextWindowKind.INPUT))
    )
    combined = _record(
        "combined", context_window=_known(ContextWindow(200, ContextWindowKind.COMBINED))
    )
    unknown_kind = _record(
        "unknown-kind", context_window=_known(ContextWindow(300, ContextWindowKind.UNKNOWN))
    )
    compatible = compare_models(
        (input_context, _record("input-two")), (input_context.samyak_id, "openai:input-two")
    )
    assert next(item for item in compatible.dimensions if item.name == "context_window").comparable
    for other in (combined, unknown_kind):
        comparison = compare_models(
            (input_context, other), (input_context.samyak_id, other.samyak_id)
        )
        dimension = next(item for item in comparison.dimensions if item.name == "context_window")
        assert dimension.comparable is False
        assert dimension.reason == "context kinds are not directly comparable"


def test_json_projection_is_deterministic_and_json_compatible_without_mutation() -> None:
    first, second = _record("first"), _record("second")
    before = (first, second)
    comparison = compare_models(before, (first.samyak_id, second.samyak_id))
    payload = comparison_to_dict(comparison)
    assert before == (first, second)
    assert json.loads(json.dumps(payload)) == payload
    assert payload["models"][0]["detail_href"] == "/models/openai:first"
    assert payload["models"][0]["facts"]["context_window"]["value"] == {
        "tokens": 100,
        "kind": "input",
    }
