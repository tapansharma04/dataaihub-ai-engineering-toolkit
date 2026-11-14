"""Offline contract tests for evidence-aware model queries."""

from __future__ import annotations

import pytest

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
from samyak.model.query import ModelQuery, explain_query_result, query_models
from samyak.model.records import EvidenceDisposition, ModelRecord


def _provenance(provider: str = "openai", source_id: str = "docs") -> Provenance:
    return Provenance(
        provider=provider,
        source_id=source_id,
        source_kind=SourceKind.PROVIDER_DOCS,
        content_hash="sha256:test",
        retrieved_at="2026-11-01T00:00:00+00:00",
        verified_at="2026-11-01T00:00:00+00:00",
        confidence=Confidence.HIGH,
    )


def _known(value: object, provider: str = "openai", source_id: str = "docs") -> Fact:
    return Fact(status=FactStatus.KNOWN, value=value, provenance=_provenance(provider, source_id))


def _unknown(provider: str = "openai") -> Fact:
    return Fact(status=FactStatus.UNKNOWN, provenance=_provenance(provider))


def _record(
    model_id: str,
    *,
    provider: str = "openai",
    disposition: EvidenceDisposition | None = EvidenceDisposition.CURRENT_DOCUMENTED,
    **facts: object,
) -> ModelRecord:
    fields: dict[str, object] = {
        "lifecycle": _known(LifecycleState.ACTIVE, provider),
        "api_access": _known(True, provider),
    }
    fields.update(facts)
    return ModelRecord(
        identity=ModelIdentity(provider_id=provider, provider_model_id=model_id),
        evidence_disposition=disposition,
        **fields,  # type: ignore[arg-type]
    )


def _capable(
    model_id: str = "capable",
    *,
    provider: str = "openai",
    disposition: EvidenceDisposition | None = EvidenceDisposition.CURRENT_DOCUMENTED,
    **changes: object,
) -> ModelRecord:
    fields: dict[str, object] = {
        "tool_calling": _known(True),
        "structured_output": _known(True),
        "input_modalities": _known((Modality.TEXT, Modality.IMAGE)),
        "output_modalities": _known((Modality.TEXT,)),
        "context_window": _known(ContextWindow(128_000, ContextWindowKind.INPUT)),
    }
    fields.update(changes)
    return _record(model_id, provider=provider, disposition=disposition, **fields)


def test_explanations_are_deterministic_and_evidence_safe() -> None:
    matched = _capable("matched")
    excluded = _record("unknown-tool", tool_calling=_unknown())
    result = query_models((excluded, matched), ModelQuery(tool_calling=True))

    explanations = explain_query_result(result)

    assert tuple(explanations) == ("openai:matched", "openai:unknown-tool")
    assert explanations["openai:matched"][1].reason_code == "matched_verified"
    assert explanations["openai:matched"][1].source_ids == ("docs",)
    assert explanations["openai:unknown-tool"][0].reason_code == "missing_verified_evidence"
    assert "false" not in explanations["openai:unknown-tool"][0].reason


@pytest.mark.parametrize(
    ("actual", "expected", "matches", "reason"),
    [
        (True, True, True, None),
        (False, False, True, None),
        (False, True, False, "tool_calling_mismatch"),
        (True, False, False, "tool_calling_mismatch"),
    ],
)
def test_boolean_explanations_preserve_verified_true_and_false_semantics(
    actual: bool, expected: bool, matches: bool, reason: str | None
) -> None:
    result = query_models(
        (_capable(tool_calling=_known(actual)),), ModelQuery(tool_calling=expected)
    )
    explanations = explain_query_result(result)
    if matches:
        explanation = explanations["openai:capable"][-1]
        assert explanation.reason_code == "matched_verified"
        assert explanation.source_ids == ("docs",)
    else:
        assert result.exclusions[0].reasons == (reason,)
        assert explanations["openai:capable"][0].reason_code == "verified_opposite"


@pytest.mark.parametrize(
    "fact",
    [
        _unknown(),
        Fact(status=FactStatus.NOT_VERIFIED),
        Fact(
            status=FactStatus.CONFLICT,
            claims=(
                FactClaim(True, _provenance()),
                FactClaim(False, _provenance(source_id="other")),
            ),
        ),
    ],
)
def test_unverified_and_conflicting_boolean_evidence_never_becomes_negative(fact: Fact) -> None:
    for expected in (True, False):
        result = query_models((_capable(tool_calling=fact),), ModelQuery(tool_calling=expected))
        assert result.matches == ()
        assert result.exclusions[0].reasons == ("tool_calling_not_verified",)
        assert (
            explain_query_result(result)["openai:capable"][0].reason_code
            == "missing_verified_evidence"
        )


def test_explanations_match_evaluator_ordering_and_provenance() -> None:
    first = _capable("first")
    second = _capable(
        "second",
        lifecycle=_known(LifecycleState.RETIRED),
        tool_calling=_known(False),
        input_modalities=_known((Modality.TEXT,)),
    )
    result = query_models(
        (second, first),
        ModelQuery(
            lifecycle=LifecycleState.ACTIVE,
            tool_calling=True,
            input_modalities=frozenset({Modality.IMAGE}),
        ),
    )
    explanations = explain_query_result(result)
    assert tuple(explanations) == ("openai:first", "openai:second")
    assert [item.predicate for item in explanations["openai:first"]] == [
        "lifecycle",
        "tool_calling",
        "input_modalities",
    ]
    assert all(item.source_ids == ("docs",) for item in explanations["openai:first"])
    assert [item.predicate for item in explanations["openai:second"]] == [
        "lifecycle",
        "tool_calling",
        "input_modalities",
    ]
    assert result.exclusions[0].reasons == (
        "lifecycle_mismatch",
        "tool_calling_mismatch",
        "input_modalities_mismatch",
    )


def test_missing_match_timestamp_is_serialized_without_invention() -> None:
    no_timestamp = Provenance(
        provider="openai",
        source_id="without-timestamp",
        source_kind=SourceKind.PROVIDER_DOCS,
    )
    result = query_models(
        (_capable(tool_calling=Fact(FactStatus.KNOWN, True, no_timestamp)),),
        ModelQuery(tool_calling=True),
    )
    explanation = explain_query_result(result)["openai:capable"][-1]
    assert explanation.source_ids == ("without-timestamp",)
    assert explanation.verified_at == ()


def test_documentation_gate_does_not_invent_capability_explanations() -> None:
    result = query_models(
        (_capable("legacy", disposition=None, tool_calling=_known(False)),),
        ModelQuery(tool_calling=True),
    )
    assert result.exclusions[0].reasons == ("documentation_not_eligible",)
    explanation = explain_query_result(result)["openai:legacy"]
    assert [item.predicate for item in explanation] == ["documentation"]
    opted = query_models(
        (_capable("legacy", disposition=None),),
        ModelQuery(
            tool_calling=True,
            documentation_dispositions=frozenset({EvidenceDisposition.CURRENT_DOCUMENTED, None}),
        ),
    )
    assert [item.samyak_id for item in opted.matches] == ["openai:legacy"]


def test_default_requires_current_documentation_and_opt_ins_are_explicit() -> None:
    records = (
        _capable("current"),
        _capable("legacy", disposition=None),
        _capable("historical", disposition=EvidenceDisposition.HISTORICAL_REFERENCE_ONLY),
        _capable("not-observed", disposition=EvidenceDisposition.NOT_OBSERVED_IN_CURRENT_CAPTURE),
    )
    default = query_models(records, ModelQuery(tool_calling=True))
    assert [item.samyak_id for item in default.matches] == ["openai:current"]
    assert default.summary.documentation_eligible_records == 1
    opted = query_models(
        records,
        ModelQuery(
            tool_calling=True,
            documentation_dispositions=frozenset(
                {
                    EvidenceDisposition.CURRENT_DOCUMENTED,
                    None,
                    EvidenceDisposition.HISTORICAL_REFERENCE_ONLY,
                    EvidenceDisposition.NOT_OBSERVED_IN_CURRENT_CAPTURE,
                }
            ),
        ),
    )
    assert [item.samyak_id for item in opted.matches] == [
        "openai:current",
        "openai:historical",
        "openai:legacy",
        "openai:not-observed",
    ]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (ModelQuery(tool_calling=True), ["openai:capable"]),
        (ModelQuery(structured_output=True), ["openai:capable"]),
        (
            ModelQuery(input_modalities=frozenset({Modality.TEXT, Modality.IMAGE})),
            ["openai:capable"],
        ),
        (
            ModelQuery(tool_calling=True, input_modalities=frozenset({Modality.IMAGE})),
            ["openai:capable"],
        ),
        (
            ModelQuery(minimum_context_window=100_000, context_window_kind=ContextWindowKind.INPUT),
            ["openai:capable"],
        ),
        (ModelQuery(providers=frozenset({"anthropic"})), []),
        (ModelQuery(lifecycle=LifecycleState.ACTIVE, api_access=True), ["openai:capable"]),
    ],
)
def test_supported_predicates_match_verified_facts(query: ModelQuery, expected: list[str]) -> None:
    result = query_models((_capable(),), query)
    assert [item.samyak_id for item in result.matches] == expected


def test_context_kind_is_required_and_never_interchangeable() -> None:
    combined = _capable(context_window=_known(ContextWindow(256_000, ContextWindowKind.COMBINED)))
    result = query_models(
        (combined,),
        ModelQuery(minimum_context_window=100_000, context_window_kind=ContextWindowKind.INPUT),
    )
    assert result.matches == ()
    assert result.exclusions[0].reasons == ("context_window_kind_mismatch",)
    with pytest.raises(ValueError, match="requires input"):
        ModelQuery(minimum_context_window=1, context_window_kind=ContextWindowKind.UNKNOWN)


@pytest.mark.parametrize(
    ("field", "fact", "reason"),
    [
        ("tool_calling", _unknown(), "tool_calling_not_verified"),
        (
            "structured_output",
            Fact(status=FactStatus.NOT_VERIFIED),
            "structured_output_not_verified",
        ),
        (
            "api_access",
            Fact(
                status=FactStatus.CONFLICT,
                claims=(
                    FactClaim(True, _provenance()),
                    FactClaim(False, _provenance(source_id="other")),
                ),
            ),
            "api_access_not_verified",
        ),
        ("tool_calling", _known(False), "tool_calling_mismatch"),
    ],
)
def test_unknown_unverified_conflicting_and_false_capabilities_do_not_match(
    field: str, fact: Fact, reason: str
) -> None:
    record = _capable(**{field: fact})
    expected = {"tool_calling": True, "structured_output": True, "api_access": True}[field]
    result = query_models((record,), ModelQuery(**{field: expected}))
    assert result.matches == ()
    assert reason in result.exclusions[0].reasons


def test_multiple_providers_ordering_provenance_and_no_input_mutation() -> None:
    first = _capable("z", provider="openai")
    second = _capable("a", provider="anthropic")
    before = (first, second)
    result = query_models(
        before,
        ModelQuery(providers=frozenset({"openai", "anthropic"}), tool_calling=True),
    )
    assert [item.samyak_id for item in result.matches] == ["anthropic:a", "openai:z"]
    assert before == (first, second)
    fact = next(item for item in result.matches[0].matched_facts if item.name == "tool_calling")
    assert fact.status is FactStatus.KNOWN
    assert fact.source_ids == ("docs",)
    assert fact.verified_at == ("2026-11-01T00:00:00+00:00",)


def test_summary_tracks_all_exclusion_reasons_and_empty_results() -> None:
    record = _capable(
        provider="openai",
        lifecycle=_known(LifecycleState.RETIRED),
        tool_calling=_known(False),
    )
    result = query_models(
        (record,),
        ModelQuery(
            providers=frozenset({"anthropic"}),
            lifecycle=LifecycleState.ACTIVE,
            tool_calling=True,
        ),
    )
    assert result.matches == ()
    assert result.summary.total_records_considered == 1
    assert result.summary.documentation_eligible_records == 1
    assert result.summary.excluded_records == 1
    assert dict(result.summary.exclusion_reason_counts) == {
        "lifecycle_mismatch": 1,
        "provider_mismatch": 1,
        "tool_calling_mismatch": 1,
    }


def test_context_below_minimum_and_missing_limit_are_distinguished() -> None:
    short = _capable(context_window=_known(ContextWindow(10, ContextWindowKind.INPUT)))
    missing = _capable("missing", context_window=_unknown())
    result = query_models(
        (short, missing),
        ModelQuery(minimum_context_window=20, context_window_kind=ContextWindowKind.INPUT),
    )
    assert [item.reasons for item in result.exclusions] == [
        ("context_window_below_minimum",),
        ("missing_compatible_context_limit",),
    ]


def test_lifecycle_unknown_and_modality_mismatch_are_not_matches() -> None:
    unknown_lifecycle = _capable(lifecycle=_unknown())
    text_only = _capable("text", input_modalities=_known((Modality.TEXT,)))
    result = query_models(
        (unknown_lifecycle, text_only),
        ModelQuery(lifecycle=LifecycleState.ACTIVE, input_modalities=frozenset({Modality.IMAGE})),
    )
    assert result.matches == ()
    assert result.exclusions[0].reasons == ("lifecycle_not_verified",)
    assert result.exclusions[1].reasons == ("input_modalities_mismatch",)


def test_query_validation_rejects_ambiguous_context_and_empty_dispositions() -> None:
    with pytest.raises(ValueError, match="context_window_kind requires"):
        ModelQuery(context_window_kind=ContextWindowKind.INPUT)
    with pytest.raises(ValueError, match="documentation_dispositions"):
        ModelQuery(documentation_dispositions=frozenset())


@pytest.mark.parametrize(
    ("record", "query"),
    [
        (
            _capable(output_modalities=_known((Modality.TEXT, Modality.IMAGE))),
            ModelQuery(output_modalities=frozenset({Modality.IMAGE})),
        ),
        (_capable(tool_calling=_known(False)), ModelQuery(tool_calling=False)),
        (_capable(api_access=_known(False)), ModelQuery(api_access=False)),
        (
            _capable(lifecycle=_known(LifecycleState.RETIRED)),
            ModelQuery(lifecycle=LifecycleState.RETIRED),
        ),
        (_capable(provider="google"), ModelQuery(providers=frozenset({"google"}))),
        (
            _capable(context_window=_known(ContextWindow(256_000, ContextWindowKind.COMBINED))),
            ModelQuery(
                minimum_context_window=128_000,
                context_window_kind=ContextWindowKind.COMBINED,
            ),
        ),
        (
            _capable(),
            ModelQuery(structured_output=True, output_modalities=frozenset({Modality.TEXT})),
        ),
    ],
)
def test_additional_verified_predicate_combinations_match(
    record: ModelRecord, query: ModelQuery
) -> None:
    result = query_models((record,), query)
    assert [item.samyak_id for item in result.matches] == [record.samyak_id]
