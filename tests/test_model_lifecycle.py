"""Update-time Model Lifecycle Intelligence diffs. No network. No persistence."""

from __future__ import annotations

from pathlib import Path

from samyak.cli import _render_update_result
from samyak.model.catalog import (
    CATALOG_SCHEMA_VERSION,
    CatalogFreshness,
    CatalogOverlay,
    FreshnessStatus,
    build_catalog,
)
from samyak.model.facts import (
    Confidence,
    ContextWindow,
    ContextWindowKind,
    Fact,
    FactClaim,
    FactStatus,
    LifecycleState,
    Provenance,
    SourceKind,
    unverified_fact,
)
from samyak.model.identity import ModelIdentity
from samyak.model.lifecycle import (
    LifecycleChange,
    LifecycleChangeType,
    diff_provider_lifecycle,
    facts_semantically_equal,
)
from samyak.model.providers.fireworks import (
    SOURCE_ID_CHANGELOG,
    SOURCE_ID_SERVING_PATHS,
    captured_markdown,
    catalog_from_fireworks_sources,
    parse_fireworks_sources,
)
from samyak.model.providers.fireworks.sources import CHANGELOG_URL, SERVING_PATHS_URL
from samyak.model.records import ModelRecord
from samyak.model.update import CatalogUpdateResult

GENERATED_AT = "2026-09-19T12:00:00+00:00"
FIREWORKS_FIXTURES = Path(__file__).parent / "fixtures" / "models" / "fireworks"


def _provenance(**overrides: object) -> Provenance:
    payload: dict[str, object] = {
        "provider": "openai",
        "source_id": "openai-docs-deprecations",
        "source_kind": SourceKind.PROVIDER_DEPRECATIONS,
        "source_url": "https://example.invalid/deprecations.md",
        "content_hash": "sha256:abc",
        "retrieved_at": "2026-09-19T11:59:00+00:00",
        "observed_at": None,
        "verified_at": "2026-09-19T12:00:00+00:00",
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


def _identity(provider_model_id: str, *, provider_id: str = "openai") -> ModelIdentity:
    return ModelIdentity(provider_id=provider_id, provider_model_id=provider_model_id)


def _model(
    provider_model_id: str = "gpt-test",
    *,
    provider_id: str = "openai",
    **fields: object,
) -> ModelRecord:
    return ModelRecord(
        identity=_identity(provider_model_id, provider_id=provider_id),
        **fields,  # type: ignore[arg-type]
    )


def _catalog(*models: ModelRecord, generated_at: str = GENERATED_AT):
    return build_catalog(
        models=models,
        generated_at=generated_at,
        freshness=CatalogFreshness(
            status=FreshnessStatus.AS_OF,
            generated_at=generated_at,
            overlay=CatalogOverlay.USER_CACHE,
        ),
    )


def _change_types(changes) -> list[tuple[str, LifecycleChangeType]]:
    return [(item.samyak_id, item.change_type) for item in changes]


def test_no_previous_catalog_produces_no_changes() -> None:
    incoming = _catalog(_model(lifecycle=_known(LifecycleState.ACTIVE)))
    assert diff_provider_lifecycle(None, incoming, provider_id="openai") == ()


def test_first_provider_snapshot_in_mixed_catalog_produces_no_changes() -> None:
    previous = _catalog(
        _model("claude-opus-5", provider_id="anthropic", lifecycle=_known(LifecycleState.ACTIVE))
    )
    incoming = _catalog(_model(lifecycle=_known(LifecycleState.ACTIVE)))
    assert diff_provider_lifecycle(previous, incoming, provider_id="openai") == ()


def test_no_differences_produces_no_changes() -> None:
    record = _model(lifecycle=_known(LifecycleState.ACTIVE))
    previous = _catalog(record)
    incoming = _catalog(
        _model(
            lifecycle=_known(LifecycleState.ACTIVE, source_id="openai-docs-models"),
            context_window=_known(ContextWindow(tokens=128_000, kind=ContextWindowKind.COMBINED)),
        )
    )
    assert diff_provider_lifecycle(previous, incoming, provider_id="openai") == ()


def test_provenance_only_difference_is_not_a_change() -> None:
    left = _known(LifecycleState.DEPRECATED, source_id="source-a", content_hash="sha256:1")
    right = _known(LifecycleState.DEPRECATED, source_id="source-b", content_hash="sha256:2")
    assert facts_semantically_equal(left, right) is True


def test_unknown_and_not_verified_are_not_lifecycle_events() -> None:
    previous = _catalog(_model(retirement_at=_unknown()))
    incoming = _catalog(_model(retirement_at=unverified_fact()))
    assert diff_provider_lifecycle(previous, incoming, provider_id="openai") == ()


def test_model_added() -> None:
    previous = _catalog(_model("kept", lifecycle=_known(LifecycleState.ACTIVE)))
    incoming = _catalog(
        _model("kept", lifecycle=_known(LifecycleState.ACTIVE)),
        _model("new-model", lifecycle=_known(LifecycleState.ACTIVE)),
    )
    changes = diff_provider_lifecycle(previous, incoming, provider_id="openai")
    assert _change_types(changes) == [("openai:new-model", LifecycleChangeType.ADDED)]
    assert changes[0].previous is None
    assert changes[0].current is not None


def test_model_removed_from_documented_catalog_is_not_retired() -> None:
    previous = _catalog(_model("gone", lifecycle=_known(LifecycleState.ACTIVE)))
    incoming = _catalog(_model("kept", lifecycle=_known(LifecycleState.ACTIVE)))
    changes = diff_provider_lifecycle(previous, incoming, provider_id="openai")
    removed = [item for item in changes if item.change_type is LifecycleChangeType.REMOVED]
    added = [item for item in changes if item.change_type is LifecycleChangeType.ADDED]
    assert len(removed) == 1
    assert removed[0].samyak_id == "openai:gone"
    assert removed[0].current is None
    assert added[0].samyak_id == "openai:kept"
    assert all(item.change_type is not LifecycleChangeType.LIFECYCLE_CHANGED for item in changes)


def test_active_to_deprecated() -> None:
    previous = _catalog(_model(lifecycle=_known(LifecycleState.ACTIVE)))
    incoming = _catalog(_model(lifecycle=_known(LifecycleState.DEPRECATED)))
    changes = diff_provider_lifecycle(previous, incoming, provider_id="openai")
    assert _change_types(changes) == [("openai:gpt-test", LifecycleChangeType.LIFECYCLE_CHANGED)]
    assert changes[0].previous.lifecycle.value is LifecycleState.ACTIVE
    assert changes[0].current.lifecycle.value is LifecycleState.DEPRECATED


def test_legacy_to_deprecated() -> None:
    previous = _catalog(_model(lifecycle=_known(LifecycleState.LEGACY)))
    incoming = _catalog(_model(lifecycle=_known(LifecycleState.DEPRECATED)))
    changes = diff_provider_lifecycle(previous, incoming, provider_id="openai")
    assert _change_types(changes) == [("openai:gpt-test", LifecycleChangeType.LIFECYCLE_CHANGED)]
    assert changes[0].previous.lifecycle.value is LifecycleState.LEGACY


def test_deprecated_to_retired() -> None:
    previous = _catalog(
        _model(
            lifecycle=_known(LifecycleState.DEPRECATED),
            retirement_at=_known("2026-01-01"),
        )
    )
    incoming = _catalog(
        _model(
            lifecycle=_known(LifecycleState.RETIRED),
            retirement_at=_known("2026-01-01"),
        )
    )
    changes = diff_provider_lifecycle(previous, incoming, provider_id="openai")
    assert _change_types(changes) == [("openai:gpt-test", LifecycleChangeType.LIFECYCLE_CHANGED)]
    assert changes[0].current.lifecycle.value is LifecycleState.RETIRED


def test_retired_model_remaining_in_catalog_is_not_removed() -> None:
    record = _model(lifecycle=_known(LifecycleState.RETIRED), retirement_at=_known("2026-01-01"))
    changes = diff_provider_lifecycle(_catalog(record), _catalog(record), provider_id="openai")
    assert changes == ()


def test_deprecated_date_added() -> None:
    previous = _catalog(_model(lifecycle=_known(LifecycleState.DEPRECATED)))
    incoming = _catalog(
        _model(
            lifecycle=_known(LifecycleState.DEPRECATED),
            deprecated_at=_known("2027-01-01"),
        )
    )
    changes = diff_provider_lifecycle(previous, incoming, provider_id="openai")
    assert _change_types(changes) == [
        ("openai:gpt-test", LifecycleChangeType.DEPRECATION_DATE_CHANGED)
    ]
    assert changes[0].current.deprecated_at.value == "2027-01-01"


def test_deprecated_date_changed() -> None:
    previous = _catalog(_model(deprecated_at=_known("2027-01-01")))
    incoming = _catalog(_model(deprecated_at=_known("2027-02-01")))
    changes = diff_provider_lifecycle(previous, incoming, provider_id="openai")
    assert _change_types(changes) == [
        ("openai:gpt-test", LifecycleChangeType.DEPRECATION_DATE_CHANGED)
    ]


def test_retirement_date_added() -> None:
    previous = _catalog(_model(lifecycle=_known(LifecycleState.DEPRECATED)))
    incoming = _catalog(
        _model(
            lifecycle=_known(LifecycleState.DEPRECATED),
            retirement_at=_known("2027-02-01"),
        )
    )
    changes = diff_provider_lifecycle(previous, incoming, provider_id="openai")
    assert _change_types(changes) == [
        ("openai:gpt-test", LifecycleChangeType.RETIREMENT_DATE_CHANGED)
    ]


def test_retirement_date_changed() -> None:
    previous = _catalog(_model(retirement_at=_known("2026-06-01")))
    incoming = _catalog(_model(retirement_at=_known("2026-09-01")))
    changes = diff_provider_lifecycle(previous, incoming, provider_id="openai")
    assert _change_types(changes) == [
        ("openai:gpt-test", LifecycleChangeType.RETIREMENT_DATE_CHANGED)
    ]


def test_replacement_added() -> None:
    successor = _identity("gpt-4.1")
    previous = _catalog(_model(lifecycle=_known(LifecycleState.DEPRECATED)))
    incoming = _catalog(
        _model(
            lifecycle=_known(LifecycleState.DEPRECATED),
            replacement=_known((successor,)),
        )
    )
    changes = diff_provider_lifecycle(previous, incoming, provider_id="openai")
    assert _change_types(changes) == [("openai:gpt-test", LifecycleChangeType.REPLACEMENT_CHANGED)]
    assert changes[0].current.replacement.value == (successor,)


def test_replacement_changed() -> None:
    previous = _catalog(_model(replacement=_known((_identity("gpt-4.1"),))))
    incoming = _catalog(_model(replacement=_known((_identity("gpt-5"),))))
    changes = diff_provider_lifecycle(previous, incoming, provider_id="openai")
    assert _change_types(changes) == [("openai:gpt-test", LifecycleChangeType.REPLACEMENT_CHANGED)]


def test_replacement_removed() -> None:
    previous = _catalog(_model(replacement=_known((_identity("gpt-4.1"),))))
    incoming = _catalog(_model(replacement=unverified_fact()))
    changes = diff_provider_lifecycle(previous, incoming, provider_id="openai")
    assert _change_types(changes) == [("openai:gpt-test", LifecycleChangeType.REPLACEMENT_CHANGED)]
    assert changes[0].current.replacement.status is FactStatus.NOT_VERIFIED


def test_known_to_conflict_is_detected() -> None:
    previous = _catalog(_model(retirement_at=_known("2026-11-24")))
    incoming = _catalog(_model(retirement_at=_conflict("2026-11-24", "2026-12-01")))
    changes = diff_provider_lifecycle(previous, incoming, provider_id="openai")
    assert _change_types(changes) == [
        ("openai:gpt-test", LifecycleChangeType.RETIREMENT_DATE_CHANGED)
    ]
    assert changes[0].current.retirement_at.status is FactStatus.CONFLICT


def test_conflict_to_known_is_detected() -> None:
    previous = _catalog(_model(replacement=_conflict((_identity("a"),), (_identity("b"),))))
    incoming = _catalog(_model(replacement=_known((_identity("a"),))))
    changes = diff_provider_lifecycle(previous, incoming, provider_id="openai")
    assert _change_types(changes) == [("openai:gpt-test", LifecycleChangeType.REPLACEMENT_CHANGED)]
    assert changes[0].current.replacement.status is FactStatus.KNOWN


def test_other_providers_do_not_contaminate_diff() -> None:
    previous = _catalog(
        _model("gpt-test", lifecycle=_known(LifecycleState.ACTIVE)),
        _model("claude-opus-5", provider_id="anthropic", lifecycle=_known(LifecycleState.ACTIVE)),
    )
    incoming = _catalog(_model("gpt-test", lifecycle=_known(LifecycleState.ACTIVE)))
    changes = diff_provider_lifecycle(previous, incoming, provider_id="openai")
    assert changes == ()
    assert all("anthropic:" not in item.samyak_id for item in changes)


def test_other_provider_lifecycle_change_is_ignored_during_openai_diff() -> None:
    previous = _catalog(
        _model("gpt-test", lifecycle=_known(LifecycleState.ACTIVE)),
        _model("claude-opus-5", provider_id="anthropic", lifecycle=_known(LifecycleState.ACTIVE)),
    )
    incoming = _catalog(
        _model("gpt-test", lifecycle=_known(LifecycleState.ACTIVE)),
        _model(
            "claude-opus-5",
            provider_id="anthropic",
            lifecycle=_known(LifecycleState.DEPRECATED),
        ),
    )
    changes = diff_provider_lifecycle(previous, incoming, provider_id="openai")
    assert changes == ()


def test_multiple_changes_on_the_same_model_are_preserved() -> None:
    successor = _identity("gpt-4.1")
    previous = _catalog(_model(lifecycle=_known(LifecycleState.ACTIVE)))
    incoming = _catalog(
        _model(
            lifecycle=_known(LifecycleState.DEPRECATED),
            retirement_at=_known("2026-12-15"),
            replacement=_known((successor,)),
        )
    )
    changes = diff_provider_lifecycle(previous, incoming, provider_id="openai")
    assert _change_types(changes) == [
        ("openai:gpt-test", LifecycleChangeType.LIFECYCLE_CHANGED),
        ("openai:gpt-test", LifecycleChangeType.RETIREMENT_DATE_CHANGED),
        ("openai:gpt-test", LifecycleChangeType.REPLACEMENT_CHANGED),
    ]


def test_capability_changes_are_not_lifecycle_events() -> None:
    previous = _catalog(
        _model(
            lifecycle=_known(LifecycleState.ACTIVE),
            context_window=_known(ContextWindow(tokens=8_000, kind=ContextWindowKind.COMBINED)),
        )
    )
    incoming = _catalog(
        _model(
            lifecycle=_known(LifecycleState.ACTIVE),
            context_window=_known(ContextWindow(tokens=128_000, kind=ContextWindowKind.COMBINED)),
        )
    )
    assert diff_provider_lifecycle(previous, incoming, provider_id="openai") == ()


def test_schema_version_is_unchanged_by_lifecycle_diff() -> None:
    previous = _catalog(_model(lifecycle=_known(LifecycleState.ACTIVE)))
    incoming = _catalog(_model(lifecycle=_known(LifecycleState.DEPRECATED)))
    diff_provider_lifecycle(previous, incoming, provider_id="openai")
    assert previous.catalog_schema_version == CATALOG_SCHEMA_VERSION == 1
    assert incoming.catalog_schema_version == 1
    assert "lifecycle_changes" not in previous.to_dict()
    assert "lifecycle_changes" not in incoming.to_dict()


def test_fireworks_name_only_changelog_produces_zero_attached_events() -> None:
    serving = (FIREWORKS_FIXTURES / "serving-paths.md").read_text(encoding="utf-8")
    name_only = (
        "# Changelog\n\n"
        "GLM 5.2 is deprecated from serverless. This display name is not a serving ID.\n"
        "DeepSeek V4 Flash is deprecated from serverless.\n"
    )
    previous_sources = (
        captured_markdown(
            source_id=SOURCE_ID_SERVING_PATHS,
            source_url=SERVING_PATHS_URL,
            body=serving,
            retrieved_at="2026-09-19T11:59:00+00:00",
        ),
        captured_markdown(
            source_id=SOURCE_ID_CHANGELOG,
            source_url=CHANGELOG_URL,
            body="# Changelog\n\nNo documented serving IDs in this changelog.\n",
            retrieved_at="2026-09-19T11:59:00+00:00",
        ),
    )
    incoming_sources = (
        previous_sources[0],
        captured_markdown(
            source_id=SOURCE_ID_CHANGELOG,
            source_url=CHANGELOG_URL,
            body=name_only,
            retrieved_at="2026-09-19T12:00:00+00:00",
        ),
    )
    observations = parse_fireworks_sources(incoming_sources)
    ids = {item.provider_model_id for item in observations}
    assert "GLM 5.2" not in ids
    assert "DeepSeek V4 Flash" not in ids
    previous = catalog_from_fireworks_sources(
        previous_sources,
        generated_at=GENERATED_AT,
        verified_at=GENERATED_AT,
        overlay=CatalogOverlay.USER_CACHE,
    )
    incoming = catalog_from_fireworks_sources(
        incoming_sources,
        generated_at=GENERATED_AT,
        verified_at=GENERATED_AT,
        overlay=CatalogOverlay.USER_CACHE,
    )
    glm = next(
        item
        for item in incoming.models
        if item.provider_model_id == "accounts/fireworks/models/glm-5p2"
    )
    assert glm.lifecycle.value is LifecycleState.ACTIVE
    changes = diff_provider_lifecycle(previous, incoming, provider_id="fireworks")
    assert changes == ()


def test_cli_omits_lifecycle_section_when_there_are_no_changes() -> None:
    catalog = _catalog(_model(lifecycle=_known(LifecycleState.ACTIVE)))
    output = _render_update_result(
        CatalogUpdateResult(
            provider_id="openai",
            ok=True,
            committed=True,
            partial=False,
            catalog=catalog,
            catalog_path=Path("/tmp/catalog.json"),
            previous_existed=False,
            error=None,
            notices=(),
        ),
        label="OpenAI",
    )
    assert "Updated OpenAI model catalog." in output
    assert "Lifecycle changes" not in output
    assert "ADDED" not in output


def test_cli_reports_removed_from_documented_catalog_not_retired() -> None:
    previous = _model("gone", lifecycle=_known(LifecycleState.ACTIVE))
    catalog = _catalog(_model("kept", lifecycle=_known(LifecycleState.ACTIVE)))
    output = _render_update_result(
        CatalogUpdateResult(
            provider_id="openai",
            ok=True,
            committed=True,
            partial=False,
            catalog=catalog,
            catalog_path=Path("/tmp/catalog.json"),
            previous_existed=True,
            error=None,
            notices=(),
            lifecycle_changes=(
                LifecycleChange(
                    change_type=LifecycleChangeType.REMOVED,
                    samyak_id="openai:gone",
                    previous=previous,
                ),
            ),
        ),
        label="OpenAI",
    )
    assert "REMOVED FROM DOCUMENTED CATALOG" in output
    assert "openai:gone" in output
    assert "Model was removed from the documented catalog." in output
    assert "RETIRED" not in output
    lowered = output.lower()
    for banned in (
        "likely obsolete",
        "at risk",
        "will be retired",
        "recommended migration",
        "best replacement",
        "predicted retirement",
        "risk score",
    ):
        assert banned not in lowered


def test_cli_reports_multiple_fact_changes_for_one_model() -> None:
    previous = _model(lifecycle=_known(LifecycleState.ACTIVE))
    current = _model(
        lifecycle=_known(LifecycleState.DEPRECATED),
        retirement_at=_known("2026-12-15"),
        replacement=_known((_identity("gpt-4.1"),)),
    )
    changes = diff_provider_lifecycle(_catalog(previous), _catalog(current), provider_id="openai")
    assert len(changes) == 3
    output = _render_update_result(
        CatalogUpdateResult(
            provider_id="openai",
            ok=True,
            committed=True,
            partial=False,
            catalog=_catalog(current),
            catalog_path=Path("/tmp/catalog.json"),
            previous_existed=True,
            error=None,
            notices=(),
            lifecycle_changes=changes,
        ),
        label="OpenAI",
    )
    assert "DEPRECATED" in output
    assert "Provider announced deprecation." in output
    assert "ACTIVE → DEPRECATED" in output
    assert "Retirement: 2026-12-15" in output
    assert "RETIREMENT DATE CHANGED" in output
    assert "REPLACEMENT ADDED" in output
    assert "Provider documented a replacement." in output
    assert "3 lifecycle changes detected." in output


def test_cli_reports_retired_without_invented_retirement_date_cause() -> None:
    previous = _model(lifecycle=_known(LifecycleState.DEPRECATED))
    current = _model(lifecycle=_known(LifecycleState.RETIRED))
    changes = diff_provider_lifecycle(_catalog(previous), _catalog(current), provider_id="openai")
    output = _render_update_result(
        CatalogUpdateResult(
            provider_id="openai",
            ok=True,
            committed=True,
            partial=False,
            catalog=_catalog(current),
            catalog_path=Path("/tmp/catalog.json"),
            previous_existed=True,
            error=None,
            notices=(),
            lifecycle_changes=changes,
        ),
        label="OpenAI",
    )
    assert "RETIRED" in output
    assert "DEPRECATED → RETIRED" in output
    assert "Lifecycle changed: deprecated → retired" in output
    assert "Provider retirement date has passed." not in output
