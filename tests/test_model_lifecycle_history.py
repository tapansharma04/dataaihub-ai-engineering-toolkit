"""Lifecycle history sidecar tests. No live network."""

from __future__ import annotations

import json
import socket
from dataclasses import replace
from pathlib import Path

import pytest

from samyak.cli import _render_update_all_results, _render_update_result, main
from samyak.model.catalog import (
    CatalogFreshness,
    CatalogNotice,
    CatalogOverlay,
    FreshnessStatus,
    build_catalog,
)
from samyak.model.errors import CatalogValidationError, HistorySchemaError, HistoryStoreError
from samyak.model.facts import (
    Confidence,
    Fact,
    FactClaim,
    FactStatus,
    LifecycleState,
    Provenance,
    SourceKind,
)
from samyak.model.history import (
    HISTORY_SCHEMA_VERSION,
    filter_history_events,
    history_events_from_changes,
    render_lifecycle_history,
)
from samyak.model.history_store import FileLifecycleHistoryStore
from samyak.model.http import FixtureHop
from samyak.model.identity import ModelIdentity
from samyak.model.lifecycle import LifecycleChange, LifecycleChangeType
from samyak.model.records import ModelRecord
from samyak.model.store import FileCatalogStore
from samyak.model.update import (
    HISTORY_WRITE_FAILED_CODE,
    CatalogUpdateResult,
    update_fireworks_catalog,
    update_openai_catalog,
)
from test_model_update import (
    FIREWORKS_KIMI_URL,
    MODELS_INDEX_URL,
    _clock,
    _fireworks_pages,
    _fireworks_transport,
    _official_pages,
    _patch_cli_transport,
    _transport,
    model_page_url,
)

GENERATED_AT = "2026-09-19T12:00:00+00:00"
_HISTORY_TARGET = "openai:gpt-4.5-preview"


@pytest.fixture(autouse=True)
def deny_live_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("Lifecycle history tests must not use the network")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr("urllib.request.urlopen", blocked)


def _provenance(**overrides: object) -> Provenance:
    payload: dict[str, object] = {
        "provider": "openai",
        "source_id": "openai-docs-deprecations",
        "source_kind": SourceKind.PROVIDER_DEPRECATIONS,
        "source_url": "https://example.invalid/deprecations.md",
        "content_hash": "sha256:abc",
        "retrieved_at": "2026-09-19T11:59:00+00:00",
        "observed_at": None,
        "verified_at": GENERATED_AT,
        "confidence": Confidence.HIGH,
    }
    payload.update(overrides)
    return Provenance(**payload)  # type: ignore[arg-type]


def _known(value: object, **overrides: object) -> Fact:
    return Fact(status=FactStatus.KNOWN, value=value, provenance=_provenance(**overrides))


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


def _catalog(*models: ModelRecord):
    return build_catalog(
        models=models,
        generated_at=GENERATED_AT,
        freshness=CatalogFreshness(
            status=FreshnessStatus.AS_OF,
            generated_at=GENERATED_AT,
            overlay=CatalogOverlay.USER_CACHE,
        ),
    )


def _events_for(change: LifecycleChange, *, provider_id: str = "openai"):
    return history_events_from_changes(
        (change,),
        provider_id=provider_id,
        observed_at=GENERATED_AT,
    )


def _rewrite_lifecycle(
    catalog_store: FileCatalogStore,
    samyak_id: str,
    state: LifecycleState,
) -> None:
    catalog = catalog_store.load()
    rewritten = [
        replace(
            record,
            lifecycle=Fact(
                status=FactStatus.KNOWN,
                value=state,
                provenance=record.lifecycle.provenance,
            ),
        )
        if record.samyak_id == samyak_id
        else record
        for record in catalog.models
    ]
    catalog_store.save(
        build_catalog(
            models=rewritten,
            generated_at=catalog.generated_at,
            freshness=catalog.freshness,
            notices=catalog.notices,
            version=catalog.version,
        )
    )


def _failing_history_append(*_args: object, **_kwargs: object) -> None:
    raise HistoryStoreError("lifecycle history could not be written")


def _failing_history_events(*_args: object, **_kwargs: object):
    raise CatalogValidationError("lifecycle history samyak_id does not match provider_id")


def test_first_lifecycle_change_is_persisted(tmp_path: Path) -> None:
    store = FileLifecycleHistoryStore(root=tmp_path / "cache")
    change = LifecycleChange(
        change_type=LifecycleChangeType.LIFECYCLE_CHANGED,
        samyak_id="openai:gpt-test",
        previous=_model(lifecycle=_known(LifecycleState.ACTIVE)),
        current=_model(
            lifecycle=_known(LifecycleState.DEPRECATED),
            retirement_at=_known("2027-01-01"),
        ),
    )
    store.append(_events_for(change))
    loaded = FileLifecycleHistoryStore(root=tmp_path / "cache").load()
    assert len(loaded.events) == 1
    event = loaded.events[0]
    assert event.change_type is LifecycleChangeType.LIFECYCLE_CHANGED
    assert event.samyak_id == "openai:gpt-test"
    assert event.previous_lifecycle is not None
    assert event.previous_lifecycle.value == "active"
    assert event.current_lifecycle is not None
    assert event.current_lifecycle.value == "deprecated"
    assert event.current_retirement_at is not None
    assert event.current_retirement_at.value == "2027-01-01"
    assert loaded.lifecycle_history_schema_version == HISTORY_SCHEMA_VERSION == 1


def test_repeating_identical_update_does_not_duplicate_history(tmp_path: Path) -> None:
    catalog_store = FileCatalogStore(root=tmp_path / "cache")
    first = update_openai_catalog(
        transport=_transport(_official_pages()),
        store=catalog_store,
        clock=_clock,
    )
    assert first.committed is True
    assert first.catalog is not None
    target = "openai:gpt-4.5-preview"
    rewritten = [
        replace(
            record,
            lifecycle=Fact(
                status=FactStatus.KNOWN,
                value=LifecycleState.ACTIVE,
                provenance=record.lifecycle.provenance,
            ),
        )
        if record.samyak_id == target
        else record
        for record in first.catalog.models
    ]
    catalog_store.save(
        build_catalog(
            models=rewritten,
            generated_at=first.catalog.generated_at,
            freshness=first.catalog.freshness,
            notices=first.catalog.notices,
            version=first.catalog.version,
        )
    )
    second = update_openai_catalog(
        transport=_transport(_official_pages()),
        store=catalog_store,
        clock=_clock,
    )
    assert any(
        item.change_type is LifecycleChangeType.LIFECYCLE_CHANGED
        for item in second.lifecycle_changes
    )
    history = FileLifecycleHistoryStore(root=tmp_path / "cache")
    after_first_change = history.load()
    assert after_first_change.events
    third = update_openai_catalog(
        transport=_transport(_official_pages()),
        store=catalog_store,
        clock=_clock,
    )
    assert third.lifecycle_changes == ()
    assert history.load().events == after_first_change.events


def test_lifecycle_transition_is_persisted(tmp_path: Path) -> None:
    store = FileLifecycleHistoryStore(root=tmp_path / "cache")
    store.append(
        _events_for(
            LifecycleChange(
                change_type=LifecycleChangeType.LIFECYCLE_CHANGED,
                samyak_id="openai:gpt-test",
                previous=_model(lifecycle=_known(LifecycleState.DEPRECATED)),
                current=_model(lifecycle=_known(LifecycleState.RETIRED)),
            )
        )
    )
    event = store.load().events[0]
    assert event.previous_lifecycle.value == "deprecated"
    assert event.current_lifecycle.value == "retired"


def test_deprecation_date_change_is_persisted(tmp_path: Path) -> None:
    store = FileLifecycleHistoryStore(root=tmp_path / "cache")
    change = LifecycleChange(
        change_type=LifecycleChangeType.DEPRECATION_DATE_CHANGED,
        samyak_id="openai:gpt-test",
        previous=_model(deprecated_at=_known("2027-01-01")),
        current=_model(deprecated_at=_known("2027-02-01")),
    )
    store.append(_events_for(change))
    event = store.load().events[0]
    assert event.change_type is LifecycleChangeType.DEPRECATION_DATE_CHANGED
    assert event.previous_deprecated_at is not None
    assert event.current_deprecated_at is not None
    assert event.previous_deprecated_at.value == "2027-01-01"
    assert event.current_deprecated_at.value == "2027-02-01"


def test_retirement_date_change_is_persisted(tmp_path: Path) -> None:
    store = FileLifecycleHistoryStore(root=tmp_path / "cache")
    change = LifecycleChange(
        change_type=LifecycleChangeType.RETIREMENT_DATE_CHANGED,
        samyak_id="openai:gpt-test",
        previous=_model(retirement_at=_known("2027-02-01")),
        current=_model(retirement_at=_known("2027-03-01")),
    )
    store.append(_events_for(change))
    event = store.load().events[0]
    assert event.change_type is LifecycleChangeType.RETIREMENT_DATE_CHANGED
    assert event.previous_retirement_at is not None
    assert event.current_retirement_at is not None
    assert event.previous_retirement_at.value == "2027-02-01"
    assert event.current_retirement_at.value == "2027-03-01"


def test_replacement_change_is_persisted(tmp_path: Path) -> None:
    store = FileLifecycleHistoryStore(root=tmp_path / "cache")
    change = LifecycleChange(
        change_type=LifecycleChangeType.REPLACEMENT_CHANGED,
        samyak_id="openai:gpt-test",
        previous=_model(replacement=_known((_identity("gpt-4.1"),))),
        current=_model(replacement=_known((_identity("gpt-5"),))),
    )
    store.append(_events_for(change))
    event = store.load().events[0]
    assert event.change_type is LifecycleChangeType.REPLACEMENT_CHANGED
    assert event.previous_replacement is not None
    assert event.current_replacement is not None
    assert event.previous_replacement.value == ["openai:gpt-4.1"]
    assert event.current_replacement.value == ["openai:gpt-5"]


def test_multiple_lifecycle_changes_on_one_model_are_preserved(tmp_path: Path) -> None:
    store = FileLifecycleHistoryStore(root=tmp_path / "cache")
    previous = _model(lifecycle=_known(LifecycleState.ACTIVE))
    current = _model(
        lifecycle=_known(LifecycleState.DEPRECATED),
        retirement_at=_known("2027-01-01"),
        replacement=_known((_identity("gpt-4.1"),)),
    )
    changes = (
        LifecycleChange(
            change_type=LifecycleChangeType.LIFECYCLE_CHANGED,
            samyak_id="openai:gpt-test",
            previous=previous,
            current=current,
        ),
        LifecycleChange(
            change_type=LifecycleChangeType.RETIREMENT_DATE_CHANGED,
            samyak_id="openai:gpt-test",
            previous=previous,
            current=current,
        ),
        LifecycleChange(
            change_type=LifecycleChangeType.REPLACEMENT_CHANGED,
            samyak_id="openai:gpt-test",
            previous=previous,
            current=current,
        ),
    )
    store.append(
        history_events_from_changes(changes, provider_id="openai", observed_at=GENERATED_AT)
    )
    types = [item.change_type for item in store.load().events]
    assert types == [
        LifecycleChangeType.LIFECYCLE_CHANGED,
        LifecycleChangeType.RETIREMENT_DATE_CHANGED,
        LifecycleChangeType.REPLACEMENT_CHANGED,
    ]


def test_known_to_conflict_is_preserved(tmp_path: Path) -> None:
    store = FileLifecycleHistoryStore(root=tmp_path / "cache")
    change = LifecycleChange(
        change_type=LifecycleChangeType.RETIREMENT_DATE_CHANGED,
        samyak_id="openai:gpt-test",
        previous=_model(retirement_at=_known("2026-11-24")),
        current=_model(retirement_at=_conflict("2026-11-24", "2026-12-01")),
    )
    store.append(_events_for(change))
    event = store.load().events[0]
    assert event.current_retirement_at is not None
    assert event.current_retirement_at.status is FactStatus.CONFLICT
    assert set(event.current_retirement_at.claims) == {"2026-11-24", "2026-12-01"}
    assert event.current_retirement_at.value is None


def test_conflict_to_known_is_preserved(tmp_path: Path) -> None:
    store = FileLifecycleHistoryStore(root=tmp_path / "cache")
    change = LifecycleChange(
        change_type=LifecycleChangeType.REPLACEMENT_CHANGED,
        samyak_id="openai:gpt-test",
        previous=_model(replacement=_conflict((_identity("a"),), (_identity("b"),))),
        current=_model(replacement=_known((_identity("a"),))),
    )
    store.append(_events_for(change))
    event = store.load().events[0]
    assert event.previous_replacement is not None
    assert event.current_replacement is not None
    assert event.previous_replacement.status is FactStatus.CONFLICT
    assert event.current_replacement.status is FactStatus.KNOWN
    assert event.current_replacement.value == ["openai:a"]


def test_removed_from_documented_catalog_is_not_retired(tmp_path: Path) -> None:
    store = FileLifecycleHistoryStore(root=tmp_path / "cache")
    store.append(
        _events_for(
            LifecycleChange(
                change_type=LifecycleChangeType.REMOVED,
                samyak_id="openai:gone",
                previous=_model("gone", lifecycle=_known(LifecycleState.ACTIVE)),
            )
        )
    )
    event = store.load().events[0]
    assert event.change_type is LifecycleChangeType.REMOVED
    assert event.previous_lifecycle is not None
    assert event.previous_lifecycle.value == "active"
    assert event.current_lifecycle is None
    rendered = render_lifecycle_history(store.load().events)
    assert "REMOVED FROM DOCUMENTED CATALOG" in rendered
    assert "RETIRED" not in rendered


def test_failed_update_produces_no_history_event(tmp_path: Path) -> None:
    catalog_store = FileCatalogStore(root=tmp_path / "cache")
    first = update_openai_catalog(
        transport=_transport(_official_pages()),
        store=catalog_store,
        clock=_clock,
    )
    assert first.committed is True
    history = FileLifecycleHistoryStore(root=tmp_path / "cache")
    before = history.load().events
    pages = _official_pages()
    pages[MODELS_INDEX_URL] = FixtureHop(status=500, body=b"error")
    failed = update_openai_catalog(
        transport=_transport(pages),
        store=catalog_store,
        clock=_clock,
    )
    assert failed.committed is False
    assert history.load().events == before


def test_partial_refresh_does_not_create_false_removal_history(tmp_path: Path) -> None:
    catalog_store = FileCatalogStore(root=tmp_path / "cache")
    first = update_openai_catalog(
        transport=_transport(_official_pages()),
        store=catalog_store,
        clock=_clock,
    )
    assert first.committed is True
    pages = _official_pages()
    del pages[model_page_url("index-only")]
    second = update_openai_catalog(
        transport=_transport(pages),
        store=catalog_store,
        clock=_clock,
    )
    assert second.partial is True
    history = FileLifecycleHistoryStore(root=tmp_path / "cache").load()
    removed = [
        item
        for item in history.events
        if item.change_type is LifecycleChangeType.REMOVED and item.samyak_id == "openai:index-only"
    ]
    assert removed == []


def test_fireworks_optional_page_failure_does_not_write_removal_history(
    tmp_path: Path,
) -> None:
    catalog_store = FileCatalogStore(root=tmp_path / "cache")
    first = update_fireworks_catalog(
        transport=_fireworks_transport(_fireworks_pages()),
        store=catalog_store,
        clock=_clock,
    )
    assert first.committed is True
    pages = _fireworks_pages()
    del pages[FIREWORKS_KIMI_URL]
    second = update_fireworks_catalog(
        transport=_fireworks_transport(pages),
        store=catalog_store,
        clock=lambda: "2026-09-12T14:00:00+00:00",
    )
    assert second.partial is True
    history = FileLifecycleHistoryStore(root=tmp_path / "cache").load()
    removed = [
        item
        for item in history.events
        if item.change_type is LifecycleChangeType.REMOVED
        and item.samyak_id
        in {
            "fireworks:accounts/fireworks/models/kimi-k2-instruct",
            "fireworks:accounts/fireworks/models/kimi-k2-instruct-latest",
        }
    ]
    assert removed == []


def test_different_providers_remain_isolated(tmp_path: Path) -> None:
    store = FileLifecycleHistoryStore(root=tmp_path / "cache")
    openai_change = LifecycleChange(
        change_type=LifecycleChangeType.LIFECYCLE_CHANGED,
        samyak_id="openai:gpt-test",
        previous=_model(lifecycle=_known(LifecycleState.ACTIVE)),
        current=_model(lifecycle=_known(LifecycleState.DEPRECATED)),
    )
    anthropic_change = LifecycleChange(
        change_type=LifecycleChangeType.LIFECYCLE_CHANGED,
        samyak_id="anthropic:claude-opus-5",
        previous=_model(
            "claude-opus-5",
            provider_id="anthropic",
            lifecycle=_known(LifecycleState.ACTIVE, provider="anthropic"),
        ),
        current=_model(
            "claude-opus-5",
            provider_id="anthropic",
            lifecycle=_known(LifecycleState.DEPRECATED, provider="anthropic"),
        ),
    )
    store.append(_events_for(openai_change))
    store.append(
        history_events_from_changes(
            (anthropic_change,),
            provider_id="anthropic",
            observed_at=GENERATED_AT,
        )
    )
    openai_only = filter_history_events(store.load().events, provider_id="openai")
    anthropic_only = filter_history_events(store.load().events, provider_id="anthropic")
    assert [item.samyak_id for item in openai_only] == ["openai:gpt-test"]
    assert [item.samyak_id for item in anthropic_only] == ["anthropic:claude-opus-5"]


def test_different_models_remain_isolated(tmp_path: Path) -> None:
    store = FileLifecycleHistoryStore(root=tmp_path / "cache")
    for model_id in ("gpt-one", "gpt-two"):
        store.append(
            _events_for(
                LifecycleChange(
                    change_type=LifecycleChangeType.ADDED,
                    samyak_id=f"openai:{model_id}",
                    current=_model(model_id, lifecycle=_known(LifecycleState.ACTIVE)),
                )
            )
        )
    selected = filter_history_events(store.load().events, samyak_id="openai:gpt-two")
    assert [item.samyak_id for item in selected] == ["openai:gpt-two"]


def test_history_survives_new_cli_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = FileLifecycleHistoryStore(root=tmp_path / "cache")
    store.append(
        _events_for(
            LifecycleChange(
                change_type=LifecycleChangeType.RETIREMENT_DATE_CHANGED,
                samyak_id="openai:gpt-4o",
                previous=_model("gpt-4o", retirement_at=_known("2027-02-01")),
                current=_model("gpt-4o", retirement_at=_known("2027-03-01")),
            )
        )
    )
    monkeypatch.setenv("SAMYAK_CACHE_DIR", str(tmp_path / "cache"))
    code = main(["model", "history", "--model", "openai:gpt-4o"])
    captured = capsys.readouterr()
    assert code == 0
    assert "Model lifecycle history" in captured.out
    assert "openai:gpt-4o" in captured.out
    assert "RETIREMENT DATE CHANGED" in captured.out
    assert "2027-02-01 → 2027-03-01" in captured.out


def test_corrupt_history_fails_closed_without_changing_catalog(tmp_path: Path) -> None:
    catalog_store = FileCatalogStore(root=tmp_path / "cache")
    catalog_store.save(_catalog(_model(lifecycle=_known(LifecycleState.ACTIVE))))
    catalog_bytes = catalog_store.catalog_path.read_bytes()
    history = FileLifecycleHistoryStore(root=tmp_path / "cache")
    history.history_path.parent.mkdir(parents=True, exist_ok=True)
    history.history_path.write_bytes(b"{not-json")
    with pytest.raises(HistoryStoreError, match="corrupt"):
        history.load()
    assert catalog_store.catalog_path.read_bytes() == catalog_bytes
    with pytest.raises(HistoryStoreError, match="corrupt"):
        history.append(
            _events_for(
                LifecycleChange(
                    change_type=LifecycleChangeType.ADDED,
                    samyak_id="openai:gpt-test",
                    current=_model(lifecycle=_known(LifecycleState.ACTIVE)),
                )
            )
        )
    assert history.history_path.read_bytes() == b"{not-json"
    assert catalog_store.catalog_path.read_bytes() == catalog_bytes


def test_unsupported_history_schema_fails_closed(tmp_path: Path) -> None:
    store = FileLifecycleHistoryStore(root=tmp_path / "cache")
    store.history_path.parent.mkdir(parents=True, exist_ok=True)
    store.history_path.write_text(
        json.dumps(
            {
                "product": "samyak",
                "capability": "model",
                "lifecycle_history_schema_version": 99,
                "events": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(HistorySchemaError):
        store.load()


def test_history_operations_do_not_mutate_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    catalog_store = FileCatalogStore(root=tmp_path / "cache")
    catalog_store.save(_catalog(_model(lifecycle=_known(LifecycleState.ACTIVE))))
    before = catalog_store.catalog_path.read_bytes()
    history = FileLifecycleHistoryStore(root=tmp_path / "cache")
    history.append(
        _events_for(
            LifecycleChange(
                change_type=LifecycleChangeType.LIFECYCLE_CHANGED,
                samyak_id="openai:gpt-test",
                previous=_model(lifecycle=_known(LifecycleState.ACTIVE)),
                current=_model(lifecycle=_known(LifecycleState.DEPRECATED)),
            )
        )
    )
    monkeypatch.setenv("SAMYAK_CACHE_DIR", str(tmp_path / "cache"))
    assert main(["model", "history"]) == 0
    assert main(["model", "history", "--provider", "openai"]) == 0
    capsys.readouterr()
    assert catalog_store.catalog_path.read_bytes() == before
    payload = json.loads(before)
    assert payload["catalog_schema_version"] == 1
    assert "events" not in payload
    assert "lifecycle_history_schema_version" not in payload


def test_cli_history_empty(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SAMYAK_CACHE_DIR", str(tmp_path / "cache"))
    assert main(["model", "history"]) == 0
    assert capsys.readouterr().out == "No documented lifecycle history.\n"


def test_cli_history_rejects_unknown_provider(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["model", "history", "--provider", "groq"]) == 2
    assert "unknown provider" in capsys.readouterr().err


def test_cli_corrupt_history_exits_nonzero_without_catalog_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    catalog_store = FileCatalogStore(root=tmp_path / "cache")
    catalog_store.save(_catalog(_model()))
    before = catalog_store.catalog_path.read_bytes()
    history_path = tmp_path / "cache" / "models" / "lifecycle-history.json"
    history_path.write_bytes(b"{not-json")
    monkeypatch.setenv("SAMYAK_CACHE_DIR", str(tmp_path / "cache"))
    assert main(["model", "history"]) == 2
    assert "corrupt" in capsys.readouterr().err
    assert catalog_store.catalog_path.read_bytes() == before
    assert history_path.read_bytes() == b"{not-json"


def test_catalog_update_succeeds_when_history_write_succeeds(tmp_path: Path) -> None:
    catalog_store = FileCatalogStore(root=tmp_path / "cache")
    first = update_openai_catalog(
        transport=_transport(_official_pages()),
        store=catalog_store,
        clock=_clock,
    )
    assert first.ok is True
    assert first.committed is True
    _rewrite_lifecycle(catalog_store, _HISTORY_TARGET, LifecycleState.ACTIVE)
    second = update_openai_catalog(
        transport=_transport(_official_pages()),
        store=catalog_store,
        clock=_clock,
    )
    assert second.ok is True
    assert second.committed is True
    assert second.catalog is not None
    assert second.lifecycle_changes
    assert all(notice.code != HISTORY_WRITE_FAILED_CODE for notice in second.notices)
    assert second.notices == second.catalog.notices
    history = FileLifecycleHistoryStore(root=tmp_path / "cache").load()
    assert history.events
    assert any(
        item.samyak_id == _HISTORY_TARGET
        and item.change_type is LifecycleChangeType.LIFECYCLE_CHANGED
        for item in second.lifecycle_changes
    )


def test_history_write_failure_keeps_catalog_committed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_store = FileCatalogStore(root=tmp_path / "cache")
    update_openai_catalog(
        transport=_transport(_official_pages()),
        store=catalog_store,
        clock=_clock,
    )
    _rewrite_lifecycle(catalog_store, _HISTORY_TARGET, LifecycleState.ACTIVE)
    seeded = update_openai_catalog(
        transport=_transport(_official_pages()),
        store=catalog_store,
        clock=_clock,
    )
    assert seeded.committed is True
    history = FileLifecycleHistoryStore(root=tmp_path / "cache")
    before_events = history.load().events
    assert before_events
    _rewrite_lifecycle(catalog_store, _HISTORY_TARGET, LifecycleState.ACTIVE)
    previous = catalog_store.load()
    previous_target = next(item for item in previous.models if item.samyak_id == _HISTORY_TARGET)
    assert previous_target.lifecycle.value is LifecycleState.ACTIVE
    monkeypatch.setattr(
        "samyak.model.update.FileLifecycleHistoryStore.append",
        _failing_history_append,
    )
    third = update_openai_catalog(
        transport=_transport(_official_pages()),
        store=catalog_store,
        clock=_clock,
    )
    assert third.ok is True
    assert third.committed is True
    assert third.error is None
    assert third.catalog is not None
    assert any(notice.code == HISTORY_WRITE_FAILED_CODE for notice in third.notices)
    assert all(notice.code != HISTORY_WRITE_FAILED_CODE for notice in third.catalog.notices)
    loaded = catalog_store.load()
    current = next(item for item in loaded.models if item.samyak_id == _HISTORY_TARGET)
    assert current.lifecycle.value is not LifecycleState.ACTIVE
    assert all(notice.code != HISTORY_WRITE_FAILED_CODE for notice in loaded.notices)
    assert any(
        item.change_type is LifecycleChangeType.LIFECYCLE_CHANGED
        and item.samyak_id == _HISTORY_TARGET
        for item in third.lifecycle_changes
    )
    assert history.load().events == before_events


def test_history_event_construction_failure_keeps_catalog_committed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_store = FileCatalogStore(root=tmp_path / "cache")
    update_openai_catalog(
        transport=_transport(_official_pages()),
        store=catalog_store,
        clock=_clock,
    )
    _rewrite_lifecycle(catalog_store, _HISTORY_TARGET, LifecycleState.ACTIVE)
    seeded = update_openai_catalog(
        transport=_transport(_official_pages()),
        store=catalog_store,
        clock=_clock,
    )
    assert seeded.committed is True
    history = FileLifecycleHistoryStore(root=tmp_path / "cache")
    before_events = history.load().events
    _rewrite_lifecycle(catalog_store, _HISTORY_TARGET, LifecycleState.ACTIVE)
    before_catalog = catalog_store.catalog_path.read_bytes()
    monkeypatch.setattr(
        "samyak.model.update.history_events_from_changes",
        _failing_history_events,
    )
    result = update_openai_catalog(
        transport=_transport(_official_pages()),
        store=catalog_store,
        clock=_clock,
    )
    assert result.ok is True
    assert result.committed is True
    assert result.error is None
    assert result.catalog is not None
    assert any(notice.code == HISTORY_WRITE_FAILED_CODE for notice in result.notices)
    assert all(notice.code != HISTORY_WRITE_FAILED_CODE for notice in result.catalog.notices)
    loaded = catalog_store.load()
    current = next(item for item in loaded.models if item.samyak_id == _HISTORY_TARGET)
    assert current.lifecycle.value is not LifecycleState.ACTIVE
    assert all(notice.code != HISTORY_WRITE_FAILED_CODE for notice in loaded.notices)
    assert catalog_store.catalog_path.read_bytes() != before_catalog
    assert history.load().events == before_events


def test_cli_displays_history_write_failure_warning(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_cli_transport(monkeypatch, _official_pages())
    assert main(["model", "update", "openai"]) == 0
    capsys.readouterr()
    catalog_store = FileCatalogStore()
    _rewrite_lifecycle(catalog_store, _HISTORY_TARGET, LifecycleState.ACTIVE)
    monkeypatch.setattr(
        "samyak.model.update.FileLifecycleHistoryStore.append",
        _failing_history_append,
    )
    code = main(["model", "update", "openai"])
    captured = capsys.readouterr()
    assert code == 0
    assert "Updated OpenAI model catalog." in captured.out
    assert "Warning: lifecycle history could not be written." in captured.out
    assert (
        "The model catalog was updated, but lifecycle changes were not persisted to history."
        in captured.out
    )
    loaded = catalog_store.load()
    current = next(item for item in loaded.models if item.samyak_id == _HISTORY_TARGET)
    assert current.lifecycle.value is not LifecycleState.ACTIVE
    assert all(notice.code != HISTORY_WRITE_FAILED_CODE for notice in loaded.notices)
    history = FileLifecycleHistoryStore()
    assert history.exists() is False or history.load().events == ()


def test_cli_displays_history_write_failure_when_event_construction_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_cli_transport(monkeypatch, _official_pages())
    assert main(["model", "update", "openai"]) == 0
    capsys.readouterr()
    catalog_store = FileCatalogStore()
    _rewrite_lifecycle(catalog_store, _HISTORY_TARGET, LifecycleState.ACTIVE)
    monkeypatch.setattr(
        "samyak.model.update.history_events_from_changes",
        _failing_history_events,
    )
    code = main(["model", "update", "openai"])
    captured = capsys.readouterr()
    assert code == 0
    assert "Updated OpenAI model catalog." in captured.out
    assert "Warning: lifecycle history could not be written." in captured.out
    loaded = catalog_store.load()
    current = next(item for item in loaded.models if item.samyak_id == _HISTORY_TARGET)
    assert current.lifecycle.value is not LifecycleState.ACTIVE
    assert all(notice.code != HISTORY_WRITE_FAILED_CODE for notice in loaded.notices)


def test_cli_history_write_warning_is_not_counted_as_catalog_notice() -> None:
    catalog = _catalog(_model(lifecycle=_known(LifecycleState.ACTIVE)))
    notice = CatalogNotice(
        code=HISTORY_WRITE_FAILED_CODE,
        message=(
            "lifecycle history could not be written. "
            "The model catalog was updated, but lifecycle changes were not persisted to history."
        ),
    )
    change = LifecycleChange(
        change_type=LifecycleChangeType.LIFECYCLE_CHANGED,
        samyak_id="openai:gpt-test",
        previous=_model(lifecycle=_known(LifecycleState.ACTIVE)),
        current=_model(lifecycle=_known(LifecycleState.DEPRECATED)),
    )
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
            notices=(notice,),
            lifecycle_changes=(change,),
        ),
        label="OpenAI",
    )
    assert "Warning: lifecycle history could not be written." in output
    assert (
        "The model catalog was updated, but lifecycle changes were not persisted to history."
        in output
    )
    assert "Notices:" not in output
    assert "LIFECYCLE CHANGED" in output or "DEPRECATED" in output


def test_cli_update_all_displays_history_write_warning() -> None:
    catalog = _catalog(_model(lifecycle=_known(LifecycleState.ACTIVE)))
    notice = CatalogNotice(
        code=HISTORY_WRITE_FAILED_CODE,
        message=(
            "lifecycle history could not be written. "
            "The model catalog was updated, but lifecycle changes were not persisted to history."
        ),
    )
    results = (
        CatalogUpdateResult(
            provider_id="openai",
            ok=True,
            committed=True,
            partial=False,
            catalog=catalog,
            catalog_path=Path("/tmp/catalog.json"),
            previous_existed=True,
            error=None,
            notices=(notice,),
            lifecycle_changes=(
                LifecycleChange(
                    change_type=LifecycleChangeType.LIFECYCLE_CHANGED,
                    samyak_id="openai:gpt-test",
                    previous=_model(lifecycle=_known(LifecycleState.ACTIVE)),
                    current=_model(lifecycle=_known(LifecycleState.DEPRECATED)),
                ),
            ),
        ),
    )
    output = _render_update_all_results(results)
    assert "Updated" in output
    assert "openai:gpt-test" in output
    assert "ACTIVE → DEPRECATED" in output
    assert "Warning: lifecycle history could not be written." in output
    assert (
        "The model catalog was updated, but lifecycle changes were not persisted to history."
        in output
    )
