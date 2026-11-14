"""Read-only Model Intelligence export. No network and no catalog mutation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from samyak.__version__ import __version__
from samyak.cli import main
from samyak.model.catalog import (
    CATALOG_SCHEMA_VERSION,
    CatalogFreshness,
    CatalogNotice,
    CatalogOverlay,
    FreshnessStatus,
    ModelCatalog,
    build_catalog,
)
from samyak.model.export import (
    EXPORT_SCHEMA_VERSION,
    PublicationStatus,
    build_model_export,
    model_export_to_json,
)
from samyak.model.facts import (
    Confidence,
    Fact,
    FactClaim,
    FactStatus,
    LifecycleState,
    Provenance,
    SourceKind,
)
from samyak.model.history import LifecycleHistory, LifecycleHistoryEvent
from samyak.model.history_store import FileLifecycleHistoryStore
from samyak.model.identity import IdentityKind, ModelIdentity
from samyak.model.lifecycle import LifecycleChangeType
from samyak.model.records import ModelRecord
from samyak.model.serialize import catalog_from_dict
from samyak.model.store import FileCatalogStore
from samyak.model.update import (
    SUPPORTED_PROVIDERS,
    history_write_failure_code,
    is_history_write_failure_code,
)

GENERATED_AT = "2026-09-09T12:00:00+00:00"
OLDER = "2026-08-01T12:00:00+00:00"
_EXPORT_KEYS = (
    "product",
    "capability",
    "export_schema_version",
    "exporter_version",
    "publication",
    "coverage",
    "catalog",
    "history",
)


def _provenance(
    provider_id: str,
    *,
    source_id: str | None = None,
    verified_at: str | None = GENERATED_AT,
) -> Provenance:
    return Provenance(
        provider=provider_id,
        source_id=source_id or f"{provider_id}-docs-models-index",
        source_kind=SourceKind.PROVIDER_DOCS,
        source_url=f"https://example.invalid/{provider_id}/models.md",
        content_hash="sha256:abc",
        retrieved_at=GENERATED_AT,
        observed_at=None,
        verified_at=verified_at,
        confidence=Confidence.HIGH,
    )


def _known_lifecycle(
    provider_id: str,
    state: LifecycleState = LifecycleState.ACTIVE,
    *,
    verified_at: str | None = GENERATED_AT,
    source_id: str | None = None,
) -> Fact[LifecycleState]:
    return Fact(
        status=FactStatus.KNOWN,
        value=state,
        provenance=_provenance(provider_id, source_id=source_id, verified_at=verified_at),
    )


def _model(
    provider_id: str,
    provider_model_id: str = "model-1",
    *,
    lifecycle: Fact[LifecycleState] | None = None,
) -> ModelRecord:
    return ModelRecord(
        identity=ModelIdentity(provider_id=provider_id, provider_model_id=provider_model_id),
        lifecycle=(_known_lifecycle(provider_id) if lifecycle is None else lifecycle),
    )


def _catalog(
    models: tuple[ModelRecord, ...] | None = None,
    notices: tuple[CatalogNotice, ...] = (),
) -> ModelCatalog:
    if models is None:
        models = tuple(_model(provider_id) for provider_id in SUPPORTED_PROVIDERS)
    return build_catalog(
        models=models,
        generated_at=GENERATED_AT,
        freshness=CatalogFreshness(
            status=FreshnessStatus.AS_OF,
            generated_at=GENERATED_AT,
            overlay=CatalogOverlay.USER_CACHE,
        ),
        notices=notices,
    )


def _history_event() -> LifecycleHistoryEvent:
    return LifecycleHistoryEvent(
        provider_id="openai",
        samyak_id="openai:model-1",
        observed_at=GENERATED_AT,
        change_type=LifecycleChangeType.LIFECYCLE_CHANGED,
        previous_lifecycle=None,
        current_lifecycle=None,
    )


def test_complete_snapshot_includes_fireworks_subset_notice() -> None:
    catalog = _catalog(
        notices=(
            CatalogNotice(
                code="DOCUMENTED_SUBSET:fireworks",
                message="Fireworks coverage is a documented subset",
            ),
        )
    )
    exported = build_model_export(catalog, None)
    payload = exported.to_dict()

    assert payload["product"] == "samyak"
    assert payload["capability"] == "model_export"
    assert payload["export_schema_version"] == EXPORT_SCHEMA_VERSION == 1
    assert payload["exporter_version"] == __version__
    assert tuple(payload) == _EXPORT_KEYS
    assert payload["publication"] == {"status": "complete", "reasons": []}
    assert exported.publication.status is PublicationStatus.COMPLETE
    assert payload["history"] is None
    assert payload["catalog"] == catalog.to_dict()
    assert payload["catalog"]["catalog_schema_version"] == CATALOG_SCHEMA_VERSION
    assert catalog_from_dict(payload["catalog"]).to_dict() == catalog.to_dict()

    coverage = {item["provider_id"]: item for item in payload["coverage"]}
    assert tuple(coverage) == SUPPORTED_PROVIDERS
    for provider_id in SUPPORTED_PROVIDERS:
        assert coverage[provider_id]["present"] is True
        assert coverage[provider_id]["refreshed_with_snapshot"] is True
        assert coverage[provider_id]["optional_source_failures"] == []
    assert coverage["fireworks"]["coverage_notices"] == ["DOCUMENTED_SUBSET:fireworks"]


def test_export_json_is_stable_and_does_not_mutate_inputs() -> None:
    catalog = _catalog()
    history = LifecycleHistory(events=(_history_event(),))
    before_catalog = catalog.to_dict()
    before_history = history.to_dict()
    first = model_export_to_json(build_model_export(catalog, history))
    second = model_export_to_json(build_model_export(catalog, history))
    assert first == second
    assert catalog.to_dict() == before_catalog
    assert history.to_dict() == before_history
    payload = json.loads(first)
    assert payload["history"] == before_history
    assert payload["catalog"]["models"][0]["lifecycle"]["status"] == "known"


def test_missing_provider_is_incomplete_without_a_failure_cause() -> None:
    models = tuple(
        _model(provider_id) for provider_id in SUPPORTED_PROVIDERS if provider_id != "google"
    )
    exported = build_model_export(_catalog(models), None)
    assert exported.publication.status is PublicationStatus.INCOMPLETE
    assert exported.publication.reasons == ("provider_missing:google",)
    google = next(item for item in exported.coverage if item.provider_id == "google")
    assert google.present is False
    assert google.refreshed_with_snapshot is False
    assert "failed" not in " ".join(exported.publication.reasons)


def test_history_write_failed_notice_blocks_publication() -> None:
    catalog = _catalog(
        notices=(
            CatalogNotice(
                code="HISTORY_WRITE_FAILED",
                message=(
                    "lifecycle history could not be written. "
                    "The model catalog was updated, but lifecycle changes "
                    "were not persisted to history."
                ),
            ),
        )
    )
    exported = build_model_export(catalog, LifecycleHistory())
    assert exported.publication.status is PublicationStatus.INCOMPLETE
    assert exported.publication.reasons == ("history_write_failed",)
    assert exported.history is not None
    assert exported.history.events == ()
    qualified = _catalog(
        notices=(
            CatalogNotice(
                code="HISTORY_WRITE_FAILED:openai:2026-09-09T12:00:00+00:00",
                message=(
                    "lifecycle history could not be written. "
                    "The model catalog was updated, but lifecycle changes "
                    "were not persisted to history."
                ),
            ),
        )
    )
    qualified_export = build_model_export(qualified, LifecycleHistory())
    assert qualified_export.publication.reasons == ("history_write_failed",)


def _marker_identity(code: str) -> tuple[str, str]:
    """Split a marker once for the provider and leave the timestamp intact."""
    prefix = "HISTORY_WRITE_FAILED:"
    rest = code.removeprefix(prefix)
    provider_id, separator, observed_at = rest.partition(":")
    assert separator == ":"
    return provider_id, observed_at


def test_history_marker_keeps_provider_and_full_timestamp() -> None:
    stamps = (
        "2026-10-30T17:00:52.530312+00:00",
        "2026-10-30T12:30:00.5-05:00",
        "2026-10-30T17:00:52Z",
    )
    cases = (("openai", stamps[0]), ("fireworks", stamps[1]), ("together", stamps[2]))
    for provider_id, stamp in cases:
        code = history_write_failure_code(provider_id, stamp)
        assert _marker_identity(code) == (provider_id, stamp)
        assert is_history_write_failure_code(code) is True


def test_malformed_and_unknown_markers_block_and_unrelated_notices_do_not() -> None:
    blockers = (
        "HISTORY_WRITE_FAILED:openai:not-a-timestamp",
        "HISTORY_WRITE_FAILED:openai",
        "HISTORY_WRITE_FAILED:",
        "HISTORY_WRITE_FAILED:groq:2026-10-30T17:00:52.530312+00:00",
    )
    unrelated = (
        "DOCUMENTED_SUBSET:fireworks",
        "PARTIAL_MODEL_PAGES:openai",
        "SOURCE_FETCH_FAILED:openai-docs-model-page:kimi-k2",
        "SOURCE_PARSE_FAILED:together-docs-model-page:glm",
        "HISTORY_WRITE_FAILED_NOTE",
        "history_write_failed",
        "NOT_HISTORY_WRITE_FAILED:openai:2026-10-30T17:00:52+00:00",
    )
    for code in blockers:
        assert is_history_write_failure_code(code) is True
        exported = build_model_export(
            _catalog(notices=(CatalogNotice(code=code, message="unresolved history write"),)),
            None,
        )
        assert exported.history is None
        assert exported.publication.status is PublicationStatus.INCOMPLETE
        assert exported.publication.reasons == ("history_write_failed",)
    informational = _catalog(
        notices=(CatalogNotice(code="DOCUMENTED_SUBSET:fireworks", message="documented subset"),)
    )
    assert build_model_export(informational, None).publication.status is PublicationStatus.COMPLETE
    for code in unrelated:
        assert is_history_write_failure_code(code) is False


def test_several_history_markers_add_one_reason_and_keep_optional_sources() -> None:
    notices = (
        CatalogNotice(
            code=history_write_failure_code("openai", "2026-10-30T17:00:52.530312+00:00"),
            message="openai history was not written",
        ),
        CatalogNotice(
            code=history_write_failure_code("together", "2026-10-30T12:30:00.5-05:00"),
            message="together history was not written",
        ),
        CatalogNotice(code="PARTIAL_MODEL_PAGES:together", message="Together pages omitted"),
        CatalogNotice(
            code="SOURCE_FETCH_FAILED:fireworks-docs-model-page:kimi-k2",
            message="Fireworks page was not retrieved",
        ),
        CatalogNotice(code="DOCUMENTED_SUBSET:fireworks", message="documented subset"),
    )
    exported = build_model_export(_catalog(notices=notices), LifecycleHistory())
    assert exported.publication.reasons.count("history_write_failed") == 1
    assert "optional_source_failed:PARTIAL_MODEL_PAGES:together" in exported.publication.reasons
    assert (
        "optional_source_failed:SOURCE_FETCH_FAILED:fireworks-docs-model-page:kimi-k2"
        in exported.publication.reasons
    )
    assert "optional_source_failed:DOCUMENTED_SUBSET:fireworks" not in exported.publication.reasons
    fireworks = next(item for item in exported.coverage if item.provider_id == "fireworks")
    assert "DOCUMENTED_SUBSET:fireworks" in fireworks.coverage_notices
    assert exported.history is not None
    assert exported.history.events == ()


def test_absent_history_without_failure_notice_stays_complete() -> None:
    exported = build_model_export(_catalog(), None)
    assert exported.history is None
    assert exported.publication.status is PublicationStatus.COMPLETE
    assert "history_write_failed" not in exported.publication.reasons


def test_optional_source_notices_block_publication() -> None:
    notices = (
        CatalogNotice(code="PARTIAL_MODEL_PAGES", message="OpenAI pages omitted"),
        CatalogNotice(code="PARTIAL_MODEL_PAGES:anthropic", message="Anthropic pages omitted"),
        CatalogNotice(
            code="SOURCE_FETCH_FAILED:openai-docs-model-page:gpt-test",
            message="OpenAI page was not retrieved",
        ),
        CatalogNotice(
            code="SOURCE_PARSE_FAILED:google-docs-model-page:gemini",
            message="Google page could not be parsed",
        ),
    )
    exported = build_model_export(_catalog(notices=notices), None)
    assert exported.publication.reasons == (
        "optional_source_failed:PARTIAL_MODEL_PAGES",
        "optional_source_failed:PARTIAL_MODEL_PAGES:anthropic",
        "optional_source_failed:SOURCE_FETCH_FAILED:openai-docs-model-page:gpt-test",
        "optional_source_failed:SOURCE_PARSE_FAILED:google-docs-model-page:gemini",
    )
    by_provider = {item.provider_id: item for item in exported.coverage}
    assert by_provider["openai"].optional_source_failures == (
        "PARTIAL_MODEL_PAGES",
        "SOURCE_FETCH_FAILED:openai-docs-model-page:gpt-test",
    )
    assert by_provider["anthropic"].optional_source_failures == ("PARTIAL_MODEL_PAGES:anthropic",)
    assert by_provider["google"].optional_source_failures == (
        "SOURCE_PARSE_FAILED:google-docs-model-page:gemini",
    )


def test_older_verified_at_marks_snapshot_incomplete() -> None:
    models = tuple(
        _model(
            provider_id,
            lifecycle=_known_lifecycle(
                provider_id,
                verified_at=OLDER if provider_id == "together" else GENERATED_AT,
            ),
        )
        for provider_id in SUPPORTED_PROVIDERS
    )
    exported = build_model_export(_catalog(models), None)
    assert exported.publication.reasons == ("facts_not_from_this_snapshot:together",)
    together = next(item for item in exported.coverage if item.provider_id == "together")
    assert together.present is True
    assert together.refreshed_with_snapshot is False


def test_not_verified_facts_do_not_block_a_complete_snapshot() -> None:
    models = []
    for provider_id in SUPPORTED_PROVIDERS:
        if provider_id == "openai":
            models.append(
                ModelRecord(
                    identity=ModelIdentity(provider_id="openai", provider_model_id="bare"),
                )
            )
        else:
            models.append(_model(provider_id))
    exported = build_model_export(_catalog(tuple(models)), None)
    assert exported.publication.status is PublicationStatus.COMPLETE
    openai = next(item for item in exported.coverage if item.provider_id == "openai")
    assert openai.refreshed_with_snapshot is True


def test_conflict_claims_must_all_match_the_snapshot_stamp() -> None:
    current = Fact(
        status=FactStatus.CONFLICT,
        claims=(
            FactClaim(
                value=LifecycleState.ACTIVE,
                provenance=_provenance("openai", source_id="openai-docs-a"),
            ),
            FactClaim(
                value=LifecycleState.DEPRECATED,
                provenance=_provenance("openai", source_id="openai-docs-b"),
            ),
        ),
    )
    stale = Fact(
        status=FactStatus.CONFLICT,
        claims=(
            FactClaim(
                value=LifecycleState.ACTIVE,
                provenance=_provenance("openai", source_id="openai-docs-a"),
            ),
            FactClaim(
                value=LifecycleState.DEPRECATED,
                provenance=_provenance("openai", source_id="openai-docs-b", verified_at=OLDER),
            ),
        ),
    )
    others = tuple(
        _model(provider_id) for provider_id in SUPPORTED_PROVIDERS if provider_id != "openai"
    )
    complete = build_model_export(_catalog((_model("openai", lifecycle=current),) + others), None)
    incomplete = build_model_export(_catalog((_model("openai", lifecycle=stale),) + others), None)
    assert complete.publication.status is PublicationStatus.COMPLETE
    assert incomplete.publication.reasons == ("facts_not_from_this_snapshot:openai",)


def test_publication_reasons_are_sorted_and_unique() -> None:
    models = tuple(
        _model(
            provider_id,
            lifecycle=_known_lifecycle(
                provider_id,
                verified_at=OLDER if provider_id == "together" else GENERATED_AT,
            ),
        )
        for provider_id in SUPPORTED_PROVIDERS
        if provider_id != "google"
    )
    notices = (
        CatalogNotice(code="SOURCE_PARSE_FAILED:together-docs-model-page:kimi", message="parse"),
        CatalogNotice(code="PARTIAL_MODEL_PAGES", message="pages"),
    )
    exported = build_model_export(_catalog(models, notices), None)
    assert exported.publication.reasons == tuple(sorted(set(exported.publication.reasons)))
    assert exported.publication.reasons == (
        "facts_not_from_this_snapshot:together",
        "optional_source_failed:PARTIAL_MODEL_PAGES",
        "optional_source_failed:SOURCE_PARSE_FAILED:together-docs-model-page:kimi",
        "provider_missing:google",
    )
    prefixes = (
        "facts_not_from_this_snapshot:",
        "optional_source_failed:",
        "provider_missing:",
    )
    assert all(reason.startswith(prefixes) for reason in exported.publication.reasons)


def test_freshness_as_of_does_not_make_an_incomplete_export_complete() -> None:
    catalog = _catalog(
        tuple(_model(provider_id) for provider_id in SUPPORTED_PROVIDERS if provider_id != "google")
    )
    assert catalog.freshness.status is FreshnessStatus.AS_OF
    exported = build_model_export(catalog, None)
    assert exported.publication.status is PublicationStatus.INCOMPLETE
    assert exported.publication.reasons == ("provider_missing:google",)


def test_absent_history_is_null_and_an_empty_log_is_an_object() -> None:
    catalog = _catalog()
    absent = build_model_export(catalog, None).to_dict()
    empty = build_model_export(catalog, LifecycleHistory()).to_dict()
    assert absent["history"] is None
    assert empty["history"]["events"] == []
    assert empty["history"]["lifecycle_history_schema_version"] == 1
    assert empty["history"]["capability"] == "model"
    assert empty["history"] is not None


def test_removed_history_event_is_not_rewritten_as_retired() -> None:
    event = LifecycleHistoryEvent(
        provider_id="openai",
        samyak_id="openai:model-1",
        observed_at=GENERATED_AT,
        change_type=LifecycleChangeType.REMOVED,
    )
    payload = build_model_export(_catalog(), LifecycleHistory(events=(event,))).to_dict()
    embedded = payload["history"]["events"][0]
    assert embedded["change_type"] == "removed"
    row = next(
        item for item in payload["catalog"]["models"] if item["samyak_id"] == "openai:model-1"
    )
    assert row["lifecycle"]["value"] == "active"


def test_serialized_export_meets_the_independent_consumer_checklist() -> None:
    complete = json.loads(model_export_to_json(build_model_export(_catalog(), None)))
    incomplete = json.loads(
        model_export_to_json(
            build_model_export(
                _catalog(
                    tuple(
                        _model(provider_id)
                        for provider_id in SUPPORTED_PROVIDERS
                        if provider_id != "google"
                    )
                ),
                LifecycleHistory(),
            )
        )
    )
    assert _consumer_can_validate(complete) is True
    assert complete["publication"]["status"] == "complete"
    assert complete["history"] is None
    assert _consumer_can_validate(incomplete) is True
    assert incomplete["publication"]["status"] == "incomplete"
    assert incomplete["history"]["events"] == []
    assert incomplete["catalog"]["freshness"]["status"] == "as_of"


def _consumer_can_validate(payload: dict[str, object]) -> bool:
    """Stdlib-only checks an importer can make without importing Samyak."""
    if payload.get("product") != "samyak" or payload.get("capability") != "model_export":
        return False
    if payload.get("export_schema_version") != 1:
        return False
    if not isinstance(payload.get("exporter_version"), str):
        return False
    publication = payload.get("publication")
    if not isinstance(publication, dict):
        return False
    status = publication.get("status")
    reasons = publication.get("reasons")
    if not isinstance(reasons, list) or any(not isinstance(item, str) for item in reasons):
        return False
    if status == "complete":
        if reasons:
            return False
    elif status == "incomplete":
        if not reasons:
            return False
    else:
        return False
    catalog = payload.get("catalog")
    if not isinstance(catalog, dict):
        return False
    if catalog.get("catalog_schema_version") != 1 or catalog.get("capability") != "model":
        return False
    if catalog.get("product") != "samyak":
        return False
    history = payload.get("history")
    if history is None:
        return True
    if not isinstance(history, dict):
        return False
    return (
        history.get("lifecycle_history_schema_version") == 1
        and history.get("capability") == "model"
        and history.get("product") == "samyak"
        and isinstance(history.get("events"), list)
    )


def test_embedded_catalog_preserves_alias_conflict_and_uncertainty() -> None:
    canonical = _model("openai", "gpt-test")
    alias = ModelRecord(
        identity=ModelIdentity(provider_id="openai", provider_model_id="gpt-alias"),
        identity_kind=IdentityKind.ALIAS,
        resolves_to=Fact(
            status=FactStatus.KNOWN,
            value=canonical.identity,
            provenance=_provenance("openai", source_id="openai-docs-alias"),
        ),
        lifecycle=_known_lifecycle("openai"),
    )
    others = tuple(
        _model(provider_id) for provider_id in SUPPORTED_PROVIDERS if provider_id != "openai"
    )
    catalog = _catalog((canonical, alias) + others)
    payload = build_model_export(catalog, None).to_dict()
    alias_row = next(
        item for item in payload["catalog"]["models"] if item["samyak_id"] == "openai:gpt-alias"
    )
    assert alias_row["identity_kind"] == "alias"
    assert alias_row["resolves_to"]["value"]["samyak_id"] == "openai:gpt-test"
    assert alias_row["deprecated_at"]["status"] == "not_verified"
    assert alias_row["deprecated_at"]["value"] is None


def _save_catalog(cache: Path, catalog: ModelCatalog, *, history: bool = False) -> FileCatalogStore:
    store = FileCatalogStore(root=cache)
    store.save(catalog)
    if history:
        FileLifecycleHistoryStore(root=cache).append((_history_event(),))
    return store


def test_cli_export_does_not_require_a_source_archive(
    samyak_cache_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _save_catalog(samyak_cache_dir, _catalog())
    archive = samyak_cache_dir / "models" / "source-archive"
    assert not archive.exists()
    assert main(["model", "export"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["publication"]["status"] == "complete"
    assert not archive.exists()


def test_cli_export_writes_json_and_leaves_files_unchanged(
    samyak_cache_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    catalog = _catalog(
        notices=(CatalogNotice(code="DOCUMENTED_SUBSET:fireworks", message="documented subset"),)
    )
    store = _save_catalog(samyak_cache_dir, catalog, history=True)
    history_path = FileLifecycleHistoryStore(root=samyak_cache_dir).history_path
    before_catalog = store.catalog_path.read_bytes()
    before_history = history_path.read_bytes()

    assert main(["model", "export"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["publication"]["status"] == "complete"
    assert payload["history"]["events"]
    assert store.catalog_path.read_bytes() == before_catalog
    assert history_path.read_bytes() == before_history


def test_cli_absent_history_is_null(
    samyak_cache_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = _save_catalog(samyak_cache_dir, _catalog())
    history_store = FileLifecycleHistoryStore(root=samyak_cache_dir)
    assert history_store.exists() is False
    assert main(["model", "export"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["history"] is None
    assert history_store.exists() is False
    assert store.catalog_path.is_file()


def test_cli_empty_history_file_is_an_object(
    samyak_cache_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _save_catalog(samyak_cache_dir, _catalog())
    history_store = FileLifecycleHistoryStore(root=samyak_cache_dir)
    history_store.history_path.write_text(
        json.dumps(LifecycleHistory().to_dict()) + "\n",
        encoding="utf-8",
    )
    assert main(["model", "export"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["history"]["events"] == []
    assert payload["history"]["lifecycle_history_schema_version"] == 1


def test_cli_incomplete_export_exits_zero_without_require_complete(
    samyak_cache_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    models = tuple(
        _model(provider_id) for provider_id in SUPPORTED_PROVIDERS if provider_id != "google"
    )
    _save_catalog(samyak_cache_dir, _catalog(models))
    assert main(["model", "export"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["publication"]["status"] == "incomplete"
    assert payload["publication"]["reasons"] == ["provider_missing:google"]


def _incomplete_catalog() -> ModelCatalog:
    models = tuple(
        _model(provider_id) for provider_id in SUPPORTED_PROVIDERS if provider_id != "google"
    )
    return _catalog(models)


def test_cli_require_complete_does_not_write_incomplete_stdout(
    samyak_cache_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _save_catalog(samyak_cache_dir, _incomplete_catalog())
    assert main(["model", "export", "--require-complete"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error: model export is incomplete" in captured.err
    assert "provider_missing:google" in captured.err


def test_cli_require_complete_does_not_create_incomplete_output(
    samyak_cache_dir: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _save_catalog(samyak_cache_dir, _incomplete_catalog())
    destination = tmp_path / "export.json"
    assert main(["model", "export", "--require-complete", "--output", str(destination)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error: model export is incomplete" in captured.err
    assert destination.exists() is False


def test_cli_require_complete_preserves_existing_incomplete_output(
    samyak_cache_dir: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _save_catalog(samyak_cache_dir, _incomplete_catalog())
    destination = tmp_path / "export.json"
    destination.write_bytes(b"keep-me")
    assert main(["model", "export", "--require-complete", "--output", str(destination)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "provider_missing:google" in captured.err
    assert destination.read_bytes() == b"keep-me"


def test_cli_incomplete_export_writes_output_without_require_complete(
    samyak_cache_dir: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _save_catalog(samyak_cache_dir, _incomplete_catalog())
    destination = tmp_path / "export.json"
    assert main(["model", "export", "--output", str(destination)]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["publication"]["status"] == "incomplete"


def test_cli_require_complete_exits_zero_for_complete_snapshot(
    samyak_cache_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _save_catalog(samyak_cache_dir, _catalog())
    assert main(["model", "export", "--require-complete"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["publication"]["status"] == "complete"
    assert captured.err == ""


def test_cli_output_path_does_not_change_catalog(
    samyak_cache_dir: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    store = _save_catalog(samyak_cache_dir, _catalog())
    before = store.catalog_path.read_bytes()
    destination = tmp_path / "export.json"
    assert main(["model", "export", "--output", str(destination)]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    written = json.loads(destination.read_text(encoding="utf-8"))
    assert written["publication"]["status"] == "complete"
    assert store.catalog_path.read_bytes() == before
    assert not FileLifecycleHistoryStore(root=samyak_cache_dir).exists()


def test_cli_tilde_output_writes_the_resolved_home_path(
    samyak_cache_dir: Path,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _save_catalog(samyak_cache_dir, _catalog())
    before = store.catalog_path.read_bytes()
    home = tmp_path / "home"
    work = tmp_path / "work"
    home.mkdir()
    work.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(work)
    assert main(["model", "export", "--output", "~/nested/../export.json"]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    written = home / "export.json"
    assert written.is_file()
    assert json.loads(written.read_text(encoding="utf-8"))["publication"]["status"] == "complete"
    assert (home / "nested").exists() is False
    assert (work / "~").exists() is False
    assert store.catalog_path.read_bytes() == before


def test_cli_tilde_output_rejects_a_resolved_protected_path(
    samyak_cache_dir: Path,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _save_catalog(samyak_cache_dir, _catalog(), history=True)
    before = _seed_protected_files(store)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("HOME", str(samyak_cache_dir))
    monkeypatch.chdir(work)
    _assert_export_output_rejected(Path("~/models/catalog.json"), capsys)
    _assert_protected_unchanged(store, before)
    assert (work / "~").exists() is False


def test_cli_require_complete_does_not_write_tilde_output_when_incomplete(
    samyak_cache_dir: Path,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_catalog(samyak_cache_dir, _incomplete_catalog())
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    assert main(["model", "export", "--require-complete", "--output", "~/export.json"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error: model export is incomplete" in captured.err
    assert (home / "export.json").exists() is False
    assert (tmp_path / "~").exists() is False


def _protected_export_paths(store: FileCatalogStore) -> dict[str, Path]:
    history = FileLifecycleHistoryStore(root=store.root)
    return {
        "catalog": store.catalog_path,
        "backup": store.backup_path,
        "history": history.history_path,
    }


def _seed_protected_files(store: FileCatalogStore) -> dict[str, bytes]:
    paths = _protected_export_paths(store)
    if not paths["backup"].is_file():
        paths["backup"].write_bytes(b"backup-sentinel")
    if not paths["history"].is_file():
        paths["history"].write_bytes(b"history-sentinel")
    return {label: path.read_bytes() for label, path in paths.items()}


def _assert_protected_unchanged(store: FileCatalogStore, before: dict[str, bytes]) -> None:
    for label, path in _protected_export_paths(store).items():
        assert path.read_bytes() == before[label]


def _assert_export_output_rejected(path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["model", "export", "--output", str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "must not replace" in captured.err


@pytest.mark.parametrize("label", ["catalog", "backup", "history"])
def test_cli_refuses_direct_protected_output_paths(
    samyak_cache_dir: Path, capsys: pytest.CaptureFixture[str], label: str
) -> None:
    store = _save_catalog(samyak_cache_dir, _catalog(), history=True)
    before = _seed_protected_files(store)
    _assert_export_output_rejected(_protected_export_paths(store)[label], capsys)
    _assert_protected_unchanged(store, before)


@pytest.mark.parametrize("label", ["catalog", "backup", "history"])
def test_cli_refuses_protected_output_paths_with_parent_segments(
    samyak_cache_dir: Path, capsys: pytest.CaptureFixture[str], label: str
) -> None:
    store = _save_catalog(samyak_cache_dir, _catalog(), history=True)
    before = _seed_protected_files(store)
    target = _protected_export_paths(store)[label]
    traversed = target.parent / "missing" / ".." / target.name
    assert traversed.parent.exists() is False
    _assert_export_output_rejected(traversed, capsys)
    _assert_protected_unchanged(store, before)


@pytest.mark.parametrize("label", ["catalog", "backup", "history"])
def test_cli_refuses_symlink_to_protected_output_paths(
    samyak_cache_dir: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path, label: str
) -> None:
    store = _save_catalog(samyak_cache_dir, _catalog(), history=True)
    before = _seed_protected_files(store)
    link = tmp_path / f"{label}-link.json"
    try:
        link.symlink_to(_protected_export_paths(store)[label])
    except OSError as exc:
        pytest.skip(f"symlinks are not supported: {exc}")
    _assert_export_output_rejected(link, capsys)
    _assert_protected_unchanged(store, before)


def test_cli_refuses_to_overwrite_catalog_or_history(
    samyak_cache_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = _save_catalog(samyak_cache_dir, _catalog(), history=True)
    history_path = FileLifecycleHistoryStore(root=samyak_cache_dir).history_path
    before_catalog = store.catalog_path.read_bytes()
    before_history = history_path.read_bytes()
    assert main(["model", "export", "--output", str(store.catalog_path)]) == 2
    assert main(["model", "export", "--output", str(history_path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "must not replace" in captured.err
    assert store.catalog_path.read_bytes() == before_catalog
    assert history_path.read_bytes() == before_history


def test_cli_corrupt_history_fails_without_rewriting_files(
    samyak_cache_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = _save_catalog(samyak_cache_dir, _catalog())
    history_path = samyak_cache_dir / "models" / "lifecycle-history.json"
    history_path.write_bytes(b"{not-json")
    before_catalog = store.catalog_path.read_bytes()
    assert main(["model", "export"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "corrupt" in captured.err
    assert store.catalog_path.read_bytes() == before_catalog
    assert history_path.read_bytes() == b"{not-json"


def test_cli_unsupported_history_schema_fails_closed(
    samyak_cache_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = _save_catalog(samyak_cache_dir, _catalog())
    history_path = samyak_cache_dir / "models" / "lifecycle-history.json"
    document = {
        "product": "samyak",
        "capability": "model",
        "lifecycle_history_schema_version": 99,
        "events": [],
    }
    history_path.write_text(json.dumps(document), encoding="utf-8")
    before = history_path.read_bytes()
    assert main(["model", "export"]) == 2
    assert "unsupported lifecycle_history_schema_version" in capsys.readouterr().err
    assert history_path.read_bytes() == before
    assert store.catalog_path.is_file()


def test_cli_missing_or_corrupt_catalog_exits_nonzero(
    samyak_cache_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["model", "export"]) == 2
    assert "no local model catalog" in capsys.readouterr().err
    assert not (samyak_cache_dir / "models" / "lifecycle-history.json").exists()

    catalog_path = samyak_cache_dir / "models" / "catalog.json"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_bytes(b"{not-json")
    assert main(["model", "export"]) == 2
    assert "corrupt" in capsys.readouterr().err
    assert catalog_path.read_bytes() == b"{not-json"


def test_cli_unsupported_catalog_schema_exits_nonzero(
    samyak_cache_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = _save_catalog(samyak_cache_dir, _catalog())
    payload = json.loads(store.catalog_path.read_text(encoding="utf-8"))
    payload["catalog_schema_version"] = 99
    store.catalog_path.write_text(json.dumps(payload), encoding="utf-8")
    before = store.catalog_path.read_bytes()
    assert main(["model", "export"]) == 2
    assert "unsupported catalog_schema_version" in capsys.readouterr().err
    assert store.catalog_path.read_bytes() == before


def test_cli_export_help_lists_require_complete(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["model", "export", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--require-complete" in help_text
    assert "--output" in help_text
