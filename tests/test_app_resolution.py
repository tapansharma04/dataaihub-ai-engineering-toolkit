"""Model Intelligence resolution and first-slice lifecycle rules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from helpers import make_project
from samyak.app.catalog_view import CatalogView, load_catalog_view
from samyak.app.models import (
    FINDING_MODEL_PROVIDER_AMBIGUOUS,
    FINDING_MODEL_REFERENCE_DEPRECATED,
    FINDING_MODEL_REFERENCE_RETIRED,
    FINDING_MODEL_REFERENCE_UNKNOWN,
    NOTICE_CATALOG_ABSENT,
    NOTICE_CATALOG_SCHEMA,
    NOTICE_CATALOG_STALE,
    NOTICE_CATALOG_UNREADABLE,
    NOTICE_MODEL_CATALOG_CANDIDATE,
    NOTICE_MODEL_LIFECYCLE_CONFLICT,
    NOTICE_MODEL_REFERENCE_UNCHECKED,
    NOTICE_PROVIDER_NOT_IN_CATALOG,
    CatalogContext,
    MatchKind,
    Severity,
)
from samyak.app.models import (
    Confidence as DetectorConfidence,
)
from samyak.app.observations import Observation, ObservationKind
from samyak.app.pipeline import extract_application
from samyak.model.catalog import CatalogFreshness, CatalogOverlay, FreshnessStatus, build_catalog
from samyak.model.facts import (
    Confidence as CatalogConfidence,
)
from samyak.model.facts import (
    Fact,
    FactClaim,
    FactStatus,
    LifecycleState,
    Provenance,
    SourceKind,
    unverified_fact,
)
from samyak.model.identity import IdentityKind, ModelIdentity
from samyak.model.records import ModelRecord
from samyak.model.store import FileCatalogStore

GENERATED_AT = "2026-09-09T12:00:00+00:00"
APP_ROOT = Path(__file__).resolve().parents[1] / "src" / "samyak" / "app"


def _provenance(provider: str = "openai", **overrides: object) -> Provenance:
    payload: dict[str, object] = {
        "provider": provider,
        "source_id": f"{provider}-docs",
        "source_kind": SourceKind.PROVIDER_DOCS,
        "source_url": f"https://example.invalid/{provider}/docs",
        "content_hash": "sha256:test",
        "retrieved_at": "2026-09-09T11:59:00+00:00",
        "verified_at": "2026-09-09T12:00:00+00:00",
        "confidence": CatalogConfidence.HIGH,
    }
    payload.update(overrides)
    return Provenance(**payload)  # type: ignore[arg-type]


def _known(value: object, provider: str = "openai", **overrides: object) -> Fact:
    return Fact(
        status=FactStatus.KNOWN,
        value=value,
        provenance=_provenance(provider, **overrides),
    )


def _unknown(provider: str = "openai") -> Fact:
    return Fact(status=FactStatus.UNKNOWN, provenance=_provenance(provider))


def _conflict(*values: object, provider: str = "openai") -> Fact:
    claims = tuple(
        FactClaim(
            value=value,
            provenance=_provenance(provider, source_id=f"{provider}-source-{index}"),
        )
        for index, value in enumerate(values)
    )
    return Fact(status=FactStatus.CONFLICT, claims=claims)


def _freshness(status: FreshnessStatus = FreshnessStatus.AS_OF) -> CatalogFreshness:
    return CatalogFreshness(
        status=status,
        generated_at=GENERATED_AT,
        overlay=CatalogOverlay.USER_CACHE,
    )


def _model(
    provider_model_id: str,
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


def _save_catalog(
    *records: ModelRecord,
    freshness_status: FreshnessStatus = FreshnessStatus.AS_OF,
) -> None:
    catalog = build_catalog(
        models=records,
        generated_at=GENERATED_AT,
        freshness=_freshness(freshness_status),
    )
    FileCatalogStore().save(catalog)


def _openai_source(model_id: str) -> str:
    return f"import openai\nclient.create(model={model_id!r})\n"


def _anthropic_source(model_id: str) -> str:
    return f"import anthropic\nclient.messages.create(model={model_id!r})\n"


def _plain_source(model_id: str) -> str:
    return f"client.create(model={model_id!r})\n"


def _finding_codes(extraction) -> list[str]:
    return [item.code for item in extraction.report.findings]


def _notice_codes(extraction) -> list[str]:
    return [item.code for item in extraction.report.notices]


def _lifecycle_codes() -> frozenset[str]:
    return frozenset(
        {
            FINDING_MODEL_REFERENCE_DEPRECATED,
            FINDING_MODEL_REFERENCE_RETIRED,
        }
    )


def test_catalog_absent_emits_notice_only(tmp_path: Path) -> None:
    project = make_project(tmp_path, {"bot.py": _openai_source("gpt-retired")})
    extraction = extract_application(project)
    assert extraction.report.catalog == CatalogContext(present=False)
    assert NOTICE_CATALOG_ABSENT in _notice_codes(extraction)
    assert extraction.report.findings == ()
    assert not (_lifecycle_codes() & set(_finding_codes(extraction)))


def test_catalog_unreadable_emits_notice_only(tmp_path: Path) -> None:
    store = FileCatalogStore()
    store.models_dir.mkdir(parents=True)
    store.catalog_path.write_text("{not json", encoding="utf-8")
    project = make_project(tmp_path, {"bot.py": _openai_source("gpt-retired")})
    extraction = extract_application(project)
    assert extraction.report.catalog.present is False
    assert NOTICE_CATALOG_UNREADABLE in _notice_codes(extraction)
    assert extraction.report.findings == ()


def test_catalog_unsupported_schema_emits_notice_only(tmp_path: Path) -> None:
    _save_catalog(_model("gpt-retired", lifecycle=_known(LifecycleState.RETIRED)))
    store = FileCatalogStore()
    payload = json.loads(store.catalog_path.read_text(encoding="utf-8"))
    payload["catalog_schema_version"] = 99
    store.catalog_path.write_text(json.dumps(payload), encoding="utf-8")
    project = make_project(tmp_path, {"bot.py": _openai_source("gpt-retired")})
    extraction = extract_application(project)
    assert NOTICE_CATALOG_SCHEMA in _notice_codes(extraction)
    assert extraction.report.findings == ()


def test_stale_catalog_uses_stored_facts_and_stays_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _save_catalog(
        _model("gpt-old", lifecycle=_known(LifecycleState.DEPRECATED)),
        freshness_status=FreshnessStatus.STALE,
    )
    original = FileCatalogStore.save

    def boom(self: FileCatalogStore, catalog: object) -> None:
        raise AssertionError("catalog must not be written during analysis")

    monkeypatch.setattr(FileCatalogStore, "save", boom)
    project = make_project(tmp_path, {"bot.py": _openai_source("gpt-old")})
    extraction = extract_application(project)
    monkeypatch.setattr(FileCatalogStore, "save", original)
    assert NOTICE_CATALOG_STALE in _notice_codes(extraction)
    assert _finding_codes(extraction) == [FINDING_MODEL_REFERENCE_DEPRECATED]
    finding = extraction.report.findings[0]
    assert finding.catalog_evidence is not None
    assert finding.catalog_evidence.catalog_freshness == {
        "status": "stale",
        "generated_at": GENERATED_AT,
        "overlay": "user_cache",
    }
    assert extraction.report.catalog.present is True
    assert extraction.report.catalog.freshness_status == "stale"


def test_provider_established_but_missing_from_catalog(tmp_path: Path) -> None:
    _save_catalog(
        _model(
            "claude-active",
            provider_id="anthropic",
            lifecycle=_known(LifecycleState.ACTIVE, provider="anthropic"),
        )
    )
    project = make_project(tmp_path, {"bot.py": _openai_source("gpt-retired")})
    extraction = extract_application(project)
    assert NOTICE_PROVIDER_NOT_IN_CATALOG in _notice_codes(extraction)
    assert extraction.report.findings == ()
    assert FINDING_MODEL_REFERENCE_UNKNOWN not in _finding_codes(extraction)


def test_exact_match_deprecated_and_retired(tmp_path: Path) -> None:
    replacement = ModelIdentity(provider_id="openai", provider_model_id="gpt-new")
    _save_catalog(
        _model(
            "gpt-dep",
            lifecycle=_known(LifecycleState.DEPRECATED),
            replacement=_known((replacement,)),
        ),
        _model("gpt-ret", lifecycle=_known(LifecycleState.RETIRED)),
        _model("gpt-new", lifecycle=_known(LifecycleState.ACTIVE)),
    )
    project = make_project(
        tmp_path,
        {
            "a.py": _openai_source("gpt-dep"),
            "z.py": _openai_source("gpt-ret"),
        },
    )
    extraction = extract_application(project)
    findings = extraction.report.findings
    assert [item.code for item in findings] == [
        FINDING_MODEL_REFERENCE_DEPRECATED,
        FINDING_MODEL_REFERENCE_RETIRED,
    ]
    assert [item.evidence.path for item in findings] == ["a.py", "z.py"]
    deprecated, retired = findings
    assert deprecated.severity is Severity.MEDIUM
    assert retired.severity is Severity.HIGH
    assert deprecated.confidence is DetectorConfidence.HIGH
    assert retired.confidence is DetectorConfidence.HIGH
    assert deprecated.catalog_evidence is not None
    assert deprecated.catalog_evidence.match_kind is MatchKind.EXACT_IDENTITY
    assert deprecated.catalog_evidence.samyak_id == "openai:gpt-dep"
    assert deprecated.catalog_evidence.provider_id == "openai"
    assert deprecated.catalog_evidence.provider_model_id == "gpt-dep"
    assert deprecated.catalog_evidence.lifecycle_status == "known"
    assert deprecated.catalog_evidence.lifecycle_value == "deprecated"
    assert deprecated.catalog_evidence.replacement_samyak_ids == ("openai:gpt-new",)
    assert deprecated.recommendation == "Replace this reference with openai:gpt-new."
    assert retired.catalog_evidence is not None
    assert retired.catalog_evidence.samyak_id == "openai:gpt-ret"
    assert "does not establish a successor" in retired.recommendation
    assert "Replace this reference with a current model" not in retired.recommendation


def test_exact_match_unknown_model_is_info_only(tmp_path: Path) -> None:
    _save_catalog(_model("gpt-active", lifecycle=_known(LifecycleState.ACTIVE)))
    project = make_project(tmp_path, {"bot.py": _openai_source("missing-id")})
    extraction = extract_application(project)
    assert _finding_codes(extraction) == [FINDING_MODEL_REFERENCE_UNKNOWN]
    finding = extraction.report.findings[0]
    assert finding.severity is Severity.INFO
    assert finding.catalog_evidence is not None
    assert finding.catalog_evidence.match_kind is MatchKind.NOT_IN_CATALOG
    assert finding.catalog_evidence.provider_id == "openai"
    assert finding.catalog_evidence.samyak_id is None
    assert finding.catalog_evidence.lifecycle_status is None
    assert "lifecycle_status" not in finding.catalog_evidence.to_dict()
    assert not (_lifecycle_codes() & set(_finding_codes(extraction)))


def test_provider_less_zero_matches_is_notice_not_per_occurrence(tmp_path: Path) -> None:
    _save_catalog(_model("gpt-active", lifecycle=_known(LifecycleState.ACTIVE)))
    project = make_project(
        tmp_path,
        {
            "a.py": _plain_source("ghost-id"),
            "b.py": _plain_source("ghost-id"),
            "c.py": _plain_source("other-ghost"),
        },
    )
    extraction = extract_application(project)
    unchecked = [
        item for item in extraction.report.notices if item.code == NOTICE_MODEL_REFERENCE_UNCHECKED
    ]
    assert len(unchecked) == 1
    assert "ghost-id" in unchecked[0].message
    assert "other-ghost" in unchecked[0].message
    assert FINDING_MODEL_REFERENCE_UNKNOWN not in _finding_codes(extraction)
    assert extraction.report.findings == ()


def test_provider_less_unique_match_is_candidate_not_lifecycle(tmp_path: Path) -> None:
    _save_catalog(_model("unique-retired", lifecycle=_known(LifecycleState.RETIRED)))
    project = make_project(tmp_path, {"bot.py": _plain_source("unique-retired")})
    extraction = extract_application(project)
    assert NOTICE_MODEL_CATALOG_CANDIDATE in _notice_codes(extraction)
    assert extraction.report.findings == ()
    candidate = next(
        item for item in extraction.report.notices if item.code == NOTICE_MODEL_CATALOG_CANDIDATE
    )
    assert "openai:unique-retired" in candidate.message
    assert "unique-retired" in candidate.message


def test_json_unique_match_does_not_create_lifecycle_finding(tmp_path: Path) -> None:
    _save_catalog(_model("cfg-retired", lifecycle=_known(LifecycleState.RETIRED)))
    project = make_project(tmp_path, {"cfg.json": '{"llm": {"model": "cfg-retired"}}\n'})
    extraction = extract_application(project)
    assert extraction.report.findings == ()
    assert NOTICE_MODEL_CATALOG_CANDIDATE in _notice_codes(extraction)
    refs = [
        item for item in extraction.observations if item.kind is ObservationKind.MODEL_REFERENCE
    ]
    assert refs[0].location.line_start is None
    assert refs[0].location.line_end is None


def test_provider_less_multiple_match_is_ambiguous(tmp_path: Path) -> None:
    _save_catalog(
        _model(
            "shared-id",
            provider_id="together",
            lifecycle=_known(LifecycleState.DEPRECATED, provider="together"),
        ),
        _model(
            "shared-id",
            provider_id="fireworks",
            lifecycle=_known(LifecycleState.RETIRED, provider="fireworks"),
        ),
    )
    project = make_project(tmp_path, {"bot.py": _plain_source("shared-id")})
    extraction = extract_application(project)
    assert _finding_codes(extraction) == [FINDING_MODEL_PROVIDER_AMBIGUOUS]
    finding = extraction.report.findings[0]
    assert finding.severity is Severity.INFO
    assert finding.catalog_evidence is not None
    assert finding.catalog_evidence.match_kind is MatchKind.MODEL_ID_AMBIGUOUS
    assert finding.catalog_evidence.candidate_samyak_ids == (
        "fireworks:shared-id",
        "together:shared-id",
    )
    assert finding.catalog_evidence.samyak_id is None
    assert finding.catalog_evidence.provider_id is None
    assert finding.catalog_evidence.lifecycle_status is None
    assert not (_lifecycle_codes() & set(_finding_codes(extraction)))


def test_json_ambiguous_finding_is_file_level(tmp_path: Path) -> None:
    _save_catalog(
        _model(
            "shared-id",
            provider_id="together",
            lifecycle=_known(LifecycleState.ACTIVE, provider="together"),
        ),
        _model(
            "shared-id",
            provider_id="fireworks",
            lifecycle=_known(LifecycleState.ACTIVE, provider="fireworks"),
        ),
    )
    project = make_project(tmp_path, {"cfg.json": '{"model": "shared-id"}\n'})
    extraction = extract_application(project)
    finding = extraction.report.findings[0]
    assert finding.code == FINDING_MODEL_PROVIDER_AMBIGUOUS
    assert finding.evidence.line_start is None
    assert finding.evidence.line_end is None
    assert finding.evidence.config_key == "model"


@pytest.mark.parametrize(
    "lifecycle",
    [
        _known(LifecycleState.ACTIVE),
        _known(LifecycleState.LEGACY),
        _unknown(),
        unverified_fact(),
    ],
)
def test_non_actionable_lifecycle_produces_no_finding(tmp_path: Path, lifecycle: Fact) -> None:
    _save_catalog(_model("gpt-test", lifecycle=lifecycle))
    project = make_project(tmp_path, {"bot.py": _openai_source("gpt-test")})
    extraction = extract_application(project)
    assert extraction.report.findings == ()
    assert NOTICE_MODEL_LIFECYCLE_CONFLICT not in _notice_codes(extraction)


def test_conflict_lifecycle_produces_notice_not_finding(tmp_path: Path) -> None:
    _save_catalog(
        _model(
            "gpt-conflict",
            lifecycle=_conflict(LifecycleState.DEPRECATED, LifecycleState.RETIRED),
        )
    )
    project = make_project(tmp_path, {"bot.py": _openai_source("gpt-conflict")})
    extraction = extract_application(project)
    assert extraction.report.findings == ()
    conflict = next(
        item for item in extraction.report.notices if item.code == NOTICE_MODEL_LIFECYCLE_CONFLICT
    )
    assert "openai:gpt-conflict" in conflict.message
    assert "deprecated" in conflict.message
    assert "retired" in conflict.message


def test_conflict_replacement_does_not_imply_successor(tmp_path: Path) -> None:
    _save_catalog(
        _model(
            "gpt-dep",
            lifecycle=_known(LifecycleState.DEPRECATED),
            replacement=_conflict(
                (ModelIdentity(provider_id="openai", provider_model_id="gpt-a"),),
                (ModelIdentity(provider_id="openai", provider_model_id="gpt-b"),),
            ),
        )
    )
    project = make_project(tmp_path, {"bot.py": _openai_source("gpt-dep")})
    extraction = extract_application(project)
    finding = extraction.report.findings[0]
    assert finding.code == FINDING_MODEL_REFERENCE_DEPRECATED
    assert "in conflict" in finding.recommendation
    assert "no successor was selected" in finding.recommendation
    assert "Replace this reference with openai:" not in finding.recommendation


def test_alias_preserves_detected_evidence(tmp_path: Path) -> None:
    canonical = _model("gpt-canonical", lifecycle=_known(LifecycleState.ACTIVE))
    alias = _model(
        "gpt-alias",
        identity_kind=IdentityKind.ALIAS,
        resolves_to=_known(canonical.identity),
        lifecycle=_known(LifecycleState.DEPRECATED),
    )
    _save_catalog(canonical, alias)
    project = make_project(tmp_path, {"bot.py": _openai_source("gpt-alias")})
    extraction = extract_application(project)
    assert _finding_codes(extraction) == [FINDING_MODEL_REFERENCE_DEPRECATED]
    finding = extraction.report.findings[0]
    assert finding.evidence.detected_value == "gpt-alias"
    assert finding.catalog_evidence is not None
    assert finding.catalog_evidence.samyak_id == "openai:gpt-alias"
    assert finding.catalog_evidence.identity_kind == "alias"
    assert finding.catalog_evidence.provider_model_id == "gpt-alias"
    assert finding.catalog_evidence.resolves_to_samyak_id == "openai:gpt-canonical"
    assert "gpt-canonical" not in finding.evidence.to_dict()["detected_value"]
    payload = finding.to_dict()
    assert payload["evidence"]["detected_value"] == "gpt-alias"
    assert payload["catalog_evidence"]["samyak_id"] == "openai:gpt-alias"


def test_anthropic_import_does_not_steal_openai_catalog_hit(tmp_path: Path) -> None:
    _save_catalog(
        _model("gpt-retired", lifecycle=_known(LifecycleState.RETIRED)),
        _model(
            "claude-active",
            provider_id="anthropic",
            lifecycle=_known(LifecycleState.ACTIVE, provider="anthropic"),
        ),
    )
    project = make_project(tmp_path, {"bot.py": _anthropic_source("gpt-retired")})
    extraction = extract_application(project)
    assert _finding_codes(extraction) == [FINDING_MODEL_REFERENCE_UNKNOWN]
    assert FINDING_MODEL_REFERENCE_RETIRED not in _finding_codes(extraction)


def test_same_file_constant_creates_lifecycle_finding(tmp_path: Path) -> None:
    _save_catalog(_model("gpt-retired", lifecycle=_known(LifecycleState.RETIRED)))
    project = make_project(
        tmp_path,
        {"bot.py": "import openai\nMODEL = 'gpt-retired'\nclient.create(model=MODEL)\n"},
    )
    extraction = extract_application(project)
    assert _finding_codes(extraction) == [FINDING_MODEL_REFERENCE_RETIRED]
    finding = extraction.report.findings[0]
    assert finding.reference_expression == "MODEL"
    assert finding.resolution_source is not None
    assert finding.evidence.detected_value == "gpt-retired"


def test_detector_confidence_is_not_catalog_fact_confidence(tmp_path: Path) -> None:
    _save_catalog(
        _model(
            "gpt-dep",
            lifecycle=_known(LifecycleState.DEPRECATED, confidence=CatalogConfidence.MEDIUM),
        )
    )
    project = make_project(tmp_path, {"bot.py": _openai_source("gpt-dep")})
    extraction = extract_application(project)
    finding = extraction.report.findings[0]
    assert finding.confidence is DetectorConfidence.HIGH
    assert finding.severity is Severity.MEDIUM
    assert finding.catalog_evidence is not None
    assert finding.catalog_evidence.fact_confidence == "medium"
    assert finding.catalog_evidence.lifecycle_status == "known"


def test_catalog_view_resolution_branches() -> None:
    catalog = build_catalog(
        models=(
            _model(
                "shared",
                provider_id="together",
                lifecycle=_known(LifecycleState.ACTIVE, provider="together"),
            ),
            _model(
                "shared",
                provider_id="fireworks",
                lifecycle=_known(LifecycleState.ACTIVE, provider="fireworks"),
            ),
            _model("only-openai", lifecycle=_known(LifecycleState.DEPRECATED)),
        ),
        generated_at=GENERATED_AT,
        freshness=_freshness(),
    )
    view = CatalogView.from_catalog(catalog)
    from samyak.app.models import EvidenceType, SourceLocation

    def ref(model_id: str, provider_id: str | None) -> Observation:
        return Observation(
            kind=ObservationKind.MODEL_REFERENCE,
            location=SourceLocation(
                path="bot.py",
                evidence_type=EvidenceType.PYTHON_KEYWORD,
                detected_value=model_id,
                line_start=1,
                line_end=1,
                config_key="model",
            ),
            provider_id=provider_id,
        )

    exact = view.resolve(ref("only-openai", "openai"))
    assert exact is not None
    assert exact.match_kind is MatchKind.EXACT_IDENTITY
    assert exact.record is not None
    assert exact.record.samyak_id == "openai:only-openai"

    unknown = view.resolve(ref("missing", "openai"))
    assert unknown is not None
    assert unknown.match_kind is MatchKind.NOT_IN_CATALOG
    assert unknown.source_provider_id == "openai"

    missing_provider = view.resolve(ref("only-openai", "google"))
    assert missing_provider is not None
    assert missing_provider.match_kind is MatchKind.PROVIDER_NOT_IN_CATALOG

    unique = view.resolve(ref("only-openai", None))
    assert unique is not None
    assert unique.match_kind is MatchKind.MODEL_ID_UNIQUE
    assert unique.record is None
    assert unique.candidates[0].samyak_id == "openai:only-openai"

    ambiguous = view.resolve(ref("shared", None))
    assert ambiguous is not None
    assert ambiguous.match_kind is MatchKind.MODEL_ID_AMBIGUOUS
    assert [item.samyak_id for item in ambiguous.candidates] == [
        "fireworks:shared",
        "together:shared",
    ]

    none = view.resolve(ref("absent", None))
    assert none is not None
    assert none.match_kind is MatchKind.NOT_IN_CATALOG
    assert none.source_provider_id is None


def test_load_catalog_view_absent_does_not_touch_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("model update/http must not run")

    monkeypatch.setattr("samyak.model.store.FileCatalogStore.save", boom)
    view, context, notices = load_catalog_view()
    assert view is None
    assert context.present is False
    assert [item.code for item in notices] == [NOTICE_CATALOG_ABSENT]


def test_analysis_does_not_import_model_network_surface() -> None:
    forbidden = (
        "samyak.model.providers",
        "samyak.model.update",
        "samyak.model.http",
    )
    for path in APP_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in forbidden:
            assert name not in text, f"{path} imports {name}"


def test_findings_and_notices_are_deterministically_ordered(tmp_path: Path) -> None:
    _save_catalog(
        _model("gpt-dep", lifecycle=_known(LifecycleState.DEPRECATED)),
        _model("gpt-ret", lifecycle=_known(LifecycleState.RETIRED)),
    )
    project = make_project(
        tmp_path,
        {
            "z.py": _openai_source("gpt-dep"),
            "a.py": _openai_source("gpt-ret"),
        },
    )
    first = extract_application(project)
    second = extract_application(project)
    assert [item.instance_key for item in first.report.findings] == [
        item.instance_key for item in second.report.findings
    ]
    assert [item.to_dict() for item in first.report.notices] == [
        item.to_dict() for item in second.report.notices
    ]
    assert [item.evidence.path for item in first.report.findings] == ["a.py", "z.py"]
    assert _finding_codes(first) == [
        FINDING_MODEL_REFERENCE_RETIRED,
        FINDING_MODEL_REFERENCE_DEPRECATED,
    ]
