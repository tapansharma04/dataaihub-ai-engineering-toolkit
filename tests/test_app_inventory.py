"""Application model inventory (Phase 4A)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from helpers import make_project
from samyak.app.catalog_view import CatalogView
from samyak.app.models import (
    FINDING_MODEL_REFERENCE_DEPRECATED,
    FINDING_MODEL_REFERENCE_RETIRED,
    FINDING_MODEL_REFERENCE_UNKNOWN,
    AiOperationKind,
    ApplicationIdentityState,
    EvidenceType,
    MatchKind,
    UnboundReason,
    UnresolvedOperation,
)
from samyak.app.observations import Observation, ObservationKind
from samyak.app.pipeline import extract_application
from samyak.app.report import render_app_report
from samyak.model.facts import LifecycleState
from samyak.model.store import FileCatalogStore
from test_app_resolution import (
    _known,
    _model,
    _openai_source,
    _plain_source,
    _save_catalog,
    _unknown,
    unverified_fact,
)


def _dependencies(extraction):
    return extraction.report.inventory.dependencies


def _unresolved(extraction):
    return extraction.report.inventory.unresolved_operations


def test_active_literal_is_in_inventory_without_findings(tmp_path: Path) -> None:
    _save_catalog(_model("gpt-active", lifecycle=_known(LifecycleState.ACTIVE)))
    project = make_project(tmp_path, {"bot.py": _openai_source("gpt-active")})
    extraction = extract_application(project)
    assert extraction.report.findings == ()
    deps = _dependencies(extraction)
    assert len(deps) == 1
    dep = deps[0]
    assert dep.application_identity is ApplicationIdentityState.ESTABLISHED
    assert dep.detected_model_id == "gpt-active"
    assert dep.provider_id == "openai"
    assert dep.samyak_id == "openai:gpt-active"
    assert dep.catalog_evidence is not None
    assert dep.catalog_evidence.lifecycle_value == "active"
    assert len(dep.operations) == 1
    assert dep.operations[0].path == "bot.py"
    text = render_app_report(extraction.report, "text")
    assert "Inventory" in text
    assert "openai:gpt-active" in text
    assert "Identity:     established" in text
    assert "No findings." in text
    assert "no AI dependencies" not in text.lower()


def test_retired_literal_is_in_inventory_and_findings(tmp_path: Path) -> None:
    _save_catalog(_model("gpt-ret", lifecycle=_known(LifecycleState.RETIRED)))
    project = make_project(tmp_path, {"bot.py": _openai_source("gpt-ret")})
    extraction = extract_application(project)
    assert [item.code for item in extraction.report.findings] == [FINDING_MODEL_REFERENCE_RETIRED]
    dep = _dependencies(extraction)[0]
    assert dep.application_identity is ApplicationIdentityState.ESTABLISHED
    assert dep.samyak_id == "openai:gpt-ret"
    assert dep.catalog_evidence is not None
    assert dep.catalog_evidence.lifecycle_value == "retired"


def test_deprecated_literal_is_in_inventory_and_findings(tmp_path: Path) -> None:
    _save_catalog(_model("gpt-dep", lifecycle=_known(LifecycleState.DEPRECATED)))
    project = make_project(tmp_path, {"bot.py": _openai_source("gpt-dep")})
    extraction = extract_application(project)
    assert [item.code for item in extraction.report.findings] == [
        FINDING_MODEL_REFERENCE_DEPRECATED
    ]
    assert _dependencies(extraction)[0].detected_model_id == "gpt-dep"


def test_multiple_established_references_group_together(tmp_path: Path) -> None:
    _save_catalog(_model("gpt-active", lifecycle=_known(LifecycleState.ACTIVE)))
    project = make_project(
        tmp_path,
        {
            "a.py": _openai_source("gpt-active"),
            "z.py": "import openai\nclient.responses.create(model='gpt-active')\n",
        },
    )
    extraction = extract_application(project)
    deps = _dependencies(extraction)
    assert len(deps) == 1
    paths = [item.path for item in deps[0].operations]
    assert paths == ["a.py", "z.py"]
    assert extraction.report.findings == ()


def test_provider_less_unique_is_candidate_not_established(tmp_path: Path) -> None:
    _save_catalog(_model("unique-id", lifecycle=_known(LifecycleState.RETIRED)))
    project = make_project(tmp_path, {"bot.py": _plain_source("unique-id")})
    extraction = extract_application(project)
    assert extraction.report.findings == ()
    dep = _dependencies(extraction)[0]
    assert dep.application_identity is ApplicationIdentityState.CANDIDATE
    assert dep.samyak_id is None
    assert dep.provider_id is None
    assert dep.match_kind is MatchKind.MODEL_ID_UNIQUE
    assert dep.catalog_evidence is not None
    assert dep.catalog_evidence.candidate_samyak_ids == ("openai:unique-id",)
    assert dep.catalog_evidence.samyak_id is None
    assert dep.catalog_evidence.lifecycle_value is None


def test_provider_less_ambiguous_is_ambiguous(tmp_path: Path) -> None:
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
    dep = _dependencies(extraction)[0]
    assert dep.application_identity is ApplicationIdentityState.AMBIGUOUS
    assert dep.samyak_id is None
    assert dep.provider_id is None
    assert dep.match_kind is MatchKind.MODEL_ID_AMBIGUOUS


def test_established_unknown_and_not_verified_lifecycle_stay_in_inventory(
    tmp_path: Path,
) -> None:
    _save_catalog(
        _model("gpt-unknown", lifecycle=_unknown()),
        _model("gpt-unverified", lifecycle=unverified_fact()),
    )
    project = make_project(
        tmp_path,
        {
            "a.py": _openai_source("gpt-unknown"),
            "b.py": _openai_source("gpt-unverified"),
        },
    )
    extraction = extract_application(project)
    assert extraction.report.findings == ()
    states = {item.detected_model_id: item for item in _dependencies(extraction)}
    assert states["gpt-unknown"].application_identity is ApplicationIdentityState.ESTABLISHED
    assert states["gpt-unknown"].catalog_evidence is not None
    assert states["gpt-unknown"].catalog_evidence.lifecycle_status == "unknown"
    assert states["gpt-unknown"].catalog_evidence.lifecycle_value is None
    assert states["gpt-unverified"].application_identity is ApplicationIdentityState.ESTABLISHED
    assert states["gpt-unverified"].catalog_evidence is not None
    assert states["gpt-unverified"].catalog_evidence.lifecycle_status == "not_verified"


def test_name_not_bound_is_unresolved(tmp_path: Path) -> None:
    _save_catalog(_model("MODEL", lifecycle=_known(LifecycleState.RETIRED)))
    project = make_project(
        tmp_path,
        {"bot.py": "import openai\nclient.create(model=MODEL)\n"},
    )
    extraction = extract_application(project)
    assert [item.detected_model_id for item in _dependencies(extraction)] == []
    ops = _unresolved(extraction)
    assert len(ops) == 1
    assert ops[0].application_identity is ApplicationIdentityState.UNRESOLVED
    assert ops[0].unbound_reason is UnboundReason.NAME_NOT_BOUND
    assert ops[0].expression == "MODEL"
    assert ops[0].path == "bot.py"
    assert ops[0].slot == "model"
    payload = extraction.report.to_dict()
    assert "detected_model_id" not in payload["inventory"]["unresolved_operations"][0]
    assert payload["inventory"]["dependencies"] == []
    assert FINDING_MODEL_REFERENCE_RETIRED not in [item.code for item in extraction.report.findings]


def test_getenv_is_dynamic_without_identity(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {"bot.py": "import os\nclient.create(model=os.getenv('MODEL'))\n"},
    )
    extraction = extract_application(project)
    ops = _unresolved(extraction)
    assert len(ops) == 1
    assert ops[0].application_identity is ApplicationIdentityState.DYNAMIC
    assert ops[0].unbound_reason is UnboundReason.DYNAMIC_EXPRESSION
    assert "getenv" in ops[0].expression
    assert _dependencies(extraction) == ()
    assert "detected_model_id" not in ops[0].to_dict()


def test_get_model_call_is_dynamic(tmp_path: Path) -> None:
    project = make_project(tmp_path, {"bot.py": "client.create(model=get_model())\n"})
    extraction = extract_application(project)
    ops = _unresolved(extraction)
    assert len(ops) == 1
    assert ops[0].application_identity is ApplicationIdentityState.DYNAMIC
    assert ops[0].unbound_reason is UnboundReason.DYNAMIC_CALL
    assert ops[0].expression == "get_model()"


def test_config_subscript_is_dynamic(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {"bot.py": "client.create(model=config['model'])\n"},
    )
    extraction = extract_application(project)
    ops = _unresolved(extraction)
    assert len(ops) == 1
    assert ops[0].application_identity is ApplicationIdentityState.DYNAMIC
    assert ops[0].unbound_reason is UnboundReason.DYNAMIC_SUBSCRIPT
    assert "config" in ops[0].expression


def test_unresolved_operations_do_not_cause_catalog_lookups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _save_catalog(_model("MODEL", lifecycle=_known(LifecycleState.RETIRED)))
    seen: list[tuple[str, str]] = []
    original = CatalogView.resolve

    def spy(self: CatalogView, observation: Observation) -> object:
        seen.append((observation.kind.value, observation.location.detected_value))
        return original(self, observation)

    monkeypatch.setattr(CatalogView, "resolve", spy)
    project = make_project(
        tmp_path,
        {
            "bot.py": (
                "import os\n"
                "client.create(model=MODEL)\n"
                "client.create(model=os.getenv('MODEL'))\n"
                "client.create(model=get_model())\n"
                "client.create(model=config['model'])\n"
            )
        },
    )
    extraction = extract_application(project)
    assert seen == []
    assert _dependencies(extraction) == ()
    assert len(_unresolved(extraction)) == 4
    ids = [item.expression for item in _unresolved(extraction)]
    assert "MODEL" not in [dep.detected_model_id for dep in _dependencies(extraction)]
    assert any(item == "MODEL" for item in ids)


def test_no_fake_model_ids_for_unbound_slots(tmp_path: Path) -> None:
    project = make_project(tmp_path, {"bot.py": "client.create(model=MODEL)\n"})
    payload = extract_application(project).report.to_dict()
    dumped = json.dumps(payload)
    assert payload["inventory"]["dependencies"] == []
    op = payload["inventory"]["unresolved_operations"][0]
    assert "detected_model_id" not in op
    assert op["expression"] == "MODEL"
    assert '"detected_model_id": "MODEL"' not in dumped


def test_json_contains_inventory_not_observations(tmp_path: Path) -> None:
    project = make_project(tmp_path, {"bot.py": _plain_source("ghost")})
    extraction = extract_application(project)
    payload = json.loads(render_app_report(extraction.report, "json"))
    assert "inventory" in payload
    assert "dependencies" in payload["inventory"]
    assert "unresolved_operations" in payload["inventory"]
    assert "observations" not in payload
    assert "ai_operation" not in json.dumps(payload)
    assert payload == extraction.report.to_dict()
    internal = extraction.to_dict()
    assert "observations" in internal


def test_text_and_json_are_deterministic(tmp_path: Path) -> None:
    _save_catalog(_model("gpt-active", lifecycle=_known(LifecycleState.ACTIVE)))
    project = make_project(
        tmp_path,
        {
            "b.py": "client.create(model=MODEL)\n",
            "a.py": _openai_source("gpt-active"),
            "z.py": "client.create(model=get_model())\n",
        },
    )
    first = extract_application(project).report
    second = extract_application(project).report
    assert render_app_report(first, "text") == render_app_report(second, "text")
    assert render_app_report(first, "json") == render_app_report(second, "json")
    assert render_app_report(first, "json").endswith("\n")
    assert first.to_dict()["inventory"] == second.to_dict()["inventory"]


def test_catalog_absent_still_lists_bound_literals(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {
            "bot.py": _openai_source("gpt-ret"),
            "plain.py": _plain_source("ghost"),
        },
    )
    extraction = extract_application(project)
    assert extraction.report.catalog.present is False
    assert extraction.report.findings == ()
    deps = {item.detected_model_id: item for item in _dependencies(extraction)}
    assert deps["gpt-ret"].application_identity is ApplicationIdentityState.ESTABLISHED
    assert deps["gpt-ret"].provider_id == "openai"
    assert deps["gpt-ret"].catalog_evidence is None
    assert deps["ghost"].application_identity is ApplicationIdentityState.BOUND
    assert deps["ghost"].provider_id is None
    text = render_app_report(extraction.report, "text")
    assert "gpt-ret" in text
    assert "ghost" in text
    assert "No findings were emitted" in text


def test_empty_inventory_is_explicit(tmp_path: Path) -> None:
    project = make_project(tmp_path, {"bot.py": "value = 1\n"})
    text = render_app_report(extract_application(project).report, "text")
    assert "No bound model dependencies." in text
    assert "No unresolved operations." in text


def test_inventory_does_not_write_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _save_catalog(_model("gpt-active", lifecycle=_known(LifecycleState.ACTIVE)))

    def boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("catalog must stay read-only")

    monkeypatch.setattr(FileCatalogStore, "save", boom)
    project = make_project(
        tmp_path,
        {
            "bot.py": _openai_source("gpt-active") + "client.create(model=MODEL)\n",
        },
    )
    extraction = extract_application(project)
    assert _dependencies(extraction)
    assert _unresolved(extraction)


def test_json_config_literal_is_a_bound_dependency(tmp_path: Path) -> None:
    _save_catalog(_model("cfg-model", lifecycle=_known(LifecycleState.ACTIVE)))
    project = make_project(tmp_path, {"cfg.json": '{"llm": {"model": "cfg-model"}}\n'})
    extraction = extract_application(project)
    dep = _dependencies(extraction)[0]
    assert dep.application_identity is ApplicationIdentityState.CANDIDATE
    assert dep.operations[0].line_start is None
    assert dep.operations[0].path == "cfg.json"


def test_unknown_finding_still_established_in_inventory(tmp_path: Path) -> None:
    _save_catalog(_model("gpt-active", lifecycle=_known(LifecycleState.ACTIVE)))
    project = make_project(tmp_path, {"bot.py": _openai_source("missing-id")})
    extraction = extract_application(project)
    assert [item.code for item in extraction.report.findings] == [FINDING_MODEL_REFERENCE_UNKNOWN]
    dep = _dependencies(extraction)[0]
    assert dep.application_identity is ApplicationIdentityState.ESTABLISHED
    assert dep.detected_model_id == "missing-id"
    assert dep.samyak_id is None
    assert dep.match_kind is MatchKind.NOT_IN_CATALOG


def test_bound_identity_serializes_and_displays_as_resolved(tmp_path: Path) -> None:
    project = make_project(tmp_path, {"plain.py": _plain_source("ghost")})
    extraction = extract_application(project)
    dep = _dependencies(extraction)[0]
    assert dep.application_identity is ApplicationIdentityState.BOUND
    payload = dep.to_dict()
    assert payload["application_identity"] == "resolved"
    report_json = json.dumps(extraction.report.to_dict())
    assert '"application_identity": "bound"' not in report_json
    assert '"application_identity": "resolved"' in report_json
    text = render_app_report(extraction.report, "text")
    assert "Identity:     resolved" in text
    assert "Identity:     bound" not in text


def test_unresolved_operation_bound_state_serializes_as_resolved() -> None:
    operation = UnresolvedOperation(
        application_identity=ApplicationIdentityState.BOUND,
        unbound_reason=UnboundReason.NAME_NOT_BOUND,
        expression="MODEL",
        path="app.py",
        evidence_type=EvidenceType.PYTHON_KEYWORD,
    )
    assert operation.to_dict()["application_identity"] == "resolved"
    assert "operation_kind" not in operation.to_dict()


def test_ai_operation_kind_is_model_inference(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {
            "bot.py": (
                "client.create(model=MODEL)\n"
                "client.create(model=os.getenv('MODEL'))\n"
                "client.create(model=get_model())\n"
                "client.create(model=config['model'])\n"
            )
        },
    )
    extraction = extract_application(project)
    observations = [
        item for item in extraction.observations if item.kind is ObservationKind.AI_OPERATION
    ]
    assert observations
    assert all(item.operation_kind is AiOperationKind.MODEL_INFERENCE for item in observations)
    ops = _unresolved(extraction)
    assert len(ops) == 4
    assert all(item.operation_kind is AiOperationKind.MODEL_INFERENCE for item in ops)
    dumped = json.dumps(extraction.report.to_dict())
    assert "operation_kind" not in dumped
    assert "observations" not in extraction.report.to_dict()


def test_public_identities_other_than_bound_are_unchanged(tmp_path: Path) -> None:
    _save_catalog(_model("gpt-active", lifecycle=_known(LifecycleState.ACTIVE)))
    project = make_project(tmp_path, {"bot.py": _openai_source("gpt-active")})
    extraction = extract_application(project)
    dep = _dependencies(extraction)[0]
    assert dep.application_identity is ApplicationIdentityState.ESTABLISHED
    assert dep.to_dict()["application_identity"] == "established"
    text = render_app_report(extraction.report, "text")
    assert "Identity:     established" in text


def test_text_humanizes_name_not_bound(tmp_path: Path) -> None:
    project = make_project(tmp_path, {"bot.py": "client.create(model=MODEL)\n"})
    extraction = extract_application(project)
    text = render_app_report(extraction.report, "text")
    payload = extraction.report.to_dict()
    assert payload["inventory"]["unresolved_operations"][0]["unbound_reason"] == "name_not_bound"
    assert "model identity could not be resolved from this file" in text
    assert "name_not_bound" not in text


def test_text_humanizes_dynamic_expression(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {"bot.py": "import os\nclient.create(model=os.getenv('MODEL'))\n"},
    )
    extraction = extract_application(project)
    text = render_app_report(extraction.report, "text")
    payload = extraction.report.to_dict()
    assert (
        payload["inventory"]["unresolved_operations"][0]["unbound_reason"] == "dynamic_expression"
    )
    assert "model identity is determined at runtime" in text
    assert "dynamic_expression" not in text


def test_text_humanizes_dynamic_call(tmp_path: Path) -> None:
    project = make_project(tmp_path, {"bot.py": "client.create(model=get_model())\n"})
    extraction = extract_application(project)
    text = render_app_report(extraction.report, "text")
    payload = extraction.report.to_dict()
    assert payload["inventory"]["unresolved_operations"][0]["unbound_reason"] == "dynamic_call"
    assert "model identity comes from a runtime call" in text
    assert "dynamic_call" not in text


def test_text_humanizes_dynamic_subscript(tmp_path: Path) -> None:
    project = make_project(tmp_path, {"bot.py": "client.create(model=config['model'])\n"})
    extraction = extract_application(project)
    text = render_app_report(extraction.report, "text")
    payload = extraction.report.to_dict()
    assert payload["inventory"]["unresolved_operations"][0]["unbound_reason"] == "dynamic_subscript"
    assert "model identity comes from a runtime lookup" in text
    assert "dynamic_subscript" not in text
