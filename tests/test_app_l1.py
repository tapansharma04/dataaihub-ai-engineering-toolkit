"""Level 1 same-file constant resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from helpers import make_project
from samyak.app.catalog_view import CatalogView
from samyak.app.models import (
    FINDING_MODEL_REFERENCE_RETIRED,
    ApplicationIdentityState,
    ResolutionMethod,
    UnboundReason,
)
from samyak.app.observations import Observation
from samyak.app.pipeline import extract_application
from samyak.app.report import render_app_report
from samyak.model.facts import LifecycleState
from samyak.model.store import FileCatalogStore
from test_app_resolution import (
    _known,
    _model,
    _openai_source,
    _save_catalog,
)


def _deps(extraction):
    return extraction.report.inventory.dependencies


def _unresolved(extraction):
    return extraction.report.inventory.unresolved_operations


def test_same_file_constant_resolves(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {"app.py": 'MODEL = "gpt-5"\nclient.responses.create(model=MODEL)\n'},
    )
    extraction = extract_application(project)
    deps = _deps(extraction)
    assert len(deps) == 1
    dep = deps[0]
    assert dep.detected_model_id == "gpt-5"
    assert _unresolved(extraction) == ()
    op = dep.operations[0]
    assert op.expression == "MODEL"
    assert op.resolution_method is ResolutionMethod.SAME_FILE_CONSTANT
    assert op.line_start == 2
    assert op.detected_model_id == "gpt-5"
    assert op.resolution_source is not None
    assert op.resolution_source.line_start == 1
    assert op.resolution_source.symbol == "MODEL"
    assert op.resolution_source.detected_value == "gpt-5"
    text = render_app_report(extraction.report, "text")
    assert "model=MODEL" in text
    assert "MODEL='gpt-5'" in text or 'MODEL="gpt-5"' in text
    assert "app.py:2" in text
    assert "app.py:1" in text


def test_annotated_constant_resolves(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {"app.py": 'MODEL: str = "gpt-5"\nclient.create(model=MODEL)\n'},
    )
    extraction = extract_application(project)
    dep = _deps(extraction)[0]
    assert dep.detected_model_id == "gpt-5"
    assert dep.operations[0].resolution_method is ResolutionMethod.SAME_FILE_CONSTANT
    assert _unresolved(extraction) == ()


def test_multiple_uses_group_to_one_dependency(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {"app.py": ('MODEL = "gpt-5"\nclient.a(model=MODEL)\nclient.b(model=MODEL)\n')},
    )
    extraction = extract_application(project)
    deps = _deps(extraction)
    assert len(deps) == 1
    assert deps[0].detected_model_id == "gpt-5"
    assert [item.line_start for item in deps[0].operations] == [2, 3]
    assert all(
        item.resolution_method is ResolutionMethod.SAME_FILE_CONSTANT for item in deps[0].operations
    )


def test_two_different_constants(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {
            "app.py": (
                'MODEL_A = "gpt-5"\n'
                'MODEL_B = "claude-3"\n'
                "client.a(model=MODEL_A)\n"
                "client.b(model=MODEL_B)\n"
            )
        },
    )
    extraction = extract_application(project)
    ids = {item.detected_model_id for item in _deps(extraction)}
    assert ids == {"gpt-5", "claude-3"}
    assert _unresolved(extraction) == ()


def test_reassignment_remains_unresolved(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {"app.py": ('MODEL = "gpt-4"\nMODEL = "gpt-5"\nclient.create(model=MODEL)\n')},
    )
    extraction = extract_application(project)
    assert _deps(extraction) == ()
    ops = _unresolved(extraction)
    assert len(ops) == 1
    assert ops[0].application_identity is ApplicationIdentityState.UNRESOLVED
    assert ops[0].unbound_reason is UnboundReason.NAME_NOT_BOUND


def test_dynamic_assignment_remains_unresolved(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {"app.py": ("import os\nMODEL = os.getenv('MODEL')\nclient.create(model=MODEL)\n")},
    )
    extraction = extract_application(project)
    assert _deps(extraction) == ()
    ops = _unresolved(extraction)
    assert len(ops) == 1
    assert ops[0].application_identity is ApplicationIdentityState.UNRESOLVED
    assert ops[0].unbound_reason is UnboundReason.NAME_NOT_BOUND
    assert ops[0].expression == "MODEL"


def test_function_call_assignment_remains_unresolved(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {"app.py": "MODEL = get_model()\nclient.create(model=MODEL)\n"},
    )
    extraction = extract_application(project)
    ops = _unresolved(extraction)
    assert len(ops) == 1
    assert ops[0].unbound_reason is UnboundReason.NAME_NOT_BOUND
    assert _deps(extraction) == ()


def test_attribute_reference_remains_unresolved(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {"app.py": ('class Config:\n    MODEL = "gpt-5"\nclient.create(model=settings.MODEL)\n')},
    )
    extraction = extract_application(project)
    ops = _unresolved(extraction)
    assert len(ops) == 1
    assert ops[0].unbound_reason is UnboundReason.UNRESOLVED_ATTRIBUTE
    assert _deps(extraction) == ()


def test_subscript_reference_remains_unresolved(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {"app.py": 'MODEL = "gpt-5"\nclient.create(model=config["model"])\n'},
    )
    extraction = extract_application(project)
    ops = _unresolved(extraction)
    assert len(ops) == 1
    assert ops[0].unbound_reason is UnboundReason.DYNAMIC_SUBSCRIPT
    assert _deps(extraction) == ()


def test_function_local_constant_is_not_resolved(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {"app.py": ('def make_client():\n    MODEL = "gpt-5"\n    client.create(model=MODEL)\n')},
    )
    extraction = extract_application(project)
    assert _deps(extraction) == ()
    ops = _unresolved(extraction)
    assert len(ops) == 1
    assert ops[0].unbound_reason is UnboundReason.NAME_NOT_BOUND


def test_cross_file_constant_is_not_resolved(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {
            "config.py": 'MODEL = "gpt-5"\n',
            "app.py": "from config import MODEL\nclient.create(model=MODEL)\n",
        },
    )
    extraction = extract_application(project)
    assert [item.detected_model_id for item in _deps(extraction)] == []
    ops = [item for item in _unresolved(extraction) if item.path == "app.py"]
    assert len(ops) == 1
    assert ops[0].unbound_reason is UnboundReason.NAME_NOT_BOUND


def test_provider_evidence_applies_to_resolved_constant(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {"app.py": ("import openai\nMODEL = 'gpt-5'\nclient.create(model=MODEL)\n")},
    )
    extraction = extract_application(project)
    dep = _deps(extraction)[0]
    assert dep.application_identity is ApplicationIdentityState.ESTABLISHED
    assert dep.provider_id == "openai"
    assert dep.detected_model_id == "gpt-5"


def test_catalog_lookup_after_successful_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _save_catalog(_model("gpt-5", lifecycle=_known(LifecycleState.ACTIVE)))
    seen: list[tuple[str, str]] = []
    original = CatalogView.resolve

    def spy(self: CatalogView, observation: Observation) -> object:
        seen.append((observation.kind.value, observation.location.detected_value))
        return original(self, observation)

    monkeypatch.setattr(CatalogView, "resolve", spy)
    project = make_project(
        tmp_path,
        {"app.py": 'MODEL = "gpt-5"\nclient.create(model=MODEL)\n'},
    )
    extract_application(project)
    assert seen
    assert all(item == ("model_reference", "gpt-5") for item in seen)
    assert ("ai_operation", "") not in seen
    assert ("model_reference", "MODEL") not in seen


def test_lifecycle_finding_matches_direct_literal(tmp_path: Path) -> None:
    _save_catalog(_model("gpt-ret", lifecycle=_known(LifecycleState.RETIRED)))
    lit_dir = tmp_path / "lit"
    const_dir = tmp_path / "const"
    lit_dir.mkdir()
    const_dir.mkdir()
    literal = make_project(lit_dir, {"bot.py": _openai_source("gpt-ret")})
    constant = make_project(
        const_dir,
        {"bot.py": "import openai\nMODEL = 'gpt-ret'\nclient.create(model=MODEL)\n"},
    )
    literal_report = extract_application(literal).report
    constant_report = extract_application(constant).report
    assert [item.code for item in literal_report.findings] == [FINDING_MODEL_REFERENCE_RETIRED]
    assert [item.code for item in constant_report.findings] == [FINDING_MODEL_REFERENCE_RETIRED]
    lit = literal_report.findings[0]
    con = constant_report.findings[0]
    assert lit.severity is con.severity
    assert lit.catalog_evidence is not None
    assert con.catalog_evidence is not None
    assert lit.catalog_evidence.samyak_id == con.catalog_evidence.samyak_id
    assert con.reference_expression == "MODEL"
    assert con.resolution_source is not None
    assert con.resolution_source.line_start == 2
    assert con.evidence.line_start == 3
    assert con.evidence.detected_value == "gpt-ret"
    text = render_app_report(constant_report, "text")
    assert "model=MODEL" in text
    assert "Resolution:" in text
    assert "model='gpt-ret'" not in text.split("Resolution:")[0]


def test_json_includes_resolution_evidence_not_observations(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {"app.py": 'MODEL = "gpt-5"\nclient.create(model=MODEL)\n'},
    )
    payload = json.loads(render_app_report(extract_application(project).report, "json"))
    assert "observations" not in payload
    op = payload["inventory"]["dependencies"][0]["operations"][0]
    assert op["detected_model_id"] == "gpt-5"
    assert op["expression"] == "MODEL"
    assert op["resolution_method"] == "same_file_constant"
    assert op["resolution_source"]["path"] == "app.py"
    assert op["resolution_source"]["detected_value"] == "gpt-5"
    assert op["resolution_source"]["symbol"] == "MODEL"
    assert op["line_start"] != op["resolution_source"]["line_start"]


def test_l1_output_is_deterministic(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {
            "b.py": 'B = "claude-3"\nclient.create(model=B)\n',
            "a.py": 'A = "gpt-5"\nclient.create(model=A)\nclient.other(model=A)\n',
        },
    )
    first = extract_application(project).report
    second = extract_application(project).report
    assert render_app_report(first, "json") == render_app_report(second, "json")
    assert render_app_report(first, "text") == render_app_report(second, "text")


def test_l1_does_not_write_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _save_catalog(_model("gpt-5", lifecycle=_known(LifecycleState.ACTIVE)))

    def boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("catalog must stay read-only")

    monkeypatch.setattr(FileCatalogStore, "save", boom)
    project = make_project(
        tmp_path,
        {"app.py": 'MODEL = "gpt-5"\nclient.create(model=MODEL)\n'},
    )
    extraction = extract_application(project)
    assert _deps(extraction)[0].detected_model_id == "gpt-5"


def test_fstring_and_concat_assignments_remain_unresolved(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {
            "a.py": 'version = "5"\nMODEL = f"gpt-{version}"\nclient.create(model=MODEL)\n',
            "b.py": 'MODEL = "gpt-" + version\nclient.create(model=MODEL)\n',
        },
    )
    extraction = extract_application(project)
    assert _deps(extraction) == ()
    reasons = [item.unbound_reason for item in _unresolved(extraction)]
    assert reasons
    assert all(item is UnboundReason.NAME_NOT_BOUND for item in reasons)


def test_walrus_and_unpacking_remain_unresolved(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {
            "walrus.py": 'if (MODEL := "gpt-5"):\n    client.create(model=MODEL)\n',
            "unpack.py": 'MODEL, OTHER = "gpt-5", "x"\nclient.create(model=MODEL)\n',
        },
    )
    extraction = extract_application(project)
    assert _deps(extraction) == ()
    assert _unresolved(extraction)


def test_literal_json_shape_unchanged_without_resolution_fields(tmp_path: Path) -> None:
    project = make_project(tmp_path, {"app.py": 'client.create(model="gpt-5")\n'})
    op = extract_application(project).report.to_dict()["inventory"]["dependencies"][0][
        "operations"
    ][0]
    assert "resolution_method" not in op
    assert "resolution_source" not in op
    assert "expression" not in op
    assert op["detected_model_id"] == "gpt-5"
