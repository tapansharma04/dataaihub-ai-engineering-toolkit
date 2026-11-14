"""Text presentation of the application inventory (not JSON schema)."""

from __future__ import annotations

from pathlib import Path

from helpers import make_project
from samyak.app.pipeline import extract_application
from samyak.app.report import render_app_report
from samyak.model.facts import LifecycleState
from test_app_resolution import _known, _model, _openai_source, _plain_source, _save_catalog


def _text(project: Path) -> str:
    return render_app_report(extract_application(project).report, "text")


def test_empty_inventory_includes_zero_summary(tmp_path: Path) -> None:
    text = _text(make_project(tmp_path, {"bot.py": "value = 1\n"}))
    assert "Application Inventory" in text
    assert "Model dependencies:     0" in text
    assert "Established:          0" in text
    assert "Candidate:            0" in text
    assert "Resolved:             0" in text
    assert "Unresolved operations:  0" in text
    assert "No bound model dependencies." in text
    assert "No unresolved operations." in text
    assert " × " not in text


def test_only_established_dependencies_in_summary(tmp_path: Path) -> None:
    _save_catalog(_model("gpt-active", lifecycle=_known(LifecycleState.ACTIVE)))
    text = _text(make_project(tmp_path, {"bot.py": _openai_source("gpt-active")}))
    assert "Model dependencies:     1" in text
    assert "Established:          1" in text
    assert "Candidate:            0" in text
    assert "Resolved:             0" in text
    assert "Unresolved operations:  0" in text
    assert "Identity:     established" in text
    assert "Unresolved operations" in text
    assert "  (none)" in text


def test_mixed_identity_counts_in_summary(tmp_path: Path) -> None:
    _save_catalog(_model("gpt-active", lifecycle=_known(LifecycleState.ACTIVE)))
    project = make_project(
        tmp_path,
        {
            "bot.py": _openai_source("gpt-active"),
            "cfg.json": '{"model": "gpt-active"}\n',
            "plain.py": _plain_source("ghost"),
        },
    )
    text = _text(project)
    assert "Model dependencies:     3" in text
    assert "Established:          1" in text
    assert "Candidate:            1" in text
    assert "Resolved:             1" in text
    payload = extract_application(project).report.to_dict()["inventory"]["dependencies"]
    assert {item["application_identity"] for item in payload} == {
        "established",
        "candidate",
        "resolved",
    }
    assert {item["detected_model_id"] for item in payload} == {"gpt-active", "ghost"}
    assert sum(1 for item in payload if item["detected_model_id"] == "gpt-active") == 2


def test_unresolved_reasons_are_counted_and_grouped(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {
            "a.py": "client.create(model=MODEL)\n",
            "b.py": "import os\nclient.create(model=os.getenv('MODEL'))\n",
            "c.py": "client.create(model=get_model())\n",
            "d.py": "client.create(model=config['model'])\n",
        },
    )
    extraction = extract_application(project)
    text = render_app_report(extraction.report, "text")
    assert "Unresolved operations:  4" in text
    assert "1 × model identity could not be resolved from this file" in text
    assert "1 × model identity is determined at runtime" in text
    assert "1 × model identity comes from a runtime call" in text
    assert "1 × model identity comes from a runtime lookup" in text
    unresolved_heading = text.index("Unresolved operations:  4")
    grouped = text.index("Unresolved operations\n")
    assert unresolved_heading < grouped
    # Enum order: name_not_bound, then dynamic_expression, call, subscript.
    phrase_not_bound = text.index("model identity could not be resolved from this file")
    phrase_runtime = text.index("model identity is determined at runtime")
    phrase_call = text.index("model identity comes from a runtime call")
    phrase_lookup = text.index("model identity comes from a runtime lookup")
    assert phrase_not_bound < phrase_runtime < phrase_call < phrase_lookup
    operations = extraction.report.to_dict()["inventory"]["unresolved_operations"]
    assert len(operations) == 4
    assert {item["unbound_reason"] for item in operations} == {
        "name_not_bound",
        "dynamic_expression",
        "dynamic_call",
        "dynamic_subscript",
    }


def test_repeated_unresolved_operations_are_aggregated(tmp_path: Path) -> None:
    files = {f"n{index:02d}.py": "client.create(model=MODEL)\n" for index in range(8)}
    files.update(
        {f"s{index:02d}.py": "client.create(model=self.model_name)\n" for index in range(5)}
    )
    files.update({f"g{index:02d}.py": "client.create(model=get_model())\n" for index in range(3)})
    project = make_project(tmp_path, files)
    extraction = extract_application(project)
    text = render_app_report(extraction.report, "text")
    assert "Unresolved operations:  16" in text
    assert "13 × model identity could not be resolved from this file" in text
    assert "3 × model identity comes from a runtime call" in text
    assert "common: model=MODEL (8), model=self.model_name (5)" in text
    assert "common: model=get_model() (3)" in text
    assert "n00.py:" not in text
    assert "s00.py:" not in text
    operations = extraction.report.to_dict()["inventory"]["unresolved_operations"]
    assert len(operations) == 16
    for item in operations:
        assert item["path"]
        assert item["expression"]
        assert item["application_identity"]
        assert item["unbound_reason"]
        assert "line_start" in item
    assert {item["path"] for item in operations if item["path"].startswith("n")} == {
        f"n{index:02d}.py" for index in range(8)
    }


def test_cookbook_like_noisy_output_stays_compact(tmp_path: Path) -> None:
    files: dict[str, str] = {}
    for index in range(20):
        files[f"synth_{index:02d}.py"] = "client.create(model=self.model_name)\n"
    for index in range(9):
        files[f"agent_{index:02d}.py"] = "client.create(model=model)\n"
    for index in range(5):
        files[f"env_{index:02d}.py"] = "import os\nclient.create(model=os.getenv('MODEL'))\n"
    for index in range(3):
        files[f"call_{index:02d}.py"] = "client.create(model=get_model())\n"
    for index in range(2):
        files[f"cfg_{index:02d}.py"] = "client.create(model=config['model'])\n"
    project = make_project(tmp_path, files)
    extraction = extract_application(project)
    text = render_app_report(extraction.report, "text")
    operations = extraction.report.to_dict()["inventory"]["unresolved_operations"]
    assert len(operations) == 39
    assert {item["unbound_reason"] for item in operations} == {
        "name_not_bound",
        "unresolved_attribute",
        "dynamic_expression",
        "dynamic_call",
        "dynamic_subscript",
    }
    assert len({item["path"] for item in operations}) == 39
    assert "Unresolved operations:  39" in text
    assert "29 × model identity could not be resolved from this file" in text
    assert "5 × model identity is determined at runtime" in text
    assert "3 × model identity comes from a runtime call" in text
    assert "2 × model identity comes from a runtime lookup" in text
    assert text.count("synth_00.py") == 0
    assert "common: model=self.model_name (20), model=model (9)" in text
    first = render_app_report(extraction.report, "text")
    second = render_app_report(extract_application(project).report, "text")
    assert first == second


def test_inventory_summary_is_deterministic(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {
            "z.py": "client.create(model=get_model())\n",
            "a.py": "client.create(model=MODEL)\n",
            "m.py": "client.create(model=MODEL)\n",
        },
    )
    first = _text(project)
    second = _text(project)
    assert first == second
    not_bound = first.index("2 × model identity could not be resolved from this file")
    runtime_call = first.index("1 × model identity comes from a runtime call")
    assert not_bound < runtime_call
