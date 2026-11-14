"""Pipeline envelope and deterministic serialization."""

from __future__ import annotations

from pathlib import Path

from helpers import make_project
from samyak.app.models import CAPABILITY_NAME, PRODUCT_NAME, REPORT_SCHEMA_VERSION
from samyak.app.observations import ObservationKind
from samyak.app.pipeline import extract_application
from samyak.app.serialize import extraction_debug_json


def test_extraction_report_identity(tmp_path: Path) -> None:
    project = make_project(tmp_path, {"bot.py": "client.create(model='m')\n"})
    extraction = extract_application(project)
    report = extraction.report
    assert report.product == PRODUCT_NAME
    assert report.capability == CAPABILITY_NAME
    assert report.report_schema_version == REPORT_SCHEMA_VERSION
    assert report.findings == ()
    payload = report.to_dict()
    assert payload["catalog"] == {"present": False}
    assert payload["findings"] == []
    assert "inventory" in payload
    assert payload["inventory"]["dependencies"]
    assert payload["inventory"]["unresolved_operations"] == []
    assert any(item.code == "CATALOG_ABSENT" for item in report.notices)


def test_same_tree_serializes_identically(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {
            "b.py": "import openai\nclient.create(model='second')\n",
            "a.py": "client.create(model='first')\n",
            "cfg.json": '{"llm": {"model": "json-model"}}\n',
            "cfg.toml": "[llm]\nmodel = 'toml-model'\n",
        },
    )
    first = extraction_debug_json(extract_application(project))
    second = extraction_debug_json(extract_application(project))
    assert first == second
    assert first.endswith("\n")


def test_observation_order_is_path_then_line(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {
            "z.py": "client.create(model='z')\n",
            "a.py": "client.create(\n    model='a',\n)\n",
        },
    )
    extraction = extract_application(project)
    refs = [
        item for item in extraction.observations if item.kind is ObservationKind.MODEL_REFERENCE
    ]
    assert [item.location.path for item in refs] == ["a.py", "z.py"]
    assert refs[0].location.line_start == 2
