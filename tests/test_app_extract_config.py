"""JSON and TOML extraction for Application Readiness."""

from __future__ import annotations

from pathlib import Path

from helpers import make_project
from samyak.app.extractors.json_config import extract_json_observations
from samyak.app.extractors.toml_config import extract_toml_observations
from samyak.app.models import NOTICE_FILE_PARSE_FAILED, EvidenceType
from samyak.app.observations import ObservationKind
from samyak.app.pipeline import extract_application


def test_json_nested_model_key_is_dotted() -> None:
    source = '{"llm": {"model": "nested-model"}}\n'
    refs = extract_json_observations("config.json", source)
    assert len(refs) == 1
    assert refs[0].kind is ObservationKind.MODEL_REFERENCE
    assert refs[0].location.config_key == "llm.model"
    assert refs[0].location.detected_value == "nested-model"
    assert refs[0].location.evidence_type is EvidenceType.JSON_VALUE


def test_json_file_level_location_has_no_invented_line() -> None:
    source = '{\n  "model": "pretty-model"\n}\n'
    refs = extract_json_observations("config.json", source)
    assert refs[0].location.path == "config.json"
    assert refs[0].location.line_start is None
    assert refs[0].location.line_end is None
    payload = refs[0].location.to_dict()
    assert payload["line_start"] is None
    assert payload["line_end"] is None


def test_json_model_id_and_model_name() -> None:
    source = '{"model_id": "id-1", "model_name": "name-1"}\n'
    refs = extract_json_observations("config.json", source)
    values = {item.location.config_key: item.location.detected_value for item in refs}
    assert values == {"model_id": "id-1", "model_name": "name-1"}


def test_malformed_json_produces_notice(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {
            "ok.py": "client.create(model='kept')\n",
            "bad.json": "{not json",
        },
    )
    extraction = extract_application(project)
    assert any(item.code == NOTICE_FILE_PARSE_FAILED for item in extraction.report.notices)
    assert any(item.path == "bad.json" for item in extraction.report.notices)
    values = [
        item.location.detected_value
        for item in extraction.observations
        if item.kind is ObservationKind.MODEL_REFERENCE
    ]
    assert "kept" in values


def test_toml_nested_model_key() -> None:
    source = b"[llm]\nmodel = 'toml-model'\n"
    refs = extract_toml_observations("config.toml", source)
    assert len(refs) == 1
    assert refs[0].location.config_key == "llm.model"
    assert refs[0].location.detected_value == "toml-model"
    assert refs[0].location.evidence_type is EvidenceType.TOML_VALUE
    assert refs[0].location.line_start is None
    assert refs[0].location.line_end is None


def test_malformed_toml_produces_notice(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {
            "ok.py": "client.create(model='kept')\n",
            "bad.toml": "model = [\n",
        },
    )
    extraction = extract_application(project)
    assert any(item.code == NOTICE_FILE_PARSE_FAILED for item in extraction.report.notices)
    assert any(item.path == "bad.toml" for item in extraction.report.notices)
    assert extraction.report.summary.parse_failures == 1
