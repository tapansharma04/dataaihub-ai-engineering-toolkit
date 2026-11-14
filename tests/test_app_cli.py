"""CLI integration for Application Readiness (`samyak app check`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from helpers import make_project
from samyak.app.errors import ENGINE_EXIT_CODE, USAGE_EXIT_CODE
from samyak.app.models import (
    FINDING_MODEL_PROVIDER_AMBIGUOUS,
    FINDING_MODEL_REFERENCE_DEPRECATED,
    FINDING_MODEL_REFERENCE_RETIRED,
    FINDING_MODEL_REFERENCE_UNKNOWN,
    NOTICE_CATALOG_ABSENT,
)
from samyak.app.report import render_app_report
from samyak.cli import main
from samyak.model.facts import LifecycleState
from samyak.model.identity import ModelIdentity
from samyak.model.store import FileCatalogStore
from test_app_resolution import (
    _known,
    _model,
    _openai_source,
    _plain_source,
    _save_catalog,
)


def test_cli_valid_directory_text_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = make_project(tmp_path, {"bot.py": _openai_source("gpt-x")})
    code = main(["app", "check", str(project)])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.startswith("Samyak Application Analysis\n")
    assert "Capability:               app" in captured.out
    assert "Catalog" in captured.out
    assert "Present:                  no" in captured.out
    assert NOTICE_CATALOG_ABSENT in captured.out
    assert "samyak model update" in captured.out
    assert "Lifecycle rules were not applied" in captured.out
    assert "No findings were emitted" in captured.out
    assert captured.out.count("No findings.") == 0
    assert "Notices" in captured.out
    assert "Findings" in captured.out
    assert "Inventory" in captured.out
    assert "Models" in captured.out
    assert "gpt-x" in captured.out
    assert "readiness score" in captured.out.lower()
    assert "72%" not in captured.out
    assert "production ready" not in captured.out.lower()
    assert "observations" not in captured.out.lower()
    assert captured.err == ""


def test_cli_invalid_path_returns_usage_exit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["app", "check", str(tmp_path / "missing")])
    captured = capsys.readouterr()
    assert code == USAGE_EXIT_CODE
    assert "error:" in captured.err
    assert "does not exist" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_cli_file_path_returns_usage_exit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "file.py"
    target.write_text("x = 1\n", encoding="utf-8")
    code = main(["app", "check", str(target)])
    captured = capsys.readouterr()
    assert code == USAGE_EXIT_CODE
    assert "not a directory" in captured.err


def test_cli_json_is_app_report_without_observations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = make_project(tmp_path, {"bot.py": _openai_source("gpt-x")})
    code = main(["app", "check", str(project), "--format", "json"])
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["product"] == "samyak"
    assert payload["capability"] == "app"
    assert payload["catalog"] == {"present": False}
    assert "observations" not in payload
    assert any(item["code"] == NOTICE_CATALOG_ABSENT for item in payload["notices"])
    assert payload["findings"] == []
    assert "inventory" in payload
    assert payload["inventory"]["dependencies"]
    assert captured.out.endswith("\n")


def test_cli_output_writes_selected_representation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], samyak_cache_dir: Path
) -> None:
    project = make_project(tmp_path, {"bot.py": "client.create(model='x')\n"})
    output = tmp_path / "report.json"
    code = main(["app", "check", str(project), "--format", "json", "--output", str(output)])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == ""
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["capability"] == "app"
    assert "observations" not in payload
    assert output.read_text(encoding="utf-8").endswith("\n")
    assert not (samyak_cache_dir / "runs").exists()


def test_cli_findings_still_exit_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
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
    code = main(["app", "check", str(project), "--format", "json"])
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    codes = [item["code"] for item in payload["findings"]]
    assert FINDING_MODEL_REFERENCE_DEPRECATED in codes
    assert FINDING_MODEL_REFERENCE_RETIRED in codes
    deprecated = next(
        item for item in payload["findings"] if item["code"] == FINDING_MODEL_REFERENCE_DEPRECATED
    )
    assert deprecated["recommendation"] == "Replace this reference with openai:gpt-new."
    retired = next(
        item for item in payload["findings"] if item["code"] == FINDING_MODEL_REFERENCE_RETIRED
    )
    assert "does not establish a successor" in retired["recommendation"]


def test_cli_text_includes_evidence_fields(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _save_catalog(_model("gpt-ret", lifecycle=_known(LifecycleState.RETIRED)))
    project = make_project(tmp_path, {"bot.py": _openai_source("gpt-ret")})
    code = main(["app", "check", str(project)])
    captured = capsys.readouterr()
    assert code == 0
    text = captured.out
    assert "HIGH  Retired model reference" in text
    assert "Code: MODEL_REFERENCE_RETIRED" in text
    assert "Source: bot.py:" in text
    assert "Catalog: openai:gpt-ret" in text
    assert "Why it matters:" in text
    assert "Recommendation:" in text
    assert "Present:                  yes" in text


def test_cli_unknown_and_ambiguous(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _save_catalog(
        _model("gpt-active", lifecycle=_known(LifecycleState.ACTIVE)),
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
    project = make_project(
        tmp_path,
        {
            "known.py": _openai_source("missing-id"),
            "cfg.json": '{"model": "shared-id"}\n',
        },
    )
    code = main(["app", "check", str(project), "--format", "json"])
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    codes = [item["code"] for item in payload["findings"]]
    assert FINDING_MODEL_REFERENCE_UNKNOWN in codes
    assert FINDING_MODEL_PROVIDER_AMBIGUOUS in codes
    unknown = next(
        item for item in payload["findings"] if item["code"] == FINDING_MODEL_REFERENCE_UNKNOWN
    )
    assert unknown["severity"] == "INFO"
    assert "samyak model update" in unknown["recommendation"]
    assert "This inspection does not refresh" in unknown["recommendation"]
    ambiguous = next(
        item for item in payload["findings"] if item["code"] == FINDING_MODEL_PROVIDER_AMBIGUOUS
    )
    assert ambiguous["evidence"]["line_start"] is None
    assert "observations" not in payload


def test_cli_output_is_deterministic(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _save_catalog(_model("gpt-dep", lifecycle=_known(LifecycleState.DEPRECATED)))
    project = make_project(tmp_path, {"bot.py": _openai_source("gpt-dep")})
    first = main(["app", "check", str(project), "--format", "text"])
    first_out = capsys.readouterr().out
    second = main(["app", "check", str(project), "--format", "text"])
    second_out = capsys.readouterr().out
    assert first == 0
    assert second == 0
    assert first_out == second_out
    json_code = main(["app", "check", str(project), "--format", "json"])
    json_first = capsys.readouterr().out
    json_code_2 = main(["app", "check", str(project), "--format", "json"])
    json_second = capsys.readouterr().out
    assert json_code == 0
    assert json_code_2 == 0
    assert json_first == json_second


def test_cli_does_not_write_or_refresh_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _save_catalog(_model("gpt-dep", lifecycle=_known(LifecycleState.DEPRECATED)))

    def boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("catalog must stay read-only")

    monkeypatch.setattr(FileCatalogStore, "save", boom)
    monkeypatch.setattr("samyak.model.update.update_openai_catalog", boom)
    monkeypatch.setattr("samyak.model.update.update_all_catalogs", boom)
    project = make_project(tmp_path, {"bot.py": _openai_source("gpt-dep")})
    code = main(["app", "check", str(project), "--format", "json"])
    capsys.readouterr()
    assert code == 0


def test_cli_engine_failure_returns_operational_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = make_project(tmp_path, {"bot.py": "x = 1\n"})

    def boom(_path: object, config: object = None) -> object:
        raise RuntimeError("engine exploded")

    monkeypatch.setattr("samyak.cli.extract_application", boom)
    code = main(["app", "check", str(project)])
    captured = capsys.readouterr()
    assert code == ENGINE_EXIT_CODE
    assert ENGINE_EXIT_CODE not in {0, USAGE_EXIT_CODE}
    assert "unexpected analysis failure" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_cli_app_without_check_returns_usage(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["app"])
    captured = capsys.readouterr()
    assert code == USAGE_EXIT_CODE
    assert "check" in captured.out.lower()


def test_cli_invalid_format_is_usage_error() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["app", "check", ".", "--format", "html"])
    assert exc.value.code == USAGE_EXIT_CODE


def test_render_json_matches_app_report_envelope(tmp_path: Path) -> None:
    project = make_project(tmp_path, {"bot.py": _plain_source("ghost")})
    from samyak.app.pipeline import extract_application

    extraction = extract_application(project)
    rendered = render_app_report(extraction.report, "json")
    payload = json.loads(rendered)
    assert payload == extraction.report.to_dict()
    assert "observations" not in payload
    assert extraction.to_dict()["observations"]
    text = render_app_report(extraction.report, "text")
    assert "Lifecycle rules were not applied" in text
    assert NOTICE_CATALOG_ABSENT in text
    assert "samyak model update" in text
    assert "maturity" not in text.lower()
    assert "checklist" not in text.lower()


def test_catalog_absent_does_not_imply_clean_lifecycle(tmp_path: Path) -> None:
    from samyak.app.pipeline import extract_application

    project = make_project(tmp_path, {"bot.py": _openai_source("gpt-ret")})
    text = render_app_report(extract_application(project).report, "text")
    assert "No findings were emitted" in text
    assert "local catalog was not available" in text
    assert "samyak model update" in text
    assert "MODEL_REFERENCE_RETIRED" not in text


def test_unknown_text_does_not_present_detected_id_as_catalog_identity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _save_catalog(_model("gpt-active", lifecycle=_known(LifecycleState.ACTIVE)))
    project = make_project(tmp_path, {"bot.py": _openai_source("missing-id")})
    code = main(["app", "check", str(project)])
    text = capsys.readouterr().out
    assert code == 0
    assert "Catalog: (not an established catalog identity)" in text
    assert "Catalog: missing-id" not in text
    assert "match: not_in_catalog" in text
    assert "samyak model update" in text
