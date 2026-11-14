"""CLI coverage for the read-only evidence-aware model query command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from samyak.cli import main
from samyak.model.catalog import CatalogFreshness, CatalogOverlay, FreshnessStatus, build_catalog
from samyak.model.facts import (
    Confidence,
    ContextWindow,
    ContextWindowKind,
    Fact,
    FactClaim,
    FactStatus,
    LifecycleState,
    Modality,
    Provenance,
    SourceKind,
)
from samyak.model.identity import ModelIdentity
from samyak.model.records import EvidenceDisposition, ModelRecord
from samyak.model.store import FileCatalogStore

STAMP = "2026-11-01T00:00:00+00:00"


def _provenance(provider: str = "fireworks", source: str = "fireworks-docs-test") -> Provenance:
    return Provenance(
        provider=provider,
        source_id=source,
        source_kind=SourceKind.PROVIDER_DOCS,
        content_hash="sha256:test",
        retrieved_at=STAMP,
        verified_at=STAMP,
        confidence=Confidence.HIGH,
    )


def _known(value: object, provider: str = "fireworks") -> Fact:
    return Fact(status=FactStatus.KNOWN, value=value, provenance=_provenance(provider))


def _model(
    name: str,
    *,
    provider: str = "fireworks",
    disposition: EvidenceDisposition | None = EvidenceDisposition.CURRENT_DOCUMENTED,
    **changes: object,
) -> ModelRecord:
    fields: dict[str, object] = {
        "display_name": _known(name.title(), provider),
        "lifecycle": _known(LifecycleState.ACTIVE, provider),
        "api_access": _known(True, provider),
        "tool_calling": _known(True, provider),
        "structured_output": _known(True, provider),
        "input_modalities": _known((Modality.TEXT, Modality.IMAGE), provider),
        "output_modalities": _known((Modality.TEXT,), provider),
        "context_window": _known(ContextWindow(128_000, ContextWindowKind.INPUT), provider),
    }
    fields.update(changes)
    return ModelRecord(
        identity=ModelIdentity(provider_id=provider, provider_model_id=name),
        evidence_disposition=disposition,
        **fields,  # type: ignore[arg-type]
    )


def _install_catalog(monkeypatch, tmp_path: Path, models: tuple[ModelRecord, ...]) -> Path:
    root = tmp_path / "cache"
    monkeypatch.setenv("SAMYAK_CACHE_DIR", str(root))
    catalog = build_catalog(
        models=models,
        generated_at=STAMP,
        freshness=CatalogFreshness(
            status=FreshnessStatus.AS_OF,
            generated_at=STAMP,
            overlay=CatalogOverlay.USER_CACHE,
        ),
    )
    FileCatalogStore(root=root).save(catalog)
    return root


def test_registration_and_help(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["model", "query", "--help"])
    assert exc_info.value.code == 0
    assert "--include-legacy" in capsys.readouterr().out


def test_explain_json_is_additive_deterministic_and_preserves_provenance(
    monkeypatch, tmp_path, capsys
) -> None:
    _install_catalog(
        monkeypatch,
        tmp_path,
        (
            _model("match"),
            _model(
                "missing", tool_calling=Fact(status=FactStatus.UNKNOWN, provenance=_provenance())
            ),
        ),
    )
    command = ["model", "query", "--tool-calling", "true", "--format", "json"]
    assert main(command) == 0
    default = json.loads(capsys.readouterr().out)
    assert main([*command, "--explain"]) == 0
    explained_text = capsys.readouterr().out
    explained = json.loads(explained_text)
    assert "explanations" not in default
    assert {key: value for key, value in explained.items() if key != "explanations"} == default
    assert explained["explanations"]["fireworks:match"][-1] == {
        "predicate": "tool_calling",
        "outcome": "matched",
        "reason_code": "matched_verified",
        "reason": "matched fact is supported by verified evidence",
        "source_ids": ["fireworks-docs-test"],
        "verified_at": [STAMP],
    }
    assert (
        explained["explanations"]["fireworks:missing"][0]["reason_code"]
        == "missing_verified_evidence"
    )
    assert main([*command, "--explain"]) == 0
    assert capsys.readouterr().out == explained_text


def test_explain_text_samples_only_ten_excluded_records_in_stable_order(
    monkeypatch, tmp_path, capsys
) -> None:
    models = tuple(
        _model(f"excluded-{index:02d}", provider="openai", tool_calling=_known(False, "openai"))
        for index in range(12)
    ) + (_model("match", provider="openai"),)
    _install_catalog(monkeypatch, tmp_path, models)
    assert (
        main(["model", "query", "--provider", "fireworks", "--tool-calling", "true", "--explain"])
        == 0
    )
    output = capsys.readouterr().out
    assert "provider_mismatch: 13" in output
    assert "tool_calling_mismatch: 12" in output
    assert "openai:excluded-00:" in output
    assert "openai:excluded-09:" in output
    assert "openai:excluded-10:" not in output
    assert "… 3 further excluded records omitted" in output


def test_default_legacy_and_explicit_documentation_policies(monkeypatch, tmp_path, capsys) -> None:
    _install_catalog(
        monkeypatch,
        tmp_path,
        (
            _model("current"),
            _model("legacy", disposition=None),
            _model("historical", disposition=EvidenceDisposition.HISTORICAL_REFERENCE_ONLY),
            _model("not-observed", disposition=EvidenceDisposition.NOT_OBSERVED_IN_CURRENT_CAPTURE),
        ),
    )
    assert main(["model", "query", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["summary"]["matching_records"] == 1
    assert main(["model", "query", "--include-legacy", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["summary"]["matching_records"] == 2
    assert (
        main(["model", "query", "--documentation", "historical_reference_only", "--format", "json"])
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["matching_records"] == 2
    assert "current_documented" in payload["documentation_eligibility"]


def test_predicates_repetition_and_json_provenance(monkeypatch, tmp_path, capsys) -> None:
    _install_catalog(monkeypatch, tmp_path, (_model("one"), _model("two", provider="openai")))
    assert (
        main(
            [
                "model",
                "query",
                "--provider",
                "fireworks",
                "--provider",
                "openai",
                "--tool-calling",
                "true",
                "--input-modality",
                "text",
                "--input-modality",
                "image",
                "--format",
                "json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert [item["samyak_id"] for item in payload["matches"]] == ["fireworks:one", "openai:two"]
    fact = next(
        item for item in payload["matches"][0]["matched_facts"] if item["name"] == "tool_calling"
    )
    assert fact["source_ids"] == ["fireworks-docs-test"]
    assert fact["verified_at"] == [STAMP]


def test_false_unknown_unverified_and_conflict_boolean_handling(
    monkeypatch, tmp_path, capsys
) -> None:
    conflict = Fact(
        status=FactStatus.CONFLICT,
        claims=(
            FactClaim(True, _provenance()),
            FactClaim(False, _provenance(source="other")),
        ),
    )
    _install_catalog(
        monkeypatch,
        tmp_path,
        (
            _model("false", tool_calling=_known(False)),
            _model(
                "unknown", tool_calling=Fact(status=FactStatus.UNKNOWN, provenance=_provenance())
            ),
            _model("unverified", tool_calling=Fact(status=FactStatus.NOT_VERIFIED)),
            _model("conflict", tool_calling=conflict),
        ),
    )
    assert main(["model", "query", "--tool-calling", "false", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [item["samyak_id"] for item in payload["matches"]] == ["fireworks:false"]
    assert payload["summary"]["exclusion_reason_counts"]["tool_calling_not_verified"] == 3


def test_lifecycle_context_text_and_errors(monkeypatch, tmp_path, capsys) -> None:
    _install_catalog(monkeypatch, tmp_path, (_model("one"),))
    assert (
        main(
            [
                "model",
                "query",
                "--lifecycle",
                "active",
                "--min-context",
                "100000",
                "--context-kind",
                "input",
            ]
        )
        == 0
    )
    assert "Matches: 1" in capsys.readouterr().out
    assert main(["model", "query", "--min-context", "100000"]) == 2
    assert "requires --context-kind" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc_info:
        main(["model", "query", "--min-context", "0", "--context-kind", "input"])
    assert exc_info.value.code == 2
    assert "positive integer" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc_info:
        main(["model", "query", "--tool-calling", "maybe"])
    assert exc_info.value.code == 2
    assert "true or false" in capsys.readouterr().err


def test_missing_corrupt_and_pending_catalog_fail_without_recovery(
    monkeypatch, tmp_path, capsys
) -> None:
    root = tmp_path / "cache"
    monkeypatch.setenv("SAMYAK_CACHE_DIR", str(root))
    assert main(["model", "query"]) == 2
    assert "no local model catalog" in capsys.readouterr().err
    models_dir = root / "models"
    models_dir.mkdir(parents=True)
    (models_dir / "catalog.json").write_text("not json")
    assert main(["model", "query"]) == 2
    assert "corrupt" in capsys.readouterr().err
    (models_dir / "catalog.json").write_text(json.dumps(_catalog_payload()))
    (models_dir / ".catalog-history-transaction.json").write_text("pending")
    assert main(["model", "query"]) == 2
    assert "transaction is pending" in capsys.readouterr().err
    assert (models_dir / ".catalog-history-transaction.json").is_file()


def _catalog_payload() -> dict[str, object]:
    """A valid payload is only needed after the pending-artifact check."""
    from samyak.model.serialize import catalog_to_json

    catalog = build_catalog(
        models=(_model("one"),),
        generated_at=STAMP,
        freshness=CatalogFreshness(
            status=FreshnessStatus.AS_OF,
            generated_at=STAMP,
            overlay=CatalogOverlay.USER_CACHE,
        ),
    )
    return json.loads(catalog_to_json(catalog))
