"""CLI contract tests for evidence-safe model comparison."""

from __future__ import annotations

import json

import pytest

from samyak.cli import main
from samyak.model.catalog import CatalogFreshness, CatalogOverlay, FreshnessStatus, build_catalog
from samyak.model.compare import compare_models, comparison_to_dict
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


def _provenance(source: str = "docs") -> Provenance:
    return Provenance(
        provider="openai",
        source_id=source,
        source_kind=SourceKind.PROVIDER_DOCS,
        verified_at=STAMP,
        confidence=Confidence.HIGH,
    )


def _known(value: object, source: str = "docs") -> Fact:
    return Fact(FactStatus.KNOWN, value, _provenance(source))


def _model(
    name: str,
    *,
    disposition: EvidenceDisposition | None = EvidenceDisposition.CURRENT_DOCUMENTED,
    **changes: object,
) -> ModelRecord:
    facts: dict[str, object] = {
        "display_name": _known(name.title()),
        "lifecycle": _known(LifecycleState.ACTIVE),
        "api_access": _known(True),
        "tool_calling": _known(True),
        "structured_output": _known(True),
        "input_modalities": _known((Modality.TEXT, Modality.IMAGE)),
        "output_modalities": _known((Modality.TEXT,)),
        "context_window": _known(ContextWindow(128_000, ContextWindowKind.COMBINED)),
    }
    facts.update(changes)
    return ModelRecord(
        identity=ModelIdentity("openai", name),
        evidence_disposition=disposition,
        **facts,  # type: ignore[arg-type]
    )


def _install(monkeypatch, tmp_path, models: tuple[ModelRecord, ...]) -> None:
    root = tmp_path / "cache"
    monkeypatch.setenv("SAMYAK_CACHE_DIR", str(root))
    FileCatalogStore(root=root).save(
        build_catalog(
            models=models,
            generated_at=STAMP,
            freshness=CatalogFreshness(
                status=FreshnessStatus.AS_OF, generated_at=STAMP, overlay=CatalogOverlay.USER_CACHE
            ),
        )
    )


def test_compare_text_json_order_and_api_projection(monkeypatch, tmp_path, capsys) -> None:
    first, second, third = _model("first"), _model("second"), _model("third")
    _install(monkeypatch, tmp_path, (first, second, third))
    command = ["model", "compare", second.samyak_id, first.samyak_id, third.samyak_id]
    assert main(command) == 0
    text = capsys.readouterr().out
    assert "Model comparison" in text
    assert text.index(second.samyak_id) < text.index(first.samyak_id)
    assert "Verified yes" in text
    assert "combined" in text
    assert main([*command, "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == comparison_to_dict(compare_models((first, second, third), command[2:]))
    assert [item["samyak_id"] for item in payload["models"]] == [
        second.samyak_id,
        first.samyak_id,
        third.samyak_id,
    ]


def test_compare_preserves_unknown_false_conflict_and_opt_in_warnings(
    monkeypatch, tmp_path, capsys
) -> None:
    conflict = Fact(
        FactStatus.CONFLICT,
        claims=(
            FactClaim(True, _provenance("true-docs")),
            FactClaim(False, _provenance("false-docs")),
        ),
    )
    current = _model("current", api_access=_known(False), tool_calling=conflict)
    legacy = _model(
        "legacy",
        disposition=None,
        structured_output=Fact(FactStatus.UNKNOWN, provenance=_provenance("unknown-docs")),
    )
    _install(monkeypatch, tmp_path, (current, legacy))
    assert main(["model", "compare", current.samyak_id, legacy.samyak_id]) == 2
    assert "legacy_uncertified" in capsys.readouterr().err
    assert main(["model", "compare", current.samyak_id, legacy.samyak_id, "--include-legacy"]) == 0
    output = capsys.readouterr().out
    assert "Verified no" in output
    assert "Conflicting evidence: True (true-docs), False (false-docs)" in output
    assert "Unknown" in output
    assert "multiple documentation certification levels" in output


@pytest.mark.parametrize(
    ("ids", "message"),
    [
        (["openai:one"], "exactly two"),
        (["openai:one", "openai:one"], "unique"),
        (["openai:one", "openai:missing"], "unknown"),
    ],
)
def test_compare_selection_errors(monkeypatch, tmp_path, capsys, ids, message) -> None:
    _install(monkeypatch, tmp_path, (_model("one"), _model("two")))
    assert main(["model", "compare", *ids]) == 2
    assert message in capsys.readouterr().err


def test_compare_missing_and_pending_catalog_fail_without_recovery(
    monkeypatch, tmp_path, capsys
) -> None:
    root = tmp_path / "cache"
    monkeypatch.setenv("SAMYAK_CACHE_DIR", str(root))
    assert main(["model", "compare", "openai:one", "openai:two"]) == 2
    assert "no local model catalog" in capsys.readouterr().err
    _install(monkeypatch, tmp_path, (_model("one"), _model("two")))
    pending = root / "models" / ".catalog-history-transaction.json"
    pending.write_text("pending")
    assert main(["model", "compare", "openai:one", "openai:two"]) == 2
    assert "transaction is pending" in capsys.readouterr().err
    assert pending.is_file()
