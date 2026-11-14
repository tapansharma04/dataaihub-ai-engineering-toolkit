"""Read-only source-override diagnostic. Uses temporary archives only."""

from __future__ import annotations

import contextlib
import hashlib
import json
from dataclasses import replace
from io import StringIO
from pathlib import Path

from samyak.cli import main
from samyak.model.catalog import CatalogOverlay
from samyak.model.facts import ContextWindow, ContextWindowKind, Fact, FactStatus, LifecycleState
from samyak.model.override_diagnostic import (
    StatedObservation,
    classify_statements,
    diagnose_overrides,
    render_override_report,
)
from samyak.model.providers.fireworks.sources import (
    CHANGELOG_URL as FIREWORKS_CHANGELOG_URL,
)
from samyak.model.providers.fireworks.sources import (
    SOURCE_ID_CHANGELOG as FIREWORKS_SOURCE_ID_CHANGELOG,
)
from samyak.model.providers.google.adapter import catalog_from_google_sources
from samyak.model.providers.google.sources import (
    MODELS_INDEX_URL,
    SOURCE_ID_MODELS_INDEX,
)
from samyak.model.providers.google.sources import (
    captured_markdown as capture_google,
)
from samyak.model.providers.openai.adapter import catalog_from_openai_sources
from samyak.model.providers.openai.sources import (
    DEPRECATIONS_URL as OPENAI_DEPRECATIONS_URL,
)
from samyak.model.providers.openai.sources import (
    SOURCE_ID_DEPRECATIONS as OPENAI_SOURCE_ID_DEPRECATIONS,
)
from samyak.model.providers.openai.sources import (
    captured_markdown as capture_openai,
)
from samyak.model.providers.together.adapter import catalog_from_together_sources
from samyak.model.providers.together.sources import (
    CHANGELOG_URL,
    DEPRECATIONS_URL,
    SERVERLESS_URL,
    SOURCE_ID_CHANGELOG,
    SOURCE_ID_DEPRECATIONS,
    SOURCE_ID_SERVERLESS,
    captured_markdown,
    model_page_source_id,
    model_page_url,
)
from samyak.model.source_archive import blob_path, refresh_id_from_stamp

FIXTURES = Path(__file__).parent / "fixtures" / "models" / "together"
GENERATED_AT = "2026-11-02T06:20:15.685704+00:00"
RETRIEVED_AT = "2026-11-02T06:20:17.381462+00:00"
OLDER = "2026-11-02T05-14-04.539777Z"
NEWER = "2026-11-02T06-20-15.685704Z"
OLDER_STAMP = "2026-11-02T05:14:04.539777+00:00"
GLOBAL_LATER = "2026-11-03T06:20:15.685704+00:00"

DEPRECATIONS = """# Deprecations

## Scheduled deprecations

| Model | Removal date |
| --- | --- |
| `zai-org/Example` | 2026-07-29 |
| `zai-org/Other` | 2026-08-01 |
"""

CHANGELOG = """# Changelog

## Deprecations

- `zai-org/Example` will be removed on July 28, 2025.
- `zai-org/Other` will be removed on August 2, 2025.
"""


def _stated(
    fact: str,
    value: object,
    source_id: str,
    rank: int,
    *,
    model: str = "together:zai-org/Example",
    generation: str = NEWER,
    value_type: str | None = None,
) -> StatedObservation:
    if value_type is None:
        if isinstance(value, bool):
            value_type = "bool"
        elif isinstance(value, int):
            value_type = "int"
        elif isinstance(value, str):
            value_type = "str"
        else:
            value_type = "sequence"
    return StatedObservation(
        provider="together",
        samyak_id=model,
        fact_name=fact,
        value=value,
        value_type=value_type,
        source_id=source_id,
        rank=rank,
        generation=generation,
        retrieved_at=RETRIEVED_AT,
        content_hash=f"sha256:{source_id}",
        confidence="high",
    )


def _kinds(fact: str, statements: list[StatedObservation]) -> list[str]:
    return [item.kind for item in classify_statements(statements) if item.fact_name == fact]


def test_same_rank_agreement_is_supporting() -> None:
    findings = classify_statements(
        [
            _stated("api_access", True, "together-docs-serverless-models", 2),
            _stated("api_access", True, "together-docs-model-page:glm-5.3-quickstart", 2),
        ]
    )
    assert [item.kind for item in findings] == ["supporting"]
    assert findings[0].selected_source_id == "together-docs-serverless-models"
    assert findings[0].other_source_id == "together-docs-model-page:glm-5.3-quickstart"
    assert findings[0].selected_source_rank == 2
    assert findings[0].other_source_rank == 2


def test_same_rank_conflict_keeps_every_claim() -> None:
    findings = classify_statements(
        [
            _stated("context_window", 1048576, "together-docs-serverless-models", 2),
            _stated("context_window", 1050000, "together-docs-model-page:kimi-k3-quickstart", 2),
        ]
    )
    assert [item.kind for item in findings] == ["winning_rank_conflict"]
    assert findings[0].selected_value is None
    assert {item.value for item in findings[0].claims} == {1048576, 1050000}
    assert {item.source_id for item in findings[0].claims} == {
        "together-docs-serverless-models",
        "together-docs-model-page:kimi-k3-quickstart",
    }


def test_higher_rank_override() -> None:
    findings = classify_statements(
        [
            _stated("retirement_at", "2026-07-29", "together-docs-deprecations", 3),
            _stated("retirement_at", "2025-08-28", "together-docs-changelog", 1),
        ]
    )
    assert [item.kind for item in findings] == ["override"]
    assert findings[0].selected_value == "2026-07-29"
    assert findings[0].selected_source_rank == 3
    assert findings[0].other_value == "2025-08-28"
    assert findings[0].other_source_rank == 1


def test_two_lower_sources_disagree_with_the_winner() -> None:
    findings = classify_statements(
        [
            _stated("retirement_at", "2026-07-29", "together-docs-deprecations", 3),
            _stated("retirement_at", "2025-08-28", "together-docs-changelog", 1),
            _stated("retirement_at", "2025-08-28", "together-docs-model-page:gpt-oss", 2),
        ]
    )
    assert [item.kind for item in findings] == ["override", "override"]
    assert {item.other_source_rank for item in findings} == {1, 2}
    assert {item.other_value for item in findings} == {"2025-08-28"}


def test_lower_ranked_sources_disagree_with_each_other() -> None:
    findings = classify_statements(
        [
            _stated("retirement_at", "2026-07-29", "together-docs-deprecations", 3),
            _stated("retirement_at", "2025-01-01", "together-docs-changelog", 1),
            _stated("retirement_at", "2025-06-01", "together-docs-model-page:gpt-oss", 1),
        ]
    )
    kinds = [item.kind for item in findings]
    assert kinds == ["override", "override", "lower_rank_disagreement"]
    grouped = findings[-1]
    assert {item.value for item in grouped.claims} == {"2025-01-01", "2025-06-01"}
    assert grouped.selected_value == "2026-07-29"


def test_only_lower_ranked_source_is_selected() -> None:
    findings = classify_statements(
        [_stated("retirement_at", "2025-08-28", "together-docs-changelog", 1)]
    )
    assert findings == ()


def test_unstated_fields_create_no_finding() -> None:
    findings = classify_statements([_stated("api_access", True, "together-docs-deprecations", 3)])
    assert findings == ()
    assert "context_window" not in {item.fact_name for item in findings}


def test_unknown_and_not_verified_are_not_competing_values(tmp_path: Path) -> None:
    refresh_id = _write_generation(
        tmp_path,
        NEWER,
        [_source("together", SOURCE_ID_DEPRECATIONS, DEPRECATIONS_URL, DEPRECATIONS)],
    )
    catalog = _catalog_from_bodies(
        (SOURCE_ID_DEPRECATIONS, DEPRECATIONS_URL, DEPRECATIONS),
    )
    record = catalog.models[0]
    unknown = Fact(status=FactStatus.UNKNOWN, provenance=record.api_access.provenance)
    not_verified = Fact(status=FactStatus.NOT_VERIFIED)
    catalog = replace(
        catalog,
        models=tuple(
            replace(item, api_access=unknown, display_name=not_verified) for item in catalog.models
        ),
    )
    report = diagnose_overrides(
        archive_root=tmp_path,
        generations=(("together", refresh_id),),
        catalog=catalog,
        fact_name="api_access",
    )
    assert report.findings
    assert {item.kind for item in report.findings} == {"not_comparable"}
    assert all(item.other_value is None for item in report.findings)
    assert "not_verified" not in {item.kind for item in report.findings}
    assert "winning_rank_conflict" not in {item.kind for item in report.findings}


def test_incompatible_types_are_not_coerced() -> None:
    findings = classify_statements(
        [
            _stated("context_window", 1048576, "together-docs-serverless-models", 2),
            _stated(
                "context_window",
                "1048576",
                "together-docs-model-page:kimi-k3-quickstart",
                2,
            ),
        ]
    )
    assert [item.kind for item in findings] == ["incomparable"]
    assert "incompatible types" in findings[0].explanation
    assert findings[0].selected_value is None


def test_valid_archived_blob_parses(tmp_path: Path) -> None:
    body = (FIXTURES / "glm-5.3-quickstart.md").read_bytes()
    source_id = model_page_source_id("glm-5.3-quickstart")
    refresh_id = _write_generation(
        tmp_path,
        NEWER,
        [_source("together", source_id, model_page_url("glm-5.3-quickstart"), body)],
    )
    report = diagnose_overrides(
        archive_root=tmp_path,
        generations=(("together", refresh_id),),
    )
    assert report.generations[0].source_count == 1
    assert report.limitations == () or all(
        item.code == "provider_not_evaluated" for item in report.limitations
    )
    assert not any(
        item.code
        in {
            "hash_mismatch",
            "byte_length_mismatch",
            "missing_blob",
            "missing_manifest",
            "parse_failed",
        }
        for item in report.limitations
    )


def test_hash_mismatch_is_reported_and_not_parsed(tmp_path: Path) -> None:
    body = b"deprecations\n"
    digest = hashlib.sha256(body).hexdigest()
    refresh_id = _write_generation(
        tmp_path,
        NEWER,
        [
            _source(
                "together",
                SOURCE_ID_DEPRECATIONS,
                DEPRECATIONS_URL,
                body,
                blob=body + b"x",
                content_hash=f"sha256:{digest}",
            )
        ],
    )
    report = diagnose_overrides(archive_root=tmp_path, generations=(("together", refresh_id),))
    assert _codes(report, "together") == {"hash_mismatch"}
    assert report.findings == ()


def test_byte_length_mismatch_is_reported_and_not_parsed(tmp_path: Path) -> None:
    body = DEPRECATIONS.encode()
    refresh_id = _write_generation(
        tmp_path,
        NEWER,
        [
            _source(
                "together",
                SOURCE_ID_DEPRECATIONS,
                DEPRECATIONS_URL,
                body,
                byte_length=len(body) + 1,
            )
        ],
    )
    report = diagnose_overrides(archive_root=tmp_path, generations=(("together", refresh_id),))
    assert _codes(report, "together") == {"byte_length_mismatch"}
    assert report.findings == ()


def test_missing_manifest_is_reported(tmp_path: Path) -> None:
    report = diagnose_overrides(archive_root=tmp_path, generations=(("together", NEWER),))
    assert _codes(report, "together") == {"missing_manifest"}
    assert report.findings == ()


def test_missing_source_blob_is_reported(tmp_path: Path) -> None:
    body = DEPRECATIONS.encode()
    refresh_id = _write_generation(
        tmp_path,
        NEWER,
        [
            _source(
                "together",
                SOURCE_ID_DEPRECATIONS,
                DEPRECATIONS_URL,
                body,
                skip_blob=True,
            )
        ],
    )
    report = diagnose_overrides(archive_root=tmp_path, generations=(("together", refresh_id),))
    assert _codes(report, "together") == {"missing_blob"}
    assert report.findings == ()


def test_missing_provider_source_is_not_an_empty_agreement(tmp_path: Path) -> None:
    refresh_id = _write_generation(
        tmp_path,
        NEWER,
        [_source("together", SOURCE_ID_DEPRECATIONS, DEPRECATIONS_URL, DEPRECATIONS)],
    )
    report = diagnose_overrides(archive_root=tmp_path, generations=(("fireworks", refresh_id),))
    assert _codes(report, "fireworks") == {"missing_provider_source"}
    assert report.findings == ()
    assert "no fireworks sources" in report.limitations[-1].message or any(
        "no fireworks sources" in item.message for item in report.limitations
    )


def test_fireworks_empty_changelog_is_not_a_diagnostic_parse_failure(tmp_path: Path) -> None:
    refresh_id = _write_generation(
        tmp_path,
        NEWER,
        [
            _source(
                "fireworks",
                FIREWORKS_SOURCE_ID_CHANGELOG,
                FIREWORKS_CHANGELOG_URL,
                "# Changelog\n\nNo serving model lifecycle event is listed.\n",
            )
        ],
    )
    report = diagnose_overrides(
        archive_root=tmp_path,
        generations=(("fireworks", refresh_id),),
    )

    assert not any(
        item.code == "parse_failed" and item.provider == "fireworks" for item in report.limitations
    )
    assert report.findings == ()


def test_fireworks_changelog_link_replays_without_parse_failure(tmp_path: Path) -> None:
    refresh_id = _write_generation(
        tmp_path,
        NEWER,
        [
            _source(
                "fireworks",
                FIREWORKS_SOURCE_ID_CHANGELOG,
                FIREWORKS_CHANGELOG_URL,
                """# Changelog

[Target](https://app.fireworks.ai/models/fireworks/target) is deprecated from
serverless. Migrate to
[Replacement](https://app.fireworks.ai/models/fireworks/replacement).
""",
            )
        ],
    )
    report = diagnose_overrides(
        archive_root=tmp_path,
        generations=(("fireworks", refresh_id),),
    )

    assert not any(
        item.code == "parse_failed" and item.provider == "fireworks" for item in report.limitations
    )


def test_fireworks_malformed_changelog_remains_a_parse_failure(tmp_path: Path) -> None:
    refresh_id = _write_generation(
        tmp_path,
        NEWER,
        [
            _source(
                "fireworks",
                FIREWORKS_SOURCE_ID_CHANGELOG,
                FIREWORKS_CHANGELOG_URL,
                "# Release notes\n\nNothing relevant is described.\n",
            )
        ],
    )
    report = diagnose_overrides(
        archive_root=tmp_path,
        generations=(("fireworks", refresh_id),),
    )

    assert "parse_failed" in _codes(report, "fireworks")


def test_invalid_source_attribution_is_not_parsed(tmp_path: Path) -> None:
    refresh_id = _write_generation(
        tmp_path,
        NEWER,
        [_source("together", "fireworks-docs-changelog", DEPRECATIONS_URL, DEPRECATIONS)],
    )
    report = diagnose_overrides(archive_root=tmp_path, generations=(("together", refresh_id),))
    assert _codes(report, "together") == {"invalid_attribution"}
    assert report.findings == ()


def test_global_generation_keeps_provider_pairs_separate(tmp_path: Path) -> None:
    refresh_id = _write_generation(
        tmp_path,
        OLDER,
        [
            _source("together", SOURCE_ID_DEPRECATIONS, DEPRECATIONS_URL, DEPRECATIONS),
            _source(
                "fireworks",
                "fireworks-docs-changelog",
                "https://docs.fireworks.ai/changelog.md",
                b"# Changelog\n",
            ),
        ],
    )
    report = diagnose_overrides(
        archive_root=tmp_path,
        generations=(("together", refresh_id), ("fireworks", refresh_id)),
    )
    assert [(item.provider, item.refresh_id) for item in report.generations] == [
        ("together", refresh_id),
        ("fireworks", refresh_id),
    ]
    assert "missing_provider_source" not in {item.code for item in report.limitations}
    assert "provider_not_evaluated" not in {
        item.code for item in report.limitations if item.provider in {"together", "fireworks"}
    }


def test_provider_only_generation_does_not_imply_other_providers(tmp_path: Path) -> None:
    refresh_id = _write_generation(
        tmp_path,
        NEWER,
        [_source("together", SOURCE_ID_DEPRECATIONS, DEPRECATIONS_URL, DEPRECATIONS)],
    )
    report = diagnose_overrides(archive_root=tmp_path, generations=(("together", refresh_id),))
    assert report.generations[0].provider == "together"
    assert report.generations[0].source_count == 1
    unevaluated = {
        item.provider for item in report.limitations if item.code == "provider_not_evaluated"
    }
    assert "fireworks" in unevaluated


def test_multiple_provider_generations_stay_labeled(tmp_path: Path) -> None:
    together_id = _write_generation(
        tmp_path,
        NEWER,
        [
            _source("together", SOURCE_ID_DEPRECATIONS, DEPRECATIONS_URL, DEPRECATIONS),
            _source("together", SOURCE_ID_CHANGELOG, CHANGELOG_URL, CHANGELOG),
        ],
    )
    fireworks_id = _write_generation(
        tmp_path,
        OLDER,
        [
            _source(
                "fireworks",
                "fireworks-docs-changelog",
                "https://docs.fireworks.ai/changelog.md",
                b"# Changelog\n",
            )
        ],
    )
    report = diagnose_overrides(
        archive_root=tmp_path,
        generations=(("together", together_id), ("fireworks", fireworks_id)),
    )
    assert {item.refresh_id for item in report.generations} == {together_id, fireworks_id}
    together_findings = [item for item in report.findings if item.provider == "together"]
    assert together_findings
    assert {item.selected_generation for item in together_findings} == {together_id}


def test_missing_provider_generation_is_not_filled_in(tmp_path: Path) -> None:
    _write_generation(
        tmp_path,
        NEWER,
        [_source("together", SOURCE_ID_DEPRECATIONS, DEPRECATIONS_URL, DEPRECATIONS)],
    )
    _write_generation(
        tmp_path,
        OLDER,
        [
            _source(
                "fireworks",
                "fireworks-docs-changelog",
                "https://docs.fireworks.ai/changelog.md",
                b"# Changelog\n",
            )
        ],
    )
    report = diagnose_overrides(archive_root=tmp_path, generations=(("together", NEWER),))
    assert [item.provider for item in report.generations] == ["together"]
    assert any(
        item.code == "provider_not_evaluated" and item.provider == "fireworks"
        for item in report.limitations
    )
    assert "newest generation was not used" in next(
        item.message for item in report.limitations if item.provider == "fireworks"
    )


def test_different_generation_timestamps_are_identified(tmp_path: Path) -> None:
    _write_pair(tmp_path)
    report = diagnose_overrides(
        archive_root=tmp_path,
        generations=(("together", NEWER), ("fireworks", OLDER)),
    )
    assert any(item.code == "generations_differ" for item in report.limitations)
    message = next(item.message for item in report.limitations if item.code == "generations_differ")
    assert NEWER in message
    assert OLDER in message
    rendered = render_override_report(report, output_format="text")
    assert NEWER in rendered
    assert OLDER in rendered


def test_no_implicit_newest_generation(tmp_path: Path) -> None:
    _write_generation(
        tmp_path,
        OLDER,
        [_source("together", SOURCE_ID_DEPRECATIONS, DEPRECATIONS_URL, DEPRECATIONS)],
    )
    _write_generation(
        tmp_path,
        NEWER,
        [
            _source("together", SOURCE_ID_DEPRECATIONS, DEPRECATIONS_URL, DEPRECATIONS),
            _source("together", SOURCE_ID_CHANGELOG, CHANGELOG_URL, CHANGELOG),
        ],
    )
    report = diagnose_overrides(archive_root=tmp_path, generations=(("together", OLDER),))
    assert [item.refresh_id for item in report.generations] == [OLDER]
    assert report.findings == ()
    assert NEWER not in render_override_report(report, output_format="text")


def test_reconstructed_observation_matches_catalog(tmp_path: Path) -> None:
    body = (FIXTURES / "glm-5.3-quickstart.md").read_bytes()
    source_id = model_page_source_id("glm-5.3-quickstart")
    url = model_page_url("glm-5.3-quickstart")
    refresh_id = _write_generation(
        tmp_path,
        NEWER,
        [_source("together", source_id, url, body)],
    )
    catalog = _catalog_from_bodies((source_id, url, body.decode()))
    report = diagnose_overrides(
        archive_root=tmp_path,
        generations=(("together", refresh_id),),
        catalog=catalog,
        model_id="together:zai-org/GLM-5.3",
        fact_name="api_access",
    )
    assert report.catalog_generated_at == GENERATED_AT
    assert report.parser_version
    checks = [item for item in report.catalog_checks if item.fact_name == "api_access"]
    assert checks
    assert {item.status for item in checks} == {"comparable"}
    assert report.findings == ()


def test_parser_output_that_differs_from_catalog_is_a_comparable_mismatch(tmp_path: Path) -> None:
    catalog, refresh_id = _disagreeing_catalog(tmp_path)
    record = next(item for item in catalog.models if item.provider_model_id == "zai-org/Example")
    mismatched = replace(
        record,
        api_access=Fact(
            status=FactStatus.KNOWN,
            value=True,
            provenance=record.api_access.provenance,
        ),
    )
    catalog = replace(
        catalog,
        models=tuple(
            mismatched if item.samyak_id == mismatched.samyak_id else item
            for item in catalog.models
        ),
    )
    report = diagnose_overrides(
        archive_root=tmp_path,
        generations=(("together", refresh_id),),
        catalog=catalog,
        model_id="together:zai-org/Example",
        fact_name="api_access",
    )
    assert [item.kind for item in report.findings] == ["mismatch"]
    assert report.findings[0].comparability_status == "comparable"
    assert "not a source conflict" in report.findings[0].explanation
    assert report.catalog_checks[0].status == "mismatching"


def test_current_parser_cannot_reproduce_a_historical_observation(tmp_path: Path) -> None:
    body = (FIXTURES / "glm-5.3-quickstart.md").read_bytes()
    source_id = model_page_source_id("glm-5.3-quickstart")
    url = model_page_url("glm-5.3-quickstart")
    refresh_id = _write_generation(tmp_path, NEWER, [_source("together", source_id, url, body)])
    catalog = _catalog_from_bodies((source_id, url, body.decode()))
    record = next(item for item in catalog.models if item.provider_model_id == "zai-org/GLM-5.3")
    invented = replace(
        record,
        context_window=Fact(
            status=FactStatus.KNOWN,
            value=ContextWindow(tokens=999_999, kind=ContextWindowKind.COMBINED),
            provenance=record.api_access.provenance,
        ),
    )
    catalog = replace(
        catalog,
        models=tuple(
            invented if item.samyak_id == invented.samyak_id else item for item in catalog.models
        ),
    )
    report = diagnose_overrides(
        archive_root=tmp_path,
        generations=(("together", refresh_id),),
        catalog=catalog,
        model_id="together:zai-org/GLM-5.3",
        fact_name="context_window",
    )
    assert [item.kind for item in report.findings] == ["mismatch"]
    assert report.findings[0].selected_value is None
    assert report.findings[0].other_value == 999_999
    assert "winning_rank_conflict" not in {item.kind for item in report.findings}


def test_parser_mismatch_is_not_classified_as_a_source_conflict(tmp_path: Path) -> None:
    catalog, refresh_id = _disagreeing_catalog(tmp_path)
    record = next(item for item in catalog.models if item.provider_model_id == "zai-org/Example")
    mismatched = replace(
        record,
        api_access=Fact(
            status=FactStatus.KNOWN,
            value=True,
            provenance=record.api_access.provenance,
        ),
    )
    catalog = replace(
        catalog,
        models=tuple(
            mismatched if item.samyak_id == mismatched.samyak_id else item
            for item in catalog.models
        ),
    )
    report = diagnose_overrides(
        archive_root=tmp_path,
        generations=(("together", refresh_id),),
        catalog=catalog,
        model_id="together:zai-org/Example",
    )
    api_access = [item.kind for item in report.findings if item.fact_name == "api_access"]
    assert api_access == ["mismatch"]
    assert "winning_rank_conflict" not in api_access
    assert "override" not in api_access
    assert "supporting" not in api_access


def test_output_order_is_stable() -> None:
    statements = [
        _stated("retirement_at", "2025-06-01", "together-docs-model-page:gpt-oss", 1),
        _stated("retirement_at", "2026-07-29", "together-docs-deprecations", 3),
        _stated("retirement_at", "2025-01-01", "together-docs-changelog", 1),
    ]
    first = classify_statements(statements)
    second = classify_statements(statements)
    assert first == second
    assert [item.kind for item in first] == ["override", "override", "lower_rank_disagreement"]
    assert [item.other_source_id for item in first if item.kind == "override"] == [
        "together-docs-changelog",
        "together-docs-model-page:gpt-oss",
    ]


def test_json_output_is_valid(tmp_path: Path) -> None:
    code, out, err = _run(tmp_path, "--format", "json")
    assert code == 0
    assert err == ""
    payload = json.loads(out)
    assert payload["capability"] == "model_override_diagnostic"
    assert payload["product"] == "samyak"
    assert isinstance(payload["findings"], list)
    assert payload["findings"]
    assert payload["findings"][0]["kind"] in {
        "supporting",
        "override",
        "lower_rank_disagreement",
        "winning_rank_conflict",
    }


def test_human_summary_names_generations_and_findings(tmp_path: Path) -> None:
    code, out, err = _run(tmp_path)
    assert code == 0
    assert err == ""
    assert out.startswith("Source override diagnostic\n")
    assert "Findings:" in out
    assert NEWER in out
    assert "override" in out


def test_model_and_fact_filters(tmp_path: Path) -> None:
    _write_generation(
        tmp_path,
        NEWER,
        [
            _source("together", SOURCE_ID_DEPRECATIONS, DEPRECATIONS_URL, DEPRECATIONS),
            _source("together", SOURCE_ID_CHANGELOG, CHANGELOG_URL, CHANGELOG),
        ],
    )
    code, out, _err = _run(
        tmp_path,
        "--model",
        "together:zai-org/Example",
        "--fact",
        "retirement_at",
        "--format",
        "json",
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["findings"]
    assert {item["samyak_id"] for item in payload["findings"]} == {"together:zai-org/Example"}
    assert {item["fact_name"] for item in payload["findings"]} == {"retirement_at"}


def test_disagreements_do_not_fail_the_command(tmp_path: Path) -> None:
    code, out, err = _run(tmp_path, "--format", "json")
    assert code == 0
    assert err == ""
    payload = json.loads(out)
    assert any(item["kind"] == "override" for item in payload["findings"])


def test_past_deprecation_is_excluded_history_and_catalog_matches(tmp_path: Path) -> None:
    deprecations = """# Deprecations

## History

| Model | Removal date |
| --- | --- |
| `zai-org/PastModel` | 2026-02-25 |
"""
    changelog = """# Changelog

## Deprecations

- `zai-org/PastModel` is no longer available.
"""
    report, catalog = _diagnose_bodies(
        tmp_path,
        (SOURCE_ID_DEPRECATIONS, DEPRECATIONS_URL, deprecations),
        (SOURCE_ID_CHANGELOG, CHANGELOG_URL, changelog),
        model_id="together:zai-org/PastModel",
        fact_name="lifecycle",
    )
    record = next(item for item in catalog.models if item.provider_model_id == "zai-org/PastModel")
    assert record.lifecycle.value is LifecycleState.RETIRED
    assert record.lifecycle.provenance is not None
    assert record.lifecycle.provenance.source_id == SOURCE_ID_CHANGELOG
    assert {item.status for item in report.catalog_checks} == {"comparable"}
    assert "not_comparable" not in {item.kind for item in report.findings}
    excluded = [item for item in report.findings if item.kind == "excluded"]
    assert len(excluded) == 1
    assert excluded[0].other_source_id == SOURCE_ID_DEPRECATIONS
    assert excluded[0].other_value == "retired"
    assert excluded[0].selected_source_id == SOURCE_ID_CHANGELOG
    assert excluded[0].selected_value == "retired"
    assert excluded[0].comparability_status == "comparable"


def test_lifecycle_provenance_mismatch_stays_not_comparable(tmp_path: Path) -> None:
    deprecations = """# Deprecations

## History

| Model | Removal date |
| --- | --- |
| `zai-org/PastModel` | 2026-02-25 |
"""
    changelog = """# Changelog

## Deprecations

- `zai-org/PastModel` is no longer available.
"""
    catalog, refresh_id = _bodies_catalog(
        tmp_path,
        (SOURCE_ID_DEPRECATIONS, DEPRECATIONS_URL, deprecations),
        (SOURCE_ID_CHANGELOG, CHANGELOG_URL, changelog),
    )
    record = next(item for item in catalog.models if item.provider_model_id == "zai-org/PastModel")
    assert record.lifecycle.provenance is not None
    mismatched = replace(
        record,
        lifecycle=replace(
            record.lifecycle,
            provenance=replace(record.lifecycle.provenance, source_id=SOURCE_ID_DEPRECATIONS),
        ),
    )
    catalog = replace(
        catalog,
        models=tuple(
            mismatched if item.samyak_id == mismatched.samyak_id else item
            for item in catalog.models
        ),
    )
    report = diagnose_overrides(
        archive_root=tmp_path,
        generations=(("together", refresh_id),),
        catalog=catalog,
        model_id="together:zai-org/PastModel",
        fact_name="lifecycle",
    )
    assert [item.kind for item in report.findings] == ["mismatch"]
    assert "not a source conflict" in report.findings[0].explanation


def test_removal_on_verification_date_promotes_to_retired(tmp_path: Path) -> None:
    deprecations = """# Deprecations

## Scheduled deprecations

| Model | Removal date |
| --- | --- |
| `zai-org/OnDate` | 2026-11-02 |
"""
    changelog = """# Changelog

## Deprecations

- `zai-org/OnDate` is no longer available.
"""
    report, catalog = _diagnose_bodies(
        tmp_path,
        (SOURCE_ID_DEPRECATIONS, DEPRECATIONS_URL, deprecations),
        (SOURCE_ID_CHANGELOG, CHANGELOG_URL, changelog),
        model_id="together:zai-org/OnDate",
        fact_name="lifecycle",
    )
    record = next(item for item in catalog.models if item.provider_model_id == "zai-org/OnDate")
    assert record.lifecycle.value is LifecycleState.RETIRED
    assert record.lifecycle.provenance is not None
    assert record.lifecycle.provenance.source_id == SOURCE_ID_DEPRECATIONS
    assert {item.status for item in report.catalog_checks} == {"comparable"}
    selected = [item for item in report.findings if item.kind == "supporting"]
    assert selected
    assert {item.selected_value for item in selected} == {"retired"}
    assert {item.selected_source_id for item in selected} == {SOURCE_ID_DEPRECATIONS}


def test_removal_after_verification_date_stays_deprecated(tmp_path: Path) -> None:
    deprecations = """# Deprecations

## Scheduled deprecations

| Model | Removal date |
| --- | --- |
| `zai-org/Later` | 2026-11-03 |
"""
    changelog = """# Changelog

## Deprecations

- `zai-org/Later` will be removed on November 3, 2026.
"""
    report, catalog = _diagnose_bodies(
        tmp_path,
        (SOURCE_ID_DEPRECATIONS, DEPRECATIONS_URL, deprecations),
        (SOURCE_ID_CHANGELOG, CHANGELOG_URL, changelog),
        model_id="together:zai-org/Later",
        fact_name="lifecycle",
    )
    record = next(item for item in catalog.models if item.provider_model_id == "zai-org/Later")
    assert record.lifecycle.value is LifecycleState.DEPRECATED
    assert {item.status for item in report.catalog_checks} == {"comparable"}
    assert {item.selected_value for item in report.findings} == {"deprecated"}
    assert "retired" not in {item.selected_value for item in report.findings}


def test_serverless_membership_derives_active(tmp_path: Path) -> None:
    body = """# Serverless models

## Chat

| API model string | Model name |
| --- | --- |
| `zai-org/Listed` | Listed |
"""
    report, catalog = _diagnose_bodies(
        tmp_path,
        (SOURCE_ID_SERVERLESS, SERVERLESS_URL, body),
        model_id="together:zai-org/Listed",
        fact_name="lifecycle",
    )
    record = next(item for item in catalog.models if item.provider_model_id == "zai-org/Listed")
    assert record.lifecycle.value is LifecycleState.ACTIVE
    assert record.lifecycle.provenance is not None
    assert record.lifecycle.provenance.source_id == SOURCE_ID_SERVERLESS
    derived = [item for item in report.findings if item.kind == "derived"]
    assert len(derived) == 1
    assert derived[0].selected_value == "active"
    assert derived[0].selected_source_id == SOURCE_ID_SERVERLESS
    assert derived[0].selected_confidence == "medium"
    assert derived[0].other_source_id is None
    assert derived[0].claims == ()
    assert {item.status for item in report.catalog_checks} == {"comparable"}


def test_past_retired_row_with_serverless_membership_derives_active(tmp_path: Path) -> None:
    deprecations = """# Deprecations

## History

| Model | Removal date |
| --- | --- |
| `zai-org/Listed` | 2025-04-14 |
"""
    serverless = """# Serverless models

## Chat

| API model string | Model name |
| --- | --- |
| `zai-org/Listed` | Listed |
"""
    report, _catalog = _diagnose_bodies(
        tmp_path,
        (SOURCE_ID_SERVERLESS, SERVERLESS_URL, serverless),
        (SOURCE_ID_DEPRECATIONS, DEPRECATIONS_URL, deprecations),
        model_id="together:zai-org/Listed",
        fact_name="lifecycle",
    )
    assert {item.kind for item in report.findings} == {"derived", "excluded"}
    derived = next(item for item in report.findings if item.kind == "derived")
    excluded = next(item for item in report.findings if item.kind == "excluded")
    assert derived.selected_value == "active"
    assert derived.selected_source_id == SOURCE_ID_SERVERLESS
    assert excluded.other_source_id == SOURCE_ID_DEPRECATIONS
    assert excluded.other_value == "retired"
    assert excluded.selected_value == "active"
    assert {item.status for item in report.catalog_checks} == {"comparable"}


def test_two_current_rank_three_lifecycle_values_conflict(tmp_path: Path) -> None:
    deprecations = """# Deprecations

## Scheduled deprecations

| Model | Removal date |
| --- | --- |
| `zai-org/Split` | 2026-12-01 |

## Scheduled removals

| Model | Removal date |
| --- | --- |
| `zai-org/Split` | 2026-01-01 |
"""
    report, catalog = _diagnose_bodies(
        tmp_path,
        (SOURCE_ID_DEPRECATIONS, DEPRECATIONS_URL, deprecations),
        model_id="together:zai-org/Split",
        fact_name="lifecycle",
    )
    record = next(item for item in catalog.models if item.provider_model_id == "zai-org/Split")
    assert record.lifecycle.status is FactStatus.CONFLICT
    assert [item.kind for item in report.findings] == ["winning_rank_conflict"]
    assert {item.value for item in report.findings[0].claims} == {"deprecated", "retired"}
    assert {item.status for item in report.catalog_checks} == {"comparable"}


def test_identical_observations_collapse_to_one_finding(tmp_path: Path) -> None:
    deprecations = """# Deprecations

## History

| Model | Removal date |
| --- | --- |
| `zai-org/Dup` | 2026-02-25 |
"""
    changelog = """# Changelog

## Deprecations

- `zai-org/Dup` is no longer available.

## More deprecations

- `zai-org/Dup` is no longer available.
"""
    refresh_id = _write_generation(
        tmp_path,
        NEWER,
        [
            _source("together", SOURCE_ID_DEPRECATIONS, DEPRECATIONS_URL, deprecations),
            _source("together", SOURCE_ID_CHANGELOG, CHANGELOG_URL, changelog),
        ],
    )
    report = diagnose_overrides(
        archive_root=tmp_path,
        generations=(("together", refresh_id),),
        model_id="together:zai-org/Dup",
        fact_name="api_access",
    )
    supporting = [item for item in report.findings if item.kind == "supporting"]
    assert len(supporting) == 1
    assert supporting[0].selected_value is False
    assert supporting[0].other_value is False
    assert supporting[0].other_source_id == SOURCE_ID_CHANGELOG
    assert "Repeated 2 times." in supporting[0].explanation


def test_distinct_values_from_one_source_stay_separate(tmp_path: Path) -> None:
    changelog = """# Changelog

## Deprecations

**June 26, 2026**

- `Qwen/Qwen3.5-397B-A17B` is no longer available.

## Deprecations

**June 13, 2026**

- `Qwen/Qwen3.5-397B-A17B` will no longer be available after June 29, 2026.
"""
    refresh_id = _write_generation(
        tmp_path,
        NEWER,
        [_source("together", SOURCE_ID_CHANGELOG, CHANGELOG_URL, changelog)],
    )
    report = diagnose_overrides(
        archive_root=tmp_path,
        generations=(("together", refresh_id),),
        model_id="together:Qwen/Qwen3.5-397B-A17B",
        fact_name="api_access",
    )
    assert [item.kind for item in report.findings] == ["winning_rank_conflict"]
    assert {item.value for item in report.findings[0].claims} == {True, False}
    rendered = json.dumps([item.to_dict() for item in report.findings])
    assert "June 26" not in rendered
    assert "June 13" not in rendered
    assert "Repeated" not in report.findings[0].explanation


def test_catalog_hash_from_another_generation_is_historical(tmp_path: Path) -> None:
    catalog, refresh_id = _disagreeing_catalog(tmp_path)
    record = next(item for item in catalog.models if item.provider_model_id == "zai-org/Example")
    assert record.api_access.provenance is not None
    historical = replace(
        record,
        api_access=Fact(
            status=FactStatus.KNOWN,
            value=False,
            provenance=replace(
                record.api_access.provenance,
                content_hash="sha256:" + ("ab" * 32),
            ),
        ),
    )
    catalog = replace(
        catalog,
        models=tuple(
            historical if item.samyak_id == historical.samyak_id else item
            for item in catalog.models
        ),
    )
    report = diagnose_overrides(
        archive_root=tmp_path,
        generations=(("together", refresh_id),),
        catalog=catalog,
        model_id="together:zai-org/Example",
        fact_name="api_access",
    )
    assert report.findings
    assert {item.kind for item in report.findings} == {"not_comparable"}
    assert {item.reason for item in report.catalog_checks} == {"required_archive_evidence_absent"}
    assert "override" not in {item.kind for item in report.findings}


def test_google_index_membership_derives_active_through_its_normalizer(tmp_path: Path) -> None:
    body = """## Gemini models

| Model | Endpoint |
|---|---|
| Listed model | `gemini-listed` |
"""
    refresh_id = _write_generation(
        tmp_path,
        NEWER,
        [_source("google", SOURCE_ID_MODELS_INDEX, MODELS_INDEX_URL, body)],
    )
    catalog = catalog_from_google_sources(
        (
            capture_google(
                source_id=SOURCE_ID_MODELS_INDEX,
                source_url=MODELS_INDEX_URL,
                body=body,
                retrieved_at=RETRIEVED_AT,
            ),
        ),
        generated_at=GENERATED_AT,
        verified_at=GENERATED_AT,
        overlay=CatalogOverlay.BUNDLED,
    )
    report = diagnose_overrides(
        archive_root=tmp_path,
        generations=(("google", refresh_id),),
        catalog=catalog,
        model_id="google:gemini-listed",
        fact_name="lifecycle",
    )
    record = next(item for item in catalog.models if item.provider_model_id == "gemini-listed")
    assert record.lifecycle.value is LifecycleState.ACTIVE
    assert record.lifecycle.provenance is not None
    assert record.lifecycle.provenance.source_id == SOURCE_ID_MODELS_INDEX
    derived = [item for item in report.findings if item.kind == "derived"]
    assert len(derived) == 1
    assert derived[0].selected_value == "active"
    assert derived[0].selected_confidence == "medium"
    assert derived[0].selected_source_id == SOURCE_ID_MODELS_INDEX
    assert {item.status for item in report.catalog_checks} == {"comparable"}


def test_provider_timestamp_differs_from_global_does_not_promote(tmp_path: Path) -> None:
    deprecations = """# Deprecations

## Scheduled deprecations

| Model | Removal date |
| --- | --- |
| `zai-org/Boundary` | 2026-11-03 |
"""
    refresh_id = _write_generation(
        tmp_path,
        OLDER,
        [_source("together", SOURCE_ID_DEPRECATIONS, DEPRECATIONS_URL, deprecations)],
    )
    catalog = _catalog_from_bodies(
        (SOURCE_ID_DEPRECATIONS, DEPRECATIONS_URL, deprecations),
        generated_at=OLDER_STAMP,
        verified_at=OLDER_STAMP,
    )
    catalog = _with_global_generated_at(catalog, GLOBAL_LATER)
    assert catalog.freshness.generated_at == GLOBAL_LATER
    assert catalog.models[0].lifecycle.provenance is not None
    assert catalog.models[0].lifecycle.provenance.verified_at == OLDER_STAMP
    report = diagnose_overrides(
        archive_root=tmp_path,
        generations=(("together", refresh_id),),
        catalog=catalog,
        model_id="together:zai-org/Boundary",
        fact_name="lifecycle",
    )
    summary = report.generations[0]
    assert summary.normalization_timestamp_status == "resolved"
    assert summary.normalization_timestamp == OLDER_STAMP
    assert summary.normalization_timestamp != GLOBAL_LATER
    assert summary.catalog_provider_verified_at == OLDER_STAMP
    record = next(item for item in catalog.models if item.provider_model_id == "zai-org/Boundary")
    assert record.lifecycle.value is LifecycleState.DEPRECATED
    assert {item.status for item in report.catalog_checks} == {"comparable"}
    assert "retired" not in {
        item.selected_value for item in report.findings if item.fact_name == "lifecycle"
    }


def test_two_providers_use_independent_timestamps(tmp_path: Path) -> None:
    together_body = """# Deprecations

## Scheduled deprecations

| Model | Removal date |
| --- | --- |
| `zai-org/TOne` | 2026-11-03 |
"""
    openai_body = """# Deprecations

## Upcoming deprecations

### 2026-01-01: Boundary

| Shutdown date | Model / system | Recommended replacement |
| --- | --- | --- |
| 2026-11-03 | `gpt-boundary` | `gpt-next` |
"""
    together_gen = _write_generation(
        tmp_path,
        NEWER,
        [_source("together", SOURCE_ID_DEPRECATIONS, DEPRECATIONS_URL, together_body)],
    )
    openai_gen = _write_generation(
        tmp_path,
        OLDER,
        [_source("openai", OPENAI_SOURCE_ID_DEPRECATIONS, OPENAI_DEPRECATIONS_URL, openai_body)],
    )
    together_catalog = _catalog_from_bodies(
        (SOURCE_ID_DEPRECATIONS, DEPRECATIONS_URL, together_body),
        generated_at=GENERATED_AT,
        verified_at=GENERATED_AT,
    )
    openai_catalog = catalog_from_openai_sources(
        (
            capture_openai(
                source_id=OPENAI_SOURCE_ID_DEPRECATIONS,
                source_url=OPENAI_DEPRECATIONS_URL,
                body=openai_body,
                retrieved_at=RETRIEVED_AT,
            ),
        ),
        generated_at=OLDER_STAMP,
        verified_at=OLDER_STAMP,
        overlay=CatalogOverlay.BUNDLED,
    )
    merged = tuple(
        sorted(
            tuple(together_catalog.models) + tuple(openai_catalog.models),
            key=lambda item: item.samyak_id,
        )
    )
    from samyak.model.catalog import ProviderSummary

    counts: dict[str, int] = {}
    for record in merged:
        counts[record.provider_id] = counts.get(record.provider_id, 0) + 1
    providers = tuple(
        ProviderSummary(id=provider_id, model_count=counts[provider_id])
        for provider_id in sorted(counts)
    )
    catalog = replace(
        together_catalog,
        generated_at=GENERATED_AT,
        freshness=replace(together_catalog.freshness, generated_at=GENERATED_AT),
        providers=providers,
        models=merged,
    )
    report = diagnose_overrides(
        archive_root=tmp_path,
        generations=(("together", together_gen), ("openai", openai_gen)),
        catalog=catalog,
        fact_name="lifecycle",
    )
    by_provider = {item.provider: item for item in report.generations}
    assert by_provider["together"].normalization_timestamp == GENERATED_AT
    assert by_provider["openai"].normalization_timestamp == OLDER_STAMP
    assert by_provider["together"].normalization_timestamp_status == "resolved"
    assert by_provider["openai"].normalization_timestamp_status == "resolved"
    together_life = next(
        item for item in catalog.models if item.samyak_id == "together:zai-org/TOne"
    )
    openai_life = next(item for item in catalog.models if item.samyak_id == "openai:gpt-boundary")
    assert together_life.lifecycle.value is LifecycleState.DEPRECATED
    assert openai_life.lifecycle.value is LifecycleState.DEPRECATED
    assert {item.status for item in report.catalog_checks} == {"comparable"}


def test_consistent_timestamp_evidence_resolves(tmp_path: Path) -> None:
    report, _catalog = _diagnose_bodies(
        tmp_path,
        (SOURCE_ID_DEPRECATIONS, DEPRECATIONS_URL, DEPRECATIONS),
        (SOURCE_ID_CHANGELOG, CHANGELOG_URL, CHANGELOG),
        model_id="together:zai-org/Example",
        fact_name="api_access",
    )
    summary = report.generations[0]
    assert summary.normalization_timestamp_status == "resolved"
    assert summary.normalization_timestamp == GENERATED_AT
    assert summary.normalization_timestamp_method == "catalog_fact_provenance.verified_at"
    assert summary.normalization_timestamp_evidence_count is not None
    assert summary.normalization_timestamp_evidence_count > 1


def test_missing_provenance_on_some_facts_uses_remaining_evidence(tmp_path: Path) -> None:
    catalog, refresh_id = _disagreeing_catalog(tmp_path)
    record = next(item for item in catalog.models if item.provider_model_id == "zai-org/Example")
    stripped = replace(record, display_name=Fact(status=FactStatus.NOT_VERIFIED))
    catalog = replace(
        catalog,
        models=tuple(
            stripped if item.samyak_id == stripped.samyak_id else item for item in catalog.models
        ),
    )
    report = diagnose_overrides(
        archive_root=tmp_path,
        generations=(("together", refresh_id),),
        catalog=catalog,
        model_id="together:zai-org/Example",
        fact_name="api_access",
    )
    assert report.generations[0].normalization_timestamp_status == "resolved"
    assert report.generations[0].normalization_timestamp == GENERATED_AT
    assert {item.status for item in report.catalog_checks if item.fact_name == "api_access"} == {
        "comparable"
    }


def test_all_timestamp_evidence_missing_is_not_guessed(tmp_path: Path) -> None:
    catalog, refresh_id = _disagreeing_catalog(tmp_path)
    catalog = _strip_all_verified_at(catalog)
    report = diagnose_overrides(
        archive_root=tmp_path,
        generations=(("together", refresh_id),),
        catalog=catalog,
        model_id="together:zai-org/Example",
        fact_name="api_access",
    )
    assert report.generations[0].normalization_timestamp_status == "missing"
    assert report.generations[0].normalization_timestamp is None
    assert any(item.code == "normalization_timestamp_unavailable" for item in report.limitations)
    assert report.catalog_generated_at == GENERATED_AT
    assert {item.status for item in report.catalog_checks} == {"not_comparable"}
    assert {item.kind for item in report.findings} == {"not_comparable"}


def test_conflicting_timestamps_are_not_arbitrarily_chosen(tmp_path: Path) -> None:
    catalog, refresh_id = _disagreeing_catalog(tmp_path)
    first = catalog.models[0]
    second = catalog.models[1]
    assert first.api_access.provenance is not None
    altered = replace(
        second,
        api_access=replace(
            second.api_access,
            provenance=replace(second.api_access.provenance, verified_at=OLDER_STAMP),
        ),
    )
    catalog = replace(
        catalog,
        models=tuple(
            altered if item.samyak_id == altered.samyak_id else item for item in catalog.models
        ),
    )
    report = diagnose_overrides(
        archive_root=tmp_path,
        generations=(("together", refresh_id),),
        catalog=catalog,
        fact_name="api_access",
    )
    assert report.generations[0].normalization_timestamp_status == "conflict"
    assert report.generations[0].normalization_timestamp is None
    assert set(report.generations[0].distinct_normalization_timestamps) == {
        GENERATED_AT,
        OLDER_STAMP,
    }
    assert any(item.code == "normalization_timestamp_conflict" for item in report.limitations)
    assert "not_comparable" in {item.status for item in report.catalog_checks}


def test_mixed_generation_facts_on_one_record_are_checked_independently(tmp_path: Path) -> None:
    """A retained field must not suppress a fresh fact on the same identity."""
    catalog, refresh_id = _disagreeing_catalog(tmp_path)
    record = next(item for item in catalog.models if item.provider_model_id == "zai-org/Example")
    assert record.api_access.provenance is not None
    retained_api_access = replace(
        record.api_access,
        provenance=replace(record.api_access.provenance, verified_at=OLDER_STAMP),
    )
    catalog = replace(
        catalog,
        models=tuple(
            replace(item, api_access=retained_api_access)
            if item.samyak_id == record.samyak_id
            else item
            for item in catalog.models
        ),
    )
    report = diagnose_overrides(
        archive_root=tmp_path,
        generations=(("together", refresh_id),),
        catalog=catalog,
        model_id=record.samyak_id,
    )
    checks = {item.fact_name: item for item in report.catalog_checks}
    assert checks["api_access"].status == "not_comparable"
    assert checks["api_access"].reason == "retained_older_generation"
    assert checks["lifecycle"].status == "comparable"
    summary = report.to_dict()["catalog_check_summary"]
    assert summary["matching_count"] >= 1
    assert summary["not_comparable_reasons"]["retained_older_generation"] == 1


def test_report_output_exposes_fact_level_coverage_consistently(tmp_path: Path) -> None:
    catalog, refresh_id = _disagreeing_catalog(tmp_path)
    report = diagnose_overrides(
        archive_root=tmp_path,
        generations=(("together", refresh_id),),
        catalog=catalog,
        model_id="together:zai-org/Example",
        fact_name="api_access",
    )
    payload = json.loads(render_override_report(report, output_format="json"))
    summary = payload["catalog_check_summary"]
    assert summary["reproducibility_coverage_denominator"] == len(report.catalog_checks)
    assert summary["matching_count"] == sum(
        item.status == "comparable" for item in report.catalog_checks
    )
    text = render_override_report(report, output_format="text")
    assert "selected-generation reproducibility:" in text
    assert "provider-wide reproducibility: not claimed" in text


def test_historical_generation_is_not_current_mismatch(tmp_path: Path) -> None:
    _write_generation(
        tmp_path,
        OLDER,
        [
            _source("together", SOURCE_ID_DEPRECATIONS, DEPRECATIONS_URL, DEPRECATIONS),
            _source("together", SOURCE_ID_CHANGELOG, CHANGELOG_URL, CHANGELOG),
        ],
    )
    refresh_newer = _write_generation(
        tmp_path,
        NEWER,
        [
            _source("together", SOURCE_ID_DEPRECATIONS, DEPRECATIONS_URL, DEPRECATIONS),
            _source("together", SOURCE_ID_CHANGELOG, CHANGELOG_URL, CHANGELOG),
        ],
    )
    catalog = _catalog_from_bodies(
        (SOURCE_ID_DEPRECATIONS, DEPRECATIONS_URL, DEPRECATIONS),
        (SOURCE_ID_CHANGELOG, CHANGELOG_URL, CHANGELOG),
    )
    report = diagnose_overrides(
        archive_root=tmp_path,
        generations=(("together", OLDER),),
        catalog=catalog,
        model_id="together:zai-org/Example",
        fact_name="api_access",
    )
    assert refresh_id_from_stamp(GENERATED_AT) == refresh_newer
    assert report.generations[0].normalization_timestamp_status == "historical"
    assert report.generations[0].normalization_timestamp is None
    assert report.generations[0].catalog_provider_verified_at == GENERATED_AT
    assert any(item.code == "historical_generation" for item in report.limitations)
    assert {item.status for item in report.catalog_checks} == {"not_comparable"}
    assert {item.reason for item in report.catalog_checks} == {"retained_older_generation"}
    assert report.findings
    assert {item.comparability_status for item in report.findings} == {"not_comparable"}
    assert {item.kind for item in report.findings} == {"not_comparable"}
    payload = report.to_dict()
    summary = payload["catalog_check_summary"]
    assert summary["count"] == 1
    assert summary["not_comparable_reasons"] == {"retained_older_generation": 1}
    assert summary["selected_generation_reproducibility"] == "partial"
    reconstruction = payload["reconstruction"]
    assert reconstruction["scope"] == "direct_archive_reconstruction"
    assert reconstruction["merge_evaluated"] is False
    assert reconstruction["retained_absence_ids"] == []
    assert reconstruction["source_absence_decisions"] == "not_evaluated_no_merge_was_run"
    assert reconstruction["record_count"] == 1
    assert reconstruction["evidence_disposition_counts"]["current_documented"] == 1
    text = render_override_report(report, output_format="text")
    assert "selected-generation reproducibility: partial" in text
    assert "Direct archive reconstruction (not a merged catalog)" in text


def test_reused_blob_hash_does_not_establish_generation_identity(tmp_path: Path) -> None:
    dep_hash = f"sha256:{hashlib.sha256(DEPRECATIONS.encode()).hexdigest()}"
    change_hash = f"sha256:{hashlib.sha256(CHANGELOG.encode()).hexdigest()}"
    sources = [
        _source(
            "together",
            SOURCE_ID_DEPRECATIONS,
            DEPRECATIONS_URL,
            DEPRECATIONS,
            content_hash=dep_hash,
        ),
        _source(
            "together",
            SOURCE_ID_CHANGELOG,
            CHANGELOG_URL,
            CHANGELOG,
            content_hash=change_hash,
        ),
    ]
    _write_generation(tmp_path, OLDER, sources)
    _write_generation(tmp_path, NEWER, sources)
    catalog = _catalog_from_bodies(
        (SOURCE_ID_DEPRECATIONS, DEPRECATIONS_URL, DEPRECATIONS),
        (SOURCE_ID_CHANGELOG, CHANGELOG_URL, CHANGELOG),
    )
    assert catalog.models[0].api_access.provenance is not None
    assert catalog.models[0].api_access.provenance.content_hash == dep_hash
    report = diagnose_overrides(
        archive_root=tmp_path,
        generations=(("together", OLDER),),
        catalog=catalog,
        model_id="together:zai-org/Example",
        fact_name="api_access",
    )
    assert report.generations[0].normalization_timestamp_status == "historical"
    assert {item.status for item in report.catalog_checks} == {"not_comparable"}
    assert {item.reason for item in report.catalog_checks} == {"retained_older_generation"}
    assert report.findings
    assert {item.comparability_status for item in report.findings} == {"not_comparable"}
    assert {item.kind for item in report.findings} == {"not_comparable"}


def test_derived_active_uses_provider_timestamp(tmp_path: Path) -> None:
    body = """# Serverless models

## Chat

| API model string | Model name |
| --- | --- |
| `zai-org/Listed` | Listed |
"""
    refresh_id = _write_generation(
        tmp_path,
        OLDER,
        [_source("together", SOURCE_ID_SERVERLESS, SERVERLESS_URL, body)],
    )
    catalog = _catalog_from_bodies(
        (SOURCE_ID_SERVERLESS, SERVERLESS_URL, body),
        generated_at=OLDER_STAMP,
        verified_at=OLDER_STAMP,
    )
    catalog = _with_global_generated_at(catalog, GLOBAL_LATER)
    report = diagnose_overrides(
        archive_root=tmp_path,
        generations=(("together", refresh_id),),
        catalog=catalog,
        model_id="together:zai-org/Listed",
        fact_name="lifecycle",
    )
    assert report.generations[0].normalization_timestamp == OLDER_STAMP
    derived = [item for item in report.findings if item.kind == "derived"]
    assert len(derived) == 1
    assert derived[0].selected_value == "active"
    assert {item.status for item in report.catalog_checks} == {"comparable"}


def test_same_rank_context_conflict_preserved_with_resolved_timestamp() -> None:
    findings = classify_statements(
        [
            _stated("context_window", 1048576, "together-docs-serverless-models", 2),
            _stated("context_window", 1050000, "together-docs-model-page:kimi-k3-quickstart", 2),
        ]
    )
    assert [item.kind for item in findings] == ["winning_rank_conflict"]
    assert findings[0].selected_value is None
    assert {item.value for item in findings[0].claims} == {1048576, 1050000}


def test_together_current_generation_timestamp_matches(tmp_path: Path) -> None:
    report, _catalog = _diagnose_bodies(
        tmp_path,
        (SOURCE_ID_DEPRECATIONS, DEPRECATIONS_URL, DEPRECATIONS),
        (SOURCE_ID_CHANGELOG, CHANGELOG_URL, CHANGELOG),
        model_id="together:zai-org/Example",
        fact_name="lifecycle",
    )
    assert report.catalog_generated_at == GENERATED_AT
    assert report.generations[0].normalization_timestamp == GENERATED_AT
    assert report.generations[0].normalization_timestamp_status == "resolved"
    assert {item.status for item in report.catalog_checks} == {"comparable"}


def test_openai_date_promotion_uses_provider_timestamp_not_global(tmp_path: Path) -> None:
    body = """# Deprecations

## Upcoming deprecations

### 2026-01-01: Boundary

| Shutdown date | Model / system | Recommended replacement |
| --- | --- | --- |
| 2026-11-03 | `gpt-boundary` | `gpt-next` |
"""
    refresh_id = _write_generation(
        tmp_path,
        OLDER,
        [_source("openai", OPENAI_SOURCE_ID_DEPRECATIONS, OPENAI_DEPRECATIONS_URL, body)],
    )
    catalog = catalog_from_openai_sources(
        (
            capture_openai(
                source_id=OPENAI_SOURCE_ID_DEPRECATIONS,
                source_url=OPENAI_DEPRECATIONS_URL,
                body=body,
                retrieved_at=RETRIEVED_AT,
            ),
        ),
        generated_at=OLDER_STAMP,
        verified_at=OLDER_STAMP,
        overlay=CatalogOverlay.BUNDLED,
    )
    catalog = _with_global_generated_at(catalog, GLOBAL_LATER)
    report = diagnose_overrides(
        archive_root=tmp_path,
        generations=(("openai", refresh_id),),
        catalog=catalog,
        model_id="openai:gpt-boundary",
        fact_name="lifecycle",
    )
    record = next(item for item in catalog.models if item.provider_model_id == "gpt-boundary")
    assert record.lifecycle.value is LifecycleState.DEPRECATED
    assert report.generations[0].normalization_timestamp == OLDER_STAMP
    assert report.catalog_generated_at == GLOBAL_LATER
    assert {item.status for item in report.catalog_checks} == {"comparable"}
    assert {item.selected_value for item in report.findings if item.selected_value is not None} <= {
        "deprecated"
    }


def test_invalid_timezone_timestamp_is_not_used(tmp_path: Path) -> None:
    catalog, refresh_id = _disagreeing_catalog(tmp_path)
    invalid_catalog = _with_forced_verified_at(catalog, "2026-11-02T06:20:15.685704")
    report = diagnose_overrides(
        archive_root=tmp_path,
        generations=(("together", refresh_id),),
        catalog=invalid_catalog,
        model_id="together:zai-org/Example",
        fact_name="api_access",
    )
    assert report.generations[0].normalization_timestamp_status == "invalid"
    assert report.generations[0].normalization_timestamp is None
    assert any(item.code == "normalization_timestamp_invalid" for item in report.limitations)
    assert {item.status for item in report.catalog_checks} == {"not_comparable"}


def test_timestamp_resolution_output_is_deterministic(tmp_path: Path) -> None:
    catalog, refresh_id = _disagreeing_catalog(tmp_path)
    first = diagnose_overrides(
        archive_root=tmp_path,
        generations=(("together", refresh_id),),
        catalog=catalog,
        model_id="together:zai-org/Example",
        fact_name="api_access",
    )
    second = diagnose_overrides(
        archive_root=tmp_path,
        generations=(("together", refresh_id),),
        catalog=catalog,
        model_id="together:zai-org/Example",
        fact_name="api_access",
    )
    assert render_override_report(first, output_format="json") == render_override_report(
        second, output_format="json"
    )


def test_missing_selection_does_not_choose_a_generation(tmp_path: Path, capsys) -> None:
    _write_generation(
        tmp_path,
        NEWER,
        [_source("together", SOURCE_ID_DEPRECATIONS, DEPRECATIONS_URL, DEPRECATIONS)],
    )
    code = main(["model", "diagnose-overrides", "--archive-root", str(tmp_path), "--no-catalog"])
    captured = capsys.readouterr()
    assert code == 2
    assert "select a provider generation" in captured.err
    assert NEWER not in captured.out


def _run(root: Path, *extra: str) -> tuple[int, str, str]:
    _write_generation(
        root,
        NEWER,
        [
            _source("together", SOURCE_ID_DEPRECATIONS, DEPRECATIONS_URL, DEPRECATIONS),
            _source("together", SOURCE_ID_CHANGELOG, CHANGELOG_URL, CHANGELOG),
        ],
    )
    stdout = StringIO()
    stderr = StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = main(
            [
                "model",
                "diagnose-overrides",
                "--provider",
                "together",
                "--refresh-id",
                NEWER,
                "--archive-root",
                str(root),
                "--no-catalog",
                *extra,
            ]
        )
    return code, stdout.getvalue(), stderr.getvalue()


def _write_pair(root: Path) -> None:
    _write_generation(
        root,
        NEWER,
        [_source("together", SOURCE_ID_DEPRECATIONS, DEPRECATIONS_URL, DEPRECATIONS)],
    )
    _write_generation(
        root,
        OLDER,
        [
            _source(
                "fireworks",
                "fireworks-docs-changelog",
                "https://docs.fireworks.ai/changelog.md",
                b"# Changelog\n",
            )
        ],
    )


def _diagnose_bodies(
    root: Path,
    *sources: tuple[str, str, str],
    model_id: str,
    fact_name: str,
):
    catalog, refresh_id = _bodies_catalog(root, *sources)
    report = diagnose_overrides(
        archive_root=root,
        generations=(("together", refresh_id),),
        catalog=catalog,
        model_id=model_id,
        fact_name=fact_name,
    )
    return report, catalog


def _bodies_catalog(root: Path, *sources: tuple[str, str, str]):
    refresh_id = _write_generation(
        root,
        NEWER,
        [_source("together", source_id, url, body) for source_id, url, body in sources],
    )
    catalog = _catalog_from_bodies(*sources)
    return catalog, refresh_id


def _disagreeing_catalog(root: Path):
    refresh_id = _write_generation(
        root,
        NEWER,
        [
            _source("together", SOURCE_ID_DEPRECATIONS, DEPRECATIONS_URL, DEPRECATIONS),
            _source("together", SOURCE_ID_CHANGELOG, CHANGELOG_URL, CHANGELOG),
        ],
    )
    catalog = _catalog_from_bodies(
        (SOURCE_ID_DEPRECATIONS, DEPRECATIONS_URL, DEPRECATIONS),
        (SOURCE_ID_CHANGELOG, CHANGELOG_URL, CHANGELOG),
    )
    return catalog, refresh_id


def _catalog_from_bodies(
    *sources: tuple[str, str, str],
    generated_at: str = GENERATED_AT,
    verified_at: str = GENERATED_AT,
):
    captured = tuple(
        captured_markdown(
            source_id=source_id,
            source_url=url,
            body=body,
            retrieved_at=RETRIEVED_AT,
        )
        for source_id, url, body in sources
    )
    return catalog_from_together_sources(
        captured,
        generated_at=generated_at,
        verified_at=verified_at,
        overlay=CatalogOverlay.BUNDLED,
    )


def _with_global_generated_at(catalog, generated_at: str):
    return replace(
        catalog,
        generated_at=generated_at,
        freshness=replace(catalog.freshness, generated_at=generated_at),
    )


def _strip_all_verified_at(catalog):
    models = []
    for record in catalog.models:
        fields = {}
        for name in (
            "display_name",
            "aliases",
            "resolves_to",
            "family",
            "lifecycle",
            "deprecated_at",
            "retirement_at",
            "replacement",
            "context_window",
            "max_input_tokens",
            "max_output_tokens",
            "input_modalities",
            "output_modalities",
            "tool_calling",
            "structured_output",
            "api_access",
            "availability_scope",
            "regions",
        ):
            fact = getattr(record, name)
            if fact.provenance is None and not fact.claims:
                continue
            provenance = None
            if fact.provenance is not None:
                provenance = replace(fact.provenance, verified_at=None)
            claims = tuple(
                replace(claim, provenance=replace(claim.provenance, verified_at=None))
                for claim in fact.claims
            )
            fields[name] = replace(fact, provenance=provenance, claims=claims)
        models.append(replace(record, **fields))
    return replace(catalog, models=tuple(models))


def _with_forced_verified_at(catalog, stamp: str):
    from samyak.model.facts import Provenance

    def force(prov: Provenance) -> Provenance:
        forced = object.__new__(Provenance)
        for field_name in Provenance.__slots__:
            object.__setattr__(
                forced,
                field_name,
                stamp if field_name == "verified_at" else getattr(prov, field_name),
            )
        return forced

    models = []
    for record in catalog.models:
        fields = {}
        for name in (
            "display_name",
            "aliases",
            "resolves_to",
            "family",
            "lifecycle",
            "deprecated_at",
            "retirement_at",
            "replacement",
            "context_window",
            "max_input_tokens",
            "max_output_tokens",
            "input_modalities",
            "output_modalities",
            "tool_calling",
            "structured_output",
            "api_access",
            "availability_scope",
            "regions",
        ):
            fact = getattr(record, name)
            if fact.provenance is None and not fact.claims:
                continue
            provenance = None if fact.provenance is None else force(fact.provenance)
            claims = tuple(
                replace(claim, provenance=force(claim.provenance)) for claim in fact.claims
            )
            fields[name] = replace(fact, provenance=provenance, claims=claims)
        models.append(replace(record, **fields))
    return replace(catalog, models=tuple(models))


def _source(
    provider: str,
    source_id: str,
    url: str,
    body: str | bytes,
    *,
    blob: bytes | None = None,
    content_hash: str | None = None,
    byte_length: int | None = None,
    skip_blob: bool = False,
) -> dict[str, object]:
    payload = body.encode() if isinstance(body, str) else body
    return {
        "provider": provider,
        "source_id": source_id,
        "url": url,
        "body": payload,
        "blob": payload if blob is None else blob,
        "content_hash": content_hash,
        "byte_length": byte_length,
        "skip_blob": skip_blob,
    }


def _write_generation(root: Path, refresh_id: str, sources: list[dict[str, object]]) -> str:
    entries: list[dict[str, object]] = []
    for source in sources:
        body = source["body"]
        assert isinstance(body, bytes)
        stored = source["blob"]
        assert isinstance(stored, bytes)
        digest = hashlib.sha256(body).hexdigest()
        content_hash = source["content_hash"] or f"sha256:{digest}"
        assert isinstance(content_hash, str)
        if not source["skip_blob"]:
            path = blob_path(root, content_hash)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(stored)
        length = source["byte_length"] if source["byte_length"] is not None else len(stored)
        entries.append(
            {
                "provider_id": source["provider"],
                "source_id": source["source_id"],
                "source_url": source["url"],
                "content_hash": content_hash,
                "retrieved_at": RETRIEVED_AT,
                "media_type": "text/markdown",
                "byte_length": length,
                "package_version": "0.3.0",
            }
        )
    directory = root / "refreshes" / refresh_id
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "refresh_id": refresh_id,
        "complete": True,
        "package_version": "0.3.0",
        "retention_keep": 5,
        "sources": entries,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return refresh_id


def _codes(report, provider: str) -> set[str]:
    return {
        item.code
        for item in report.limitations
        if item.provider == provider and item.code != "provider_not_evaluated"
    }
