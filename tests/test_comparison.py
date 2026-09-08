"""Behavioral tests for saved-run comparison semantics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from samyak.comparison.compare import compare_runs, earlier_run
from samyak.comparison.errors import IncompatibleRunsError
from samyak.comparison.models import (
    CORPUS_CROSS,
    CORPUS_SAME_LOCATION,
    CORPUS_SAME_NAME_UNCONFIRMED,
    STATUS_CHANGED,
    STATUS_NEW,
    STATUS_NO_LONGER_DETECTED,
    STATUS_UNCHANGED,
)
from samyak.corpus.models import AnalysisReport, CorpusSummary, Finding, Severity
from samyak.server.compare_page import render_comparison_page
from samyak.store.models import StoredRun, metadata_from_report

_T0 = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)


def _finding(**overrides: object) -> Finding:
    data: dict[str, object] = {
        "code": "EMPTY_DOCUMENTS",
        "category": "size",
        "severity": Severity.HIGH,
        "title": "Empty documents",
        "message": "1 empty document.",
        "why_it_matters": "Empty files add no retrieval value.",
        "recommendation": "Remove or replace empty files.",
        "evidence": {"count": 1, "affected_document_count": 1},
        "affected_documents": ("empty.txt",),
    }
    data.update(overrides)
    return Finding(**data)  # type: ignore[arg-type]


def _summary(**overrides: object) -> CorpusSummary:
    data: dict[str, object] = {
        "corpus_root": "docs",
        "total_discovered_files": 3,
        "supported_files": 3,
        "unsupported_files": 0,
        "analyzed_documents": 3,
        "load_errors": 0,
        "total_characters": 300,
        "total_bytes": 300,
        "average_characters": 100.0,
        "median_characters": 100.0,
        "min_characters": 80,
        "max_characters": 120,
    }
    data.update(overrides)
    return CorpusSummary(**data)  # type: ignore[arg-type]


def _report(
    *,
    findings: tuple[Finding, ...] = (),
    summary: CorpusSummary | None = None,
    **overrides: object,
) -> AnalysisReport:
    data: dict[str, object] = {
        "product": "samyak",
        "capability": "corpus",
        "version": "0.1.0",
        "summary": summary or _summary(),
        "findings": findings,
        "config": {"small_document_chars": 100},
    }
    data.update(overrides)
    return AnalysisReport(**data)  # type: ignore[arg-type]


def _stored(
    report: AnalysisReport,
    *,
    run_id: str,
    created_at: datetime,
    corpus_path: str | None = "/tmp/example/docs",
    corpus_label: str | None = "docs",
) -> StoredRun:
    stamp = created_at.isoformat()
    metadata = metadata_from_report(
        run_id=run_id,
        report=report,
        created_at=stamp,
        completed_at=stamp,
        duration_seconds=1.0,
        corpus_label=corpus_label,
        corpus_path=corpus_path,
    )
    return StoredRun(metadata=metadata, report=report)


def _pair(
    left: AnalysisReport, right: AnalysisReport, **kwargs: object
) -> tuple[StoredRun, StoredRun]:
    older = _stored(
        left,
        run_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        created_at=_T0,
        **kwargs,
    )
    newer = _stored(
        right,
        run_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        created_at=_T0 + timedelta(minutes=5),
        **kwargs,
    )
    return older, newer


def test_identical_reports_are_unchanged() -> None:
    finding = _finding()
    baseline, current = _pair(_report(findings=(finding,)), _report(findings=(finding,)))
    result = compare_runs(baseline, current)
    assert result.finding_counts == (0, 0, 0, 1)
    assert result.unchanged[0].code == "EMPTY_DOCUMENTS"
    assert result.unchanged[0].status == STATUS_UNCHANGED
    assert result.corpus.kind == CORPUS_SAME_LOCATION
    assert result.config_changes == ()
    assert all(not item.changed for item in result.metrics)
    html = render_comparison_page(result)
    assert "No longer detected: 0" in html
    assert "New: 0" in html
    assert "Unchanged: 1" in html
    assert "Baseline" in html
    assert "Current" in html
    assert "Older run" not in html
    assert "Resolved" not in html


def test_new_finding() -> None:
    baseline, current = _pair(_report(), _report(findings=(_finding(),)))
    result = compare_runs(baseline, current)
    assert result.finding_counts == (1, 0, 0, 0)
    assert result.new[0].status == STATUS_NEW
    html = render_comparison_page(result)
    assert "New findings" in html
    assert "EMPTY_DOCUMENTS" in html


def test_finding_no_longer_detected() -> None:
    baseline, current = _pair(_report(findings=(_finding(),)), _report())
    result = compare_runs(baseline, current)
    assert result.finding_counts == (0, 1, 0, 0)
    assert result.no_longer_detected[0].status == STATUS_NO_LONGER_DETECTED
    html = render_comparison_page(result)
    assert "No longer detected" in html
    assert "Resolved" not in html


def test_changed_severity() -> None:
    baseline, current = _pair(
        _report(findings=(_finding(severity=Severity.MEDIUM),)),
        _report(findings=(_finding(severity=Severity.HIGH),)),
    )
    result = compare_runs(baseline, current)
    match = result.changed[0]
    assert match.status == STATUS_CHANGED
    assert (match.changes[0].before, match.changes[0].after) == ("MEDIUM", "HIGH")
    html = render_comparison_page(result)
    assert "MEDIUM → HIGH" in html


def test_changed_title_and_message() -> None:
    baseline, current = _pair(
        _report(findings=(_finding(title="Old title", message="old msg"),)),
        _report(findings=(_finding(title="New title", message="new msg"),)),
    )
    fields = {change.field: change for change in compare_runs(baseline, current).changed[0].changes}
    assert fields["title"].after == "New title"
    assert fields["message"].after == "new msg"


def test_changed_recommendation_and_evidence() -> None:
    older = _finding(recommendation="Remove files.", evidence={"count": 1, "threshold": 0.2})
    newer = _finding(recommendation="Rewrite files.", evidence={"count": 3, "threshold": 0.5})
    result = compare_runs(*_pair(_report(findings=(older,)), _report(findings=(newer,))))
    fields = {change.field: change for change in result.changed[0].changes}
    assert fields["recommendation"].before == "Remove files."
    assert fields["evidence.count"].after == "3"


def test_changed_affected_documents() -> None:
    older = _finding(
        affected_documents=("a.txt", "b.txt"),
        evidence={"affected_document_count": 2},
    )
    newer = _finding(
        affected_documents=("b.txt", "c.txt", "d.txt"),
        evidence={"affected_document_count": 3},
    )
    change = (
        compare_runs(*_pair(_report(findings=(older,)), _report(findings=(newer,))))
        .changed[0]
        .changes[0]
    )
    assert change.field == "affected documents"
    assert change.added_paths == ("c.txt", "d.txt")
    assert change.removed_paths == ("a.txt",)


def test_reordered_findings_remain_unchanged() -> None:
    empty = _finding()
    load = _finding(code="LOAD_ERRORS", title="Load errors", message="1 load error.")
    left = _report(findings=(empty, load))
    right = _report(findings=(load, empty))
    result = compare_runs(*_pair(left, right))
    assert result.finding_counts == (0, 0, 0, 2)
    assert [item.code for item in result.unchanged] == ["EMPTY_DOCUMENTS", "LOAD_ERRORS"]


def test_evidence_list_order_only_is_unchanged() -> None:
    older = _finding(evidence={"count": 2, "tags": ["b", "a"]})
    newer = _finding(evidence={"count": 2, "tags": ["a", "b"]})
    result = compare_runs(*_pair(_report(findings=(older,)), _report(findings=(newer,))))
    assert result.finding_counts == (0, 0, 0, 1)


def test_findings_matched_by_code_not_title() -> None:
    result = compare_runs(
        *_pair(
            _report(findings=(_finding(title="Old title"),)),
            _report(findings=(_finding(title="New title"),)),
        )
    )
    assert result.changed[0].code == "EMPTY_DOCUMENTS"


def test_different_codes_are_not_the_same_finding() -> None:
    result = compare_runs(
        *_pair(
            _report(findings=(_finding(code="EMPTY_DOCUMENTS"),)),
            _report(findings=(_finding(code="LOAD_ERRORS", title="Load errors"),)),
        )
    )
    assert [item.code for item in result.no_longer_detected] == ["EMPTY_DOCUMENTS"]
    assert [item.code for item in result.new] == ["LOAD_ERRORS"]


def test_duplicate_codes_are_rejected() -> None:
    dupes = (
        _finding(title="First"),
        _finding(title="Second", affected_documents=("b.txt",)),
    )
    with pytest.raises(IncompatibleRunsError, match="duplicate finding codes"):
        compare_runs(*_pair(_report(findings=dupes), _report(findings=(_finding(),))))


def test_inventory_metric_changes() -> None:
    result = compare_runs(
        *_pair(
            _report(
                summary=_summary(
                    total_discovered_files=2,
                    supported_files=2,
                    unsupported_files=0,
                    analyzed_documents=2,
                    load_errors=0,
                    discovery_errors=1,
                )
            ),
            _report(
                summary=_summary(
                    total_discovered_files=9,
                    supported_files=7,
                    unsupported_files=2,
                    analyzed_documents=8,
                    load_errors=2,
                    discovery_errors=0,
                )
            ),
        )
    )
    by_label = {item.label: item for item in result.metrics}
    assert (by_label["Files discovered"].before, by_label["Files discovered"].after) == (2, 9)
    assert (by_label["Supported files"].before, by_label["Supported files"].after) == (2, 7)
    assert (by_label["Unsupported files"].before, by_label["Unsupported files"].after) == (0, 2)
    assert (by_label["Files analyzed"].before, by_label["Files analyzed"].after) == (2, 8)


def test_config_change_shows_values() -> None:
    baseline, current = _pair(
        _report(config={"small_document_chars": 200, "flag": False, "ratio": 0.0}),
        _report(config={"small_document_chars": 500, "flag": True, "ratio": 0.0}),
    )
    result = compare_runs(baseline, current)
    assert [item.key for item in result.config_changes] == ["flag", "small_document_chars"]
    by_key = {item.key: item for item in result.config_changes}
    assert (by_key["small_document_chars"].before, by_key["small_document_chars"].after) == (
        "200",
        "500",
    )
    assert (by_key["flag"].before, by_key["flag"].after) == ("false", "true")
    html = render_comparison_page(result)
    assert "200 → 500" in html
    assert "false → true" in html
    assert "threshold changes rather than from the corpus" in html
    assert "<code>ratio</code>" not in html


def test_no_config_changes() -> None:
    result = compare_runs(*_pair(_report(), _report()))
    assert result.config_changes == ()
    assert "Analysis configuration changed" not in render_comparison_page(result)


def test_config_change_ordering_is_deterministic() -> None:
    left = _report(config={"z_key": 1, "a_key": 1, "m_key": 1})
    right = _report(config={"z_key": 2, "a_key": 2, "m_key": 2})
    first = compare_runs(*_pair(left, right))
    second = compare_runs(*_pair(left, right))
    assert [item.key for item in first.config_changes] == ["a_key", "m_key", "z_key"]
    assert first.config_changes == second.config_changes


def test_cross_corpus_is_allowed_and_labelled() -> None:
    older = _stored(
        _report(),
        run_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        created_at=_T0,
        corpus_path="/tmp/one",
        corpus_label="docs",
    )
    newer = _stored(
        _report(),
        run_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        created_at=_T0 + timedelta(seconds=1),
        corpus_path="/tmp/two",
        corpus_label="docs",
    )
    result = compare_runs(older, newer)
    assert result.corpus.kind == CORPUS_CROSS
    html = render_comparison_page(result)
    assert "Different saved corpus locations" in html
    assert "/tmp/one" in html
    assert "<script>" not in html


def test_matching_basename_without_paths_is_unconfirmed() -> None:
    result = compare_runs(
        _stored(
            _report(),
            run_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            created_at=_T0,
            corpus_path=None,
            corpus_label="docs",
        ),
        _stored(
            _report(),
            run_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            created_at=_T0 + timedelta(seconds=1),
            corpus_path=None,
            corpus_label="docs",
        ),
    )
    assert result.corpus.kind == CORPUS_SAME_NAME_UNCONFIRMED


def test_incompatible_schema() -> None:
    left, right = _pair(_report(), _report())
    right = StoredRun(
        metadata=right.metadata.__class__(
            **{**right.metadata.to_dict(), "report_schema_version": 2}
        ),
        report=right.report,
    )
    with pytest.raises(IncompatibleRunsError, match="schema"):
        compare_runs(left, right)


def test_incompatible_capability() -> None:
    with pytest.raises(IncompatibleRunsError, match="capabilities"):
        compare_runs(*_pair(_report(), _report(capability="other")))


def test_same_run_rejected() -> None:
    run = _stored(_report(), run_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", created_at=_T0)
    with pytest.raises(IncompatibleRunsError, match="different"):
        compare_runs(run, run)


def test_compare_runs_uses_caller_baseline() -> None:
    finding = _finding()
    older, newer = _pair(_report(findings=(finding,)), _report())
    result = compare_runs(newer, older)
    assert result.baseline.metadata.run_id == newer.metadata.run_id
    assert result.new[0].status == STATUS_NEW


def test_earlier_run_orders_by_timestamp() -> None:
    older, newer = _pair(_report(), _report())
    first, second = earlier_run(newer, older)
    assert first.metadata.run_id == older.metadata.run_id
    assert second.metadata.run_id == newer.metadata.run_id


def test_comparison_output_is_deterministic() -> None:
    findings = (
        _finding(code="LOAD_ERRORS", title="Load errors"),
        _finding(),
    )
    left, right = _pair(_report(findings=findings), _report(findings=tuple(reversed(findings))))
    first = compare_runs(left, right)
    second = compare_runs(left, right)
    assert first.unchanged == second.unchanged
    assert [item.code for item in first.unchanged] == ["EMPTY_DOCUMENTS", "LOAD_ERRORS"]
    assert render_comparison_page(first) == render_comparison_page(second)
