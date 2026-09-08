"""Compare two reconstructed saved runs using AnalysisReport semantics.

``analyze_corpus`` emits at most one finding per ``Finding.code``. Comparison
matches findings by that code. Duplicate codes in a reconstructed snapshot are
rejected rather than paired by list position.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from samyak.comparison.errors import IncompatibleRunsError
from samyak.comparison.models import (
    CORPUS_CROSS,
    CORPUS_DIFFERENT_NAME_UNCONFIRMED,
    CORPUS_SAME_LOCATION,
    CORPUS_SAME_NAME_UNCONFIRMED,
    STATUS_CHANGED,
    STATUS_NEW,
    STATUS_NO_LONGER_DETECTED,
    STATUS_UNCHANGED,
    ConfigChange,
    CorpusRelation,
    FieldChange,
    FindingMatch,
    MetricChange,
    RunComparison,
)
from samyak.corpus.models import AnalysisReport, Finding
from samyak.store.models import StoredRun

_EVIDENCE_SHOWN_VIA_AFFECTED = frozenset(
    {
        "sample_paths",
        "affected_document_count",
        "affected_documents_truncated",
    }
)


def compare_runs(baseline: StoredRun, current: StoredRun) -> RunComparison:
    """Compare *current* against *baseline*.

    Deltas are baseline → current. The caller chooses which run is the
    baseline. Findings are matched by unique ``Finding.code``.
    """
    _assert_compatible(baseline, current)
    return RunComparison(
        baseline=baseline,
        current=current,
        corpus=_corpus_relation(baseline, current),
        version_note=_version_note(baseline, current),
        config_changes=_config_changes(baseline.report, current.report),
        metrics=_metrics(baseline.report, current.report),
        **_finding_groups(baseline.report.findings, current.report.findings),
    )


def _assert_compatible(baseline: StoredRun, current: StoredRun) -> None:
    if baseline.metadata.run_id == current.metadata.run_id:
        raise IncompatibleRunsError("Select two different runs to compare.")
    if baseline.metadata.report_schema_version != current.metadata.report_schema_version:
        raise IncompatibleRunsError(
            "These runs use different report schemas and cannot be compared."
        )
    if baseline.report.product != current.report.product:
        raise IncompatibleRunsError(
            "These runs are not Samyak corpus reports and cannot be compared."
        )
    if baseline.report.capability != current.report.capability:
        raise IncompatibleRunsError(
            "These runs use different analysis capabilities and cannot be compared."
        )


def _version_note(baseline: StoredRun, current: StoredRun) -> str | None:
    left = baseline.report.version
    right = current.report.version
    if left == right:
        return None
    return f"These runs were produced by different Samyak versions ({left} → {right})."


def _config_changes(left: AnalysisReport, right: AnalysisReport) -> tuple[ConfigChange, ...]:
    keys = sorted(set(left.config) | set(right.config), key=str)
    changes: list[ConfigChange] = []
    for key in keys:
        before = left.config.get(key)
        after = right.config.get(key)
        if before != after:
            changes.append(ConfigChange(key, _fmt_value(before), _fmt_value(after)))
    return tuple(changes)


def _corpus_relation(baseline: StoredRun, current: StoredRun) -> CorpusRelation:
    left_path = baseline.metadata.corpus_path
    right_path = current.metadata.corpus_path
    left_label = baseline.metadata.corpus_label or baseline.report.summary.corpus_root
    right_label = current.metadata.corpus_label or current.report.summary.corpus_root
    if left_path and right_path:
        kind = CORPUS_SAME_LOCATION if left_path == right_path else CORPUS_CROSS
    elif left_label == right_label:
        kind = CORPUS_SAME_NAME_UNCONFIRMED
    else:
        kind = CORPUS_DIFFERENT_NAME_UNCONFIRMED
    return CorpusRelation(
        kind=kind,
        left_label=left_label,
        right_label=right_label,
        left_path=left_path,
        right_path=right_path,
    )


def _metrics(left: AnalysisReport, right: AnalysisReport) -> tuple[MetricChange, ...]:
    ls = left.summary
    rs = right.summary
    rows = [
        MetricChange("Files discovered", ls.total_discovered_files, rs.total_discovered_files),
        MetricChange("Supported files", ls.supported_files, rs.supported_files),
        MetricChange("Unsupported files", ls.unsupported_files, rs.unsupported_files),
        MetricChange("Files analyzed", ls.analyzed_documents, rs.analyzed_documents),
        MetricChange("Findings", len(left.findings), len(right.findings)),
        MetricChange("Load errors", ls.load_errors, rs.load_errors),
        MetricChange("Discovery errors", ls.discovery_errors, rs.discovery_errors),
    ]
    left_sev = left.severity_counts()
    right_sev = right.severity_counts()
    for severity in ("HIGH", "MEDIUM", "LOW", "INFO"):
        rows.append(
            MetricChange(
                f"{severity} findings",
                left_sev.get(severity, 0),
                right_sev.get(severity, 0),
            )
        )
    return tuple(rows)


def _finding_groups(
    before: tuple[Finding, ...],
    after: tuple[Finding, ...],
) -> dict[str, tuple[FindingMatch, ...]]:
    left_map = _index_by_code(before)
    right_map = _index_by_code(after)
    codes = sorted(set(left_map) | set(right_map))
    new: list[FindingMatch] = []
    gone: list[FindingMatch] = []
    changed: list[FindingMatch] = []
    unchanged: list[FindingMatch] = []
    for code in codes:
        left_item = left_map.get(code)
        right_item = right_map.get(code)
        if left_item is None and right_item is not None:
            new.append(FindingMatch(code=code, status=STATUS_NEW, before=None, after=right_item))
        elif left_item is not None and right_item is None:
            gone.append(
                FindingMatch(
                    code=code,
                    status=STATUS_NO_LONGER_DETECTED,
                    before=left_item,
                    after=None,
                )
            )
        elif left_item is not None and right_item is not None:
            match = _pair_findings(left_item, right_item)
            if match.status == STATUS_CHANGED:
                changed.append(match)
            else:
                unchanged.append(match)
    return {
        "new": tuple(_sort_matches(new)),
        "no_longer_detected": tuple(_sort_matches(gone)),
        "changed": tuple(_sort_matches(changed)),
        "unchanged": tuple(_sort_matches(unchanged)),
    }


def _index_by_code(findings: tuple[Finding, ...]) -> dict[str, Finding]:
    indexed: dict[str, Finding] = {}
    for finding in findings:
        if finding.code in indexed:
            raise IncompatibleRunsError(
                "A saved report contains duplicate finding codes and cannot be compared."
            )
        indexed[finding.code] = finding
    return indexed


def _pair_findings(before: Finding, after: Finding) -> FindingMatch:
    changes = _field_changes(before, after)
    status = STATUS_CHANGED if changes else STATUS_UNCHANGED
    return FindingMatch(
        code=before.code,
        status=status,
        before=before,
        after=after,
        changes=changes,
    )


def _field_changes(before: Finding, after: Finding) -> tuple[FieldChange, ...]:
    changes: list[FieldChange] = []
    if before.severity != after.severity:
        changes.append(FieldChange("severity", before.severity.value, after.severity.value))
    for name in ("title", "message", "why_it_matters", "recommendation"):
        left = getattr(before, name)
        right = getattr(after, name)
        if left != right:
            changes.append(FieldChange(name.replace("_", " "), left, right))
    left_paths = set(before.affected_documents)
    right_paths = set(after.affected_documents)
    left_count = _affected_count(before)
    right_count = _affected_count(after)
    if left_count != right_count or left_paths != right_paths:
        changes.append(
            FieldChange(
                "affected documents",
                str(left_count),
                str(right_count),
                added_paths=tuple(sorted(right_paths - left_paths)),
                removed_paths=tuple(sorted(left_paths - right_paths)),
            )
        )
    changes.extend(_evidence_changes(before.evidence, after.evidence))
    return tuple(changes)


def _affected_count(finding: Finding) -> int:
    raw = finding.evidence.get("affected_document_count")
    if isinstance(raw, bool) or not isinstance(raw, int):
        return len(finding.affected_documents)
    return raw


def _evidence_changes(left: dict[str, Any], right: dict[str, Any]) -> list[FieldChange]:
    keys = sorted((set(left) | set(right)) - _EVIDENCE_SHOWN_VIA_AFFECTED, key=str)
    changes: list[FieldChange] = []
    for key in keys:
        if key not in left:
            changes.append(FieldChange(f"evidence.{key}", "(missing)", _fmt_value(right[key])))
            continue
        if key not in right:
            changes.append(FieldChange(f"evidence.{key}", _fmt_value(left[key]), "(missing)"))
            continue
        if not _values_equal(left[key], right[key]):
            changes.append(
                FieldChange(f"evidence.{key}", _fmt_value(left[key]), _fmt_value(right[key]))
            )
    return changes


def _values_equal(left: object, right: object) -> bool:
    return _canonical(left) == _canonical(right)


def _canonical(value: object) -> object:
    """Order-insensitive form for evidence collections."""
    if isinstance(value, dict):
        return tuple((str(key), _canonical(value[key])) for key in sorted(value, key=str))
    if isinstance(value, list | tuple):
        items = [_canonical(item) for item in value]
        return tuple(sorted(items, key=repr))
    return value


def _fmt_value(value: object) -> str:
    if value is None:
        return "(missing)"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float | str):
        text = str(value)
        if len(text) > 240:
            return text[:237] + "..."
        return text
    if isinstance(value, list | tuple):
        if not value:
            return "none"
        items = [_fmt_value(item) for item in value]
        items.sort()
        if len(items) <= 12:
            return ", ".join(items)
        return f"{len(items)} values"
    if isinstance(value, dict):
        return f"{len(value)} fields"
    return "changed"


def _sort_matches(matches: list[FindingMatch]) -> list[FindingMatch]:
    return sorted(matches, key=lambda item: (item.code, item.title))


def earlier_run(first: StoredRun, second: StoredRun) -> tuple[StoredRun, StoredRun]:
    """Return (earlier, later) by created_at, then run_id. Used by the viewer form."""
    first_key = (_created_epoch(first.metadata.created_at), first.metadata.run_id)
    second_key = (_created_epoch(second.metadata.created_at), second.metadata.run_id)
    if first_key <= second_key:
        return first, second
    return second, first


def _created_epoch(created_at: str) -> float:
    parsed = datetime.fromisoformat(created_at)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()
