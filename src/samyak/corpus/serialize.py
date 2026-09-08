"""Internal AnalysisReport JSON reconstruction.

Not part of the public Samyak API. Persisted ``report.json`` files use the same
dictionary produced by ``AnalysisReport.to_dict()`` (report schema version 1).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from samyak.corpus.models import AnalysisReport, CorpusSummary, Finding, Severity

REPORT_SCHEMA_VERSION = 1
SUPPORTED_REPORT_SCHEMA_VERSIONS = frozenset({REPORT_SCHEMA_VERSION})


class ReportDecodeError(ValueError):
    """Raised when persisted report JSON cannot be reconstructed."""


def report_from_dict(data: Mapping[str, Any]) -> AnalysisReport:
    """Rebuild an :class:`AnalysisReport` from ``AnalysisReport.to_dict()`` output."""
    if not isinstance(data, Mapping):
        raise ReportDecodeError("report JSON must be an object")

    product = _require_str("product", data.get("product"))
    capability = _require_str("capability", data.get("capability"))
    version = _require_str("version", data.get("version"))
    config = data.get("config")
    if not isinstance(config, dict):
        raise ReportDecodeError("config must be an object")
    summary_raw = data.get("summary")
    if not isinstance(summary_raw, Mapping):
        raise ReportDecodeError("summary must be an object")
    findings_raw = data.get("findings")
    if not isinstance(findings_raw, list):
        raise ReportDecodeError("findings must be an array")

    findings = tuple(_finding_from_dict(item, index) for index, item in enumerate(findings_raw))
    codes = [finding.code for finding in findings]
    if len(codes) != len(set(codes)):
        raise ReportDecodeError("findings must have unique codes")
    return AnalysisReport(
        product=product,
        capability=capability,
        version=version,
        summary=_summary_from_dict(summary_raw),
        findings=findings,
        config=dict(config),
    )


def _summary_from_dict(data: Mapping[str, Any]) -> CorpusSummary:
    return CorpusSummary(
        corpus_root=_require_str("summary.corpus_root", data.get("corpus_root")),
        total_discovered_files=_require_int(
            "summary.total_discovered_files", data.get("total_discovered_files")
        ),
        supported_files=_require_int("summary.supported_files", data.get("supported_files")),
        unsupported_files=_require_int("summary.unsupported_files", data.get("unsupported_files")),
        analyzed_documents=_require_int(
            "summary.analyzed_documents", data.get("analyzed_documents")
        ),
        load_errors=_require_int("summary.load_errors", data.get("load_errors")),
        total_characters=_require_int("summary.total_characters", data.get("total_characters")),
        total_bytes=_require_int("summary.total_bytes", data.get("total_bytes")),
        average_characters=_require_float(
            "summary.average_characters", data.get("average_characters")
        ),
        median_characters=_require_float(
            "summary.median_characters", data.get("median_characters")
        ),
        min_characters=_optional_int("summary.min_characters", data.get("min_characters")),
        max_characters=_optional_int("summary.max_characters", data.get("max_characters")),
        unsupported_by_extension=_str_int_map(
            "summary.unsupported_by_extension", data.get("unsupported_by_extension", {})
        ),
        analyzed_by_extension=_str_int_map(
            "summary.analyzed_by_extension", data.get("analyzed_by_extension", {})
        ),
        load_error_paths=_str_tuple("summary.load_error_paths", data.get("load_error_paths", ())),
        discovery_errors=_require_int("summary.discovery_errors", data.get("discovery_errors", 0)),
        discovery_error_paths=_str_tuple(
            "summary.discovery_error_paths", data.get("discovery_error_paths", ())
        ),
    )


def _finding_from_dict(data: object, index: int) -> Finding:
    if not isinstance(data, Mapping):
        raise ReportDecodeError(f"findings[{index}] must be an object")
    prefix = f"findings[{index}]"
    severity_raw = _require_str(f"{prefix}.severity", data.get("severity"))
    try:
        severity = Severity(severity_raw)
    except ValueError as exc:
        raise ReportDecodeError(f"{prefix}.severity is not a known severity") from exc
    evidence = data.get("evidence", {})
    if not isinstance(evidence, dict):
        raise ReportDecodeError(f"{prefix}.evidence must be an object")
    return Finding(
        code=_require_str(f"{prefix}.code", data.get("code")),
        category=_require_str(f"{prefix}.category", data.get("category")),
        severity=severity,
        title=_require_str(f"{prefix}.title", data.get("title")),
        message=_require_str(f"{prefix}.message", data.get("message")),
        why_it_matters=_require_str(f"{prefix}.why_it_matters", data.get("why_it_matters")),
        recommendation=_require_str(f"{prefix}.recommendation", data.get("recommendation")),
        evidence=dict(evidence),
        affected_documents=_str_tuple(
            f"{prefix}.affected_documents", data.get("affected_documents", ())
        ),
    )


def _require_str(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise ReportDecodeError(f"{name} must be a string")
    return value


def _require_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReportDecodeError(f"{name} must be an integer")
    return value


def _optional_int(name: str, value: object) -> int | None:
    if value is None:
        return None
    return _require_int(name, value)


def _require_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ReportDecodeError(f"{name} must be a number")
    return float(value)


def _str_tuple(name: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise ReportDecodeError(f"{name} must be an array of strings")
    items: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ReportDecodeError(f"{name}[{index}] must be a string")
        items.append(item)
    return tuple(items)


def _str_int_map(name: str, value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ReportDecodeError(f"{name} must be an object")
    result: dict[str, int] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ReportDecodeError(f"{name} keys must be strings")
        result[key] = _require_int(f"{name}.{key}", item)
    return result
