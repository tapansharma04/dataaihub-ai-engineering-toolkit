"""Internal run metadata models.

These are persistence records, not part of the public Samyak API.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from samyak.corpus.models import AnalysisReport
from samyak.corpus.serialize import REPORT_SCHEMA_VERSION, SUPPORTED_REPORT_SCHEMA_VERSIONS
from samyak.store.errors import RunCorruptError, RunSchemaError

RUN_STATUS_COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class RunMetadata:
    """Local summary of a persisted analysis run.

    Listing validates this record against ``report.json`` (then discards the
    reconstructed report). Opening a run reconstructs ``AnalysisReport`` for
    rendering.

    ``corpus_path`` is optional local-machine metadata. It is not part of
    ``AnalysisReport`` and is never written to report stdout/JSON/HTML.
    """

    run_id: str
    created_at: str
    completed_at: str
    duration_seconds: float
    samyak_version: str
    report_schema_version: int
    status: str
    capability: str
    corpus_label: str | None
    findings_count: int
    files_discovered: int
    files_analyzed: int
    load_errors: int
    discovery_errors: int
    corpus_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "samyak_version": self.samyak_version,
            "report_schema_version": self.report_schema_version,
            "status": self.status,
            "capability": self.capability,
            "corpus_label": self.corpus_label,
            "corpus_path": self.corpus_path,
            "findings_count": self.findings_count,
            "files_discovered": self.files_discovered,
            "files_analyzed": self.files_analyzed,
            "load_errors": self.load_errors,
            "discovery_errors": self.discovery_errors,
        }

    @classmethod
    def from_dict(cls, data: object, *, directory_name: str | None = None) -> RunMetadata:
        if not isinstance(data, dict):
            raise RunCorruptError("metadata JSON must be an object")
        run_id = _require_run_id(data.get("run_id"))
        schema_version = _require_int("report_schema_version", data.get("report_schema_version"))
        if schema_version not in SUPPORTED_REPORT_SCHEMA_VERSIONS:
            raise RunSchemaError(run_id, schema_version)
        if directory_name is not None and run_id != directory_name:
            raise RunCorruptError("metadata run_id does not match directory name")
        corpus_label = data.get("corpus_label")
        if corpus_label is not None:
            corpus_label = _require_str("corpus_label", corpus_label)
        corpus_path = data.get("corpus_path")
        if corpus_path is not None:
            corpus_path = _require_str("corpus_path", corpus_path)
            if "\x00" in corpus_path:
                raise RunCorruptError("corpus_path is invalid")
            corpus_path = corpus_path.strip() or None
        created_at = _require_timestamp("created_at", data.get("created_at"))
        completed_at = _require_timestamp("completed_at", data.get("completed_at"))
        _assert_timestamp_order(created_at, completed_at)
        status = _require_str("status", data.get("status"))
        if status not in _ALLOWED_STATUSES:
            raise RunCorruptError("status is not a recognized run status")
        return cls(
            run_id=run_id,
            created_at=created_at,
            completed_at=completed_at,
            duration_seconds=_require_duration(data.get("duration_seconds")),
            samyak_version=_require_str("samyak_version", data.get("samyak_version")),
            report_schema_version=schema_version,
            status=status,
            capability=_require_str("capability", data.get("capability")),
            corpus_label=corpus_label,
            corpus_path=corpus_path,
            findings_count=_require_count("findings_count", data.get("findings_count")),
            files_discovered=_require_count("files_discovered", data.get("files_discovered")),
            files_analyzed=_require_count("files_analyzed", data.get("files_analyzed")),
            load_errors=_require_count("load_errors", data.get("load_errors")),
            discovery_errors=_require_count("discovery_errors", data.get("discovery_errors")),
        )


@dataclass(frozen=True, slots=True)
class StoredRun:
    """A persisted run: metadata plus the reconstructed analysis report."""

    metadata: RunMetadata
    report: AnalysisReport


@dataclass(frozen=True, slots=True)
class RunPage:
    """Newest-first slice of run metadata for the history page."""

    runs: tuple[RunMetadata, ...]
    total: int
    offset: int
    limit: int
    skipped: int = 0


def metadata_from_report(
    *,
    run_id: str,
    report: AnalysisReport,
    created_at: str,
    completed_at: str,
    duration_seconds: float,
    corpus_label: str | None,
    corpus_path: str | None = None,
) -> RunMetadata:
    summary = report.summary
    return RunMetadata(
        run_id=run_id,
        created_at=created_at,
        completed_at=completed_at,
        duration_seconds=duration_seconds,
        samyak_version=report.version,
        report_schema_version=REPORT_SCHEMA_VERSION,
        status=RUN_STATUS_COMPLETED,
        capability=report.capability,
        corpus_label=corpus_label,
        corpus_path=corpus_path,
        findings_count=len(report.findings),
        files_discovered=summary.total_discovered_files,
        files_analyzed=summary.analyzed_documents,
        load_errors=summary.load_errors,
        discovery_errors=summary.discovery_errors,
    )


def _require_str(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise RunCorruptError(f"{name} must be a string")
    return value


def _require_run_id(value: object) -> str:
    run_id = _require_str("run_id", value)
    try:
        uuid.UUID(run_id)
    except ValueError as exc:
        raise RunCorruptError("run_id is not a valid UUID") from exc
    return run_id


def _require_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RunCorruptError(f"{name} must be an integer")
    return value


def _require_count(name: str, value: object) -> int:
    number = _require_int(name, value)
    if number < 0:
        raise RunCorruptError(f"{name} must be >= 0")
    return number


def _require_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RunCorruptError(f"{name} must be a number")
    return float(value)


def _require_duration(value: object) -> float:
    seconds = _require_float("duration_seconds", value)
    if not math.isfinite(seconds) or seconds < 0:
        raise RunCorruptError("duration_seconds is invalid")
    return seconds


def _require_timestamp(name: str, value: object) -> str:
    text = _require_str(name, value)
    try:
        datetime.fromisoformat(text)
    except ValueError as exc:
        raise RunCorruptError(f"{name} is not a valid timestamp") from exc
    return text


def _assert_timestamp_order(created_at: str, completed_at: str) -> None:
    created = datetime.fromisoformat(created_at)
    completed = datetime.fromisoformat(completed_at)
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    if completed.tzinfo is None:
        completed = completed.replace(tzinfo=UTC)
    if completed < created:
        raise RunCorruptError("completed_at is before created_at")


_ALLOWED_STATUSES = frozenset({RUN_STATUS_COMPLETED})
