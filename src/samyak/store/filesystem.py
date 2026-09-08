"""Filesystem-backed :class:`RunStore` implementation."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from samyak.corpus.models import AnalysisReport
from samyak.corpus.serialize import ReportDecodeError, report_from_dict
from samyak.store.atomic import atomic_write_json
from samyak.store.errors import RunCorruptError, RunNotFoundError, RunSchemaError
from samyak.store.models import RunMetadata, RunPage, StoredRun, metadata_from_report
from samyak.store.paths import default_cache_dir


class RunStore(Protocol):
    """Persistence for completed analysis runs.

    Intentionally small so a later API-backed or database-backed store can
    implement the same methods without changing the analysis engine.
    """

    def save_run(
        self,
        report: AnalysisReport,
        *,
        duration_seconds: float,
        created_at: datetime | None = None,
        completed_at: datetime | None = None,
        corpus_label: str | None = None,
        corpus_path: str | None = None,
    ) -> RunMetadata: ...

    def get_run(self, run_id: str) -> StoredRun: ...

    def list_runs(self, *, limit: int = 50, offset: int = 0) -> RunPage: ...


class FileRunStore:
    """Store each run as ``<cache>/runs/<run-id>/{report.json,metadata.json}``."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_cache_dir()
        self.runs_dir = self.root / "runs"

    def save_run(
        self,
        report: AnalysisReport,
        *,
        duration_seconds: float,
        created_at: datetime | None = None,
        completed_at: datetime | None = None,
        corpus_label: str | None = None,
        corpus_path: str | None = None,
    ) -> RunMetadata:
        finished = completed_at or datetime.now(UTC)
        started = created_at or finished
        if corpus_label is None:
            corpus_label = report.summary.corpus_root
        label = _safe_corpus_label(corpus_label)
        stored_path = _safe_corpus_path(corpus_path)
        run_dir, run_id = self._create_run_dir()
        metadata = metadata_from_report(
            run_id=run_id,
            report=report,
            created_at=_isoformat(started),
            completed_at=_isoformat(finished),
            duration_seconds=float(duration_seconds),
            corpus_label=label,
            corpus_path=stored_path,
        )
        # report.json first; metadata.json is the commit marker for listing.
        atomic_write_json(run_dir / "report.json", report.to_dict())
        atomic_write_json(run_dir / "metadata.json", metadata.to_dict())
        return metadata

    def get_run(self, run_id: str) -> StoredRun:
        run_dir = self._run_dir(run_id)
        if run_dir is None or not run_dir.is_dir():
            raise RunNotFoundError(run_id)
        metadata = self._load_metadata(run_dir)
        report_path = run_dir / "report.json"
        if not report_path.is_file():
            raise RunCorruptError("missing report.json")
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RunCorruptError("invalid report JSON") from exc
        try:
            report = report_from_dict(payload)
        except ReportDecodeError as exc:
            raise RunCorruptError("report JSON could not be reconstructed") from exc
        _assert_metadata_matches_report(metadata, report)
        return StoredRun(metadata=metadata, report=report)

    def list_runs(self, *, limit: int = 50, offset: int = 0) -> RunPage:
        if limit < 1:
            limit = 50
        if offset < 0:
            offset = 0
        if not self.runs_dir.is_dir():
            return RunPage(runs=(), total=0, offset=offset, limit=limit, skipped=0)

        metadatas: list[RunMetadata] = []
        skipped = 0
        try:
            children = list(self.runs_dir.iterdir())
        except OSError:
            return RunPage(runs=(), total=0, offset=offset, limit=limit, skipped=0)
        for child in children:
            if not child.is_dir():
                continue
            if not (child / "metadata.json").is_file() or not (child / "report.json").is_file():
                skipped += 1
                continue
            try:
                metadata = self._load_metadata(child)
                payload = json.loads((child / "report.json").read_text(encoding="utf-8"))
                report = report_from_dict(payload)
                _assert_metadata_matches_report(metadata, report)
            except (
                RunCorruptError,
                RunSchemaError,
                ReportDecodeError,
                OSError,
                json.JSONDecodeError,
            ):
                skipped += 1
                continue
            metadatas.append(metadata)

        metadatas.sort(key=lambda item: (-_sort_epoch(item.created_at), item.run_id))
        total = len(metadatas)
        page = tuple(metadatas[offset : offset + limit])
        return RunPage(runs=page, total=total, offset=offset, limit=limit, skipped=skipped)

    def _create_run_dir(self) -> tuple[Path, str]:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(8):
            run_id = str(uuid.uuid4())
            run_dir = self.runs_dir / run_id
            try:
                run_dir.mkdir(exist_ok=False)
            except FileExistsError:
                continue
            return run_dir, run_id
        raise RuntimeError("could not allocate a unique run id")

    def _run_dir(self, run_id: str) -> Path | None:
        normalized = _normalized_run_id(run_id)
        if normalized is None:
            return None
        runs_root = self.runs_dir.resolve()
        path = (self.runs_dir / normalized).resolve()
        try:
            path.relative_to(runs_root)
        except ValueError:
            return None
        return path

    def _load_metadata(self, run_dir: Path) -> RunMetadata:
        path = run_dir / "metadata.json"
        if not path.is_file():
            raise RunCorruptError("missing metadata.json")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RunCorruptError("invalid metadata JSON") from exc
        return RunMetadata.from_dict(raw, directory_name=run_dir.name)


def _normalized_run_id(run_id: str) -> str | None:
    try:
        return str(uuid.UUID(run_id))
    except ValueError:
        return None


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _sort_epoch(created_at: str) -> float:
    parsed = datetime.fromisoformat(created_at)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def _assert_metadata_matches_report(metadata: RunMetadata, report: AnalysisReport) -> None:
    """Reject snapshots whose listing metadata does not match the stored report."""
    if not _report_payload_matches_metadata(metadata, report.to_dict()):
        raise RunCorruptError("metadata does not match report")


def _report_payload_matches_metadata(metadata: RunMetadata, payload: object) -> bool:
    """Compare metadata to report JSON using only fields derived from AnalysisReport.

    Applied after reconstruction (via ``to_dict()``) so listing and open-run
    share the same consistency rules. Schema version lives only on metadata;
    the public report dict has no schema field to compare.
    """
    if not isinstance(payload, dict):
        return False
    if payload.get("capability") != metadata.capability:
        return False
    if payload.get("version") != metadata.samyak_version:
        return False
    findings = payload.get("findings")
    if not isinstance(findings, list) or len(findings) != metadata.findings_count:
        return False
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return False
    try:
        discovered = summary["total_discovered_files"]
        analyzed = summary["analyzed_documents"]
        load_errors = summary["load_errors"]
        discovery_errors = summary.get("discovery_errors", 0)
    except KeyError:
        return False
    if isinstance(discovered, bool) or not isinstance(discovered, int):
        return False
    if isinstance(analyzed, bool) or not isinstance(analyzed, int):
        return False
    if isinstance(load_errors, bool) or not isinstance(load_errors, int):
        return False
    if isinstance(discovery_errors, bool) or not isinstance(discovery_errors, int):
        return False
    return (
        discovered == metadata.files_discovered
        and analyzed == metadata.files_analyzed
        and load_errors == metadata.load_errors
        and discovery_errors == metadata.discovery_errors
    )


def _safe_corpus_path(path: str | None) -> str | None:
    """Persist a caller-supplied local corpus path without inventing one."""
    if path is None:
        return None
    text = str(path).strip()
    if not text or "\x00" in text:
        return None
    return text


def _safe_corpus_label(label: str | None) -> str | None:
    """Keep a directory basename only; never persist an absolute path as the label."""
    if label is None:
        return None
    text = str(label).strip()
    if not text:
        return None
    candidate = Path(text)
    if candidate.is_absolute() or ".." in candidate.parts:
        name = candidate.name
        return name or None
    if "/" in text or "\\" in text:
        return candidate.name or None
    return text
