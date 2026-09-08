"""Tests for the internal filesystem run store."""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from samyak import analyze_corpus
from samyak.store.atomic import atomic_write_json
from samyak.store.errors import RunCorruptError, RunNotFoundError, RunSchemaError
from samyak.store.filesystem import FileRunStore
from samyak.store.paths import default_cache_dir


def _report(tmp_path: Path, name: str = "docs"):
    corpus = tmp_path / name
    corpus.mkdir()
    (corpus / "ok.txt").write_text(
        "Store test document with enough content about shipping policies.\n" * 3,
        encoding="utf-8",
    )
    return analyze_corpus(corpus), corpus


def test_save_and_load_run(tmp_path: Path) -> None:
    report, corpus = _report(tmp_path)
    store = FileRunStore(root=tmp_path / "cache")
    created = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    meta = store.save_run(
        report,
        duration_seconds=1.25,
        created_at=created,
        completed_at=created + timedelta(seconds=1),
    )
    UUID(meta.run_id)
    loaded = store.get_run(meta.run_id)
    assert loaded.metadata.run_id == meta.run_id
    assert loaded.report.to_dict() == report.to_dict()
    assert loaded.metadata.findings_count == len(report.findings)
    assert loaded.metadata.files_discovered == report.summary.total_discovered_files
    assert loaded.metadata.files_analyzed == report.summary.analyzed_documents
    assert loaded.metadata.status == "completed"
    assert loaded.metadata.report_schema_version == 1
    run_dir = tmp_path / "cache" / "runs" / meta.run_id
    payload = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert payload == report.to_dict()
    assert payload["product"] == "samyak"
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["run_id"] == meta.run_id
    assert metadata.get("corpus_path") is None
    assert str(corpus.resolve()) not in json.dumps(metadata)
    assert str(corpus.resolve()) not in json.dumps(payload)


def test_run_ids_are_unique(tmp_path: Path) -> None:
    report, _ = _report(tmp_path)
    store = FileRunStore(root=tmp_path / "cache")
    ids = [store.save_run(report, duration_seconds=0.1).run_id for _ in range(12)]
    assert len(set(ids)) == 12
    for run_id in ids:
        UUID(run_id)


def test_list_runs_newest_first_and_deterministic(tmp_path: Path) -> None:
    report, _ = _report(tmp_path)
    store = FileRunStore(root=tmp_path / "cache")
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    first = store.save_run(report, duration_seconds=1, created_at=t0, completed_at=t0)
    second = store.save_run(
        report,
        duration_seconds=2,
        created_at=t0 + timedelta(seconds=10),
        completed_at=t0 + timedelta(seconds=11),
    )
    same_time_a = store.save_run(
        report,
        duration_seconds=3,
        created_at=t0 + timedelta(seconds=20),
        completed_at=t0 + timedelta(seconds=20),
    )
    same_time_b = store.save_run(
        report,
        duration_seconds=4,
        created_at=t0 + timedelta(seconds=20),
        completed_at=t0 + timedelta(seconds=21),
    )
    page = store.list_runs()
    assert [item.run_id for item in page.runs][:2] == sorted(
        [same_time_a.run_id, same_time_b.run_id]
    )  # same timestamp: run_id ascending
    # After the two timestamp-tied runs, newer-than-first then first.
    remaining = [item.run_id for item in page.runs][2:]
    assert remaining == [second.run_id, first.run_id]
    again = store.list_runs()
    assert [item.run_id for item in again.runs] == [item.run_id for item in page.runs]


def test_list_runs_limit_and_multiple_runs(tmp_path: Path) -> None:
    report, _ = _report(tmp_path)
    store = FileRunStore(root=tmp_path / "cache")
    base = datetime(2026, 2, 1, tzinfo=UTC)
    for index in range(5):
        store.save_run(
            report,
            duration_seconds=index,
            created_at=base + timedelta(seconds=index),
            completed_at=base + timedelta(seconds=index),
        )
    page = store.list_runs(limit=2, offset=0)
    assert page.total == 5
    assert len(page.runs) == 2
    next_page = store.list_runs(limit=2, offset=2)
    assert len(next_page.runs) == 2
    assert {item.run_id for item in page.runs}.isdisjoint({item.run_id for item in next_page.runs})


def test_corrupt_report_json_is_skipped_on_list(tmp_path: Path) -> None:
    report, _ = _report(tmp_path)
    store = FileRunStore(root=tmp_path / "cache")
    meta = store.save_run(report, duration_seconds=0.5)
    (tmp_path / "cache" / "runs" / meta.run_id / "report.json").write_text(
        "{not-json",
        encoding="utf-8",
    )
    page = store.list_runs()
    assert meta.run_id not in {item.run_id for item in page.runs}
    assert page.skipped >= 1
    with pytest.raises(RunCorruptError):
        store.get_run(meta.run_id)


def test_corrupt_metadata_does_not_break_listing(tmp_path: Path) -> None:
    report, _ = _report(tmp_path)
    store = FileRunStore(root=tmp_path / "cache")
    healthy = store.save_run(report, duration_seconds=0.2)
    bad_dir = tmp_path / "cache" / "runs" / "00000000-0000-0000-0000-000000000099"
    bad_dir.mkdir(parents=True)
    (bad_dir / "metadata.json").write_text("{not json", encoding="utf-8")
    (bad_dir / "report.json").write_text("{}", encoding="utf-8")
    page = store.list_runs()
    assert [item.run_id for item in page.runs] == [healthy.run_id]
    assert page.skipped == 1
    assert page.total == 1


def test_unknown_schema_version_is_skipped_and_get_fails(tmp_path: Path) -> None:
    report, _ = _report(tmp_path)
    store = FileRunStore(root=tmp_path / "cache")
    healthy = store.save_run(report, duration_seconds=0.2)
    future_id = "11111111-1111-1111-1111-111111111111"
    future_dir = tmp_path / "cache" / "runs" / future_id
    future_dir.mkdir(parents=True)
    (future_dir / "report.json").write_text("{}", encoding="utf-8")
    (future_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": future_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "completed_at": "2026-01-01T00:00:01+00:00",
                "duration_seconds": 1,
                "samyak_version": "9.0.0",
                "report_schema_version": 99,
                "status": "completed",
                "capability": "corpus",
                "corpus_label": "docs",
                "findings_count": 0,
                "files_discovered": 0,
                "files_analyzed": 0,
                "load_errors": 0,
                "discovery_errors": 0,
            }
        ),
        encoding="utf-8",
    )
    page = store.list_runs()
    assert [item.run_id for item in page.runs] == [healthy.run_id]
    assert page.skipped == 1
    with pytest.raises(RunSchemaError):
        store.get_run(future_id)


def test_missing_run_raises_not_found(tmp_path: Path) -> None:
    store = FileRunStore(root=tmp_path / "cache")
    with pytest.raises(RunNotFoundError):
        store.get_run("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    with pytest.raises(RunNotFoundError):
        store.get_run("../etc/passwd")
    with pytest.raises(RunNotFoundError):
        store.get_run("not-a-uuid")


def test_empty_history(tmp_path: Path) -> None:
    store = FileRunStore(root=tmp_path / "cache")
    page = store.list_runs()
    assert page.runs == ()
    assert page.total == 0
    assert page.skipped == 0


def test_partial_run_without_metadata_is_skipped(tmp_path: Path) -> None:
    report, _ = _report(tmp_path)
    store = FileRunStore(root=tmp_path / "cache")
    healthy = store.save_run(report, duration_seconds=0.1)
    partial = tmp_path / "cache" / "runs" / "22222222-2222-2222-2222-222222222222"
    partial.mkdir(parents=True)
    (partial / "report.json").write_text("{}", encoding="utf-8")
    page = store.list_runs()
    assert [item.run_id for item in page.runs] == [healthy.run_id]
    assert page.skipped == 1
    with pytest.raises(RunCorruptError, match="missing metadata"):
        store.get_run("22222222-2222-2222-2222-222222222222")


def test_report_json_never_stores_absolute_corpus_path(tmp_path: Path) -> None:
    report, corpus = _report(tmp_path, name="invoices")
    store = FileRunStore(root=tmp_path / "cache")
    meta = store.save_run(
        report,
        duration_seconds=0.4,
        corpus_label="invoices",
        corpus_path=str(corpus.resolve()),
    )
    run_dir = tmp_path / "cache" / "runs" / meta.run_id
    report_blob = (run_dir / "report.json").read_text(encoding="utf-8")
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert str(corpus) not in report_blob
    assert str(corpus.resolve()) not in report_blob
    assert "corpus_path" not in json.loads(report_blob)
    assert metadata["corpus_label"] == "invoices"
    assert metadata["corpus_path"] == str(corpus.resolve())
    assert "/" not in metadata["corpus_label"]
    assert metadata["corpus_path"] is not None


def test_absolute_corpus_label_is_reduced_to_basename(tmp_path: Path) -> None:
    report, _ = _report(tmp_path)
    store = FileRunStore(root=tmp_path / "cache")
    meta = store.save_run(
        report,
        duration_seconds=0.1,
        corpus_label=str(tmp_path / "secret-corpus"),
    )
    assert meta.corpus_label == "secret-corpus"
    assert meta.corpus_path is None
    assert str(tmp_path / "secret-corpus") not in json.dumps(meta.to_dict())


def test_atomic_write_leaves_no_partial_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "report.json"
    target.write_text('{"ok": true}\n', encoding="utf-8")

    def boom(_fd: int) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("samyak.store.atomic.os.fsync", boom)
    with pytest.raises(OSError, match="disk full"):
        atomic_write_json(target, {"broken": True})
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}
    assert list(tmp_path.glob(".report.json.*.tmp")) == []


def test_cache_dir_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    custom = tmp_path / "custom-cache"
    monkeypatch.setenv("SAMYAK_CACHE_DIR", str(custom))
    assert default_cache_dir() == custom


def test_cache_dir_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SAMYAK_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(sys, "platform", "linux")
    assert default_cache_dir() == tmp_path / "xdg" / "samyak"


def test_cache_dir_windows_localappdata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SAMYAK_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
    monkeypatch.setattr(sys, "platform", "win32")
    assert default_cache_dir() == tmp_path / "AppData" / "Local" / "samyak"


def test_default_store_uses_env_cache(samyak_cache_dir: Path, tmp_path: Path) -> None:
    report, _ = _report(tmp_path)
    meta = FileRunStore().save_run(report, duration_seconds=0.1)
    assert (samyak_cache_dir / "runs" / meta.run_id / "metadata.json").is_file()
    assert os.environ["SAMYAK_CACHE_DIR"] == str(samyak_cache_dir)


def _rewrite_metadata(run_dir: Path, **updates: object) -> None:
    path = run_dir / "metadata.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(updates)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_get_run_rejects_mismatched_summary_counts(tmp_path: Path) -> None:
    report, _ = _report(tmp_path)
    store = FileRunStore(root=tmp_path / "cache")
    meta = store.save_run(report, duration_seconds=0.2)
    run_dir = tmp_path / "cache" / "runs" / meta.run_id
    _rewrite_metadata(run_dir, findings_count=meta.findings_count + 7)
    with pytest.raises(RunCorruptError, match="does not match"):
        store.get_run(meta.run_id)
    page = store.list_runs()
    assert meta.run_id not in {item.run_id for item in page.runs}
    assert page.skipped >= 1


def test_get_run_rejects_mismatched_run_id(tmp_path: Path) -> None:
    report, _ = _report(tmp_path)
    store = FileRunStore(root=tmp_path / "cache")
    meta = store.save_run(report, duration_seconds=0.2)
    run_dir = tmp_path / "cache" / "runs" / meta.run_id
    _rewrite_metadata(run_dir, run_id="dddddddd-dddd-dddd-dddd-dddddddddddd")
    with pytest.raises(RunCorruptError, match="run_id"):
        store.get_run(meta.run_id)
    page = store.list_runs()
    assert meta.run_id not in {item.run_id for item in page.runs}
    assert page.skipped >= 1


def test_missing_report_is_skipped_on_list(tmp_path: Path) -> None:
    report, _ = _report(tmp_path)
    store = FileRunStore(root=tmp_path / "cache")
    healthy = store.save_run(report, duration_seconds=0.1)
    incomplete = tmp_path / "cache" / "runs" / "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"
    incomplete.mkdir(parents=True)
    (incomplete / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee",
                "created_at": "2026-01-01T00:00:00+00:00",
                "completed_at": "2026-01-01T00:00:01+00:00",
                "duration_seconds": 1,
                "samyak_version": "0.1.0",
                "report_schema_version": 1,
                "status": "completed",
                "capability": "corpus",
                "corpus_label": "docs",
                "findings_count": 0,
                "files_discovered": 0,
                "files_analyzed": 0,
                "load_errors": 0,
                "discovery_errors": 0,
            }
        ),
        encoding="utf-8",
    )
    page = store.list_runs()
    assert [item.run_id for item in page.runs] == [healthy.run_id]
    assert page.skipped == 1
    with pytest.raises(RunCorruptError, match="missing report"):
        store.get_run("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee")


def test_malformed_report_object_is_skipped_even_when_counts_match(tmp_path: Path) -> None:
    report, _ = _report(tmp_path)
    store = FileRunStore(root=tmp_path / "cache")
    meta = store.save_run(report, duration_seconds=0.2)
    run_dir = tmp_path / "cache" / "runs" / meta.run_id
    payload = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    findings = list(payload.get("findings") or [])
    if findings and isinstance(findings[0], dict):
        findings[0] = {**findings[0], "severity": "not-a-severity"}
    else:
        findings = [
            {
                "code": "X",
                "category": "x",
                "severity": "not-a-severity",
                "title": "t",
                "message": "m",
                "why_it_matters": "w",
                "recommendation": "r",
                "evidence": {},
                "affected_documents": [],
            }
        ]
        _rewrite_metadata(run_dir, findings_count=1)
    payload["findings"] = findings
    (run_dir / "report.json").write_text(json.dumps(payload), encoding="utf-8")
    page = store.list_runs()
    assert meta.run_id not in {item.run_id for item in page.runs}
    assert page.skipped >= 1
    with pytest.raises(RunCorruptError):
        store.get_run(meta.run_id)


def test_tampered_schema_version_is_skipped(tmp_path: Path) -> None:
    report, _ = _report(tmp_path)
    store = FileRunStore(root=tmp_path / "cache")
    meta = store.save_run(report, duration_seconds=0.2)
    _rewrite_metadata(tmp_path / "cache" / "runs" / meta.run_id, report_schema_version=2)
    page = store.list_runs()
    assert meta.run_id not in {item.run_id for item in page.runs}
    assert page.skipped >= 1
    with pytest.raises(RunSchemaError):
        store.get_run(meta.run_id)


def test_mismatched_samyak_version_is_skipped(tmp_path: Path) -> None:
    report, _ = _report(tmp_path)
    store = FileRunStore(root=tmp_path / "cache")
    meta = store.save_run(report, duration_seconds=0.2)
    _rewrite_metadata(tmp_path / "cache" / "runs" / meta.run_id, samyak_version="9.9.9")
    page = store.list_runs()
    assert meta.run_id not in {item.run_id for item in page.runs}
    assert page.skipped >= 1
    with pytest.raises(RunCorruptError, match="does not match"):
        store.get_run(meta.run_id)


def test_legacy_metadata_without_corpus_path_still_loads(tmp_path: Path) -> None:
    report, _ = _report(tmp_path)
    store = FileRunStore(root=tmp_path / "cache")
    meta = store.save_run(report, duration_seconds=0.2)
    run_dir = tmp_path / "cache" / "runs" / meta.run_id
    data = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    data.pop("corpus_path", None)
    (run_dir / "metadata.json").write_text(json.dumps(data), encoding="utf-8")
    loaded = store.get_run(meta.run_id)
    assert loaded.metadata.corpus_path is None
    assert loaded.report.to_dict() == report.to_dict()


def test_repeated_saves_are_independent(tmp_path: Path) -> None:
    report, corpus = _report(tmp_path)
    store = FileRunStore(root=tmp_path / "cache")
    t0 = datetime(2026, 4, 1, tzinfo=UTC)
    first = store.save_run(
        report,
        duration_seconds=1,
        created_at=t0,
        completed_at=t0,
        corpus_path=str(corpus / "first"),
    )
    second = store.save_run(
        report,
        duration_seconds=2,
        created_at=t0 + timedelta(seconds=5),
        completed_at=t0 + timedelta(seconds=5),
        corpus_path=str(corpus / "second"),
    )
    third = store.save_run(
        report,
        duration_seconds=3,
        created_at=t0 + timedelta(seconds=9),
        completed_at=t0 + timedelta(seconds=9),
        corpus_path=str(corpus / "third"),
    )
    assert len({first.run_id, second.run_id, third.run_id}) == 3
    page = store.list_runs()
    assert [item.run_id for item in page.runs] == [third.run_id, second.run_id, first.run_id]
    assert store.get_run(first.run_id).metadata.corpus_path.endswith("first")
    assert store.get_run(second.run_id).metadata.corpus_path.endswith("second")
    _rewrite_metadata(tmp_path / "cache" / "runs" / second.run_id, findings_count=99)
    with pytest.raises(RunCorruptError):
        store.get_run(second.run_id)
    assert store.get_run(first.run_id).metadata.run_id == first.run_id
    assert store.get_run(third.run_id).metadata.run_id == third.run_id
    assert {item.run_id for item in store.list_runs().runs} >= {first.run_id, third.run_id}


def test_concurrent_saves_get_unique_ids(tmp_path: Path) -> None:
    report, _ = _report(tmp_path)
    store = FileRunStore(root=tmp_path / "cache")

    def _save(_: int) -> str:
        return store.save_run(report, duration_seconds=0.01).run_id

    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(_save, range(24)))
    assert len(ids) == 24
    assert len(set(ids)) == 24
    assert store.list_runs(limit=50).total == 24


def test_malformed_metadata_fields_are_skipped(tmp_path: Path) -> None:
    report, _ = _report(tmp_path)
    store = FileRunStore(root=tmp_path / "cache")
    cases: list[tuple[str, dict[str, object]]] = [
        ("created_at", {"created_at": "not-a-timestamp"}),
        ("completed_at", {"completed_at": "yesterday"}),
        ("duration", {"duration_seconds": -1}),
        ("duration-nan", {"duration_seconds": float("nan")}),
        ("count", {"files_discovered": -3}),
        ("status", {"status": "running"}),
        (
            "order",
            {
                "created_at": "2026-02-01T00:00:00+00:00",
                "completed_at": "2026-01-01T00:00:00+00:00",
            },
        ),
    ]
    for label, updates in cases:
        meta = store.save_run(report, duration_seconds=0.2)
        run_dir = tmp_path / "cache" / "runs" / meta.run_id
        _rewrite_metadata(run_dir, **updates)
        page = store.list_runs()
        assert meta.run_id not in {item.run_id for item in page.runs}, label
        with pytest.raises(RunCorruptError):
            store.get_run(meta.run_id)
