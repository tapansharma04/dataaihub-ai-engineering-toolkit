"""Diagnostic source archive. Fixture transport only; no live network."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import socket
from pathlib import Path
from typing import Any

import pytest

from samyak.cli import build_parser, main
from samyak.model.export import build_model_export
from samyak.model.facts import FactStatus
from samyak.model.history_store import FileLifecycleHistoryStore
from samyak.model.http import FixtureHop
from samyak.model.providers.together.errors import TogetherParseError
from samyak.model.providers.together.sources import (
    DEPRECATIONS_URL as TOGETHER_DEPRECATIONS_URL,
)
from samyak.model.providers.together.sources import (
    SERVERLESS_URL as TOGETHER_SERVERLESS_URL,
)
from samyak.model.providers.together.sources import model_page_url
from samyak.model.source_archive import (
    DEFAULT_ARCHIVE_KEEP,
    SourceArchive,
    blob_path,
    completed_refresh_ids,
    load_failures,
    load_manifest,
    note_parse_failure,
    refresh_id_from_stamp,
)
from samyak.model.store import FileCatalogStore
from samyak.model.update import update_all_catalogs, update_together_catalog
from test_model_update import (
    STAMP,
    _together_pages,
    _together_transport,
)
from test_model_update_all import _failed_google_pages, _patch_all_transports

SECRET = "secret-token"
SORA_ID = "together:openai/sora-2"
STAMP_LATER = "2026-10-30T13:00:00+00:00"


@pytest.fixture(autouse=True)
def deny_live_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("source archive tests must not use the network")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr("urllib.request.urlopen", blocked)


def _archive_root(store: FileCatalogStore) -> Path:
    return store.models_dir / "source-archive"


def _blob_files(root: Path) -> list[Path]:
    blob_root = root / "blobs" / "sha256"
    if not blob_root.is_dir():
        return []
    return [path for path in blob_root.rglob("*") if path.is_file()]


def _archive_text(root: Path) -> str:
    parts: list[str] = []
    if not root.exists():
        return ""
    for path in sorted(root.rglob("*")):
        if path.is_file():
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


def _publication(store: FileCatalogStore) -> tuple[str, tuple[str, ...]]:
    catalog = store.load()
    history_store = FileLifecycleHistoryStore(root=store.root)
    history = history_store.load() if history_store.exists() else None
    export = build_model_export(catalog, history)
    return export.publication.status.value, export.publication.reasons


def _update(
    store: FileCatalogStore,
    pages: dict[str, FixtureHop],
    *,
    stamp: str = STAMP,
    archive_sources: bool = False,
    archive_keep: int = DEFAULT_ARCHIVE_KEEP,
):
    return update_together_catalog(
        transport=_together_transport(pages, clock=lambda: stamp),
        store=store,
        clock=lambda: stamp,
        archive_sources=archive_sources,
        archive_keep=archive_keep,
    )


def _known_hash(store: FileCatalogStore) -> str:
    catalog = store.load()
    for model in catalog.models:
        if model.provider_id != "together":
            continue
        fact = model.lifecycle
        if fact.status is FactStatus.KNOWN and fact.provenance and fact.provenance.content_hash:
            return fact.provenance.content_hash
        if fact.status is FactStatus.CONFLICT:
            for claim in fact.claims:
                if claim.provenance.content_hash:
                    return claim.provenance.content_hash
    raise AssertionError("expected a Together fact with a content hash")


class _FakeSource:
    def __init__(
        self,
        body: str,
        source_id: str = "example-source",
        source_url: str = "https://docs.together.ai/docs/example.md",
    ) -> None:
        self.source_id = source_id
        self.source_url = source_url
        self.retrieved_at = STAMP
        self.media_type = "text/markdown"
        self._body = body.encode("utf-8")

    @property
    def body_bytes(self) -> bytes:
        return self._body

    @property
    def content_hash(self) -> str:
        return "sha256:" + hashlib.sha256(self._body).hexdigest()


def _commit_generation(
    root: Path,
    refresh_id: str,
    bodies: tuple[tuple[str, str], ...],
    *,
    keep: int = 2,
) -> None:
    archive = SourceArchive(
        root=root,
        refresh_id=refresh_id,
        package_version="0.3.0",
        keep=keep,
    )
    for source_id, body in bodies:
        archive.record_captured("together", _FakeSource(body, source_id))
        assert not archive.write_failed
    archive.finalize()
    assert not archive.write_failed


def _plant_blob(root: Path, body: str) -> Path:
    source = _FakeSource(body, source_id="planted")
    path = blob_path(root, source.content_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(source.body_bytes)
    return path


def _retain(root: Path, *, keep: int) -> None:
    SourceArchive(
        root=root,
        refresh_id="retain-pass",
        package_version="0.3.0",
        keep=keep,
    )._retain()


def _spawn_context() -> tuple[multiprocessing.context.SpawnContext, str | None]:
    tests_dir = str(Path(__file__).resolve().parent)
    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    previous = os.environ.get("PYTHONPATH")
    parts = [tests_dir, src_dir]
    if previous:
        parts.append(previous)
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)
    return multiprocessing.get_context("spawn"), previous


def _restore_pythonpath(previous: str | None) -> None:
    if previous is None:
        os.environ.pop("PYTHONPATH", None)
    else:
        os.environ["PYTHONPATH"] = previous


def _lock_holder(root: str, held: Any, release: Any) -> None:
    archive = SourceArchive(
        root=Path(root),
        refresh_id="holder-gen",
        package_version="0.3.0",
        keep=1,
    )
    archive.record_captured("together", _FakeSource("holder-unpublished", "holder-source"))
    held.set()
    release.wait()
    archive.finalize()


def _lock_challenger(root: str) -> None:
    archive = SourceArchive(
        root=Path(root),
        refresh_id="challenger-gen",
        package_version="0.3.0",
        keep=1,
    )
    archive.record_captured("together", _FakeSource("challenger-body", "challenger-source"))
    archive.finalize()


def _lock_release_after_failure(root: str) -> None:
    archive = SourceArchive(
        root=Path(root),
        refresh_id="failed-holder",
        package_version="0.3.0",
        keep=1,
    )
    archive.record_captured("together", _FakeSource("failed-holder-body", "failed-holder"))
    archive.write_failed = True
    archive.finalize()


def _lock_release_after_success(root: str) -> None:
    archive = SourceArchive(
        root=Path(root),
        refresh_id="success-holder",
        package_version="0.3.0",
        keep=2,
    )
    archive.record_captured("together", _FakeSource("success-holder-body", "success-holder"))
    archive.finalize()


def test_flag_disabled_writes_no_archive_and_matches_catalog(tmp_path: Path) -> None:
    pages = _together_pages()
    plain = FileCatalogStore(root=tmp_path / "plain")
    archived = FileCatalogStore(root=tmp_path / "archived")
    plain_result = _update(plain, pages, archive_sources=False)
    archived_result = _update(archived, pages, archive_sources=False)
    assert plain_result.ok and archived_result.ok
    assert not _archive_root(plain).exists()
    assert not _archive_root(archived).exists()
    assert plain.catalog_path.read_bytes() == archived.catalog_path.read_bytes()
    assert _publication(plain) == _publication(archived)


def test_successful_capture_is_addressed_by_provenance_hash(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    result = _update(store, _together_pages(), archive_sources=True)
    assert result.ok
    root = _archive_root(store)
    content_hash = _known_hash(store)
    path = blob_path(root, content_hash)
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == content_hash.removeprefix("sha256:")
    manifest = load_manifest(root, completed_refresh_ids(root)[-1])
    assert manifest is not None
    hashes = {item["content_hash"] for item in manifest["sources"]}  # type: ignore[index]
    assert content_hash in hashes
    recorded = next(
        item
        for item in manifest["sources"]
        if item["content_hash"] == content_hash  # type: ignore[index]
    )
    assert recorded["provider_id"] == "together"
    assert recorded["package_version"]
    assert recorded["byte_length"] == len(payload)
    refresh_dir = root / "refreshes" / str(manifest["refresh_id"])
    assert not (refresh_dir / "failures.json").exists()


def test_identical_bodies_share_a_blob_and_previous_generations_remain(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    _update(store, _together_pages(), stamp=STAMP, archive_sources=True, archive_keep=2)
    root = _archive_root(store)
    first_id = completed_refresh_ids(root)[-1]
    first_blobs = {path.read_bytes() for path in _blob_files(root)}
    _update(store, _together_pages(), stamp=STAMP_LATER, archive_sources=True, archive_keep=2)
    assert len(completed_refresh_ids(root)) == 2
    assert completed_refresh_ids(root)[0] == first_id
    assert {path.read_bytes() for path in _blob_files(root)} == first_blobs
    assert load_manifest(root, first_id) is not None


def test_changed_body_keeps_the_removed_models_old_source(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    _update(store, _together_pages(), stamp=STAMP, archive_sources=True, archive_keep=2)
    root = _archive_root(store)
    first_id = completed_refresh_ids(root)[-1]
    first_manifest = load_manifest(root, first_id)
    assert first_manifest is not None
    serverless = next(
        item
        for item in first_manifest["sources"]  # type: ignore[index]
        if item["source_id"] == "together-docs-serverless-models"
    )
    old_hash = str(serverless["content_hash"])
    old_blob = blob_path(root, old_hash).read_bytes()
    assert b"openai/sora-2" in old_blob
    assert any(model.samyak_id == SORA_ID for model in store.load().models)

    revised = _together_pages()
    original = revised[TOGETHER_SERVERLESS_URL].body
    revised[TOGETHER_SERVERLESS_URL] = FixtureHop(
        body=original.replace(b"`openai/sora-2`", b"`openai/sora-2-removed`")
    )
    _update(store, revised, stamp=STAMP_LATER, archive_sources=True, archive_keep=2)
    assert all(model.samyak_id != SORA_ID for model in store.load().models)
    assert blob_path(root, old_hash).read_bytes() == old_blob
    second = load_manifest(root, completed_refresh_ids(root)[-1])
    assert second is not None
    new_hash = next(
        item["content_hash"]
        for item in second["sources"]  # type: ignore[index]
        if item["source_id"] == "together-docs-serverless-models"
    )
    assert new_hash != old_hash
    assert b"openai/sora-2`" not in blob_path(root, str(new_hash)).read_bytes()


def test_garbage_collection_drops_expired_blobs_and_keeps_orphans(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    _update(store, _together_pages(), stamp=STAMP, archive_sources=True, archive_keep=1)
    root = _archive_root(store)
    first_id = completed_refresh_ids(root)[-1]
    first_manifest = load_manifest(root, first_id)
    assert first_manifest is not None
    old_hash = next(
        item["content_hash"]
        for item in first_manifest["sources"]  # type: ignore[index]
        if item["source_id"] == "together-docs-serverless-models"
    )
    orphan = blob_path(root, "sha256:" + ("ab" * 32))
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"orphan")
    revised = _together_pages()
    revised[TOGETHER_SERVERLESS_URL] = FixtureHop(
        body=revised[TOGETHER_SERVERLESS_URL].body + b"\n<!-- changed -->\n"
    )
    _update(store, revised, stamp=STAMP_LATER, archive_sources=True, archive_keep=1)
    assert completed_refresh_ids(root) == (refresh_id_from_stamp(STAMP_LATER),)
    assert not blob_path(root, str(old_hash)).exists()
    assert orphan.exists()
    kept = load_manifest(root, refresh_id_from_stamp(STAMP_LATER))
    assert kept is not None
    for item in kept["sources"]:  # type: ignore[index]
        assert blob_path(root, str(item["content_hash"])).is_file()


def test_optional_http_status_timeout_parse_and_content_type(tmp_path: Path) -> None:
    quickstart = model_page_url("glm-5.2-quickstart")
    status_store = FileCatalogStore(root=tmp_path / "status")
    status_pages = _together_pages()
    status_pages[quickstart] = FixtureHop(
        status=404,
        body=b"secret-token",
        headers=(
            ("Content-Type", "text/markdown"),
            ("Set-Cookie", "session=secret-token"),
            ("Authorization", "Bearer secret-token"),
        ),
    )
    status_result = _update(status_store, status_pages, archive_sources=True)
    assert status_result.committed
    status_root = _archive_root(status_store)
    status_id = completed_refresh_ids(status_root)[-1]
    status_failures = load_failures(status_root, status_id)
    status_row = next(item for item in status_failures if item["source_url"] == quickstart)
    assert status_row["stage"] == "http_status"
    assert status_row["http_status"] == 404
    assert status_row["exception_class"] == "TogetherFetchError"
    assert SECRET not in _archive_text(status_root)
    assert "Set-Cookie" not in _archive_text(status_root)
    assert "Authorization" not in json.dumps(status_row)
    plain = FileCatalogStore(root=tmp_path / "status-plain")
    _update(plain, status_pages, archive_sources=False)
    assert status_store.catalog_path.read_bytes() == plain.catalog_path.read_bytes()
    assert _publication(status_store) == _publication(plain)

    timeout_store = FileCatalogStore(root=tmp_path / "timeout")
    timeout_pages = _together_pages()
    timeout_pages[quickstart] = FixtureHop(timeout=True)
    _update(timeout_store, timeout_pages, archive_sources=True)
    timeout_root = _archive_root(timeout_store)
    timeout_row = next(
        item
        for item in load_failures(timeout_root, completed_refresh_ids(timeout_root)[-1])
        if item["source_url"] == quickstart
    )
    assert timeout_row["stage"] == "timeout"
    assert timeout_row["http_status"] is None

    parse_store = FileCatalogStore(root=tmp_path / "parse")
    parse_pages = _together_pages()
    parse_body = b"# hello\n"
    parse_pages[quickstart] = FixtureHop(body=parse_body)
    _update(parse_store, parse_pages, archive_sources=True)
    parse_root = _archive_root(parse_store)
    parse_id = completed_refresh_ids(parse_root)[-1]
    parse_row = next(
        item
        for item in load_failures(parse_root, parse_id)
        if "glm-5.2-quickstart" in str(item["source_id"])
    )
    assert parse_row["stage"] == "parse"
    assert parse_row["exception_class"] == "TogetherParseError"
    digest = hashlib.sha256(parse_body).hexdigest()
    assert blob_path(parse_root, f"sha256:{digest}").read_bytes() == parse_body

    type_store = FileCatalogStore(root=tmp_path / "type")
    type_pages = _together_pages()
    type_pages[quickstart] = FixtureHop(
        headers=(("Content-Type", "text/html"),),
        body=b"SECRET_HTML_BODY",
    )
    _update(type_store, type_pages, archive_sources=True)
    type_root = _archive_root(type_store)
    type_row = next(
        item
        for item in load_failures(type_root, completed_refresh_ids(type_root)[-1])
        if item["source_url"] == quickstart
    )
    assert type_row["stage"] == "content_type"
    assert type_row["http_status"] is None
    assert b"SECRET_HTML_BODY" not in b"".join(path.read_bytes() for path in _blob_files(type_root))


def test_invalid_utf8_is_a_decode_failure_without_a_blob(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    pages = _together_pages()
    quickstart = model_page_url("glm-5.2-quickstart")
    pages[quickstart] = FixtureHop(body=b"\xff\xfe not utf-8")
    _update(store, pages, archive_sources=True)
    root = _archive_root(store)
    row = next(
        item
        for item in load_failures(root, completed_refresh_ids(root)[-1])
        if item["source_url"] == quickstart
    )
    assert row["stage"] == "decode"
    assert row["http_status"] is None
    assert row["exception_class"] == "TogetherFetchError"
    assert b"\xff\xfe" not in b"".join(path.read_bytes() for path in _blob_files(root))


def test_required_source_failure_keeps_diagnostics(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    pages = _together_pages()
    pages[TOGETHER_DEPRECATIONS_URL] = FixtureHop(
        status=500,
        body=b"secret-token",
        headers=(("Set-Cookie", "session=secret-token"),),
    )
    result = _update(store, pages, archive_sources=True)
    assert not result.ok
    assert not result.committed
    assert not store.catalog_path.exists()
    root = _archive_root(store)
    refresh_id = completed_refresh_ids(root)[-1]
    manifest = load_manifest(root, refresh_id)
    assert manifest is not None
    sources = manifest["sources"]
    assert isinstance(sources, list)
    assert any(item["source_id"] == "together-docs-serverless-models" for item in sources)
    failure = next(
        item
        for item in load_failures(root, refresh_id)
        if item["source_id"] == "together-docs-deprecations"
    )
    assert failure["stage"] == "http_status"
    assert failure["http_status"] == 500
    assert failure["exception_class"] == "TogetherFetchError"
    assert SECRET not in _archive_text(root)


def test_rejected_same_origin_path_keeps_sanitized_redirect_diagnostics(tmp_path: Path) -> None:
    requested = model_page_url("glm-5.2-quickstart")
    destination = model_page_url("kimi-k3-quickstart")
    store = FileCatalogStore(root=tmp_path / "cache")
    pages = _together_pages()
    pages[requested] = FixtureHop(
        status=302,
        location=destination,
        headers=(("Set-Cookie", f"session={SECRET}"), ("Content-Type", "text/markdown")),
        body=SECRET.encode(),
    )
    result = _update(store, pages, archive_sources=True)
    assert result.committed
    status, reasons = _publication(store)
    assert status == "incomplete"
    assert (
        "optional_source_failed:SOURCE_FETCH_FAILED:together-docs-model-page:glm-5.2-quickstart"
        in reasons
    )
    root = _archive_root(store)
    failure = next(
        item
        for item in load_failures(root, completed_refresh_ids(root)[-1])
        if item["source_url"] == requested
    )
    assert failure["stage"] == "security"
    assert failure["source_url"] == requested
    assert (
        failure["message"] == "Together documentation redirected to a different official document"
    )
    assert failure["redirects"] == [
        {
            "status": 302,
            "destination": destination,
            "accepted": False,
            "reason": "Together documentation redirected to a different official document",
        }
    ]
    assert SECRET not in _archive_text(root)
    assert "Set-Cookie" not in _archive_text(root)


def test_failure_without_redirects_remains_readable(tmp_path: Path) -> None:
    quickstart = model_page_url("glm-5.2-quickstart")
    store = FileCatalogStore(root=tmp_path / "cache")
    pages = _together_pages()
    pages[quickstart] = FixtureHop(status=404, body=b"missing")
    _update(store, pages, archive_sources=True)
    root = _archive_root(store)
    failure = next(
        item
        for item in load_failures(root, completed_refresh_ids(root)[-1])
        if item["source_url"] == quickstart
    )
    assert "redirects" not in failure
    assert failure["stage"] == "http_status"
    assert failure["http_status"] == 404


def test_redirect_record_omits_credentials_and_headers(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    pages = _together_pages()
    pages[TOGETHER_SERVERLESS_URL] = FixtureHop(
        status=302,
        location=(
            "https://secret-token:secret-token@docs.together.ai"
            "/docs/serverless/models.md?token=secret-token#notes"
        ),
        headers=(("Set-Cookie", "session=secret-token"), ("Content-Type", "text/markdown")),
        body=b"secret-token",
    )
    result = _update(store, pages, archive_sources=True)
    assert not result.ok
    root = _archive_root(store)
    failure = load_failures(root, completed_refresh_ids(root)[-1])[0]
    assert failure["stage"] == "security"
    assert failure["http_status"] is None
    assert failure["redirects"] == [
        {
            "status": 302,
            "destination": "https://docs.together.ai/docs/serverless/models.md",
            "accepted": False,
            "reason": "URLs must not include credentials",
        }
    ]
    assert SECRET not in _archive_text(root)
    assert "Set-Cookie" not in _archive_text(root)
    assert "token=" not in _archive_text(root)


def test_archive_write_failure_does_not_change_catalog_or_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(self: SourceArchive, source: object) -> None:
        raise OSError("diagnostic disk full")

    monkeypatch.setattr(SourceArchive, "_store_blob", broken)
    pages = _together_pages()
    plain = FileCatalogStore(root=tmp_path / "plain")
    broken_store = FileCatalogStore(root=tmp_path / "broken")
    _update(plain, pages, archive_sources=False)
    _update(broken_store, pages, archive_sources=True)
    assert plain.catalog_path.read_bytes() == broken_store.catalog_path.read_bytes()
    assert _publication(plain) == _publication(broken_store)
    assert "source archive" not in {notice.code for notice in broken_store.load().notices}
    assert "diagnostic source archive" in capsys.readouterr().err


def test_provider_failure_keeps_other_providers_in_the_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_all_transports(monkeypatch, google_pages=_failed_google_pages())
    store = FileCatalogStore(root=tmp_path / "cache")
    results = update_all_catalogs(store=store, clock=lambda: STAMP, archive_sources=True)
    assert any(item.provider_id == "google" and not item.ok for item in results)
    assert any(item.provider_id == "anthropic" and item.committed for item in results)
    root = _archive_root(store)
    refresh_id = completed_refresh_ids(root)[-1]
    manifest = load_manifest(root, refresh_id)
    assert manifest is not None
    providers = {item["provider_id"] for item in manifest["sources"]}  # type: ignore[index]
    assert "anthropic" in providers
    assert "fireworks" in providers
    failures = load_failures(root, refresh_id)
    assert any(
        item["provider_id"] == "google" and item["stage"] == "http_status" for item in failures
    )


def test_interrupted_manifest_is_not_a_completed_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "source-archive"
    archive = SourceArchive(
        root=root,
        refresh_id=refresh_id_from_stamp(STAMP),
        package_version="0.3.0",
        keep=2,
    )
    archive.record_captured("together", _FakeSource("shared"))
    archive.record_captured("together", _FakeSource("shared", source_id="other-source"))
    assert len(_blob_files(root)) == 1

    def fail_replace(src: object, dst: object) -> None:
        raise OSError("interrupted rename")

    monkeypatch.setattr("samyak.model.source_archive._replace", fail_replace)
    with pytest.raises(OSError, match="interrupted rename"):
        archive.finalize()
    assert completed_refresh_ids(root) == ()
    published = [
        path for path in root.rglob("manifest.json") if not path.parent.name.startswith(".")
    ]
    assert published == []

    unfinished = root / "refreshes" / "unfinished"
    unfinished.mkdir(parents=True)
    (unfinished / "manifest.json").write_text("{", encoding="utf-8")
    hidden = root / "refreshes" / ".hidden.partial"
    hidden.mkdir()
    (hidden / "manifest.json").write_text(
        json.dumps({"refresh_id": ".hidden.partial", "complete": True, "sources": []}),
        encoding="utf-8",
    )
    incomplete = root / "refreshes" / "incomplete-gen"
    incomplete.mkdir()
    (incomplete / "manifest.json").write_text(
        json.dumps({"refresh_id": "incomplete-gen", "complete": False, "sources": []}),
        encoding="utf-8",
    )
    assert completed_refresh_ids(root) == ()


def test_cli_archive_flag_placement() -> None:
    parser = build_parser()
    all_args = parser.parse_args(
        ["model", "update", "all", "--archive-sources", "--archive-keep", "3"]
    )
    assert all_args.provider == "all"
    assert all_args.archive_sources is True
    assert all_args.archive_keep == 3
    one = parser.parse_args(["model", "update", "--archive-sources", "together"])
    assert one.provider == "together"
    assert one.archive_sources is True
    assert one.archive_keep == DEFAULT_ARCHIVE_KEEP
    with pytest.raises(SystemExit) as exc:
        main(["model", "update", "--help"])
    assert exc.value.code == 0


def test_cli_archive_help_documents_retention(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["model", "update", "--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    assert "--archive-sources" in text
    assert "--archive-keep" in text
    assert "Off by default" in text or "off by default" in text.lower()
    assert str(DEFAULT_ARCHIVE_KEEP) in text
    assert "SHA-256" in text


def test_cli_rejects_archive_keep_below_one(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["model", "update", "together", "--archive-sources", "--archive-keep", "0"])
    assert code == 2
    assert "at least 1" in capsys.readouterr().err


def test_orphan_blob_survives_retention(tmp_path: Path) -> None:
    root = tmp_path / "source-archive"
    _commit_generation(root, "2026-01-01T00-00-00Z", (("old-source", "old-only"),))
    orphan = _plant_blob(root, "nobody-listed-this")
    _commit_generation(root, "2026-02-01T00-00-00Z", (("new-source", "new-only"),), keep=1)
    assert orphan.is_file()
    assert not blob_path(root, _FakeSource("old-only", "old-source").content_hash).exists()
    assert blob_path(root, _FakeSource("new-only", "new-source").content_hash).is_file()


def test_unpublished_blob_survives_next_retention(tmp_path: Path) -> None:
    root = tmp_path / "source-archive"
    _commit_generation(root, "2026-01-01T00-00-00Z", (("old-source", "old-only"),))
    unpublished = _plant_blob(root, "captured-but-never-published")
    _commit_generation(root, "2026-02-01T00-00-00Z", (("new-source", "new-only"),), keep=1)
    assert unpublished.is_file()
    assert not (root / "refreshes" / "2026-01-01T00-00-00Z").exists()
    assert not blob_path(root, _FakeSource("old-only", "old-source").content_hash).exists()


def test_retained_hash_survives_when_another_generation_expires(tmp_path: Path) -> None:
    root = tmp_path / "source-archive"
    _commit_generation(
        root,
        "2026-01-01T00-00-00Z",
        (("shared-source", "shared-body"), ("old-source", "old-only")),
    )
    _commit_generation(
        root,
        "2026-02-01T00-00-00Z",
        (("shared-source", "shared-body"), ("new-source", "new-only")),
        keep=1,
    )
    assert not (root / "refreshes" / "2026-01-01T00-00-00Z").exists()
    assert (root / "refreshes" / "2026-02-01T00-00-00Z").is_dir()
    assert blob_path(root, _FakeSource("shared-body", "shared-source").content_hash).is_file()
    assert blob_path(root, _FakeSource("new-only", "new-source").content_hash).is_file()
    assert not blob_path(root, _FakeSource("old-only", "old-source").content_hash).exists()


def test_unreadable_retained_manifest_skips_blob_deletion(tmp_path: Path) -> None:
    root = tmp_path / "source-archive"
    _commit_generation(root, "2026-01-01T00-00-00Z", (("old-source", "old-only"),))
    _commit_generation(root, "2026-02-01T00-00-00Z", (("new-source", "new-only"),))
    orphan = _plant_blob(root, "orphan-beside-corrupt-manifest")
    manifest_path = root / "refreshes" / "2026-02-01T00-00-00Z" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"] = [{"content_hash": "not-a-hash"}]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _retain(root, keep=1)
    assert (root / "refreshes" / "2026-01-01T00-00-00Z").is_dir()
    assert blob_path(root, _FakeSource("old-only", "old-source").content_hash).is_file()
    assert blob_path(root, _FakeSource("new-only", "new-source").content_hash).is_file()
    assert orphan.is_file()


def test_expired_directory_that_remains_skips_blob_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "source-archive"
    _commit_generation(root, "2026-01-01T00-00-00Z", (("old-source", "old-only"),))
    _commit_generation(root, "2026-02-01T00-00-00Z", (("new-source", "new-only"),))
    orphan = _plant_blob(root, "orphan-beside-stuck-generation")

    def leave_directory(path: Path, ignore_errors: bool = False) -> None:
        return

    monkeypatch.setattr("samyak.model.source_archive.shutil.rmtree", leave_directory)
    _retain(root, keep=1)
    assert (root / "refreshes" / "2026-01-01T00-00-00Z").is_dir()
    assert blob_path(root, _FakeSource("old-only", "old-source").content_hash).is_file()
    assert blob_path(root, _FakeSource("new-only", "new-source").content_hash).is_file()
    assert orphan.is_file()


def test_blob_write_failure_does_not_evict_previous_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    _update(store, _together_pages(), stamp=STAMP, archive_sources=True, archive_keep=1)
    root = _archive_root(store)
    first_id = completed_refresh_ids(root)[-1]
    first_manifest = load_manifest(root, first_id)
    assert first_manifest is not None
    hashes = [str(item["content_hash"]) for item in first_manifest["sources"]]  # type: ignore[index]

    def broken(self: SourceArchive, source: object) -> None:
        raise OSError("diagnostic disk full")

    monkeypatch.setattr(SourceArchive, "_store_blob", broken)
    result = _update(
        store,
        _together_pages(),
        stamp=STAMP_LATER,
        archive_sources=True,
        archive_keep=1,
    )
    assert result.committed
    assert completed_refresh_ids(root) == (first_id,)
    for content_hash in hashes:
        assert blob_path(root, content_hash).is_file()
    assert all("archive" not in notice.code for notice in store.load().notices)
    assert "diagnostic source archive" in capsys.readouterr().err


def test_incomplete_generation_does_not_count_against_keep(tmp_path: Path) -> None:
    root = tmp_path / "source-archive"
    _commit_generation(root, "2026-01-01T00-00-00Z", (("old-source", "old-body"),), keep=3)
    _commit_generation(root, "2026-03-01T00-00-00Z", (("new-source", "new-body"),), keep=3)
    incomplete = root / "refreshes" / "2026-02-01T00-00-00Z"
    incomplete.mkdir()
    (incomplete / "manifest.json").write_text(
        json.dumps(
            {
                "refresh_id": "2026-02-01T00-00-00Z",
                "complete": False,
                "sources": [{"content_hash": "sha256:" + ("cd" * 32)}],
            }
        ),
        encoding="utf-8",
    )
    _retain(root, keep=2)
    assert completed_refresh_ids(root) == ("2026-01-01T00-00-00Z", "2026-03-01T00-00-00Z")
    assert (root / "refreshes" / "2026-01-01T00-00-00Z").is_dir()
    assert incomplete.is_dir()
    assert blob_path(root, _FakeSource("old-body", "old-source").content_hash).is_file()


def test_archive_enabled_matches_disabled_catalog_and_history(tmp_path: Path) -> None:
    pages = _together_pages()
    revised = _together_pages()
    revised[TOGETHER_SERVERLESS_URL] = FixtureHop(
        body=revised[TOGETHER_SERVERLESS_URL].body.replace(
            b"`openai/sora-2`",
            b"`openai/sora-2-removed`",
        )
    )
    plain = FileCatalogStore(root=tmp_path / "plain")
    archived = FileCatalogStore(root=tmp_path / "archived")
    for store, enabled in ((plain, False), (archived, True)):
        _update(store, pages, stamp=STAMP, archive_sources=enabled)
        _update(store, revised, stamp=STAMP_LATER, archive_sources=enabled)
    assert plain.catalog_path.read_bytes() == archived.catalog_path.read_bytes()
    plain_history = plain.models_dir / "lifecycle-history.json"
    archived_history = archived.models_dir / "lifecycle-history.json"
    assert plain_history.is_file() and archived_history.is_file()
    assert plain_history.read_bytes() == archived_history.read_bytes()
    assert _publication(plain) == _publication(archived)
    assert not _archive_root(plain).exists()
    assert not (_archive_root(plain) / ".lock").exists()
    assert len(completed_refresh_ids(_archive_root(archived))) == 2


def test_catalog_survives_archive_publication_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_replace(src: object, dst: object) -> None:
        raise OSError("interrupted rename")

    monkeypatch.setattr("samyak.model.source_archive._replace", fail_replace)
    pages = _together_pages()
    revised = _together_pages()
    revised[TOGETHER_SERVERLESS_URL] = FixtureHop(
        body=revised[TOGETHER_SERVERLESS_URL].body.replace(
            b"`openai/sora-2`",
            b"`openai/sora-2-removed`",
        )
    )
    plain = FileCatalogStore(root=tmp_path / "plain")
    archived = FileCatalogStore(root=tmp_path / "archived")
    _update(plain, pages, stamp=STAMP, archive_sources=False)
    _update(archived, pages, stamp=STAMP, archive_sources=False)
    plain_result = _update(plain, revised, stamp=STAMP_LATER, archive_sources=False)
    archived_result = _update(archived, revised, stamp=STAMP_LATER, archive_sources=True)
    assert plain_result.committed and archived_result.committed
    assert archived.catalog_path.is_file()
    assert plain.catalog_path.read_bytes() == archived.catalog_path.read_bytes()
    plain_history = plain.models_dir / "lifecycle-history.json"
    archived_history = archived.models_dir / "lifecycle-history.json"
    assert plain_history.is_file() and archived_history.is_file()
    assert plain_history.read_bytes() == archived_history.read_bytes()
    assert _publication(plain) == _publication(archived)
    assert all("archive" not in notice.code for notice in archived.load().notices)
    assert completed_refresh_ids(_archive_root(archived)) == ()
    assert "diagnostic source archive" in capsys.readouterr().err


def test_named_batch_parse_failure_keeps_source_attribution(tmp_path: Path) -> None:
    archive = SourceArchive(
        root=tmp_path / "archive",
        refresh_id="parse-named",
        package_version="0.3.0",
        keep=2,
    )
    serverless = _FakeSource(
        "serverless",
        "together-docs-serverless-models",
        "https://docs.together.ai/docs/serverless/models.md",
    )
    deprecations = _FakeSource(
        "deprecations",
        "together-docs-deprecations",
        "https://docs.together.ai/docs/deprecations.md",
    )
    note_parse_failure(
        archive,
        "together",
        [serverless, deprecations],
        TogetherParseError("Together documentation together-docs-deprecations could not be parsed"),
    )
    row = archive._failures[0]
    assert row["source_id"] == "together-docs-deprecations"
    assert row["source_url"] == deprecations.source_url
    assert row["stage"] == "parse"


def test_unnamed_together_batch_parse_failure_uses_provider(tmp_path: Path) -> None:
    archive = SourceArchive(
        root=tmp_path / "archive",
        refresh_id="parse-unnamed",
        package_version="0.3.0",
        keep=2,
    )
    serverless = _FakeSource(
        "serverless",
        "together-docs-serverless-models",
        "https://docs.together.ai/docs/serverless/models.md",
    )
    deprecations = _FakeSource(
        "deprecations",
        "together-docs-deprecations",
        "https://docs.together.ai/docs/deprecations.md",
    )
    note_parse_failure(
        archive,
        "together",
        [serverless, deprecations],
        TogetherParseError("Together deprecations page is missing deprecation documentation"),
    )
    row = archive._failures[0]
    assert row["source_id"] == "together"
    assert row["source_url"] == ""
    assert row["provider_id"] == "together"
    assert row["stage"] == "parse"
    assert row["exception_class"] == "TogetherParseError"


def test_second_process_cannot_write_or_collect_while_lock_is_held(tmp_path: Path) -> None:
    root = tmp_path / "source-archive"
    _commit_generation(root, "2026-01-01T00-00-00Z", (("expired-source", "expired-body"),))
    expired_blob = blob_path(root, _FakeSource("expired-body", "expired-source").content_hash)
    holder_blob = blob_path(root, _FakeSource("holder-unpublished", "holder-source").content_hash)
    challenger_source = _FakeSource("challenger-body", "challenger-source")
    challenger_blob = blob_path(root, challenger_source.content_hash)
    context, previous = _spawn_context()
    held = context.Event()
    release = context.Event()
    holder = context.Process(target=_lock_holder, args=(str(root), held, release))
    challenger = context.Process(target=_lock_challenger, args=(str(root),))
    holder.start()
    try:
        assert held.wait(20)
        challenger.start()
        challenger.join(20)
        assert challenger.exitcode == 0
        assert expired_blob.is_file()
        assert (root / "refreshes" / "2026-01-01T00-00-00Z").is_dir()
        assert holder_blob.is_file()
        assert not challenger_blob.exists()
        assert "challenger-gen" not in completed_refresh_ids(root)
    finally:
        _restore_pythonpath(previous)
        release.set()
        holder.join(20)
    assert holder.exitcode == 0


def test_archive_lock_is_released_after_success(tmp_path: Path) -> None:
    root = tmp_path / "source-archive"
    context, previous = _spawn_context()
    proc = context.Process(target=_lock_release_after_success, args=(str(root),))
    proc.start()
    _restore_pythonpath(previous)
    proc.join(10)
    assert proc.exitcode == 0
    follower = SourceArchive(
        root=root,
        refresh_id="follower",
        package_version="0.3.0",
        keep=2,
    )
    follower.record_captured("together", _FakeSource("follower-body", "follower-source"))
    assert not follower.write_failed
    follower.finalize()
    assert blob_path(root, _FakeSource("follower-body", "follower-source").content_hash).is_file()


def test_archive_lock_is_released_after_write_failure(tmp_path: Path) -> None:
    root = tmp_path / "source-archive"
    _commit_generation(root, "2026-01-01T00-00-00Z", (("kept-source", "kept-body"),), keep=1)
    kept_blob = blob_path(root, _FakeSource("kept-body", "kept-source").content_hash)
    context, previous = _spawn_context()
    proc = context.Process(target=_lock_release_after_failure, args=(str(root),))
    proc.start()
    _restore_pythonpath(previous)
    proc.join(10)
    assert proc.exitcode == 0
    assert (root / "refreshes" / "2026-01-01T00-00-00Z").is_dir()
    assert kept_blob.is_file()
    assert "failed-holder" not in completed_refresh_ids(root)
    follower = SourceArchive(
        root=root,
        refresh_id="follower-after-failure",
        package_version="0.3.0",
        keep=2,
    )
    follower.record_captured("together", _FakeSource("after-failure", "after-failure-source"))
    assert not follower.write_failed
    follower.finalize()
