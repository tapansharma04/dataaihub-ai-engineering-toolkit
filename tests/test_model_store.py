"""Filesystem overlay tests for the Model Intelligence catalog store."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from samyak.model.catalog import CatalogFreshness, CatalogOverlay, FreshnessStatus, build_catalog
from samyak.model.errors import CatalogNotFoundError, CatalogSchemaError, CatalogStoreError
from samyak.model.identity import IdentityKind, ModelIdentity
from samyak.model.records import ModelRecord
from samyak.model.serialize import catalog_from_json
from samyak.model.store import FileCatalogStore
from samyak.store.paths import default_cache_dir

GENERATED_AT = "2026-09-10T18:00:00+00:00"
REPLACED_AT = "2026-09-10T19:00:00+00:00"


def _model(provider_model_id: str = "gpt-test") -> ModelRecord:
    return ModelRecord(
        identity=ModelIdentity(provider_id="openai", provider_model_id=provider_model_id),
        identity_kind=IdentityKind.CANONICAL,
    )


def _catalog(*, generated_at: str, model_id: str = "gpt-test"):
    return build_catalog(
        models=(_model(model_id),),
        generated_at=generated_at,
        freshness=CatalogFreshness(
            status=FreshnessStatus.AS_OF,
            generated_at=generated_at,
            overlay=CatalogOverlay.USER_CACHE,
        ),
    )


def test_no_existing_catalog(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    assert store.exists() is False
    with pytest.raises(CatalogNotFoundError, match="no local model catalog"):
        store.load()
    assert not (tmp_path / "cache" / "runs").exists()


def test_successful_first_write(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    catalog = _catalog(generated_at=GENERATED_AT)
    store.save(catalog)
    assert store.exists() is True
    assert store.catalog_path == tmp_path / "cache" / "models" / "catalog.json"
    loaded = store.load()
    assert loaded.to_dict() == catalog.to_dict()
    assert loaded.freshness.overlay is CatalogOverlay.USER_CACHE
    assert loaded.generated_at == GENERATED_AT
    assert not store.backup_path.exists()
    assert not (tmp_path / "cache" / "runs").exists()


def test_successful_replacement_keeps_backup(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    first = _catalog(generated_at=GENERATED_AT, model_id="gpt-one")
    second = _catalog(generated_at=REPLACED_AT, model_id="gpt-two")
    store.save(first)
    first_bytes = store.catalog_path.read_bytes()
    store.save(second)
    assert store.load().models[0].provider_model_id == "gpt-two"
    assert store.load().generated_at == REPLACED_AT
    assert store.backup_path.is_file()
    assert store.backup_path.read_bytes() == first_bytes
    restored = catalog_from_json(store.backup_path.read_text(encoding="utf-8"))
    assert restored.models[0].provider_model_id == "gpt-one"


def test_backup_write_failure_does_not_block_or_invalidate_live_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    store.save(_catalog(generated_at=GENERATED_AT, model_id="gpt-one"))
    previous_live = store.catalog_path.read_bytes()

    def boom_backup(path: Path, payload: bytes) -> None:
        if path == store.backup_path:
            raise OSError("backup disk full")
        raise AssertionError(f"unexpected atomic_write_bytes target {path}")

    monkeypatch.setattr("samyak.model.store.atomic_write_bytes", boom_backup)
    store.save(_catalog(generated_at=REPLACED_AT, model_id="gpt-two"))
    loaded = store.load()
    assert loaded.generated_at == REPLACED_AT
    assert loaded.models[0].provider_model_id == "gpt-two"
    assert store.catalog_path.read_bytes() != previous_live
    round_trip = catalog_from_json(store.catalog_path.read_text(encoding="utf-8"))
    assert round_trip.to_dict() == loaded.to_dict()
    assert not store.backup_path.exists() or store.backup_path.read_bytes() == previous_live


def test_atomic_write_preserves_previous_on_fsync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    store.save(_catalog(generated_at=GENERATED_AT))
    previous = store.catalog_path.read_text(encoding="utf-8")

    def boom(_fd: int) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("samyak.store.atomic.os.fsync", boom)
    with pytest.raises(CatalogStoreError, match="could not be written"):
        store.save(_catalog(generated_at=REPLACED_AT, model_id="gpt-new"))
    assert store.catalog_path.read_text(encoding="utf-8") == previous
    leftover = list((tmp_path / "cache" / "models").glob(".catalog.json.*.tmp"))
    assert leftover == []


def test_corrupted_existing_catalog(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    store.catalog_path.parent.mkdir(parents=True)
    store.catalog_path.write_text("{not-json", encoding="utf-8")
    assert store.exists() is True
    with pytest.raises(CatalogStoreError, match="corrupt"):
        store.load()


def test_non_utf8_existing_catalog_is_corrupt(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    store.catalog_path.parent.mkdir(parents=True)
    store.catalog_path.write_bytes(b"\xff\xfecatalog")
    assert store.exists() is True
    with pytest.raises(CatalogStoreError, match="corrupt"):
        store.load()


def test_unsupported_schema_is_schema_error(tmp_path: Path) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    store.save(_catalog(generated_at=GENERATED_AT))
    payload = json.loads(store.catalog_path.read_text(encoding="utf-8"))
    payload["catalog_schema_version"] = 99
    store.catalog_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CatalogSchemaError, match="unsupported catalog_schema_version"):
        store.load()


def test_failed_write_preserves_previous_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    store.save(_catalog(generated_at=GENERATED_AT))
    previous = store.catalog_path.read_bytes()

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("samyak.model.store.atomic_write_json", boom)
    with pytest.raises(CatalogStoreError, match="could not be written"):
        store.save(_catalog(generated_at=REPLACED_AT, model_id="gpt-new"))
    assert store.catalog_path.read_bytes() == previous
    assert json.loads(previous)["generated_at"] == GENERATED_AT


def test_serialization_failure_does_not_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FileCatalogStore(root=tmp_path / "cache")
    store.save(_catalog(generated_at=GENERATED_AT))
    previous = store.catalog_path.read_bytes()

    monkeypatch.setattr(
        "samyak.model.store.catalog_from_dict",
        lambda payload: (_ for _ in ()).throw(TypeError("cannot encode")),
    )
    with pytest.raises(CatalogStoreError, match="could not be serialized"):
        store.save(_catalog(generated_at=REPLACED_AT, model_id="gpt-new"))
    assert store.catalog_path.read_bytes() == previous


def test_default_root_uses_samyak_cache_dir(samyak_cache_dir: Path) -> None:
    store = FileCatalogStore()
    assert store.root == default_cache_dir()
    assert store.root == samyak_cache_dir
    assert store.catalog_path == samyak_cache_dir / "models" / "catalog.json"
    assert store.backup_path == samyak_cache_dir / "models" / "catalog.json.bak"
