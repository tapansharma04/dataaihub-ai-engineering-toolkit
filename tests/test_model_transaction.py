"""Crash-recovery checks for coordinated catalog/history persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from samyak.model.catalog import CatalogFreshness, CatalogOverlay, FreshnessStatus, build_catalog
from samyak.model.identity import IdentityKind, ModelIdentity
from samyak.model.records import ModelRecord
from samyak.model.store import FileCatalogStore
from samyak.model.transaction import CatalogHistoryTransaction, CatalogHistoryTransactionError

STAMP_ONE = "2026-11-07T10:00:00+00:00"
STAMP_TWO = "2026-11-07T11:00:00+00:00"


def _catalog(stamp: str, model_id: str):
    return build_catalog(
        models=(
            ModelRecord(
                identity=ModelIdentity(provider_id="openai", provider_model_id=model_id),
                identity_kind=IdentityKind.CANONICAL,
            ),
        ),
        generated_at=stamp,
        freshness=CatalogFreshness(
            status=FreshnessStatus.AS_OF,
            generated_at=stamp,
            overlay=CatalogOverlay.USER_CACHE,
        ),
    )


def test_transaction_commits_a_valid_pair(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    FileCatalogStore(root=root).save(_catalog(STAMP_ONE, "before"))
    with CatalogHistoryTransaction(root) as transaction:
        _before, history = transaction.load_pair()
        transaction.commit(_catalog(STAMP_TWO, "after"), history)
    with CatalogHistoryTransaction(root) as transaction:
        catalog, history = transaction.load_pair()
    assert catalog is not None
    assert catalog.models[0].provider_model_id == "after"
    assert history.events == ()
    assert not (root / "models" / ".catalog-history-transaction.json").exists()


def test_recovery_completes_after_first_visible_transition(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "cache"
    FileCatalogStore(root=root).save(_catalog(STAMP_ONE, "before"))
    original = __import__("samyak.model.transaction", fromlist=["_write_durable"])._write_durable
    failed = False

    def fail_history(path: Path, payload: bytes) -> None:
        nonlocal failed
        if path.name == "lifecycle-history.json" and not failed:
            failed = True
            raise OSError("simulated interruption")
        original(path, payload)

    monkeypatch.setattr("samyak.model.transaction._write_durable", fail_history)
    with (
        pytest.raises(OSError, match="simulated interruption"),
        CatalogHistoryTransaction(root) as transaction,
    ):
        _before, history = transaction.load_pair()
        transaction.commit(_catalog(STAMP_TWO, "after"), history)
    monkeypatch.setattr("samyak.model.transaction._write_durable", original)
    with CatalogHistoryTransaction(root) as transaction:
        catalog, history = transaction.load_pair()
    assert catalog is not None
    assert catalog.models[0].provider_model_id == "after"
    assert history.events == ()


def test_corrupt_pending_journal_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    models = root / "models"
    models.mkdir(parents=True)
    (models / ".catalog-history-transaction.json").write_text("not-json", encoding="utf-8")
    with (
        pytest.raises(CatalogHistoryTransactionError, match="journal is corrupt"),
        CatalogHistoryTransaction(root),
    ):
        pass


def test_writer_lock_has_bounded_failure(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    with CatalogHistoryTransaction(root) as held:
        with (
            pytest.raises(CatalogHistoryTransactionError, match="timed out"),
            CatalogHistoryTransaction(root, wait_seconds=0),
        ):
            pass
        assert held is not None
