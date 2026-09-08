"""Pytest configuration."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def samyak_cache_dir(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Keep tests from writing into the developer's real Samyak cache."""
    cache = tmp_path_factory.mktemp("samyak-cache")
    monkeypatch.setenv("SAMYAK_CACHE_DIR", str(cache))
    return cache
