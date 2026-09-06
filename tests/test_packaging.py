"""Lightweight installed-package contract checks (no wheel rebuild)."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import samyak


def test_console_script_entry_point() -> None:
    eps = importlib.metadata.entry_points(group="console_scripts")
    matches = [ep for ep in eps if ep.name == "samyak"]
    assert matches, "console script 'samyak' is missing from package metadata"
    assert matches[0].value == "samyak.cli:main"


def test_requires_python_and_license_metadata() -> None:
    dist = importlib.metadata.metadata("samyak")
    assert dist["Name"].lower() == "samyak"
    license_text = dist.get("License-Expression") or dist.get("License") or ""
    assert "MIT" in license_text
    requires = dist["Requires-Python"]
    assert "3.12" in requires


def test_runtime_package_excludes_private_and_includes_py_typed() -> None:
    files = importlib.metadata.files("samyak")
    package_root = Path(samyak.__file__).resolve().parent
    assert (package_root / "py.typed").is_file()
    assert not any(part == ".private" for part in package_root.parts)

    if files is None:
        return

    names = [str(item).replace("\\", "/") for item in files]
    assert not any(".private" in name for name in names)
    assert not any("/benchmarks/" in name for name in names)
    # Editable installs list dist-info + the console script, not package modules.
    # py.typed is verified on the importable package path above.
