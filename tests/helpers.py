"""Shared test helpers."""

from __future__ import annotations

from pathlib import Path


def write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def make_corpus(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    for relative, content in files.items():
        write_text(root / relative, content)
    return root
