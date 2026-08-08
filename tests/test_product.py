"""Samyak productization and CLI foundation tests."""

from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

from helpers import make_corpus
from samyak import __version__, analyze_corpus
from samyak.cli import main
from samyak.corpus.report import render_json_report, render_text_report


def test_package_version_constant() -> None:
    assert __version__ == "0.1.0"


def test_package_metadata_name() -> None:
    try:
        dist = importlib.metadata.metadata("samyak")
    except importlib.metadata.PackageNotFoundError:
        # Editable install may be absent when running via PYTHONPATH only.
        return
    assert dist["Name"].lower() == "samyak"
    assert dist["Version"] == __version__


def test_report_product_capability_identity(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {"doc.txt": "Product identity should appear in machine-readable reports.\n" * 3},
    )
    report = analyze_corpus(corpus)
    payload = json.loads(render_json_report(report))
    assert payload["product"] == "samyak"
    assert payload["capability"] == "corpus"
    assert payload["version"] == "0.1.0"
    assert "utility" not in payload
    assert report.product == "samyak"
    assert report.capability == "corpus"
    assert report.version == "0.1.0"

    text = render_text_report(report)
    assert "Samyak Corpus Intelligence Report" in text
    assert "Product:                  samyak" in text
    assert "Capability:               corpus" in text
    assert "Version:                  0.1.0" in text


def test_cli_help() -> None:
    env = os.environ.copy()
    src = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "samyak", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert "corpus" in result.stdout.lower()
    assert "Samyak" in result.stdout


def test_cli_version() -> None:
    env = os.environ.copy()
    src = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "samyak", "--version"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert "samyak" in result.stdout.lower()
    assert __version__ in result.stdout
    assert "0.1.0" in result.stdout


def test_cli_corpus_subcommand_help() -> None:
    env = os.environ.copy()
    src = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "samyak", "corpus", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert "--format" in result.stdout
    assert "--output" in result.stdout
    assert "--no-progress" in result.stdout


def test_cli_requires_corpus_subcommand() -> None:
    code = main([])
    assert code == 2


def test_cli_corpus_invocation(tmp_path: Path) -> None:
    corpus = make_corpus(
        tmp_path,
        {"doc.txt": "Corpus subcommand integration document with enough content.\n"},
    )
    output = tmp_path / "out.json"
    code = main(["corpus", str(corpus), "--format", "json", "-o", str(output)])
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["product"] == "samyak"
    assert payload["capability"] == "corpus"
    assert payload["version"] == "0.1.0"
    assert "utility" not in payload
