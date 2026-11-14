"""Application Readiness project discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from helpers import make_project, write_text
from samyak.app.discovery import discover_project, validate_project_root
from samyak.app.errors import AppPathError
from samyak.app.models import (
    NOTICE_ANALYSIS_TRUNCATED,
    NOTICE_FILE_SKIPPED_BINARY,
    NOTICE_FILE_SKIPPED_TOO_LARGE,
    AppConfig,
)
from samyak.app.pipeline import extract_application


def _relative_paths(project: Path, config: AppConfig | None = None) -> list[str]:
    result = discover_project(project, config)
    return [item.relative_path for item in result.files]


def test_missing_project_path(tmp_path: Path) -> None:
    with pytest.raises(AppPathError, match="does not exist"):
        validate_project_root(tmp_path / "missing")


def test_file_is_not_a_project_root(tmp_path: Path) -> None:
    target = write_text(tmp_path / "file.py", "x = 1\n")
    with pytest.raises(AppPathError, match="not a directory"):
        validate_project_root(target)


def test_skips_git_directory(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {
            "ok.py": "client.create(model='ok')\n",
            ".git/hidden.py": "client.create(model='hidden')\n",
        },
    )
    assert _relative_paths(project) == ["ok.py"]


def test_skips_node_modules(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {
            "ok.py": "client.create(model='ok')\n",
            "node_modules/pkg/index.py": "client.create(model='vendored')\n",
        },
    )
    assert _relative_paths(project) == ["ok.py"]


def test_skips_venv(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {
            "ok.py": "client.create(model='ok')\n",
            ".venv/lib/site.py": "client.create(model='venv')\n",
        },
    )
    assert _relative_paths(project) == ["ok.py"]


@pytest.mark.parametrize(
    "directory",
    (
        "build",
        "dist",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        "env",
        "vendor",
        "target",
    ),
)
def test_skips_build_dist_and_cache_directories(tmp_path: Path, directory: str) -> None:
    project = make_project(
        tmp_path,
        {
            "ok.py": "client.create(model='ok')\n",
            f"{directory}/skip.py": "client.create(model='skip')\n",
        },
    )
    assert _relative_paths(project) == ["ok.py"]


def test_external_file_symlink_is_skipped(tmp_path: Path) -> None:
    project = make_project(tmp_path, {"ok.py": "client.create(model='ok')\n"})
    outside = tmp_path / "outside.py"
    outside.write_text("client.create(model='outside')\n", encoding="utf-8")
    link = project / "linked.py"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks not supported in this environment")
    assert _relative_paths(project) == ["ok.py"]


def test_directory_symlink_is_not_followed(tmp_path: Path) -> None:
    project = make_project(tmp_path, {"ok.py": "client.create(model='ok')\n"})
    outside_dir = tmp_path / "outside"
    write_text(outside_dir / "hidden.py", "client.create(model='hidden')\n")
    link = project / "linked_dir"
    try:
        link.symlink_to(outside_dir)
    except OSError:
        pytest.skip("symlinks not supported in this environment")
    assert _relative_paths(project) == ["ok.py"]


def test_broken_symlink_is_skipped(tmp_path: Path) -> None:
    project = make_project(tmp_path, {"ok.py": "client.create(model='ok')\n"})
    broken = project / "broken.py"
    try:
        broken.symlink_to(project / "does-not-exist.py")
    except OSError:
        pytest.skip("symlinks not supported in this environment")
    assert _relative_paths(project) == ["ok.py"]


def test_oversized_file_is_skipped_with_notice(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {
            "ok.py": "client.create(model='ok')\n",
            "huge.py": "x" * 50,
        },
    )
    config = AppConfig(max_file_bytes=40)
    result = discover_project(project, config)
    assert [item.relative_path for item in result.files] == ["ok.py"]
    assert result.skipped_oversized == 1
    assert any(item.code == NOTICE_FILE_SKIPPED_TOO_LARGE for item in result.notices)
    assert any(item.path == "huge.py" for item in result.notices)


def test_nul_binary_file_is_skipped_with_notice(tmp_path: Path) -> None:
    project = make_project(tmp_path, {"ok.py": "client.create(model='ok')\n"})
    (project / "binary.py").write_bytes(b"print('ok')\n\x00more")
    result = discover_project(project)
    assert [item.relative_path for item in result.files] == ["ok.py"]
    assert result.skipped_binary == 1
    assert any(item.code == NOTICE_FILE_SKIPPED_BINARY for item in result.notices)


def test_relative_posix_paths(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {"src/pkg/app.py": "client.create(model='ok')\n"},
    )
    paths = _relative_paths(project)
    assert paths == ["src/pkg/app.py"]
    assert "\\" not in paths[0]


def test_file_count_truncation_notice(tmp_path: Path) -> None:
    project = make_project(
        tmp_path,
        {
            "a.py": "client.create(model='a')\n",
            "b.py": "client.create(model='b')\n",
            "c.py": "client.create(model='c')\n",
        },
    )
    config = AppConfig(max_analyzed_files=2)
    result = discover_project(project, config)
    assert result.truncated is True
    assert len(result.files) == 2
    assert any(item.code == NOTICE_ANALYSIS_TRUNCATED for item in result.notices)
    extraction = extract_application(project, config)
    assert extraction.report.summary.truncated is True
    assert extraction.report.summary.analyzed_files == 2
