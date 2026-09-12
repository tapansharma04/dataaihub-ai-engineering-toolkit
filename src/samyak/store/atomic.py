"""Atomic JSON writes for persisted run files."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON via a temporary file, fsync, and atomic replace."""
    text = json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False) + "\n"
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write bytes via a temporary file, fsync, and atomic replace.

    The destination is replaced only after the temporary file is fully written
    and flushed. A failure leaves the original file intact when it existed.
    ``os.replace`` is used so the swap is atomic on POSIX and Windows when the
    temporary file is on the same directory/volume.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        with suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        raise
