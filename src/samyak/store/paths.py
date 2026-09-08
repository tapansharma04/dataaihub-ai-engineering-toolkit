"""Cache directory resolution for local Samyak data."""

from __future__ import annotations

import os
import sys
from pathlib import Path

CACHE_ENV_VAR = "SAMYAK_CACHE_DIR"


def default_cache_dir() -> Path:
    """Return the per-user Samyak cache directory.

    Resolution order:

    1. ``SAMYAK_CACHE_DIR``
    2. Windows: ``%LOCALAPPDATA%/samyak``
    3. ``$XDG_CACHE_HOME/samyak`` when that variable is set
    4. ``~/.cache/samyak``
    """
    override = os.environ.get(CACHE_ENV_VAR)
    if override:
        return Path(override).expanduser()

    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "samyak"
        return Path.home() / "AppData" / "Local" / "samyak"

    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "samyak"
    return Path.home() / ".cache" / "samyak"
