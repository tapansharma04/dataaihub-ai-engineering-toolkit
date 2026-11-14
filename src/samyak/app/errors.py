"""Errors for Application Readiness.

Not part of the public Samyak API.
"""

from __future__ import annotations

USAGE_EXIT_CODE = 2
ENGINE_EXIT_CODE = 3


class AppPathError(ValueError):
    """Raised when the application project path is invalid."""
