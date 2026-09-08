"""Errors raised when two saved runs cannot be compared."""

from __future__ import annotations


class IncompatibleRunsError(ValueError):
    """The two runs cannot be compared under the current report model."""

    def __init__(self, message: str) -> None:
        self.user_message = message
        super().__init__(message)
