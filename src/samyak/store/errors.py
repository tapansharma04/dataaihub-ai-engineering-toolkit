"""Errors for the internal run store."""

from __future__ import annotations


class RunStoreError(Exception):
    """Base class for run-store failures."""


class RunNotFoundError(RunStoreError):
    """The requested run id does not exist."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(f"run not found: {run_id}")


class RunCorruptError(RunStoreError):
    """A run directory exists but cannot be read as a complete run."""


class RunSchemaError(RunStoreError):
    """The stored report schema version is not supported by this Samyak version."""

    def __init__(self, run_id: str, schema_version: object) -> None:
        self.run_id = run_id
        self.schema_version = schema_version
        super().__init__(f"run {run_id} uses unsupported report schema version {schema_version!r}")
