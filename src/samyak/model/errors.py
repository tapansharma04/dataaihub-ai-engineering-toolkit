"""Errors for the Model Intelligence catalog domain.

Not part of the public Samyak API in this milestone.
"""

from __future__ import annotations


class ModelCatalogError(ValueError):
    """Base error for catalog domain, validation, and decode failures."""


class CatalogValidationError(ModelCatalogError):
    """A catalog or model record violates domain invariants."""


class CatalogDecodeError(ModelCatalogError):
    """Catalog JSON / dict cannot be reconstructed."""


class CatalogStoreError(ModelCatalogError):
    """The local catalog file could not be read or written."""


class CatalogNotFoundError(CatalogStoreError):
    """No local model catalog overlay exists yet."""


class CatalogSchemaError(ModelCatalogError):
    """The stored catalog schema version is not supported by this Samyak version."""

    def __init__(self, schema_version: object) -> None:
        self.schema_version = schema_version
        super().__init__(f"unsupported catalog_schema_version {schema_version!r}")
