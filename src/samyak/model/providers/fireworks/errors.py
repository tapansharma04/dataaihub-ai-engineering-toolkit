"""Fireworks AI adapter errors. Not part of the public Samyak API."""

from __future__ import annotations

from samyak.model.errors import ModelCatalogError


class FireworksAdapterError(ModelCatalogError):
    """Failure while reading Fireworks documentation into catalog records."""


class FireworksParseError(FireworksAdapterError):
    """Official Fireworks documentation could not be parsed.

    Raised when a required identity field is missing or the document structure
    is not the labeled Markdown this adapter understands. Messages name the
    document, not parser internals.
    """


class FireworksFetchError(FireworksAdapterError):
    """Official Fireworks documentation could not be retrieved or captured."""
