"""Anthropic adapter errors. Not part of the public Samyak API."""

from __future__ import annotations

from samyak.model.errors import ModelCatalogError


class AnthropicAdapterError(ModelCatalogError):
    """Failure while reading Anthropic documentation into catalog records."""


class AnthropicParseError(AnthropicAdapterError):
    """Official Anthropic documentation could not be parsed.

    Raised when a required identity field is missing or the document structure
    is not the labeled Markdown this adapter understands. Messages name the
    document, not parser internals.
    """


class AnthropicFetchError(AnthropicAdapterError):
    """Official Anthropic documentation could not be retrieved or captured."""
