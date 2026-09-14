"""Custom exception hierarchy for physicskit.chaos.

All exceptions physicskit.chaos raises intentionally (as opposed to exceptions that
bubble up unexpectedly from a dependency) inherit from :class:`ChaoskitError`,
so callers can catch "any physicskit.chaos-raised problem" with a single ``except
ChaoskitError``. Each concrete exception also inherits from the closest
matching built-in exception type (e.g. :class:`InvalidParameterError` is
also a :class:`ValueError`), so existing code that catches the built-in type
continues to work unchanged.
"""

from __future__ import annotations


class ChaoskitError(Exception):
    """Base class for all exceptions raised intentionally by physicskit.chaos."""


class InvalidParameterError(ChaoskitError, ValueError):
    """A system or tool was constructed or called with an invalid parameter.

    Examples include a billiard's scatterer not fitting inside its cell, or a
    map's parameter falling outside its valid range.
    """
