"""Custom exception hierarchy for physicskit.fluids.

All exceptions physicskit.fluids raises intentionally (as opposed to
exceptions that bubble up unexpectedly from a dependency) inherit from
:class:`FluidskitError`, so callers can catch "any physicskit.fluids-raised
problem" with a single ``except FluidskitError``. Each concrete exception
also inherits from the closest matching built-in exception type (e.g.
:class:`InvalidParameterError` is also a :class:`ValueError`), so existing
code that catches the built-in type continues to work unchanged.
"""

from __future__ import annotations


class FluidskitError(Exception):
    """Base class for all exceptions raised intentionally by physicskit.fluids."""


class InvalidParameterError(FluidskitError, ValueError):
    """A system or tool was constructed or called with a physically invalid parameter.

    Examples include a non-positive Reynolds number or viscosity, a grid
    size incompatible with the doubly-periodic pseudo-spectral method's FFT
    assumptions, a cylinder radius too large to fit its sampling grid, or an
    explicit time step that violates a CFL/Courant stability bound.
    """
