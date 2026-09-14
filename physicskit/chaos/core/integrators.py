"""Numba-accelerated numerical integrators.

Re-exports the shared, params-array-convention integrators from
:mod:`physicskit.integrators`, which is where the implementations live. See
that module for the ``rhs(state, t, params) -> ndarray`` /
``force(pos, t, params) -> ndarray`` calling convention required by every
function here, and for why it's kept explicit rather than closing over
Python scalars (it lets Numba compile these integrators once and reuse
them, as first-class functions, across every system defined in
:mod:`physicskit.chaos.systems.continuous`).

The two symplectic integrators (:func:`leapfrog_step`, 2nd order, and
:func:`yoshida4_step`, 4th order) only apply to *separable* Hamiltonian
systems of the form ``pos'' = force(pos, t)`` (the force may not depend on
velocity); RK4 is required for non-separable systems (e.g. anything with a
Coriolis-like velocity-dependent force, such as
:class:`physicskit.chaos.systems.continuous.RestrictedThreeBody`).
"""

from __future__ import annotations

from physicskit.integrators.fixed_step import (
    RHSFunc,
    leapfrog_integrate,
    leapfrog_step,
    rk4_integrate,
    rk4_step,
    yoshida4_integrate,
    yoshida4_step,
)

__all__ = [
    "RHSFunc",
    "rk4_step",
    "rk4_integrate",
    "leapfrog_step",
    "leapfrog_integrate",
    "yoshida4_step",
    "yoshida4_integrate",
]
