"""Numba-accelerated ODE integrators shared across physicskit subpackages.

Right-hand-side / force callbacks use the ``f(state_or_pos, t, params) ->
ndarray`` convention throughout, so a single compiled callback can be reused
across systems with different parameter values without recompiling (see
:mod:`physicskit.integrators.fixed_step`).

* :func:`rk4_integrate` -- classical 4th-order Runge-Kutta (not
  energy-preserving).
* :func:`leapfrog_integrate` (alias :func:`velocity_verlet_integrate`) --
  2nd-order symplectic Stormer-Verlet, for separable systems
  ``pos'' = force(pos, t)``.
* :func:`yoshida4_integrate` -- 4th-order symplectic, built from three
  leapfrog sub-steps.
* :func:`dopri5_integrate` -- adaptive-step-size embedded Dormand-Prince
  RK5(4), for non-conservative or accuracy-sensitive systems (see
  :mod:`physicskit.integrators.adaptive` for why this isn't offered for the
  symplectic integrators).
* :func:`euler_maruyama_integrate` and :func:`milstein_integrate` -- for Itô
  stochastic differential equations with diagonal noise, driven by
  caller-supplied Wiener increments from :func:`wiener_increments` (see
  :mod:`physicskit.integrators.stochastic`); :func:`baoab_integrate` for
  underdamped Langevin dynamics with a ``force(pos, t, params)`` callback.

:mod:`physicskit.classical.core.integrators` uses its own closure-based
calling convention (``force_func(q, t)`` plus a per-instance ``mass_inv``)
and :mod:`physicskit.statphys.core.md_engine` its own periodic-boundary,
force-caching variant; both are documented at their own definitions rather
than adapted to the convention here.
"""

from physicskit.integrators.adaptive import dopri5_integrate, dopri5_step
from physicskit.integrators.fixed_step import (
    RHSFunc,
    leapfrog_integrate,
    leapfrog_step,
    rk4_integrate,
    rk4_step,
    velocity_verlet_integrate,
    velocity_verlet_step,
    yoshida4_integrate,
    yoshida4_step,
)
from physicskit.integrators.stochastic import (
    baoab_integrate,
    baoab_step,
    euler_maruyama_integrate,
    euler_maruyama_step,
    milstein_integrate,
    milstein_step,
    wiener_increments,
)

__all__ = [
    "RHSFunc",
    "rk4_step",
    "rk4_integrate",
    "leapfrog_step",
    "leapfrog_integrate",
    "velocity_verlet_step",
    "velocity_verlet_integrate",
    "yoshida4_step",
    "yoshida4_integrate",
    "dopri5_step",
    "dopri5_integrate",
    "wiener_increments",
    "euler_maruyama_step",
    "euler_maruyama_integrate",
    "milstein_step",
    "milstein_integrate",
    "baoab_step",
    "baoab_integrate",
]
