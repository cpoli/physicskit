"""Numba-accelerated time integrators.

Standard RK4 is provided for reference/comparison, but it is not
energy-preserving: over long integrations the numerically computed
Hamiltonian drifts (typically grows) because RK4 is not a symplectic
map. The default backends for conservative systems in ``physicskit.classical`` are:

* :func:`velocity_verlet_integrate` -- 2nd-order symplectic Stormer-Verlet,
  valid for *separable* Hamiltonians H(q, p) = T(p) + V(q).
* :func:`yoshida4_integrate` -- 4th-order symplectic integrator built by
  composing three Verlet sub-steps with Yoshida's (1990) coefficients.
* :func:`implicit_midpoint_integrate` -- a general-purpose symplectic
  integrator (2nd order) that works for *any* Hamiltonian system,
  including non-separable ones (coupled generalized coordinates such as
  the double pendulum, or rigid-body/Euler-type equations). It is solved
  by fixed-point iteration each step.

All step functions accept plain ``@njit``-compiled callback functions so
the whole time loop can run at native speed with no Python overhead.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = [
    "rk4_step",
    "rk4_integrate",
    "velocity_verlet_step",
    "velocity_verlet_integrate",
    "yoshida4_step",
    "yoshida4_integrate",
    "implicit_midpoint_step",
    "implicit_midpoint_integrate",
    "make_separable_derivatives",
]

# ---------------------------------------------------------------------------
# Classical Runge-Kutta 4 (non-symplectic reference integrator)
# ---------------------------------------------------------------------------


@njit(cache=False)
def rk4_step(deriv_func, t, y, dt):
    """Single RK4 step for ``dy/dt = deriv_func(t, y)``.

    Parameters
    ----------
    deriv_func : callable
        ``@njit`` dispatcher ``(t, y) -> dy``.
    t : float
        Current time.
    y : ndarray
        Current state.
    dt : float
        Step size.

    Returns
    -------
    ndarray
        State at ``t + dt``.
    """
    k1 = deriv_func(t, y)
    k2 = deriv_func(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = deriv_func(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = deriv_func(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@njit(cache=False)
def rk4_integrate(deriv_func, y0, t0, n_steps, dt):
    """Integrate ``dy/dt = deriv_func(t, y)`` for ``n_steps`` of size ``dt``.

    Parameters
    ----------
    deriv_func : callable
        ``@njit`` dispatcher ``(t, y) -> dy``.
    y0 : ndarray
        Initial state.
    t0 : float
        Initial time.
    n_steps : int
        Number of steps to take.
    dt : float
        Step size.

    Returns
    -------
    t : ndarray, shape (n_steps + 1,)
    y : ndarray, shape (n_steps + 1, len(y0))
    """
    dim = y0.shape[0]
    ts = np.empty(n_steps + 1)
    ys = np.empty((n_steps + 1, dim))
    ts[0] = t0
    ys[0, :] = y0
    y = y0.copy()
    t = t0
    for i in range(n_steps):
        y = rk4_step(deriv_func, t, y, dt)
        t = t0 + (i + 1) * dt
        ts[i + 1] = t
        ys[i + 1, :] = y
    return ts, ys


# ---------------------------------------------------------------------------
# Symplectic (velocity) Verlet -- separable Hamiltonians H = T(p) + V(q)
# ---------------------------------------------------------------------------


@njit(cache=False)
def velocity_verlet_step(force_func, mass_inv, q, p, dt, t=0.0):
    """One Stormer-Verlet / velocity-Verlet step.

    Parameters
    ----------
    force_func : callable
        ``@njit`` dispatcher ``(q, t) -> force``, returning -dV/dq.
    mass_inv : float or ndarray
        1/m per coordinate, giving ``dT/dp = mass_inv * p`` for the
        standard kinetic form.
    q, p : ndarray
        Current position and momentum.
    dt : float
        Step size.
    t : float, default 0.0
        Current time.

    Returns
    -------
    q, p : ndarray
        Position and momentum at ``t + dt``.
    """
    f0 = force_func(q, t)
    p_half = p + 0.5 * dt * f0
    q_new = q + dt * (mass_inv * p_half)
    f1 = force_func(q_new, t + dt)
    p_new = p_half + 0.5 * dt * f1
    return q_new, p_new


@njit(cache=False)
def velocity_verlet_integrate(force_func, mass_inv, q0, p0, t0, n_steps, dt):
    """Integrate a separable Hamiltonian system with velocity Verlet.

    Parameters
    ----------
    force_func : callable
        ``@njit`` dispatcher ``(q, t) -> force``.
    mass_inv : float or ndarray
        1/m per coordinate.
    q0, p0 : ndarray
        Initial position and momentum.
    t0 : float
        Initial time.
    n_steps : int
        Number of steps to take.
    dt : float
        Step size.

    Returns
    -------
    t : ndarray, shape (n_steps + 1,)
    q, p : ndarray, shape (n_steps + 1, len(q0))
    """
    dim = q0.shape[0]
    ts = np.empty(n_steps + 1)
    qs = np.empty((n_steps + 1, dim))
    ps = np.empty((n_steps + 1, dim))
    ts[0] = t0
    qs[0, :] = q0
    ps[0, :] = p0
    q, p = q0.copy(), p0.copy()
    t = t0
    for i in range(n_steps):
        q, p = velocity_verlet_step(force_func, mass_inv, q, p, dt, t)
        t = t0 + (i + 1) * dt
        ts[i + 1] = t
        qs[i + 1, :] = q
        ps[i + 1, :] = p
    return ts, qs, ps


# ---------------------------------------------------------------------------
# 4th-order Yoshida integrator (triple-composition of Verlet)
# ---------------------------------------------------------------------------

_YOSHIDA_CUBE_ROOT = 2.0 ** (1.0 / 3.0)
_YOSHIDA_W1 = 1.0 / (2.0 - _YOSHIDA_CUBE_ROOT)
_YOSHIDA_W0 = -_YOSHIDA_CUBE_ROOT / (2.0 - _YOSHIDA_CUBE_ROOT)


@njit(cache=False)
def yoshida4_step(force_func, mass_inv, q, p, dt, t=0.0):
    """One 4th-order Yoshida step: composition of 3 Verlet sub-steps.

    Parameters
    ----------
    force_func : callable
        ``@njit`` dispatcher ``(q, t) -> force``.
    mass_inv : float or ndarray
        1/m per coordinate.
    q, p : ndarray
        Current position and momentum.
    dt : float
        Step size.
    t : float, default 0.0
        Current time.

    Returns
    -------
    q, p : ndarray
        Position and momentum at ``t + dt``.
    """
    q, p = velocity_verlet_step(force_func, mass_inv, q, p, _YOSHIDA_W1 * dt, t)
    t1 = t + _YOSHIDA_W1 * dt
    q, p = velocity_verlet_step(force_func, mass_inv, q, p, _YOSHIDA_W0 * dt, t1)
    t2 = t1 + _YOSHIDA_W0 * dt
    q, p = velocity_verlet_step(force_func, mass_inv, q, p, _YOSHIDA_W1 * dt, t2)
    return q, p


@njit(cache=False)
def yoshida4_integrate(force_func, mass_inv, q0, p0, t0, n_steps, dt):
    """Integrate a separable Hamiltonian system with 4th-order Yoshida.

    Parameters
    ----------
    force_func : callable
        ``@njit`` dispatcher ``(q, t) -> force``.
    mass_inv : float or ndarray
        1/m per coordinate.
    q0, p0 : ndarray
        Initial position and momentum.
    t0 : float
        Initial time.
    n_steps : int
        Number of steps to take.
    dt : float
        Step size.

    Returns
    -------
    t : ndarray, shape (n_steps + 1,)
    q, p : ndarray, shape (n_steps + 1, len(q0))
    """
    dim = q0.shape[0]
    ts = np.empty(n_steps + 1)
    qs = np.empty((n_steps + 1, dim))
    ps = np.empty((n_steps + 1, dim))
    ts[0] = t0
    qs[0, :] = q0
    ps[0, :] = p0
    q, p = q0.copy(), p0.copy()
    t = t0
    for i in range(n_steps):
        q, p = yoshida4_step(force_func, mass_inv, q, p, dt, t)
        t = t0 + (i + 1) * dt
        ts[i + 1] = t
        qs[i + 1, :] = q
        ps[i + 1, :] = p
    return ts, qs, ps


# ---------------------------------------------------------------------------
# Implicit midpoint -- general symplectic integrator (non-separable OK)
# ---------------------------------------------------------------------------


@njit(cache=False)
def implicit_midpoint_step(deriv_func, t, y, dt, tol=1e-12, max_iter=100):
    """One implicit-midpoint step for ``dy/dt = deriv_func(t, y)``.

    Solved by fixed-point (Picard) iteration:
    ``y_{n+1} = y_n + dt * f(t + dt/2, (y_n + y_{n+1}) / 2)``.

    The implicit midpoint rule is symplectic for canonical Hamiltonian
    systems (any H(q, p), separable or not) and exactly preserves
    quadratic invariants of bilinear ODEs (e.g. Euler's rigid-body
    equations), making it a robust general-purpose fallback wherever a
    closed-form separable split is unavailable.

    Parameters
    ----------
    deriv_func : callable
        ``@njit`` dispatcher ``(t, y) -> dy``.
    t : float
        Current time.
    y : ndarray
        Current state.
    dt : float
        Step size.
    tol : float, default 1e-12
        Fixed-point convergence tolerance (max abs component change).
    max_iter : int, default 100
        Maximum Picard iterations per step.

    Returns
    -------
    ndarray
        State at ``t + dt``.
    """
    tm = t + 0.5 * dt
    y_next = y + dt * deriv_func(t, y)  # explicit-Euler predictor
    for _ in range(max_iter):
        y_mid = 0.5 * (y + y_next)
        y_new = y + dt * deriv_func(tm, y_mid)
        if np.max(np.abs(y_new - y_next)) < tol:
            y_next = y_new
            break
        y_next = y_new
    return y_next


@njit(cache=False)
def implicit_midpoint_integrate(deriv_func, y0, t0, n_steps, dt, tol=1e-12, max_iter=100):
    """Integrate ``dy/dt = deriv_func(t, y)`` with the implicit midpoint rule.

    Parameters
    ----------
    deriv_func : callable
        ``@njit`` dispatcher ``(t, y) -> dy``.
    y0 : ndarray
        Initial state.
    t0 : float
        Initial time.
    n_steps : int
        Number of steps to take.
    dt : float
        Step size.
    tol : float, default 1e-12
        Fixed-point convergence tolerance, passed to
        :func:`implicit_midpoint_step`.
    max_iter : int, default 100
        Maximum Picard iterations per step.

    Returns
    -------
    t : ndarray, shape (n_steps + 1,)
    y : ndarray, shape (n_steps + 1, len(y0))
    """
    dim = y0.shape[0]
    ts = np.empty(n_steps + 1)
    ys = np.empty((n_steps + 1, dim))
    ts[0] = t0
    ys[0, :] = y0
    y = y0.copy()
    t = t0
    for i in range(n_steps):
        y = implicit_midpoint_step(deriv_func, t, y, dt, tol, max_iter)
        t = t0 + (i + 1) * dt
        ts[i + 1] = t
        ys[i + 1, :] = y
    return ts, ys


# ---------------------------------------------------------------------------
# Helper: build a generic dy/dt callback from a separable force function
# ---------------------------------------------------------------------------


def make_separable_derivatives(force_func, mass_inv, ndof: int):
    """Wrap a separable-Hamiltonian force function as a stacked-state derivative.

    Wraps ``force_func(q, t)`` as a stacked-state ``deriv_func(t, y)``
    (``y = [q, p]``) usable by :func:`rk4_integrate` and
    :func:`implicit_midpoint_integrate`.

    Parameters
    ----------
    force_func : callable
        ``@njit`` dispatcher ``(q, t) -> force``.
    mass_inv : float or ndarray
        1/m per coordinate.
    ndof : int
        Number of degrees of freedom (``len(y) == 2 * ndof``).

    Returns
    -------
    callable
        A new ``@njit`` dispatcher ``(t, y) -> dy``, closing over
        ``force_func``, ``mass_inv`` and ``ndof`` as compile-time
        constants, so callers get the same first-class-function speed
        as a hand-written derivative.
    """

    @njit(cache=False)
    def deriv(t, y):
        q = y[:ndof]
        p = y[ndof:]
        dq = mass_inv * p
        dp = force_func(q, t)
        return np.concatenate((dq, dp))

    return deriv
