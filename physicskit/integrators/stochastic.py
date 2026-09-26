"""Numba-accelerated integrators for Itô stochastic differential equations.

Solves systems with diagonal noise,

.. math::

    dX_i = a_i(X, t)\\,dt + b_i(X, t)\\,dW_i, \\qquad i = 1, \\ldots, d,

where the :math:`W_i` are independent Wiener processes. The drift and
diffusion callbacks follow the same ``f(state, t, params) -> ndarray``
convention as :mod:`physicskit.integrators.fixed_step`: module-level
``@njit`` functions taking a ``float64`` parameter array, so one compiled
integrator serves every parameter value.

The Wiener increments ``dW`` are passed in rather than drawn inside the
integrator. That keeps runs reproducible with any NumPy ``Generator``,
and lets a convergence study reuse one Brownian path at several step
sizes (sum consecutive increments to coarsen it). :func:`wiener_increments`
draws them.

* :func:`euler_maruyama_step` / :func:`euler_maruyama_integrate` -- strong
  order 1/2, weak order 1 (Maruyama 1955).
* :func:`milstein_step` / :func:`milstein_integrate` -- strong order 1 for
  diagonal noise, using the derivative :math:`\\partial b_i/\\partial x_i`
  (Milstein 1974). For additive noise (``b`` independent of the state) the
  correction vanishes and it reduces to Euler-Maruyama.
* :func:`baoab_step` / :func:`baoab_integrate` -- the BAOAB splitting for
  underdamped Langevin dynamics :math:`m\\ddot x = F(x) - \\gamma\\dot x +
  \\sqrt{2\\gamma k_BT}\\,\\xi`, taking the same ``force(pos, t, params)``
  callback as :func:`~physicskit.integrators.fixed_step.leapfrog_step`.
  Euler-Maruyama applied to an oscillator of frequency :math:`\\omega`
  pumps in energy at a rate :math:`\\sim\\omega^2\\Delta t`, which heats a
  weakly damped system well above the bath temperature; BAOAB samples the
  configurational Boltzmann distribution of a harmonic well exactly for any
  stable step (Leimkuhler and Matthews, Appl. Math. Res. Express 2013, 34).

See Kloeden and Platen, *Numerical Solution of Stochastic Differential
Equations* (Springer, 1992), §§10.2-10.3, for both schemes and their
convergence orders. Like the ODE integrators, these are deliberately not
``cache=True`` (see :mod:`physicskit.integrators.fixed_step`).
"""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray

from physicskit.integrators.fixed_step import RHSFunc

__all__ = [
    "wiener_increments",
    "euler_maruyama_step",
    "euler_maruyama_integrate",
    "milstein_step",
    "milstein_integrate",
    "baoab_step",
    "baoab_integrate",
]


def wiener_increments(n_steps: int, dim: int, dt: float, rng=None) -> NDArray[np.float64]:
    """Draw independent Wiener-process increments :math:`\\Delta W \\sim \\mathcal N(0, \\Delta t)`.

    Parameters
    ----------
    n_steps : int
        Number of time steps.
    dim : int
        Number of independent noise components (the state dimension, for
        diagonal noise).
    dt : float
        Step size; each increment has variance ``dt``.
    rng : numpy.random.Generator or int, optional
        Generator or seed passed to :func:`numpy.random.default_rng`.

    Returns
    -------
    ndarray of float, shape (n_steps, dim)
        The increments.

    Examples
    --------
    >>> dW = wiener_increments(200_000, 1, 0.01, rng=0)
    >>> round(float(dW.var()), 3)
    0.01
    """
    gen = np.random.default_rng(rng)
    return np.sqrt(dt) * gen.standard_normal((n_steps, dim))


@njit
def euler_maruyama_step(
    drift: RHSFunc,
    diffusion: RHSFunc,
    state: NDArray[np.float64],
    t: float,
    dt: float,
    dW: NDArray[np.float64],
    params: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Single Euler-Maruyama step for a diagonal-noise Itô SDE.

    .. math::

        X_{n+1} = X_n + a(X_n, t_n)\\,\\Delta t + b(X_n, t_n)\\,\\Delta W_n

    (Kloeden and Platen 1992, eq. 10.2.1).

    Parameters
    ----------
    drift : callable
        Numba-jitted ``drift(state, t, params) -> ndarray``, the :math:`a_i`.
    diffusion : callable
        Numba-jitted ``diffusion(state, t, params) -> ndarray``, the
        :math:`b_i` (same shape as ``state``).
    state : ndarray of float, shape (dim,)
        Current state.
    t : float
        Current time.
    dt : float
        Step size.
    dW : ndarray of float, shape (dim,)
        Wiener increments for this step, each :math:`\\mathcal N(0, dt)`.
    params : ndarray of float
        Parameter vector passed through to the callbacks.

    Returns
    -------
    ndarray of float, shape (dim,)
        The state after one step.
    """
    return state + drift(state, t, params) * dt + diffusion(state, t, params) * dW


@njit
def milstein_step(
    drift: RHSFunc,
    diffusion: RHSFunc,
    diffusion_derivative: RHSFunc,
    state: NDArray[np.float64],
    t: float,
    dt: float,
    dW: NDArray[np.float64],
    params: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Single Milstein step for a diagonal-noise Itô SDE.

    .. math::

        X_{n+1} = X_n + a\\,\\Delta t + b\\,\\Delta W_n
        + \\tfrac12\\, b\\, \\frac{\\partial b}{\\partial x}\\,(\\Delta W_n^2 - \\Delta t)

    componentwise (Kloeden and Platen 1992, eq. 10.3.1), with every
    coefficient evaluated at :math:`(X_n, t_n)`.

    Parameters
    ----------
    drift : callable
        Numba-jitted ``drift(state, t, params) -> ndarray``.
    diffusion : callable
        Numba-jitted ``diffusion(state, t, params) -> ndarray``.
    diffusion_derivative : callable
        Numba-jitted ``diffusion_derivative(state, t, params) -> ndarray``
        returning :math:`\\partial b_i / \\partial x_i` for each component.
    state : ndarray of float, shape (dim,)
        Current state.
    t : float
        Current time.
    dt : float
        Step size.
    dW : ndarray of float, shape (dim,)
        Wiener increments for this step.
    params : ndarray of float
        Parameter vector passed through to the callbacks.

    Returns
    -------
    ndarray of float, shape (dim,)
        The state after one step.
    """
    b = diffusion(state, t, params)
    db = diffusion_derivative(state, t, params)
    return state + drift(state, t, params) * dt + b * dW + 0.5 * b * db * (dW * dW - dt)


@njit
def euler_maruyama_integrate(
    drift: RHSFunc,
    diffusion: RHSFunc,
    state0: NDArray[np.float64],
    t0: float,
    dt: float,
    dW: NDArray[np.float64],
    params: NDArray[np.float64],
    save_every: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Integrate a diagonal-noise Itô SDE with Euler-Maruyama.

    Takes one step per row of ``dW``.

    Parameters
    ----------
    drift, diffusion : callable
        Numba-jitted ``f(state, t, params) -> ndarray`` callbacks.
    state0 : ndarray of float, shape (dim,)
        Initial state.
    t0 : float
        Initial time.
    dt : float
        Step size.
    dW : ndarray of float, shape (n_steps, dim)
        Wiener increments, e.g. from :func:`wiener_increments`.
    params : ndarray of float
        Parameter vector passed through to the callbacks.
    save_every : int, default=1
        Record the state every ``save_every`` steps.

    Returns
    -------
    times : ndarray of float, shape (n_steps // save_every + 1,)
        Recorded times, starting at ``t0``.
    states : ndarray of float, shape (n_steps // save_every + 1, dim)
        Recorded states, starting at ``state0``.

    Examples
    --------
    Geometric Brownian motion :math:`dX = \\mu X\\,dt + \\sigma X\\,dW`:

    >>> from numba import njit
    >>> @njit
    ... def a(x, t, p):
    ...     return p[0] * x
    >>> @njit
    ... def b(x, t, p):
    ...     return p[1] * x
    >>> dW = wiener_increments(1000, 1, 1e-3, rng=1)
    >>> t, X = euler_maruyama_integrate(a, b, np.array([1.0]), 0.0, 1e-3, dW, np.array([0.1, 0.2]))
    >>> exact = np.exp((0.1 - 0.5 * 0.2**2) * 1.0 + 0.2 * dW.sum())
    >>> bool(abs(X[-1, 0] - exact) < 1e-2)
    True
    """
    n_steps = dW.shape[0]
    n_saved = n_steps // save_every + 1
    states = np.empty((n_saved, state0.shape[0]))
    times = np.empty(n_saved)
    states[0] = state0
    times[0] = t0
    state = state0.copy()
    k = 1
    for i in range(n_steps):
        t = t0 + i * dt
        state = euler_maruyama_step(drift, diffusion, state, t, dt, dW[i], params)
        if (i + 1) % save_every == 0:
            states[k] = state
            times[k] = t0 + (i + 1) * dt
            k += 1
    return times, states


@njit
def milstein_integrate(
    drift: RHSFunc,
    diffusion: RHSFunc,
    diffusion_derivative: RHSFunc,
    state0: NDArray[np.float64],
    t0: float,
    dt: float,
    dW: NDArray[np.float64],
    params: NDArray[np.float64],
    save_every: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Integrate a diagonal-noise Itô SDE with the Milstein scheme.

    Takes one step per row of ``dW``.

    Parameters
    ----------
    drift, diffusion, diffusion_derivative : callable
        Numba-jitted ``f(state, t, params) -> ndarray`` callbacks; the last
        returns :math:`\\partial b_i/\\partial x_i`.
    state0 : ndarray of float, shape (dim,)
        Initial state.
    t0 : float
        Initial time.
    dt : float
        Step size.
    dW : ndarray of float, shape (n_steps, dim)
        Wiener increments.
    params : ndarray of float
        Parameter vector passed through to the callbacks.
    save_every : int, default=1
        Record the state every ``save_every`` steps.

    Returns
    -------
    times : ndarray of float, shape (n_steps // save_every + 1,)
        Recorded times, starting at ``t0``.
    states : ndarray of float, shape (n_steps // save_every + 1, dim)
        Recorded states, starting at ``state0``.

    Examples
    --------
    >>> from numba import njit
    >>> @njit
    ... def a(x, t, p):
    ...     return p[0] * x
    >>> @njit
    ... def b(x, t, p):
    ...     return p[1] * x
    >>> @njit
    ... def db(x, t, p):
    ...     return p[1] * np.ones_like(x)
    >>> dW = wiener_increments(1000, 1, 1e-3, rng=1)
    >>> t, X = milstein_integrate(a, b, db, np.array([1.0]), 0.0, 1e-3, dW, np.array([0.1, 0.2]))
    >>> exact = np.exp((0.1 - 0.5 * 0.2**2) * 1.0 + 0.2 * dW.sum())
    >>> bool(abs(X[-1, 0] - exact) < 1e-4)
    True
    """
    n_steps = dW.shape[0]
    n_saved = n_steps // save_every + 1
    states = np.empty((n_saved, state0.shape[0]))
    times = np.empty(n_saved)
    states[0] = state0
    times[0] = t0
    state = state0.copy()
    k = 1
    for i in range(n_steps):
        t = t0 + i * dt
        state = milstein_step(drift, diffusion, diffusion_derivative, state, t, dt, dW[i], params)
        if (i + 1) % save_every == 0:
            states[k] = state
            times[k] = t0 + (i + 1) * dt
            k += 1
    return times, states


@njit
def baoab_step(
    force: RHSFunc,
    pos: NDArray[np.float64],
    vel: NDArray[np.float64],
    t: float,
    dt: float,
    mass: float,
    gamma: float,
    kT: float,
    noise: NDArray[np.float64],
    params: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Single BAOAB step of underdamped Langevin dynamics.

    Splits :math:`m\\,dv = F\\,dt - \\gamma v\\,dt + \\sqrt{2\\gamma k_BT}\\,dW`,
    :math:`dx = v\\,dt` into half kicks (B), half drifts (A), and an exact
    Ornstein-Uhlenbeck velocity update (O) in the middle:

    .. math::

        v \\leftarrow v + \\tfrac{\\Delta t}{2m}F(x), \\quad
        x \\leftarrow x + \\tfrac{\\Delta t}{2}v, \\quad
        v \\leftarrow e^{-\\gamma\\Delta t/m}v + \\sqrt{(1 - e^{-2\\gamma\\Delta t/m})k_BT/m}\\;\\eta,

    then the A and B half steps again (B. Leimkuhler and C. Matthews, Appl.
    Math. Res. Express 2013, 34-56, §2). For :math:`\\gamma = 0` it is the
    velocity-Verlet step.

    Parameters
    ----------
    force : callable
        Numba-jitted ``force(pos, t, params) -> ndarray`` (a force, not an
        acceleration: it is divided by ``mass``).
    pos, vel : ndarray of float, shape (dim,)
        Current positions and velocities.
    t : float
        Current time.
    dt : float
        Step size.
    mass, gamma, kT : float
        Particle mass, friction coefficient, and bath temperature
        (:math:`k_B = 1`).
    noise : ndarray of float, shape (dim,)
        Independent standard normal draws :math:`\\eta` (unit variance, *not*
        scaled by ``dt``).
    params : ndarray of float
        Parameter vector passed through to ``force``.

    Returns
    -------
    pos_new, vel_new : ndarray of float, shape (dim,)
    """
    c = np.exp(-gamma * dt / mass)
    kick = np.sqrt((1.0 - c * c) * kT / mass)
    v = vel + 0.5 * dt * force(pos, t, params) / mass
    x = pos + 0.5 * dt * v
    v = c * v + kick * noise
    x = x + 0.5 * dt * v
    v = v + 0.5 * dt * force(x, t + dt, params) / mass
    return x, v


@njit
def baoab_integrate(
    force: RHSFunc,
    pos0: NDArray[np.float64],
    vel0: NDArray[np.float64],
    t0: float,
    dt: float,
    mass: float,
    gamma: float,
    kT: float,
    noise: NDArray[np.float64],
    params: NDArray[np.float64],
    save_every: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Integrate underdamped Langevin dynamics with BAOAB.

    Takes one :func:`baoab_step` per row of ``noise``.

    Parameters
    ----------
    force : callable
        Numba-jitted ``force(pos, t, params) -> ndarray``.
    pos0, vel0 : ndarray of float, shape (dim,)
        Initial positions and velocities.
    t0 : float
        Initial time.
    dt : float
        Step size.
    mass, gamma, kT : float
        Particle mass, friction coefficient, and bath temperature.
    noise : ndarray of float, shape (n_steps, dim)
        Standard normal draws, e.g. ``rng.standard_normal((n_steps, dim))``.
    params : ndarray of float
        Parameter vector passed through to ``force``.
    save_every : int, default=1
        Record every ``save_every`` steps.

    Returns
    -------
    times : ndarray of float, shape (n_steps // save_every + 1,)
    positions, velocities : ndarray of float, shape (n_steps // save_every + 1, dim)

    Examples
    --------
    A weakly damped harmonic oscillator thermalizes to :math:`k\\langle x^2\\rangle = k_BT`:

    >>> from numba import njit
    >>> @njit
    ... def spring(x, t, p):
    ...     return -p[0] * x
    >>> eta = np.random.default_rng(0).standard_normal((4000, 2000))
    >>> t, x, v = baoab_integrate(spring, np.zeros(2000), np.zeros(2000), 0.0, 0.05, 1.0, 0.1, 0.5, eta, np.array([4.0]), 4000)
    >>> round(float(4.0 * np.mean(x[-1] ** 2)), 1), round(float(np.mean(v[-1] ** 2)), 1)
    (0.5, 0.5)
    """
    n_steps = noise.shape[0]
    n_saved = n_steps // save_every + 1
    dim = pos0.shape[0]
    positions = np.empty((n_saved, dim))
    velocities = np.empty((n_saved, dim))
    times = np.empty(n_saved)
    positions[0] = pos0
    velocities[0] = vel0
    times[0] = t0
    x = pos0.copy()
    v = vel0.copy()
    k = 1
    for i in range(n_steps):
        x, v = baoab_step(force, x, v, t0 + i * dt, dt, mass, gamma, kT, noise[i], params)
        if (i + 1) % save_every == 0:
            positions[k] = x
            velocities[k] = v
            times[k] = t0 + (i + 1) * dt
            k += 1
    return times, positions, velocities
