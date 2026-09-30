"""The one-dimensional Fokker-Planck equation for the probability density of a diffusion process.

For the Itô SDE :math:`dX = A(X)\\,dt + \\sqrt{2D(X)}\\,dW` the density
:math:`p(x, t)` obeys

.. math::

    \\frac{\\partial p}{\\partial t} = -\\frac{\\partial J}{\\partial x}, \\qquad
    J = A(x)\\,p - \\frac{\\partial}{\\partial x}\\big[D(x)\\,p\\big]

(Fokker 1914, Planck 1917; H. Risken, *The Fokker-Planck Equation*, 2nd
ed., Springer 1989, eqs. 4.1 and 5.1). The overdamped particle of
:class:`~physicskit.statphys.chapters.langevin.BrownianMotion` has
:math:`A = F/\\gamma` and :math:`D = k_BT/\\gamma`, and its stationary
density is the Boltzmann distribution. :func:`fokker_planck_1d` solves the
equation on a bounded interval with reflecting (zero-flux) walls, and
:func:`fokker_planck_stationary` gives the closed-form zero-flux
stationary state.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.sparse.linalg import splu

__all__ = ["fokker_planck_operator", "fokker_planck_1d", "fokker_planck_stationary"]


def _coefficient(c, points) -> np.ndarray:
    if callable(c):
        return np.asarray(c(points), dtype=np.float64) * np.ones_like(points)
    return np.broadcast_to(np.asarray(c, dtype=np.float64), points.shape).copy()


def _check_grid(x) -> tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=np.float64)
    dx = x[1] - x[0]
    if x.ndim != 1 or x.size < 3 or not np.allclose(np.diff(x), dx):
        raise ValueError("x must be a uniform 1D grid of at least 3 points")
    return x, float(dx)


def fokker_planck_operator(x, drift, diffusion) -> sp.csr_matrix:
    """Finite-volume Fokker-Planck operator :math:`L` with :math:`dp/dt = Lp` and zero-flux walls.

    Each grid point is the center of a cell of width :math:`\\Delta x`. The
    flux through the face between cells :math:`i` and :math:`i+1` is

    .. math::

        J_{i+1/2} = A_{i+1/2}\\,\\frac{p_i + p_{i+1}}{2}
        - \\frac{D_{i+1}p_{i+1} - D_i p_i}{\\Delta x},

    with :math:`J = 0` through the two outer walls, so
    :math:`\\sum_i p_i\\,\\Delta x` is conserved exactly. Central
    differencing of the drift needs a cell Péclet number
    :math:`|A|\\Delta x / D` below about 2 to keep :math:`p \\ge 0`.

    Parameters
    ----------
    x : array_like, shape (n,)
        Uniform grid of cell centers.
    drift : float, array_like of shape (n,), or callable
        :math:`A(x)`. A callable is evaluated at the cell faces; an array is
        averaged onto them.
    diffusion : float, array_like of shape (n,), or callable
        :math:`D(x) > 0`, evaluated at the cell centers.

    Returns
    -------
    scipy.sparse.csr_matrix, shape (n, n)

    Examples
    --------
    Columns sum to zero: probability is conserved.

    >>> L = fokker_planck_operator(np.linspace(-1, 1, 11), lambda x: -x, 0.5)
    >>> bool(np.allclose(np.asarray(L.sum(axis=0)).ravel(), 0.0))
    True
    """
    x, dx = _check_grid(x)
    n = x.size
    if callable(drift):
        a_face = _coefficient(drift, 0.5 * (x[:-1] + x[1:]))
    else:
        a_node = _coefficient(drift, x)
        a_face = 0.5 * (a_node[:-1] + a_node[1:])
    d = _coefficient(diffusion, x)
    # J_{i+1/2} = c_left[i] p_i + c_right[i] p_{i+1}
    c_left = 0.5 * a_face + d[:-1] / dx
    c_right = 0.5 * a_face - d[1:] / dx
    # dp_i/dt = (J_{i-1/2} - J_{i+1/2}) / dx
    main = np.zeros(n)
    main[:-1] -= c_left / dx
    main[1:] += c_right / dx
    upper = -c_right / dx  # coefficient of p_{i+1} in dp_i/dt
    lower = c_left / dx  # coefficient of p_{i-1} in dp_i/dt
    return sp.diags([lower, main, upper], [-1, 0, 1], format="csr")


def fokker_planck_1d(p0, x, drift, diffusion, t_max, dt, n_frames=100, theta=0.5):
    """Evolve a 1D probability density under the Fokker-Planck equation.

    Uses the finite-volume operator of :func:`fokker_planck_operator` with
    the :math:`\\theta`-scheme
    :math:`(I - \\theta\\Delta t L)p^{n+1} = (I + (1-\\theta)\\Delta t L)p^n`:
    Crank-Nicolson (second order, :math:`\\theta = 1/2`) by default, or
    backward Euler (:math:`\\theta = 1`, strongly damping) for rough
    initial data. Both are unconditionally stable; the matrix is factored
    once.

    Parameters
    ----------
    p0 : array_like, shape (n,)
        Initial density on the grid (normalized to unit mass on return).
    x : array_like, shape (n,)
        Uniform grid of cell centers; the walls sit half a cell outside
        the end points.
    drift, diffusion : float, array_like, or callable
        :math:`A(x)` and :math:`D(x)`, as in :func:`fokker_planck_operator`.
    t_max : float
        Final time (rounded to a whole number of steps per frame).
    dt : float
        Time step.
    n_frames : int, default=100
        Number of recorded frames after the initial one.
    theta : float, default=0.5
        Implicitness, in ``[0.5, 1]``.

    Returns
    -------
    times : ndarray, shape (n_frames + 1,)
    p : ndarray, shape (n_frames + 1, n)
        Density at each frame, each with :math:`\\sum p\\,\\Delta x = 1`.

    Examples
    --------
    Free diffusion spreads a Gaussian as :math:`\\sigma^2(t) = \\sigma_0^2 + 2Dt`:

    >>> x = np.linspace(-10, 10, 401)
    >>> p0 = np.exp(-x**2 / 2)
    >>> t, p = fokker_planck_1d(p0, x, 0.0, 0.5, t_max=2.0, dt=0.01, n_frames=4)
    >>> var = np.sum(x**2 * p[-1]) * (x[1] - x[0])
    >>> round(float(var), 3)  # 1 + 2 * 0.5 * 2
    3.0
    """
    x, dx = _check_grid(x)
    if not 0.5 <= theta <= 1.0:
        raise ValueError("theta must lie in [0.5, 1]")
    L = fokker_planck_operator(x, drift, diffusion)
    eye = sp.identity(x.size, format="csc")
    lhs = splu((eye - theta * dt * L).tocsc())
    rhs = (eye + (1.0 - theta) * dt * L).tocsr()
    stride = max(1, int(round(t_max / (n_frames * dt))))
    p = np.asarray(p0, dtype=np.float64).copy()
    p /= p.sum() * dx
    out = np.empty((n_frames + 1, x.size))
    out[0] = p
    for f in range(n_frames):
        for _ in range(stride):
            p = lhs.solve(rhs @ p)
        out[f + 1] = p
    return np.arange(n_frames + 1) * stride * dt, out


def fokker_planck_stationary(x, drift, diffusion) -> np.ndarray:
    """Zero-flux stationary solution of the 1D Fokker-Planck equation.

    Setting :math:`J = 0` gives

    .. math::

        p_{\\text{st}}(x) = \\frac{N}{D(x)}\\exp\\!\\int^x \\frac{A(y)}{D(y)}\\,dy

    (Risken 1989, eq. 5.6). For :math:`A = -U'/\\gamma` and
    :math:`D = k_BT/\\gamma` this is the Boltzmann distribution
    :math:`e^{-U/k_BT}/Z`.

    Parameters
    ----------
    x : array_like, shape (n,)
        Grid.
    drift, diffusion : float, array_like of shape (n,), or callable
        :math:`A(x)` and :math:`D(x)`, evaluated on the grid.

    Returns
    -------
    ndarray, shape (n,)
        Stationary density normalized by the trapezoid rule on ``x``.

    Examples
    --------
    An Ornstein-Uhlenbeck process, :math:`A = -\\theta x`,
    :math:`D = \\sigma^2/2`, is Gaussian with variance :math:`\\sigma^2/2\\theta`:

    >>> x = np.linspace(-6, 6, 2001)
    >>> p = fokker_planck_stationary(x, lambda y: -2.0 * y, 0.5)
    >>> round(float(np.sum(x**2 * p) * (x[1] - x[0])), 4)
    0.25
    """
    x = np.asarray(x, dtype=np.float64)
    a = _coefficient(drift, x)
    d = _coefficient(diffusion, x)
    log_p = cumulative_trapezoid(a / d, x, initial=0.0) - np.log(d)
    p = np.exp(log_p - log_p.max())
    return p / trapezoid(p, x)
