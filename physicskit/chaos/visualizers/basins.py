"""Basin-of-attraction mapping: which stable state does each initial condition end up in?

For a multistable, dissipative system with several stable attractors (e.g.
:class:`physicskit.chaos.systems.continuous.MagneticPendulum`, with one attractor per
magnet), the basin of attraction of an attractor is the set of initial
conditions that end up there. Basin boundaries between attractors are often
*fractal* -- an arbitrarily small change in initial position can flip which
attractor "wins" -- and this module maps that structure over a 2D grid.

Because this means integrating one trajectory per pixel of the map, the
integration itself is a Numba-jitted, parallelized loop rather than the
:func:`scipy.integrate.solve_ivp`-based approach used by
:mod:`physicskit.chaos.visualizers.bifurcation`: `rhs` here must be a module-level
``@njit`` function (state, t, params) -> ndarray``, exactly like the ones
:mod:`physicskit.chaos.systems.continuous` exposes publicly (e.g.
:func:`physicskit.chaos.systems.continuous.magnetic_pendulum_rhs`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numba import njit, prange
from numpy.typing import ArrayLike, NDArray

from physicskit.chaos.core.integrators import rk4_step
from physicskit.chaos.visualizers import theme


@njit(cache=True, parallel=True)
def _compute_basin_grid(rhs, xs, ys, vx0, vy0, dt, n_steps, attractors, params) -> NDArray[np.int64]:
    n_y = ys.shape[0]
    n_x = xs.shape[0]
    n_attractors = attractors.shape[0]
    labels = np.empty((n_y, n_x), dtype=np.int64)
    for i in prange(n_y):  # type: ignore[attr-defined]  # numba lacks type stubs for prange
        for j in range(n_x):
            state = np.array([xs[j], ys[i], vx0, vy0])
            for _ in range(n_steps):
                state = rk4_step(rhs, state, 0.0, dt, params)
            best_label = 0
            best_dist = 1e18
            for k in range(n_attractors):
                dx = state[0] - attractors[k, 0]
                dy = state[1] - attractors[k, 1]
                dist = dx * dx + dy * dy
                if dist < best_dist:
                    best_dist = dist
                    best_label = k
            labels[i, j] = best_label
    return labels


def basin_of_attraction(
    rhs: Callable,
    params: ArrayLike,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    attractors: ArrayLike,
    resolution: int = 200,
    velocity0: tuple[float, float] = (0.0, 0.0),
    dt: float = 0.02,
    n_steps: int = 2000,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Compute a basin-of-attraction map over a 2D grid of initial positions.

    For every point on an ``resolution x resolution`` grid spanning
    `x_range` x `y_range`, integrates a trajectory of the system defined by
    `rhs` starting at that position (with initial velocity `velocity0`) for
    `n_steps` of size `dt`, then labels the grid point by whichever entry of
    `attractors` its final position ends up closest to.

    Parameters
    ----------
    rhs : callable
        A module-level Numba ``@njit`` function ``rhs(state, t, params) ->
        ndarray``, with a 4-dimensional state ``(x, y, vx, vy)`` (e.g.
        :func:`physicskit.chaos.systems.continuous.magnetic_pendulum_rhs`).
    params : array_like of float
        Parameter vector passed through to `rhs`.
    x_range, y_range : tuple of float
        ``(min, max)`` extent of the grid along each axis.
    attractors : array_like of float, shape (n_attractors, 2)
        Known ``(x, y)`` positions of the system's stable attractors.
    resolution : int, default 200
        Number of grid points along each axis (the grid has
        ``resolution**2`` points total, each requiring one full
        trajectory integration).
    velocity0 : tuple of float, default (0.0, 0.0)
        Initial velocity ``(vx, vy)``, the same for every grid point.
    dt : float, default 0.02
        Integration step size.
    n_steps : int, default 2000
        Number of integration steps per grid point; should be long enough
        for the trajectory to have settled near an attractor.

    Returns
    -------
    xs, ys : ndarray of float, shape (resolution,)
        Grid coordinates along each axis.
    labels : ndarray of int64, shape (resolution, resolution)
        ``labels[i, j]`` is the index into `attractors` of the attractor
        closest to the trajectory launched from ``(xs[j], ys[i])``.
    """
    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)
    labels = _compute_basin_grid(
        rhs,
        xs,
        ys,
        float(velocity0[0]),
        float(velocity0[1]),
        dt,
        n_steps,
        np.asarray(attractors, dtype=np.float64),
        np.asarray(params, dtype=np.float64),
    )
    return xs, ys, labels


def plot_basin_of_attraction(
    rhs: Callable,
    params: ArrayLike,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    attractors: ArrayLike,
    resolution: int = 200,
    velocity0: tuple[float, float] = (0.0, 0.0),
    dt: float = 0.02,
    n_steps: int = 2000,
    ax: Axes | None = None,
    cmap: str = theme.QUALITATIVE_CMAP,
) -> tuple[Figure, Axes]:
    """Plot a basin-of-attraction map.

    Parameters
    ----------
    rhs, params, x_range, y_range, attractors, resolution, velocity0, dt, n_steps
        See :func:`basin_of_attraction`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure and axes are created if omitted.
    cmap : str, default "tab10" (:data:`physicskit.chaos.visualizers.theme.QUALITATIVE_CMAP`)
        Colormap used for the (discrete) attractor labels.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    xs, ys, labels = basin_of_attraction(
        rhs,
        params,
        x_range,
        y_range,
        attractors,
        resolution=resolution,
        velocity0=velocity0,
        dt=dt,
        n_steps=n_steps,
    )
    attractors = np.asarray(attractors, dtype=np.float64)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = cast(Figure, ax.figure)

    ax.imshow(
        labels,
        extent=(xs.min(), xs.max(), ys.min(), ys.max()),
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
        aspect="equal",
    )
    ax.scatter(attractors[:, 0], attractors[:, 1], c=theme.STRUCTURE, marker="x", s=60, zorder=3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Basin of attraction")
    return fig, ax
