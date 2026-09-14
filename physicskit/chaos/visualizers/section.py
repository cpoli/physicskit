"""Poincare sections (surface-of-section crossings) for continuous-time flows.

Unlike :mod:`physicskit.chaos.visualizers.phase_space` (which builds a billiard's
boundary phase space from its ray-tracing bounces), this module implements
the original form of the technique Poincare introduced while studying the
restricted three-body problem: sample a continuous flow only at the instants
it pierces a fixed surface in state space, turning the ODE's trajectory into
a lower-dimensional map whose structure -- smooth invariant curves (or a
handful of points) for regular motion, a scattered "chaotic sea" otherwise --
is far easier to read than the raw, tangled trajectory.
"""

from __future__ import annotations

from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

from physicskit.chaos.core.base_system import DynamicalSystem
from physicskit.chaos.visualizers import theme


def poincare_crossings(
    system: DynamicalSystem,
    state0: ArrayLike,
    coord: int,
    value: float = 0.0,
    direction: float = 1.0,
    t_max: float = 200.0,
    dt: float = 0.001,
) -> NDArray[np.float64]:
    """Integrate one trajectory and record its state at every crossing of a
    fixed surface ``state[coord] == value``.

    Parameters
    ----------
    system : DynamicalSystem
        System to integrate, via its own ``trajectory(state0, dt, n_steps)``.
    state0 : array_like of float, shape (dim,)
        Initial state.
    coord : int
        Index of the state component that defines the section surface.
    value : float, default 0.0
        Value of ``state[coord]`` the surface sits at.
    direction : float, default 1.0
        Sign of ``d(state[coord])/dt`` required to keep a crossing (e.g. the
        default ``1.0`` keeps only crossings where `coord` is increasing,
        discarding the return crossing on the way back, so each pass through
        the surface is counted once instead of twice).
    t_max : float, default 200.0
        Total integration time.
    dt : float, default 0.001
        Fixed integration step size.

    Returns
    -------
    ndarray of float, shape (n_crossings, dim)
        The full state at each qualifying crossing, linearly interpolated
        between the two bracketing integration steps.
    """
    n_steps = int(round(t_max / dt))
    _, states = system.trajectory(state0=np.asarray(state0, dtype=np.float64), dt=dt, n_steps=n_steps)

    f = states[:, coord] - value
    crosses = (np.sign(f[:-1]) != np.sign(f[1:])) & (f[:-1] != 0.0)
    if direction > 0:
        crosses &= f[1:] > f[:-1]
    elif direction < 0:
        crosses &= f[1:] < f[:-1]
    idx = np.flatnonzero(crosses)
    if idx.size == 0:
        return np.empty((0, states.shape[1]))

    f0, f1 = f[idx], f[idx + 1]
    frac = f0 / (f0 - f1)
    return states[idx] + frac[:, None] * (states[idx + 1] - states[idx])


def plot_poincare_map(
    system: DynamicalSystem,
    initial_states: ArrayLike,
    coord: int = 1,
    value: float = 0.0,
    direction: float = 1.0,
    plot_coords: tuple[int, int] = (0, 2),
    t_max: float = 200.0,
    dt: float = 0.001,
    labels: list[str] | None = None,
    ax: Axes | None = None,
    **scatter_kwargs: Any,
) -> tuple[Figure, Axes]:
    """Overlay Poincare-section crossings from several initial conditions.

    Each initial condition in `initial_states` is integrated independently
    and its :func:`poincare_crossings` plotted as its own scatter series
    (colored from :data:`~physicskit.chaos.visualizers.theme.QUALITATIVE_CMAP`), so
    a regular orbit's crossings -- collapsing onto a smooth curve, or even a
    handful of points, for a periodic orbit -- can be compared directly
    against a chaotic orbit's crossings, which scatter to fill a 2D patch.

    Parameters
    ----------
    system : DynamicalSystem
        System to integrate.
    initial_states : array_like of float, shape (n_series, dim)
        Initial states, one per series to overlay.
    coord : int, default 1
        Index of the state component that defines the section surface
        (default ``1``, i.e. ``y`` for a state ``(x, y, vx, vy)``).
    value : float, default 0.0
        Value of ``state[coord]`` the surface sits at.
    direction : float, default 1.0
        Sign of ``d(state[coord])/dt`` required to keep a crossing; see
        :func:`poincare_crossings`.
    plot_coords : tuple of int, default (0, 2)
        Indices of the two state components to scatter against each other
        (default: ``x`` vs. ``vx`` for a state ``(x, y, vx, vy)``).
    t_max : float, default 200.0
        Total integration time per series.
    dt : float, default 0.001
        Fixed integration step size.
    labels : list of str, optional
        Legend label for each series; the legend is omitted entirely if not
        given.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure and axes are created if omitted.
    **scatter_kwargs
        Additional keyword arguments forwarded to every ``ax.scatter`` call.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    initial_states = np.atleast_2d(np.asarray(initial_states, dtype=np.float64))

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = cast(Figure, ax.figure)

    cmap = plt.get_cmap(theme.QUALITATIVE_CMAP)
    i0, i1 = plot_coords
    for i, s0 in enumerate(initial_states):
        crossings = poincare_crossings(system, s0, coord=coord, value=value, direction=direction, t_max=t_max, dt=dt)
        if crossings.shape[0] == 0:
            continue
        kwargs: dict[str, Any] = {"s": 6, "alpha": 0.7, "color": cmap(i % cmap.N)}
        if labels is not None:
            kwargs["label"] = labels[i]
        kwargs.update(scatter_kwargs)
        ax.scatter(crossings[:, i0], crossings[:, i1], **kwargs)

    if labels is not None:
        ax.legend(fontsize=8)
    ax.set_title(f"{system.__class__.__name__} Poincare section")
    return fig, ax
