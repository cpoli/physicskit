"""Visualizations of Poincare recurrence: recurrence plots and return-time histograms."""

from __future__ import annotations

from typing import cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from physicskit.chaos.utils.recurrence import recurrence_matrix, recurrence_times
from physicskit.chaos.visualizers import theme


def plot_recurrence_matrix(states: ArrayLike, epsilon: float, ax: Axes | None = None) -> tuple[Figure, Axes]:
    """Plot the classic black-and-white recurrence plot of a trajectory.

    Parameters
    ----------
    states : array_like of float, shape (n, dim)
        Trajectory samples.
    epsilon : float
        Neighborhood radius defining a "recurrence".
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure and axes are created if omitted.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    matrix = recurrence_matrix(states, epsilon)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = cast(Figure, ax.figure)

    ax.imshow(matrix, cmap="Greys", origin="lower", interpolation="nearest")
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    ax.set_title(f"Recurrence plot ($\\epsilon$ = {epsilon:g})")
    return fig, ax


def plot_recurrence_times(
    states: ArrayLike,
    epsilon: float,
    reference_idx: int = 0,
    dt: float = 1.0,
    bins: int = 40,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a histogram of the recurrence-time distribution to a reference point.

    Parameters
    ----------
    states : array_like of float, shape (n, dim)
        Trajectory samples.
    epsilon : float
        Neighborhood radius defining a "return".
    reference_idx : int, default 0
        Index into `states` of the reference point to measure returns to.
    dt : float, default 1.0
        Time between consecutive rows of `states`.
    bins : int, default 40
        Number of histogram bins.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure and axes are created if omitted.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    times = recurrence_times(states, epsilon, reference_idx=reference_idx, dt=dt)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = cast(Figure, ax.figure)

    ax.hist(times, bins=bins, color=theme.PRIMARY, alpha=0.8)
    if times.size:
        ax.axvline(times.mean(), color=theme.ACCENT, linestyle="--", label=f"mean = {times.mean():.3g}")
        ax.legend()
    ax.set_xlabel("recurrence time")
    ax.set_ylabel("count")
    ax.set_title(f"Recurrence-time distribution ($\\epsilon$ = {epsilon:g})")
    return fig, ax
