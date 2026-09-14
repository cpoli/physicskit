r"""Animated growth of an entanglement measure over time."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

__all__ = ["animate_entanglement_growth"]


def animate_entanglement_growth(t_values: np.ndarray, metric_values: np.ndarray, ylabel: str = "concurrence", interval: int = 30, ax=None) -> FuncAnimation:
    """Animate a point tracing out an entanglement-measure curve as it builds up over time.

    Draws the curve ``metric_values`` vs. ``t_values`` progressively (rather
    than all at once), with a marker at the current time -- e.g. the
    concurrence or reduced-state purity of
    :class:`~physicskit.quantum.chapters.entanglement.IsingEntangler` growing
    from 0 (product state) toward 1 (maximally entangled) as the coupling acts.

    Parameters
    ----------
    t_values : numpy.ndarray
        Times.
    metric_values : numpy.ndarray
        The entanglement measure evaluated at each time in ``t_values``.
    ylabel : str, default='concurrence'
        Label for the y-axis.
    interval : int, default=30
        Delay between frames, in milliseconds.
    ax : matplotlib.axes.Axes or None, optional
        Axis to draw on; a new figure/axis is created if omitted.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    ax.set_xlim(t_values.min(), t_values.max())
    ymax = metric_values.max() * 1.1 if metric_values.max() > 0 else 1.0
    ax.set_ylim(0, ymax)
    ax.set_xlabel("t")
    ax.set_ylabel(ylabel)

    (line,) = ax.plot([], [], color="C0", lw=2)
    (point,) = ax.plot([], [], "o", color="darkred", markersize=8)

    def update(i):
        line.set_data(t_values[: i + 1], metric_values[: i + 1])
        point.set_data([t_values[i]], [metric_values[i]])
        ax.set_title(f"t = {t_values[i]:.3f}")
        return line, point

    return FuncAnimation(fig, update, frames=len(t_values), interval=interval, blit=False)
