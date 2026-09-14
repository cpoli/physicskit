"""Pressure-coefficient plots for potential flow."""

from __future__ import annotations

from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from physicskit.fluids.visualizers import theme

__all__ = ["plot_pressure_coefficient"]


def plot_pressure_coefficient(theta: NDArray[np.float64], Cp: NDArray[np.float64], ax: Axes | None = None, **plot_kwargs: Any) -> tuple[Figure, Axes]:
    """Plot the surface pressure coefficient :math:`C_p(\\theta)` around a body.

    Parameters
    ----------
    theta : ndarray of float
        Angular position around the body (radians), typically ``[0, 2*pi]``.
    Cp : ndarray of float
        Pressure coefficient at each `theta`, e.g. from
        :func:`physicskit.fluids.systems.potential_flow.pressure_coefficient`
        evaluated on a circle.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.
    **plot_kwargs
        Additional keyword arguments forwarded to ``ax.plot``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = cast(Figure, ax.figure)
    kwargs: dict[str, Any] = {"color": theme.PRIMARY}
    kwargs.update(plot_kwargs)
    ax.plot(np.degrees(theta), Cp, **kwargs)
    ax.axhline(0.0, color=theme.MUTED, lw=0.8, ls="--")
    ax.invert_yaxis()
    ax.set_xlabel(r"$\theta$ (degrees)")
    ax.set_ylabel(r"$C_p$")
    ax.set_title("Surface pressure coefficient")
    return fig, ax
