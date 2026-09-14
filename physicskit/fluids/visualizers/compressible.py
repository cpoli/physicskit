"""Shock-tube profile plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from physicskit.fluids.visualizers import theme

__all__ = ["plot_shock_tube_profiles"]


def plot_shock_tube_profiles(x: NDArray[np.float64], rho: NDArray[np.float64], u: NDArray[np.float64], p: NDArray[np.float64]) -> tuple[Figure, NDArray]:
    """Plot density, velocity, and pressure profiles from a shock-tube solution.

    Parameters
    ----------
    x : ndarray of float
        Spatial grid.
    rho, u, p : ndarray of float
        Density, velocity, and pressure profiles, e.g. from
        :func:`physicskit.fluids.systems.compressible_flow.sod_shock_tube`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of matplotlib.axes.Axes, shape (3,)
    """
    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    for ax, field, label in zip(axes, (rho, u, p), (r"$\rho$", "u", "p")):
        ax.plot(x, field, color=theme.PRIMARY, lw=1.2)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("x")
    fig.suptitle("Sod shock tube")
    fig.tight_layout()
    return fig, axes
