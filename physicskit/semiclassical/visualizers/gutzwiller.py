r"""Plotting helpers for the Gutzwiller trace-formula density of states."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_density_of_states"]


def plot_density_of_states(E_grid: np.ndarray, dos: np.ndarray, exact_energies: np.ndarray | None = None, ax=None):
    r"""Plot the Gutzwiller trace-formula density of states, with exact levels marked.

    Parameters
    ----------
    E_grid : ndarray
        Energy grid.
    dos : ndarray
        Density of states, e.g. from
        :func:`physicskit.semiclassical.core.gutzwiller.gutzwiller_density_of_states`.
    exact_energies : ndarray, optional
        Exact (or Bohr-Sommerfeld) energy levels to mark with vertical
        dashed lines.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.semiclassical.core.gutzwiller import gutzwiller_density_of_states
    >>> from physicskit.semiclassical.core.wkb import bohr_sommerfeld_energies
    >>> V = lambda x: 0.5 * x ** 2
    >>> E_grid = np.linspace(0.2, 3.5, 200)
    >>> dos = gutzwiller_density_of_states(E_grid, V, m=1.0, x_min=-20, x_max=20)
    >>> energies = bohr_sommerfeld_energies(V, m=1.0, x_min=-20, x_max=20, n_max=3)
    >>> fig, ax = plot_density_of_states(E_grid, dos, exact_energies=energies)
    >>> isinstance(fig, plt.Figure)
    True
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(E_grid, dos, color="C0")
    if exact_energies is not None:
        for E_n in exact_energies:
            ax.axvline(E_n, color="C3", linestyle="--", linewidth=0.8)
    ax.set_xlabel("E")
    ax.set_ylabel("g(E)")
    return fig, ax
