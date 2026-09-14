r"""Plotting helpers for WKB wavefunctions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_wkb_wavefunction"]


def plot_wkb_wavefunction(x: np.ndarray, psi: np.ndarray, V=None, ax=None):
    r"""Plot a WKB (or exact) 1D wavefunction, optionally with the potential on a twin axis.

    Parameters
    ----------
    x : ndarray
        Position grid.
    psi : ndarray
        Wavefunction values, e.g. from
        :func:`physicskit.semiclassical.core.wkb.wkb_wavefunction`.
    V : callable, optional
        Potential energy function ``V(x)``, drawn on a secondary y-axis
        if given.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.semiclassical.core.wkb import bohr_sommerfeld_energies, wkb_wavefunction
    >>> V = lambda x: 0.5 * x ** 2
    >>> E2 = bohr_sommerfeld_energies(V, m=1.0, x_min=-20, x_max=20, n_max=3)[2]
    >>> x = np.linspace(-6, 6, 1000)
    >>> psi = wkb_wavefunction(x, E2, V)
    >>> fig, ax = plot_wkb_wavefunction(x, psi, V=V)
    >>> isinstance(fig, plt.Figure)
    True
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(x, psi, color="C0", label=r"$\psi_{\mathrm{WKB}}(x)$")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\psi(x)$", color="C0")
    if V is not None:
        ax2 = ax.twinx()
        ax2.plot(x, V(x), color="C1", linestyle="--", label="V(x)")
        ax2.set_ylabel("V(x)", color="C1")
    return fig, ax
