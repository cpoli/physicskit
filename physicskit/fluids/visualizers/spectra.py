"""Turbulent energy-spectrum plots."""

from __future__ import annotations

from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from physicskit.fluids.utils.spectral_analysis import kolmogorov_reference_slope
from physicskit.fluids.visualizers import theme

__all__ = ["plot_energy_spectrum"]


def plot_energy_spectrum(
    k: NDArray[np.float64], E: NDArray[np.float64], ax: Axes | None = None, show_kolmogorov: bool = True, **plot_kwargs: Any
) -> tuple[Figure, Axes]:
    """Log-log plot of a kinetic energy spectrum, with an optional Kolmogorov -5/3 reference line.

    Parameters
    ----------
    k : ndarray of float
        Wavenumbers, e.g. from :func:`physicskit.fluids.utils.spectral_analysis.energy_spectrum`.
    E : ndarray of float
        Spectral energy density at each `k`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.
    show_kolmogorov : bool, default True
        If True, overlay a :math:`k^{-5/3}` reference line anchored at the
        lowest plotted (nonzero-energy) wavenumber.
    **plot_kwargs
        Additional keyword arguments forwarded to ``ax.loglog`` for the
        measured spectrum.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = cast(Figure, ax.figure)
    k = np.asarray(k, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    mask = (k > 0) & (E > 0)
    kwargs: dict[str, Any] = {"color": theme.PRIMARY, "marker": "o", "markersize": 3, "label": "E(k)"}
    kwargs.update(plot_kwargs)
    ax.loglog(k[mask], E[mask], **kwargs)
    if show_kolmogorov and np.any(mask):
        k0 = k[mask][0]
        E0 = E[mask][0]
        ax.loglog(k[mask], kolmogorov_reference_slope(k[mask], k0, E0), color=theme.ACCENT, ls="--", label=r"$k^{-5/3}$")
    ax.set_xlabel("k")
    ax.set_ylabel("E(k)")
    ax.set_title("Kinetic energy spectrum")
    ax.legend()
    return fig, ax
