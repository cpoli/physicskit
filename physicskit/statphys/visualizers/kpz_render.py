"""Visualizations of KPZ/RSOS interface growth: height profiles and width scaling."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_interface_profile", "plot_width_growth"]


def plot_interface_profile(interface, ax=None):
    """Plot the current 1D interface height profile.

    Parameters
    ----------
    interface : KPZInterface
        Interface instance to render.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new figure and axes are created if not given.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3))
    x = np.arange(interface.L)
    ax.plot(x, interface.heights, drawstyle="steps-mid")
    ax.set_xlabel("site")
    ax.set_ylabel("height h(x)")
    ax.set_title(f"RSOS interface at t={interface.time:.0f} sweeps")
    return ax


def plot_width_growth(times, widths, ax=None, show_theory=True, beta=1.0 / 3.0):
    """Log-log plot of interface width vs. time, with a t^beta growth reference line.

    Parameters
    ----------
    times, widths : array_like
        Growth curve, e.g. from :meth:`KPZInterface.run_growth_curve`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.
    show_theory : bool, default=True
        Overlay a :math:`t^\\beta` reference line (matched in amplitude to
        the first data point) for the KPZ growth exponent.
    beta : float, default=1/3
        Growth exponent for the reference line (the 1+1D KPZ value).

    Returns
    -------
    matplotlib.axes.Axes
    """
    times = np.asarray(times)
    widths = np.asarray(widths)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    ax.loglog(times, widths, marker="o", ms=3, linestyle="none", label="simulation")
    if show_theory:
        amplitude = widths[0] / times[0] ** beta
        ax.loglog(times, amplitude * times**beta, linestyle="--", label=rf"$t^{{{beta:.2f}}}$")
    ax.set_xlabel("t (sweeps)")
    ax.set_ylabel("interface width w(t)")
    ax.legend()
    return ax
