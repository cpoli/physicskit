r"""Plotting helpers for quantum scars: eigenstate density maps and Husimi phase-space projections.

:func:`plot_scar_map_interactive` returns a Plotly figure for pan/zoom
exploration of a scarred eigenstate density; the rest return a
Matplotlib figure and axes rather than calling ``show()``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

__all__ = ["plot_scar_map", "plot_husimi_1d", "plot_scar_map_interactive"]


def plot_scar_map(
    X: np.ndarray,
    Y: np.ndarray,
    density: np.ndarray,
    mask: np.ndarray | None = None,
    orbit_x: np.ndarray | None = None,
    orbit_y: np.ndarray | None = None,
    ax=None,
):
    r"""Heatmap an eigenstate density with a classical periodic orbit overlaid.

    Parameters
    ----------
    X, Y : ndarray
        Coordinate meshgrid.
    density : ndarray
        Probability density :math:`|\psi|^2`, same shape as ``X``.
    mask : ndarray of bool, optional
        ``True`` inside the billiard; density outside is not drawn.
    orbit_x, orbit_y : ndarray, optional
        Classical orbit coordinates, e.g. from
        :func:`physicskit.semiclassical.systems.scarring.bouncing_ball_orbit_points`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.semiclassical.systems.scarring import bouncing_ball_orbit_points
    >>> x = np.linspace(-2, 2, 100)
    >>> y = np.linspace(-1, 1, 60)
    >>> X, Y = np.meshgrid(x, y, indexing="ij")
    >>> density = np.exp(-(X - 0.3) ** 2 / (2 * 0.2 ** 2))
    >>> orbit_x, orbit_y = bouncing_ball_orbit_points(x0=0.3, R=1.0)
    >>> fig, ax = plot_scar_map(X, Y, density, orbit_x=orbit_x, orbit_y=orbit_y)
    >>> isinstance(fig, plt.Figure)
    True
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    plot_density = np.where(mask, density, np.nan) if mask is not None else density
    im = ax.pcolormesh(X, Y, plot_density, cmap="inferno", shading="auto")
    fig.colorbar(im, ax=ax, label=r"$|\psi|^2$")
    if orbit_x is not None:
        ax.plot(orbit_x, orbit_y, color="c", linewidth=1.2, label="classical orbit")
        ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    return fig, ax


def plot_husimi_1d(S0: np.ndarray, P0: np.ndarray, husimi: np.ndarray, ax=None):
    r"""Contour-plot a 1D Husimi phase-space projection.

    Parameters
    ----------
    S0, P0 : ndarray
        Phase-space grid, from :func:`physicskit.semiclassical.systems.scarring.husimi_projection_1d`.
    husimi : ndarray
        Husimi distribution, same shape as ``S0``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.semiclassical.systems.scarring import husimi_projection_1d
    >>> s = np.linspace(-10, 10, 1000)
    >>> psi = np.exp(-(s - 2.0) ** 2 / 2.0) * np.exp(1j * 3.0 * s)
    >>> S0, P0, H = husimi_projection_1d(psi, s, sigma=1.0, resolution=40, s0_range=(-2, 6), p0_range=(-2, 8))
    >>> fig, ax = plot_husimi_1d(S0, P0, H)
    >>> isinstance(fig, plt.Figure)
    True
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    im = ax.pcolormesh(S0, P0, husimi, cmap="viridis", shading="auto")
    fig.colorbar(im, ax=ax, label="Husimi (peak-normalized)")
    ax.set_xlabel("s")
    ax.set_ylabel("p")
    return fig, ax


def plot_scar_map_interactive(X: np.ndarray, Y: np.ndarray, density: np.ndarray):
    r"""Interactive Plotly heatmap of an eigenstate density, for zooming into scarred structure.

    Parameters
    ----------
    X, Y : ndarray
        Coordinate meshgrid.
    density : ndarray
        Probability density :math:`|\psi|^2`, same shape as ``X``.

    Returns
    -------
    plotly.graph_objects.Figure

    See Also
    --------
    plot_scar_map : The static Matplotlib heatmap equivalent, with orbit overlay support.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-2, 2, 60)
    >>> y = np.linspace(-1, 1, 40)
    >>> X, Y = np.meshgrid(x, y, indexing="ij")
    >>> density = np.exp(-(X - 0.3) ** 2 / (2 * 0.2 ** 2))
    >>> fig = plot_scar_map_interactive(X, Y, density)
    >>> isinstance(fig, go.Figure)
    True
    """
    fig = go.Figure(data=go.Heatmap(x=X[:, 0], y=Y[0, :], z=density.T, colorscale="Inferno"))
    fig.update_layout(xaxis_title="x", yaxis_title="y")
    return fig
