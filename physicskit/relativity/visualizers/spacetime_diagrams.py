"""Kruskal-Szekeres and Penrose-Carter conformal diagrams for the Schwarzschild spacetime.

Schwarzschild coordinates :math:`(t, r)` are singular at the horizon
:math:`r = 2M` -- a mere coordinate artifact, not a physical singularity,
but one that hides the true global structure of the spacetime. Kruskal and
Szekeres independently found coordinates :math:`(X, T)` that are perfectly
regular at the horizon and reveal that maximally extended Schwarzschild
spacetime actually contains *two* asymptotically flat regions connected by
a non-traversable wormhole (the Einstein-Rosen bridge). Compactifying
Kruskal coordinates further via Penrose and Carter's conformal
transformation brings the entire (infinite) spacetime into a single finite
diagram, making the causal structure -- including the true curvature
singularity at :math:`r=0` and the boundaries at infinity -- visible at a
glance.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "kruskal_coordinates",
    "penrose_carter_coordinates",
    "plot_kruskal_diagram",
    "plot_penrose_diagram",
]


def kruskal_coordinates(t, r, M):
    """Convert Schwarzschild :math:`(t, r)` to Kruskal-Szekeres :math:`(X, T)`.

    For the exterior region (:math:`r > 2M`):

    .. math::

        X = \\sqrt{r/2M - 1}\\, e^{r/4M} \\cosh(t/4M), \\qquad
        T = \\sqrt{r/2M - 1}\\, e^{r/4M} \\sinh(t/4M)

    and for the interior (:math:`r < 2M`), :math:`\\sinh` and
    :math:`\\cosh` swap roles. Curves of constant :math:`r` are hyperbolae
    :math:`X^2 - T^2 = (r/2M - 1)e^{r/2M}`; curves of constant :math:`t` are
    straight lines through the origin; the horizon :math:`r=2M` maps to the
    lines :math:`X = \\pm T`.

    Parameters
    ----------
    t, r : float or array_like
        Schwarzschild time and radial coordinates.
    M : float
        Mass, in geometrized units.

    Returns
    -------
    X, T : float or ndarray
        Kruskal-Szekeres coordinates.
    """
    t = np.asarray(t, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    exterior = r > 2.0 * M
    prefactor = np.sqrt(np.abs(r / (2.0 * M) - 1.0)) * np.exp(r / (4.0 * M))
    X = np.where(exterior, prefactor * np.cosh(t / (4.0 * M)), prefactor * np.sinh(t / (4.0 * M)))
    T = np.where(exterior, prefactor * np.sinh(t / (4.0 * M)), prefactor * np.cosh(t / (4.0 * M)))
    return X, T


def penrose_carter_coordinates(t, r, M):
    """Compactified Penrose-Carter diagram coordinates.

    Applies :math:`\\arctan` to the Kruskal null coordinates
    :math:`u = T - X`, :math:`v = T + X`, bringing the entire
    (infinite-range) spacetime into a finite diamond
    :math:`(\\text{space}, \\text{time}) \\in (-\\pi/2, \\pi/2)^2`.

    Parameters
    ----------
    t, r : float or array_like
        Schwarzschild time and radial coordinates.
    M : float
        Mass, in geometrized units.

    Returns
    -------
    space, time : float or ndarray
        Compactified diagram coordinates; light rays travel at
        :math:`\\pm 45^\\circ` exactly as in the Kruskal diagram.
    """
    X, T = kruskal_coordinates(t, r, M)
    u = T - X
    v = T + X
    up = np.arctan(u)
    vp = np.arctan(v)
    space = (vp - up) / 2.0
    time = (vp + up) / 2.0
    return space, time


def _plot_r_and_t_lines(ax, coord_transform, M, r_lines, t_lines, n_dense=400, t_span=40.0, r_span=(1.0e-3, 30.0)):
    t_dense = np.linspace(-t_span * M, t_span * M, n_dense)
    for rl in r_lines:
        x, y = coord_transform(t_dense, np.full_like(t_dense, rl), M)
        ax.plot(x, y, color="steelblue", linewidth=1.0)

    r_dense = np.linspace(r_span[0] * M, r_span[1] * M, n_dense)
    for tl in t_lines:
        x, y = coord_transform(np.full_like(r_dense, tl), r_dense, M)
        ax.plot(x, y, color="salmon", linewidth=0.7)


def plot_kruskal_diagram(M, ax=None, r_lines=None, t_lines=None, lim=6.0):
    """Plot a Kruskal-Szekeres diagram: constant-:math:`r` and constant-:math:`t` grid lines,
    the horizon light cone, and the :math:`r=0` singularity.

    Parameters
    ----------
    M : float
        Mass, in geometrized units.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new figure and axes are created if not given.
    r_lines : array_like, optional
        Radii (in units of ``M``) at which to draw constant-:math:`r` hyperbolae.
    t_lines : array_like, optional
        Times (in units of ``M``) at which to draw constant-:math:`t` rays.
    lim : float, default=6.0
        Axis limits, :math:`\\pm` this value.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    if r_lines is None:
        r_lines = np.array([0.5, 1.0, 1.5, 2.5, 3.5, 5.0]) * M
    if t_lines is None:
        t_lines = np.linspace(-8.0, 8.0, 9) * M

    _plot_r_and_t_lines(ax, kruskal_coordinates, M, r_lines, t_lines)

    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1.2, label="horizon r=2M")
    ax.plot([-lim, lim], [lim, -lim], "k--", linewidth=1.2)

    Xs = np.linspace(-lim, lim, 200)
    ax.plot(Xs, np.sqrt(Xs**2 + 1.0), "r-", linewidth=2, label="singularity r=0")
    ax.plot(Xs, -np.sqrt(Xs**2 + 1.0), "r-", linewidth=2)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("T")
    ax.set_title("Kruskal-Szekeres diagram")
    ax.legend(loc="upper left", fontsize=8)
    return ax


def plot_penrose_diagram(M, ax=None, r_lines=None, t_lines=None):
    """Plot a compactified Penrose-Carter diagram for maximally extended Schwarzschild spacetime.

    Parameters
    ----------
    M : float
        Mass, in geometrized units.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new figure and axes are created if not given.
    r_lines : array_like, optional
        Radii (in units of ``M``) at which to draw constant-:math:`r` curves.
    t_lines : array_like, optional
        Times (in units of ``M``) at which to draw constant-:math:`t` curves.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    if r_lines is None:
        r_lines = np.array([0.5, 1.0, 2.0, 4.0, 8.0]) * M
    if t_lines is None:
        t_lines = np.linspace(-10.0, 10.0, 9) * M

    _plot_r_and_t_lines(ax, penrose_carter_coordinates, M, r_lines, t_lines, t_span=200.0, r_span=(1.0e-3, 400.0))

    lim = np.pi / 2.0
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1.2, label="horizon r=2M")
    ax.plot([-lim, lim], [lim, -lim], "k--", linewidth=1.2)

    ax.plot([-lim, 0], [0, lim], "r-", linewidth=2, label="singularity r=0")
    ax.plot([0, lim], [lim, 0], "r-", linewidth=2)

    ax.set_xlim(-lim - 0.1, lim + 0.1)
    ax.set_ylim(-lim - 0.1, lim + 0.1)
    ax.set_aspect("equal")
    ax.set_xlabel("space")
    ax.set_ylabel("time")
    ax.set_title("Penrose-Carter conformal diagram")
    ax.legend(loc="upper left", fontsize=8)
    return ax
