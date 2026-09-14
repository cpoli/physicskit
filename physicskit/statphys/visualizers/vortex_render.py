"""Visualizations of the XY model: spin vector fields and topological vortices."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_vortices", "plot_xy_vector_field"]


def plot_xy_vector_field(theta, ax=None, step=1, cmap="hsv"):
    """Draw the XY model's planar spins as a quiver plot, colored by angle.

    Parameters
    ----------
    theta : ndarray of shape (L, L)
        Spin angles in radians, e.g. ``XYModel2D.theta``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new figure and axes are created if not given.
    step : int, default=1
        Subsample every ``step``-th spin along each axis, useful for large
        lattices where a dense quiver plot is unreadable.
    cmap : str, default="hsv"
        Colormap used to color arrows by their angle (a cyclic colormap is
        the natural choice for a periodic quantity).

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    L = theta.shape[0]
    idx = np.arange(0, L, step)
    ii, jj = np.meshgrid(idx, idx, indexing="ij")
    sub_theta = theta[np.ix_(idx, idx)]
    u = np.cos(sub_theta)
    v = np.sin(sub_theta)
    ax.quiver(jj, ii, u, v, sub_theta, cmap=cmap, pivot="mid", angles="xy", clim=(0, 2 * np.pi))
    ax.set_xlim(-1, L)
    ax.set_ylim(-1, L)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def plot_vortices(theta, vorticity=None, ax=None, threshold=0.5, step=1):
    """Overlay the XY spin field with markers at vortex and antivortex cores.

    Parameters
    ----------
    theta : ndarray of shape (L, L)
        Spin angles in radians.
    vorticity : ndarray of shape (L, L), optional
        Precomputed plaquette vorticity (see
        :func:`physicskit.statphys.core.monte_carlo.xy_plaquette_vorticity`); computed
        from ``theta`` if not given.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.
    threshold : float, default=0.5
        Minimum ``|vorticity|`` to mark as a topological charge.
    step : int, default=1
        Subsampling for the underlying vector field, forwarded to
        :func:`plot_xy_vector_field`.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if vorticity is None:
        from physicskit.statphys.core.monte_carlo import xy_plaquette_vorticity

        vorticity = xy_plaquette_vorticity(theta)

    ax = plot_xy_vector_field(theta, ax=ax, step=step)
    vi, vj = np.nonzero(vorticity > threshold)
    ai, aj = np.nonzero(vorticity < -threshold)
    ax.scatter(vj + 0.5, vi + 0.5, c="red", marker="o", s=60, label="vortex (+1)", zorder=5)
    ax.scatter(aj + 0.5, ai + 0.5, c="blue", marker="x", s=60, label="antivortex (-1)", zorder=5)
    if len(vi) or len(ai):
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    return ax
