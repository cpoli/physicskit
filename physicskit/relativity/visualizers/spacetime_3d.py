"""3D embedding diagrams: visualizing spacetime curvature as physical geometry."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["flamm_paraboloid", "plot_flamm_paraboloid"]


def flamm_paraboloid(M, r_min=None, r_max=None, n_r=100, n_phi=100):
    """Compute Flamm's paraboloid: an embedding of the Schwarzschild equatorial slice in flat 3D space.

    The spatial part of the Schwarzschild metric restricted to
    :math:`t = \\text{const}`, :math:`\\theta = \\pi/2` is

    .. math::

        d\\ell^2 = \\frac{dr^2}{1 - 2M/r} + r^2 d\\phi^2

    Embedding this curved 2-surface as a surface of revolution
    :math:`z(r)` in flat 3D Euclidean space (matching the proper radial
    distance along the surface to that of the curved metric) gives Flamm's
    paraboloid,

    .. math::

        z(r) = 2\\sqrt{2M(r - 2M)}, \\qquad r \\ge 2M

    the classic "rubber sheet" funnel shape used to visualize gravity as
    curved geometry -- though unlike the popular rubber-sheet cartoon, this
    is an exact embedding of the *actual* spatial curvature, not gravity
    acting on a sheet.

    Parameters
    ----------
    M : float
        Mass, in geometrized units.
    r_min : float, optional
        Inner radius. Defaults to the horizon, ``2M``.
    r_max : float, optional
        Outer radius. Defaults to ``20M``.
    n_r, n_phi : int, default=100
        Grid resolution in the radial and azimuthal directions.

    Returns
    -------
    X, Y, Z : ndarray of shape (n_phi, n_r)
        Cartesian embedding coordinates, suitable for
        ``Axes3D.plot_surface``.
    """
    if r_min is None:
        r_min = 2.0 * M
    if r_max is None:
        r_max = 20.0 * M
    r = np.linspace(r_min, r_max, n_r)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    R, Phi = np.meshgrid(r, phi)
    Z = 2.0 * np.sqrt(2.0 * M * np.clip(R - 2.0 * M, 0.0, None))
    X = R * np.cos(Phi)
    Y = R * np.sin(Phi)
    return X, Y, Z


def plot_flamm_paraboloid(M, ax=None, r_max=None, cmap="viridis"):
    """Plot Flamm's paraboloid as a 3D surface.

    Parameters
    ----------
    M : float
        Mass, in geometrized units.
    ax : mpl_toolkits.mplot3d.Axes3D, optional
        3D axes to draw into. A new figure and axes are created if not given.
    r_max : float, optional
        Outer radius, forwarded to :func:`flamm_paraboloid`.
    cmap : str, default="viridis"
        Matplotlib colormap name.

    Returns
    -------
    mpl_toolkits.mplot3d.Axes3D
    """
    X, Y, Z = flamm_paraboloid(M, r_max=r_max)
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, -Z, cmap=cmap, linewidth=0, antialiased=True, alpha=0.95)
    ax.set_xlabel("x [M]")
    ax.set_ylabel("y [M]")
    ax.set_zlabel("z (embedding depth) [M]")
    ax.set_title("Flamm's paraboloid: Schwarzschild spatial curvature")
    return ax
