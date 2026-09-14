"""Visualizations of gravitational wave strain time series and spatial ripple patterns."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from physicskit.relativity.utils.constants import check_geometrized_distance, check_geometrized_mass

__all__ = [
    "animate_wave_ripple",
    "plot_strain_waveform",
    "plot_wave_ripple",
    "wave_ripple_snapshot",
]


def plot_strain_waveform(t, h_plus, h_cross=None, ax=None, t_merger=None):
    """Plot a gravitational wave strain time series.

    Parameters
    ----------
    t : array_like
        Time samples.
    h_plus : array_like
        Plus-polarization strain.
    h_cross : array_like, optional
        Cross-polarization strain, plotted alongside ``h_plus`` if given.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new figure and axes are created if not given.
    t_merger : float, optional
        Merger time, marked with a vertical dashed line.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, h_plus, label=r"$h_+$", linewidth=1.2)
    if h_cross is not None:
        ax.plot(t, h_cross, label=r"$h_\times$", linewidth=1.0, alpha=0.75)
    if t_merger is not None:
        ax.axvline(t_merger, color="k", linestyle="--", linewidth=1, alpha=0.6, label="merger")
    ax.set_xlabel("time")
    ax.set_ylabel("strain")
    ax.legend()
    return ax


def wave_ripple_snapshot(merger, t, t_merger, grid_size=150, extent=40.0):
    """Compute a 2D spatial snapshot of a merger's radiated wave pattern in the orbital plane.

    Uses the actual (oscillating) :math:`h_+` strain from
    ``merger.full_waveform`` evaluated at each grid point's retarded time
    :math:`t_{\\text{ret}} = t - r`, with :math:`1/r` amplitude falloff and
    the schematic quadrupolar :math:`\\cos(2\\varphi)` azimuthal pattern
    characteristic of the dominant (l=2, m=2) gravitational wave mode as
    seen face-on. Using the oscillating strain itself, rather than its
    envelope, is what makes concentric ripples appear at all: at fixed
    ``t``, sweeping outward in ``r`` sweeps ``t_ret`` backward through the
    chirp's accumulating phase, so the wave's rising frequency shows up
    directly as rings that pack tighter close to the source. Warns (via
    :func:`~physicskit.relativity.utils.constants.check_geometrized_mass` /
    :func:`~physicskit.relativity.utils.constants.check_geometrized_distance`)
    if ``merger``'s mass or distance looks like an unconverted physical
    value rather than a geometrized length -- this is exactly the failure
    mode that once made this function's animation silently non-oscillating
    (see the package history).

    Parameters
    ----------
    merger : BinaryMerger
        The source binary.
    t : float
        Observation time.
    t_merger : float
        Coalescence time, forwarded to ``merger.full_waveform``.
    grid_size : int, default=150
        Number of grid points per spatial axis.
    extent : float, default=40.0
        Half-width of the spatial grid, in units of ``M``.

    Returns
    -------
    X, Y : ndarray of shape (grid_size, grid_size)
        Spatial grid coordinates.
    pattern : ndarray of shape (grid_size, grid_size)
        The schematic strain pattern at time ``t``.
    """
    check_geometrized_mass(merger.m1, name="merger.m1")
    check_geometrized_mass(merger.m2, name="merger.m2")
    check_geometrized_distance(merger.distance, name="merger.distance")

    x = np.linspace(-extent, extent, grid_size)
    y = np.linspace(-extent, extent, grid_size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    Phi = np.arctan2(Y, X)
    R_safe = np.clip(R, 1.0, None)

    t_ret = t - R_safe
    hp_flat, _ = merger.full_waveform(t_ret.ravel(), t_merger)
    hp = hp_flat.reshape(R.shape)
    pattern = hp * (merger.distance / R_safe) * np.cos(2.0 * Phi)
    return X, Y, pattern


def plot_wave_ripple(merger, t, t_merger, ax=None, grid_size=150, extent=40.0, cmap="RdBu"):
    """Plot one snapshot of a merger's spatial wave ripple pattern.

    Parameters
    ----------
    merger : BinaryMerger
        The source binary.
    t : float
        Observation time.
    t_merger : float
        Coalescence time.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.
    grid_size : int, default=150
        Spatial grid resolution.
    extent : float, default=40.0
        Half-width of the spatial grid, in units of ``M``.
    cmap : str, default="RdBu"
        Diverging colormap name.

    Returns
    -------
    matplotlib.axes.Axes
    """
    X, Y, pattern = wave_ripple_snapshot(merger, t, t_merger, grid_size=grid_size, extent=extent)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    vmax = np.max(np.abs(pattern)) or 1.0
    im = ax.pcolormesh(X, Y, pattern, cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto")
    ax.set_aspect("equal")
    ax.set_xlabel("x [M]")
    ax.set_ylabel("y [M]")
    ax.set_title(f"GW spatial ripple pattern, t={t:.1f}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="strain (schematic)")
    return ax


def animate_wave_ripple(merger, t_merger, t_values, grid_size=100, extent=40.0, cmap="RdBu"):
    """Animate a merger's spatial wave ripple pattern over a sequence of times.

    Parameters
    ----------
    merger : BinaryMerger
        The source binary.
    t_merger : float
        Coalescence time.
    t_values : array_like
        Sequence of observation times, one per animation frame.
    grid_size : int, default=100
        Spatial grid resolution.
    extent : float, default=40.0
        Half-width of the spatial grid, in units of ``M``.
    cmap : str, default="RdBu"
        Diverging colormap name.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    X, Y, pattern0 = wave_ripple_snapshot(merger, t_values[0], t_merger, grid_size=grid_size, extent=extent)
    vmax = np.max(np.abs(pattern0)) or 1.0
    mesh = ax.pcolormesh(X, Y, pattern0, cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto")
    ax.set_aspect("equal")
    ax.set_xlabel("x [M]")
    ax.set_ylabel("y [M]")
    title = ax.set_title(f"t={t_values[0]:.1f}")

    def update(frame):
        t = t_values[frame]
        _, _, pattern = wave_ripple_snapshot(merger, t, t_merger, grid_size=grid_size, extent=extent)
        mesh.set_array(pattern.ravel())
        title.set_text(f"t={t:.1f}")
        return mesh, title

    return FuncAnimation(fig, update, frames=len(t_values), blit=False, interval=60)
