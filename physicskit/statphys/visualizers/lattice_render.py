"""Static and animated visualizations of lattice spin configurations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

__all__ = [
    "animate_lattice_sweeps",
    "plot_cluster_size_distribution",
    "plot_percolation_clusters",
    "plot_spin_grid",
    "plot_thermodynamics",
]


def plot_spin_grid(spins, ax=None, cmap="coolwarm", title=None, colorbar=False):
    """Render a spin (or Potts state) configuration as a heatmap.

    Parameters
    ----------
    spins : ndarray of shape (L, L)
        Spin configuration, e.g. ``Ising2D.spins`` or ``PottsModel2D.spins``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new figure and axes are created if not given.
    cmap : str, default="coolwarm"
        Matplotlib colormap name.
    title : str, optional
        Axes title.
    colorbar : bool, default=False
        Whether to attach a colorbar.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> from physicskit.statphys.chapters import Ising2D
    >>> model = Ising2D(L=16, seed=0)
    >>> ax = plot_spin_grid(model.spins)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(spins, cmap=cmap, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def animate_lattice_sweeps(model, beta, n_frames=100, sweeps_per_frame=1, algorithm="metropolis", cmap="coolwarm"):
    """Animate a lattice model relaxing under repeated Monte Carlo sweeps.

    Parameters
    ----------
    model : Ising2D or PottsModel2D
        A model instance exposing a ``sweep(beta, ...)`` method and a
        ``spins`` attribute.
    beta : float
        Inverse temperature at which to sweep.
    n_frames : int, default=100
        Number of animation frames.
    sweeps_per_frame : int, default=1
        Sweeps performed between consecutive frames.
    algorithm : str, default="metropolis"
        Forwarded to ``model.sweep`` when supported (``Ising2D`` only).
    cmap : str, default="coolwarm"
        Matplotlib colormap name.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(model.spins, cmap=cmap, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title("sweep 0")

    def update(frame):
        try:
            model.sweep(beta, algorithm=algorithm, n_sweeps=sweeps_per_frame)
        except TypeError:
            model.sweep(beta, n_sweeps=sweeps_per_frame)
        im.set_data(model.spins)
        title.set_text(f"sweep {(frame + 1) * sweeps_per_frame}")
        return im, title

    return FuncAnimation(fig, update, frames=n_frames, blit=False, interval=50)


def plot_thermodynamics(result, T_c=None, ax=None):
    """Plot energy, magnetization, specific heat, and susceptibility versus temperature.

    Parameters
    ----------
    result : dict of str -> ndarray
        Output of ``Ising2D.run_temperature_sweep`` (or the analogous Potts
        method, in which case ``"M"``/``"chi"`` keys may be absent).
    T_c : float, optional
        Critical temperature to mark with a vertical dashed line on every
        panel.
    ax : ndarray of matplotlib.axes.Axes, optional
        A ``(2, 2)`` array of axes to draw into. A new figure is created if
        not given.

    Returns
    -------
    ndarray of matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
    T = result["T"]

    panels = [
        (ax[0, 0], "E", "Energy per site"),
        (ax[0, 1], "M" if "M" in result else "m", "Magnetization / order parameter"),
        (ax[1, 0], "C_v", "Specific heat"),
        (ax[1, 1], "chi", "Susceptibility"),
    ]
    for axis, key, label in panels:
        if key not in result:
            axis.axis("off")
            continue
        axis.plot(T, result[key], marker="o", ms=3)
        axis.set_ylabel(label)
        axis.set_xlabel("Temperature")
        if T_c is not None:
            axis.axvline(T_c, color="k", linestyle="--", linewidth=1, alpha=0.6)
    plt.tight_layout()
    return ax


def plot_percolation_clusters(percolation, ax=None, highlight_spanning=True, cmap="nipy_spectral"):
    """Render a percolation lattice with each connected cluster in a distinct color.

    Parameters
    ----------
    percolation : Percolation2D
        Percolation instance whose current ``labels`` should be drawn.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new figure and axes are created if not given.
    highlight_spanning : bool, default=True
        Whether to redraw spanning clusters (if any) in solid black on top
        of the color-coded background.
    cmap : str, default="nipy_spectral"
        Colormap used to distinguish clusters. Unoccupied sites are drawn in
        white.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    labels = percolation.labels
    display = np.ma.masked_equal(labels, 0)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("white")
    ax.imshow(display, cmap=cmap_obj, interpolation="nearest")
    if highlight_spanning:
        spanning = percolation.spanning_labels()
        if len(spanning):
            mask = np.isin(labels, spanning)
            spanning_display = np.ma.masked_where(~mask, mask.astype(float))
            ax.imshow(spanning_display, cmap="Greys", vmin=0, vmax=1, alpha=0.7, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{percolation.mode} percolation, p = {percolation.p:.3f}")
    return ax


def plot_cluster_size_distribution(sizes, ax=None, bins=30, tau_theory=187 / 91):
    """Log-log histogram of percolation cluster sizes against the theoretical Fisher exponent.

    Parameters
    ----------
    sizes : array_like
        Cluster sizes, e.g. from ``Percolation2D.cluster_size_distribution``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.
    bins : int, default=30
        Number of logarithmically spaced bins.
    tau_theory : float, default=187/91
        Theoretical Fisher exponent to overlay as a reference power law.
        Pass ``None`` to omit it.

    Returns
    -------
    matplotlib.axes.Axes
    """
    sizes = np.asarray(sizes)
    sizes = sizes[sizes > 0]
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    log_bins = np.logspace(0, np.log10(sizes.max()), bins)
    counts, edges = np.histogram(sizes, bins=log_bins, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:])
    mask = counts > 0
    ax.loglog(centers[mask], counts[mask], marker="o", ms=3, linestyle="none", label="simulation")
    if tau_theory is not None and np.any(mask):
        ref = counts[mask][0] * (centers[mask] / centers[mask][0]) ** (-tau_theory)
        ax.loglog(centers[mask], ref, linestyle="--", color="k", label=f"$s^{{-{tau_theory:.3f}}}$")
    ax.set_xlabel("cluster size s")
    ax.set_ylabel("P(s)")
    ax.legend()
    return ax
