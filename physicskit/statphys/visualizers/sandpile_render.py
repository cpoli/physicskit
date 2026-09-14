"""Visualizations of the BTW sandpile: pile heights and avalanche statistics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_avalanche_size_distribution", "plot_sandpile_heights"]


def plot_sandpile_heights(sandpile, ax=None, cmap="viridis"):
    """Render the current pile heights as a heatmap.

    Parameters
    ----------
    sandpile : BTWSandpile
        Sandpile instance to render.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new figure and axes are created if not given.
    cmap : str, default="viridis"
        Matplotlib colormap name.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(sandpile.heights, cmap=cmap, vmin=0, vmax=sandpile.threshold - 1, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="height")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Sandpile: {sandpile.total_grains()} grains")
    return ax


def plot_avalanche_size_distribution(sizes, ax=None, bins=40):
    """Log-log histogram of avalanche sizes, the self-organized-criticality power-law signature.

    Parameters
    ----------
    sizes : array_like
        Avalanche sizes, e.g. from :meth:`BTWSandpile.run`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.
    bins : int, default=40
        Number of logarithmically spaced bins.

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
    ax.loglog(centers[mask], counts[mask], marker="o", ms=3, linestyle="none")
    ax.set_xlabel("avalanche size s")
    ax.set_ylabel("P(s)")
    ax.set_title("Avalanche size distribution (SOC power law)")
    return ax
