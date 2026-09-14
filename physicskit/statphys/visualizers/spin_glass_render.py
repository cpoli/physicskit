"""Visualizations of spin-glass replica overlaps."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_overlap_distribution"]


def plot_overlap_distribution(samples, ax=None, bins=40, label=None):
    """Histogram the replica-overlap samples P(q).

    Parameters
    ----------
    samples : array_like
        Overlap samples in ``[-1, 1]``, e.g. from
        :meth:`SherringtonKirkpatrick.overlap_distribution`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new figure and axes are created if not given.
    bins : int, default=40
        Number of histogram bins.
    label : str, optional
        Legend label for this histogram (useful when overlaying several
        temperatures on the same axes).

    Returns
    -------
    matplotlib.axes.Axes
    """
    samples = np.asarray(samples)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.hist(samples, bins=bins, range=(-1, 1), density=True, alpha=0.6, label=label)
    ax.set_xlabel("overlap q")
    ax.set_ylabel("P(q)")
    ax.set_xlim(-1, 1)
    if label is not None:
        ax.legend()
    return ax
