"""Visualization of block-spin renormalization group flows."""

from __future__ import annotations

import matplotlib.pyplot as plt

__all__ = ["plot_rg_flow"]


def plot_rg_flow(grids, titles=None, cmap="coolwarm"):
    """Plot a sequence of coarse-grained lattices side by side.

    Parameters
    ----------
    grids : list of ndarray
        Sequence of spin configurations at increasing coarse-graining
        levels, e.g. the output of ``BlockSpinRG.iterate``.
    titles : list of str, optional
        One title per panel. Defaults to ``"L = {size}"`` for each grid.
    cmap : str, default="coolwarm"
        Matplotlib colormap name.

    Returns
    -------
    ndarray of matplotlib.axes.Axes

    Examples
    --------
    >>> from physicskit.statphys.chapters import BlockSpinRG
    >>> rg = BlockSpinRG(L=16, T=2.269, seed=0)
    >>> axes = plot_rg_flow(rg.iterate())
    """
    n = len(grids)
    _fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    if titles is None:
        titles = [f"L = {g.shape[0]}" for g in grids]
    for ax, grid, title in zip(axes, grids, titles):
        ax.imshow(grid, cmap=cmap, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    return axes
