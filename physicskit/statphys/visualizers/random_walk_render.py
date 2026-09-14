"""Visualizations of random walk trajectories and diffusive scaling."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_clt_histogram", "plot_msd", "plot_trajectories_2d"]


def plot_trajectories_2d(walk, ax=None, n_show=20):
    """Plot a subset of a 2D random walk ensemble's trajectories.

    Parameters
    ----------
    walk : RandomWalk
        A walk instance with ``dim=2``; run automatically if not already.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new figure and axes are created if not given.
    n_show : int, default=20
        Number of individual trajectories to draw.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if walk.dim != 2:
        raise ValueError("plot_trajectories_2d requires a RandomWalk with dim=2")
    if walk.trajectories is None:
        walk.run()
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    n_show = min(n_show, walk.n_walkers)
    for w in range(n_show):
        ax.plot(walk.trajectories[:, w, 0], walk.trajectories[:, w, 1], alpha=0.6, linewidth=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_title(f"{n_show} random walk trajectories")
    return ax


def plot_msd(walk, ax=None, show_theory=True):
    """Plot the ensemble mean squared displacement against the Einstein-relation prediction.

    Parameters
    ----------
    walk : RandomWalk
        Walk instance to summarize.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.
    show_theory : bool, default=True
        Whether to overlay the fitted line :math:`2 d D t`.

    Returns
    -------
    matplotlib.axes.Axes
    """
    t, msd = walk.mean_squared_displacement()
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    ax.plot(t, msd, label="simulation")
    if show_theory:
        D = walk.diffusion_coefficient()
        ax.plot(t, 2 * walk.dim * D * t, linestyle="--", label=f"$2dDt$, D={D:.3f}")
    ax.set_xlabel("t")
    ax.set_ylabel("MSD(t)")
    ax.legend()
    return ax


def plot_clt_histogram(walk, ax=None, axis=0, bins=40):
    """Histogram the final-time displacement against the Gaussian predicted by the CLT.

    Parameters
    ----------
    walk : RandomWalk
        Walk instance to summarize.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.
    axis : int, default=0
        Spatial axis to histogram.
    bins : int, default=40
        Number of histogram bins.

    Returns
    -------
    matplotlib.axes.Axes
    """
    from physicskit.statphys.utils.thermodynamics import maxwell_boltzmann_component_pdf

    centers, density = walk.final_displacement_histogram(axis=axis, bins=bins)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    ax.bar(centers, density, width=(centers[1] - centers[0]), alpha=0.6, label="simulation")

    if walk.trajectories is None:
        walk.run()
    var = np.var(walk.trajectories[-1, :, axis])
    xs = np.linspace(centers.min(), centers.max(), 200)
    gaussian = maxwell_boltzmann_component_pdf(xs, temperature=var, mass=1.0, kB=1.0)
    ax.plot(xs, gaussian, color="crimson", label="Gaussian (CLT)")
    ax.set_xlabel(f"final displacement (axis {axis})")
    ax.set_ylabel("density")
    ax.legend()
    return ax
