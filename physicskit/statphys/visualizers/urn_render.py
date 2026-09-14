"""Visualization of the Ehrenfest urn model's relaxation to equilibrium."""

from __future__ import annotations

import matplotlib.pyplot as plt

__all__ = ["plot_ehrenfest_history"]


def plot_ehrenfest_history(history, n_balls, ax=None):
    """Plot the box occupancy and entropy history of an Ehrenfest urn run.

    Parameters
    ----------
    history : dict of str -> ndarray
        Output of :meth:`EhrenfestUrn.run`.
    n_balls : int
        Total ball count, used to mark the equilibrium level N/2.
    ax : ndarray of matplotlib.axes.Axes, optional
        A pair of axes to draw into; a new figure is created if not given.

    Returns
    -------
    ndarray of matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["t"], history["n_left"], linewidth=0.8)
    ax[0].axhline(n_balls / 2, color="k", linestyle="--", linewidth=1, alpha=0.6)
    ax[0].set_xlabel("step")
    ax[0].set_ylabel("balls in left box")
    ax[0].set_title("Relaxation to equilibrium")

    ax[1].plot(history["t"], history["entropy"], color="crimson")
    ax[1].set_xlabel("step")
    ax[1].set_ylabel("entropy $S/k_B$")
    ax[1].set_title("Monotonic entropy growth")
    return ax
