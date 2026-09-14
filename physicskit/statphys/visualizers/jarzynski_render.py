"""Visualizations of the Jarzynski equality: nonequilibrium work distributions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_work_distribution"]


def plot_work_distribution(work, dF_true=0.0, dF_estimate=None, ax=None, bins=50):
    """Histogram nonequilibrium work samples against the true and Jarzynski-estimated free energy.

    Parameters
    ----------
    work : array_like
        Work samples, e.g. from :meth:`JarzynskiHarmonicTrap.run_protocol`.
    dF_true : float, default=0.0
        Known equilibrium free energy difference, marked with a solid line.
    dF_estimate : float, optional
        Jarzynski-equality estimate of the free energy (e.g. from
        :meth:`JarzynskiHarmonicTrap.jarzynski_free_energy_estimate`),
        marked with a dashed line if given.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.
    bins : int, default=50
        Number of histogram bins.

    Returns
    -------
    matplotlib.axes.Axes
    """
    work = np.asarray(work)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.hist(work, bins=bins, density=True, alpha=0.6, label="work samples")
    ax.axvline(work.mean(), color="tab:blue", linestyle=":", label=rf"$\langle W \rangle$={work.mean():.2f}")
    ax.axvline(dF_true, color="black", linewidth=2, label=rf"$\Delta F$={dF_true:.2f}")
    if dF_estimate is not None:
        ax.axvline(dF_estimate, color="tab:red", linestyle="--", label=rf"Jarzynski $\Delta F$={dF_estimate:.2f}")
    ax.set_xlabel("work W")
    ax.set_ylabel("density")
    ax.legend()
    return ax
