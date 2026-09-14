"""Static plotting helpers: decay chain populations and differential cross sections.

Matplotlib is used throughout. Every function returns its figure object
rather than calling ``show()``, so it composes cleanly into larger
figures or headless pipelines.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_decay_chain", "plot_differential_cross_section"]


def plot_decay_chain(t, populations, labels=None, ax=None):
    """Plot each species' population over time in a radioactive decay chain.

    Parameters
    ----------
    t : ndarray of shape (n_t,)
        Times.
    populations : ndarray of shape (n_species, n_t)
        Population of each species over time, e.g. from
        :func:`physicskit.particle.decays.bateman_decay_chain`.
    labels : sequence of str, optional
        Label for each species; defaults to ``"species 1"``, ``"species 2"``, ...
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.particle.decays import bateman_decay_chain
    >>> t = np.linspace(0, 10, 50)
    >>> N = bateman_decay_chain(1000.0, [0.5, 0.2], t)
    >>> fig, ax = plot_decay_chain(t, N, labels=["Parent", "Daughter"])
    >>> isinstance(fig, plt.Figure)
    True
    """
    populations = np.asarray(populations)
    if labels is None:
        labels = [f"species {i + 1}" for i in range(populations.shape[0])]
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    for row, label in zip(populations, labels, strict=True):
        ax.plot(t, row, label=label)
    ax.set_xlabel("t")
    ax.set_ylabel("population")
    ax.legend()
    return fig, ax


def plot_differential_cross_section(theta, dsigma_domega, ax=None, log_scale=True):
    """Plot a differential cross section :math:`d\\sigma/d\\Omega` vs. scattering angle.

    Parameters
    ----------
    theta : ndarray of shape (n,)
        Scattering angles (radians).
    dsigma_domega : ndarray of shape (n,)
        Differential cross section values, e.g. from
        :func:`physicskit.particle.scattering.rutherford_dsigma_domega`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.
    log_scale : bool, default=True
        Plot the cross section on a log y-axis, appropriate for the
        Rutherford formula's steep small-angle divergence.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.particle.scattering import rutherford_dsigma_domega
    >>> theta = np.linspace(0.1, np.pi - 0.1, 50)
    >>> vals = rutherford_dsigma_domega(theta, Z1=2, Z2=79, E_kin=5.0)
    >>> fig, ax = plot_differential_cross_section(theta, vals)
    >>> isinstance(fig, plt.Figure)
    True
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(theta, dsigma_domega)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$d\sigma/d\Omega$")
    return fig, ax
