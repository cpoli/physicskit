r"""Plotting helpers comparing classical trajectories with the exact quantum Wigner function."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_classical_trajectory_on_wigner"]


def _wigner_transform(x: np.ndarray, psi: np.ndarray, hbar: float = 1.0, n_p: int = 200, p_max: float | None = None):
    r"""Wigner quasi-probability distribution :math:`W(x,p)` of a 1D wavefunction.

    A local copy of :func:`physicskit.quantum.visualizers.phase_space.wigner_transform`,
    kept here so :mod:`physicskit.semiclassical` has no dependency on
    :mod:`physicskit.quantum`.
    """
    x = np.asarray(x)
    psi = np.asarray(psi, dtype=complex)
    N = x.shape[0]
    dx = x[1] - x[0]
    if p_max is None:
        p_max = np.pi * hbar / (2 * dx)

    p = np.linspace(-p_max, p_max, n_p)
    W = np.zeros((N, n_p))

    for i in range(N):
        kmax = min(i, N - 1 - i)
        k = np.arange(-kmax, kmax + 1)
        integrand = np.conj(psi[i + k]) * psi[i - k]
        phase = np.exp(2j * np.outer(p, k * dx) / hbar)
        W[i, :] = (dx / (np.pi * hbar)) * np.real(phase @ integrand)

    return x, p, W


def plot_classical_trajectory_on_wigner(x: np.ndarray, psi: np.ndarray, q_hist: np.ndarray, p_hist: np.ndarray, hbar: float = 1.0, ax=None):
    r"""Overlay a classical phase-space trajectory on the exact quantum Wigner function.

    Compares the classical and quantum pictures directly: the Wigner
    function of a quantum state, with the classical trajectory
    (:func:`physicskit.semiclassical.core.propagators.propagate_trajectory_monodromy_action`)
    that WKB/Van Vleck theory builds that state's semiclassical
    approximation from, drawn on top -- the two agree closely wherever
    the semiclassical approximation is accurate.

    Parameters
    ----------
    x : ndarray
        Position grid the wavefunction is sampled on.
    psi : ndarray of complex
        Quantum wavefunction sampled on ``x``.
    q_hist, p_hist : ndarray
        Classical trajectory phase-space coordinates.
    hbar : float, default=1.0
        Value of :math:`\hbar` to use.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.semiclassical.core.propagators import frozen_gaussian_1d
    >>> x = np.linspace(-8, 8, 400)
    >>> psi = frozen_gaussian_1d(x, qc=1.0, pc=0.5, gamma=1.0)
    >>> theta = np.linspace(0, 2 * np.pi, 100)
    >>> q_hist, p_hist = np.cos(theta), np.sin(theta)
    >>> fig, ax = plot_classical_trajectory_on_wigner(x, psi, q_hist, p_hist)
    >>> isinstance(fig, plt.Figure)
    True
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    _x, p, W = _wigner_transform(x, psi, hbar=hbar)
    im = ax.pcolormesh(x, p, W.T, cmap="RdBu_r", shading="auto")
    fig.colorbar(im, ax=ax, label="W(x,p)")
    ax.plot(q_hist, p_hist, color="k", linewidth=1.5, label="classical orbit")
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    ax.legend()
    return fig, ax
