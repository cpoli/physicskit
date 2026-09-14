r"""Wigner quasi-probability distribution :math:`W(x,p)` and 3D phase-space plots."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["wigner_transform", "WignerVisualizer"]


def wigner_transform(x: np.ndarray, psi: np.ndarray, hbar: float = 1.0, n_p: int = 200, p_max: float | None = None):
    r"""Compute the Wigner quasi-probability distribution.

    .. math::

        W(x,p) = \frac{1}{\pi\hbar} \int_{-\infty}^{\infty}
            \psi^*(x+y)\,\psi(x-y)\, e^{2ipy/\hbar}\, dy,

    on the same x-grid the wavefunction is given on, by direct quadrature
    over the shift :math:`y=k\,dx` (:math:`\psi` is treated as zero outside
    the grid).

    Parameters
    ----------
    x : numpy.ndarray
        Positions the wavefunction is sampled on (uniform grid).
    psi : numpy.ndarray
        Complex-valued wavefunction sampled on ``x``.
    hbar : float, default=1.0
        Value of :math:`\hbar` to use.
    n_p : int, default=200
        Number of momentum grid points.
    p_max : float or None, optional
        Half-width of the momentum grid; defaults to the Nyquist limit set
        by the position-grid spacing, :math:`\pi\hbar/(2\,dx)`.

    Returns
    -------
    x : numpy.ndarray
        The (unchanged) position grid.
    p : numpy.ndarray
        The momentum grid, shape ``(n_p,)``.
    W : numpy.ndarray
        The Wigner function, shape ``(len(x), n_p)``.
    """
    x = np.asarray(x)
    psi = np.asarray(psi, dtype=complex)
    N = x.shape[0]
    dx = x[1] - x[0]
    if p_max is None:
        # The transform's kernel is e^{2ipy/hbar} with y=k*dx, so the
        # conjugate variable is 2p/hbar; its Nyquist limit pi/dx gives
        # p_max = pi*hbar/(2*dx) (using the full pi*hbar/dx aliases W(x,p)
        # periodically back onto itself well before that point).
        p_max = np.pi * hbar / (2 * dx)

    p = np.linspace(-p_max, p_max, n_p)
    W = np.zeros((N, n_p))

    for i in range(N):
        kmax = min(i, N - 1 - i)
        k = np.arange(-kmax, kmax + 1)
        integrand = np.conj(psi[i + k]) * psi[i - k]
        phase = np.exp(2j * np.outer(p, k * dx) / hbar)  # (n_p, len(k))
        W[i, :] = (dx / (np.pi * hbar)) * np.real(phase @ integrand)

    return x, p, W


@dataclass
class WignerVisualizer:
    """Convenience wrapper around :func:`wigner_transform` with plotting helpers.

    Parameters
    ----------
    hbar : float, default=1.0
        Value of :math:`\\hbar` to use.
    n_p : int, default=200
        Number of momentum grid points.
    """

    hbar: float = 1.0
    n_p: int = 200

    def compute(self, x: np.ndarray, psi: np.ndarray, p_max: float | None = None):
        """See :func:`wigner_transform`.

        Parameters
        ----------
        x : numpy.ndarray
        psi : numpy.ndarray
        p_max : float or None, optional

        Returns
        -------
        x, p, W : numpy.ndarray
        """
        return wigner_transform(x, psi, self.hbar, self.n_p, p_max)

    def position_marginal(self, p: np.ndarray, W: np.ndarray) -> np.ndarray:
        r"""Integrate :math:`W(x,p)` over :math:`p`.

        Should recover :math:`\lvert\psi(x)\rvert^2` -- a standard
        correctness check for a Wigner-function implementation.

        Parameters
        ----------
        p : numpy.ndarray
            Momentum grid.
        W : numpy.ndarray
            Wigner function, shape ``(len(x), len(p))``.

        Returns
        -------
        numpy.ndarray
        """
        dp = p[1] - p[0]
        return np.sum(W, axis=1) * dp

    def plot_surface(self, x: np.ndarray, p: np.ndarray, W: np.ndarray, ax=None, cmap: str = "RdBu_r"):
        """3D surface plot of :math:`W(x,p)`.

        Uses a diverging colormap so the negative (non-classical
        interference) regions stand out from the positive ones.

        Parameters
        ----------
        x, p : numpy.ndarray
            Position and momentum grids.
        W : numpy.ndarray
            Wigner function, shape ``(len(x), len(p))``.
        ax : matplotlib.axes.Axes or None, optional
            A 3D axis to draw on; a new figure/axis is created if omitted.
        cmap : str, default='RdBu_r'
            Diverging colormap name.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
        X, P = np.meshgrid(x, p, indexing="ij")
        vmax = np.abs(W).max()
        ax.plot_surface(X, P, W, cmap=cmap, vmin=-vmax, vmax=vmax, linewidth=0, antialiased=True)
        ax.set_xlabel("x")
        ax.set_ylabel("p")
        ax.set_zlabel("W(x,p)")
        return ax

    def plot_contour(self, x: np.ndarray, p: np.ndarray, W: np.ndarray, ax=None, cmap: str = "RdBu_r"):
        """2D filled-contour plot of :math:`W(x,p)`.

        Parameters
        ----------
        x, p : numpy.ndarray
            Position and momentum grids.
        W : numpy.ndarray
            Wigner function, shape ``(len(x), len(p))``.
        ax : matplotlib.axes.Axes or None, optional
            Axis to draw on; a new figure/axis is created if omitted.
        cmap : str, default='RdBu_r'
            Diverging colormap name.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))
        vmax = np.abs(W).max()
        cs = ax.contourf(x, p, W.T, levels=60, cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_xlabel("x")
        ax.set_ylabel("p")
        plt.colorbar(cs, ax=ax, label="W(x,p)")
        return ax
