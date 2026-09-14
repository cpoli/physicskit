r"""Quantum measurement simulator and expectation-value / uncertainty monitors."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .._compat import trapz

__all__ = ["expectation_value", "uncertainty", "momentum_density", "ExpectationMonitor", "simulate_position_measurement"]


def expectation_value(operator_action, x: np.ndarray, psi: np.ndarray) -> complex:
    r"""Expectation value of a local/differential operator.

    .. math::

        \langle\psi\rvert\hat A\lvert\psi\rangle
            = \int \psi^*(x)\, [\hat A\psi](x)\, dx.

    Parameters
    ----------
    operator_action : callable
        A function ``operator_action(x, psi) -> A_psi`` applying
        :math:`\hat A` to ``psi``.
    x : numpy.ndarray
        Positions.
    psi : numpy.ndarray
        Wavefunction sampled on ``x``.

    Returns
    -------
    complex
    """
    A_psi = operator_action(x, psi)
    return complex(trapz(np.conj(psi) * A_psi, x))


def position_expectation(x: np.ndarray, psi: np.ndarray) -> float:
    r"""Position expectation value :math:`\langle x\rangle = \int x\,\lvert\psi(x)\rvert^2\,dx`.

    Parameters
    ----------
    x : numpy.ndarray
    psi : numpy.ndarray

    Returns
    -------
    float
    """
    return float(np.real(trapz(x * np.abs(psi) ** 2, x)))


def position_variance(x: np.ndarray, psi: np.ndarray) -> float:
    r"""Position variance :math:`\langle x^2\rangle - \langle x\rangle^2`.

    Parameters
    ----------
    x : numpy.ndarray
    psi : numpy.ndarray

    Returns
    -------
    float
    """
    density = np.abs(psi) ** 2
    mean = trapz(x * density, x)
    mean_sq = trapz(x**2 * density, x)
    return float(np.real(mean_sq - mean**2))


def momentum_density(x: np.ndarray, psi: np.ndarray, hbar: float = 1.0):
    """FFT-based momentum-space wavefunction and probability density.

    Parameters
    ----------
    x : numpy.ndarray
        Positions (uniform grid).
    psi : numpy.ndarray
        Wavefunction sampled on ``x``.
    hbar : float, default=1.0
        Value of :math:`\\hbar` to use.

    Returns
    -------
    p : numpy.ndarray
        Momentum grid, ascending.
    psi_p : numpy.ndarray
        Normalized momentum-space wavefunction.
    """
    dx = x[1] - x[0]
    N = x.shape[0]
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    psi_k = np.fft.fft(psi) * dx / np.sqrt(2 * np.pi)
    p = hbar * k
    order = np.argsort(p)
    p_sorted, psi_k_sorted = p[order], psi_k[order]
    norm = np.sqrt(trapz(np.abs(psi_k_sorted) ** 2, p_sorted))
    return p_sorted, psi_k_sorted / norm


def momentum_expectation(x: np.ndarray, psi: np.ndarray, hbar: float = 1.0) -> float:
    r"""Momentum expectation value :math:`\langle p\rangle`, via :func:`momentum_density`.

    Parameters
    ----------
    x : numpy.ndarray
    psi : numpy.ndarray
    hbar : float, default=1.0

    Returns
    -------
    float
    """
    p, psi_p = momentum_density(x, psi, hbar)
    return float(np.real(trapz(p * np.abs(psi_p) ** 2, p)))


def momentum_variance(x: np.ndarray, psi: np.ndarray, hbar: float = 1.0) -> float:
    r"""Momentum variance :math:`\langle p^2\rangle - \langle p\rangle^2`, via :func:`momentum_density`.

    Parameters
    ----------
    x : numpy.ndarray
    psi : numpy.ndarray
    hbar : float, default=1.0

    Returns
    -------
    float
    """
    p, psi_p = momentum_density(x, psi, hbar)
    density = np.abs(psi_p) ** 2
    mean = trapz(p * density, p)
    mean_sq = trapz(p**2 * density, p)
    return float(np.real(mean_sq - mean**2))


def uncertainty(x: np.ndarray, psi: np.ndarray, hbar: float = 1.0) -> tuple[float, float, float]:
    r"""Position/momentum uncertainties and their product.

    Parameters
    ----------
    x : numpy.ndarray
    psi : numpy.ndarray
    hbar : float, default=1.0

    Returns
    -------
    dx, dp, product : float
        :math:`\Delta x`, :math:`\Delta p`, and :math:`\Delta x\,\Delta p`,
        for checking the Heisenberg bound :math:`\Delta x\,\Delta p \ge \hbar/2`.
    """
    dx = np.sqrt(position_variance(x, psi))
    dp = np.sqrt(momentum_variance(x, psi, hbar))
    return dx, dp, dx * dp


@dataclass
class ExpectationMonitor:
    """Tracks expectation values along a time-evolution trajectory.

    Records :math:`\\langle x\\rangle(t)`, :math:`\\langle p\\rangle(t)`,
    :math:`\\Delta x(t)`, :math:`\\Delta p(t)`, and the norm, e.g. from the
    frames returned by a
    :meth:`~physicskit.quantum.core.solvers.SplitOperatorSolver1D.propagate` call.

    Parameters
    ----------
    x : numpy.ndarray
        Positions the wavefunction is sampled on.
    hbar : float, default=1.0
        Value of :math:`\\hbar` to use.
    """

    x: np.ndarray
    hbar: float = 1.0
    history: dict = field(default_factory=lambda: {"t": [], "norm": [], "x": [], "p": [], "dx": [], "dp": []})

    def record(self, t: float, psi: np.ndarray):
        """Record one snapshot.

        Parameters
        ----------
        t : float
            Time of the snapshot.
        psi : numpy.ndarray
            Wavefunction at time ``t``, sampled on ``self.x``.
        """
        norm = trapz(np.abs(psi) ** 2, self.x)
        dx, dp, _ = uncertainty(self.x, psi, self.hbar)
        self.history["t"].append(t)
        self.history["norm"].append(float(norm))
        self.history["x"].append(position_expectation(self.x, psi))
        self.history["p"].append(momentum_expectation(self.x, psi, self.hbar))
        self.history["dx"].append(dx)
        self.history["dp"].append(dp)

    def record_all(self, times: np.ndarray, frames: np.ndarray):
        """Record a full trajectory.

        Parameters
        ----------
        times : numpy.ndarray
            Times of each snapshot.
        frames : numpy.ndarray
            Wavefunction snapshots, shape ``(len(times), len(self.x))``.
        """
        for t, psi in zip(times, frames):
            self.record(t, psi)

    def as_arrays(self) -> dict:
        """Return the recorded history as arrays.

        Returns
        -------
        dict
            Maps ``'t'``, ``'norm'``, ``'x'``, ``'p'``, ``'dx'``, ``'dp'``
            to :class:`numpy.ndarray`.
        """
        return {k: np.array(v) for k, v in self.history.items()}


def simulate_position_measurement(x: np.ndarray, psi: np.ndarray, n_samples: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
    r"""Sample simulated projective position measurements.

    Draws :math:`x_i \sim \lvert\psi(x)\rvert^2` (the Born rule), via
    inverse-CDF sampling on the discretized density.

    Parameters
    ----------
    x : numpy.ndarray
        Positions.
    psi : numpy.ndarray
        Wavefunction sampled on ``x``.
    n_samples : int, default=1
        Number of simulated measurement outcomes to draw.
    rng : numpy.random.Generator or None, optional
        Random number generator; a fresh default one is used if omitted.

    Returns
    -------
    numpy.ndarray
        Simulated measurement outcomes, shape ``(n_samples,)``.
    """
    rng = rng or np.random.default_rng()
    density = np.abs(psi) ** 2
    density = density / trapz(density, x)
    cdf = np.concatenate([[0.0], np.cumsum((density[1:] + density[:-1]) / 2 * np.diff(x))])
    cdf /= cdf[-1]
    u = rng.random(n_samples)
    return np.interp(u, cdf, x)
