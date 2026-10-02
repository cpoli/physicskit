"""Path-integral Monte Carlo for a quantum particle in a one-dimensional well.

Feynman's path integral maps the quantum partition function of one
particle onto the classical partition function of a closed *ring
polymer*: discretize imaginary time :math:`\\beta\\hbar` into :math:`P`
slices of width :math:`\\tau = \\beta/P` and, in the primitive
approximation (units :math:`\\hbar = m = k_B = 1`),

.. math::

    Z \\approx \\left(\\frac{P}{2\\pi\\beta}\\right)^{P/2} \\int dx_1 \\cdots dx_P\\,
    \\exp\\left[-\\sum_{j=1}^{P} \\left(\\frac{P}{2\\beta}(x_{j+1} - x_j)^2
    + \\frac{\\beta}{P} V(x_j)\\right)\\right],
    \\qquad x_{P+1} = x_1,

which becomes exact as :math:`P \\to \\infty` (Barker, J. Chem. Phys. 70,
2914 (1979); Chandler and Wolynes, J. Chem. Phys. 74, 4078 (1981)).
:class:`PathIntegralParticle` samples this ring polymer by Metropolis bead
moves for the anharmonic well :math:`V(x) = \\tfrac12\\omega^2 x^2 +
\\lambda x^4`.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from physicskit.statphys.core.monte_carlo import seed_numba_random

__all__ = ["PathIntegralParticle", "harmonic_x2_exact", "harmonic_x2_primitive"]


@njit(cache=True)
def _potential(x, omega, lam):
    return 0.5 * omega * omega * x * x + lam * x**4


@njit(cache=True)
def _pimc_sweeps(path, beta, omega, lam, delta, shift_delta, n_sweeps, x2_out, energy_out):
    P = path.shape[0]
    tau = beta / P
    spring = 1.0 / (2.0 * tau)
    accepted = 0
    for sweep in range(n_sweeps):
        for _ in range(P):
            j = np.random.randint(0, P)
            x_old = path[j]
            x_new = x_old + delta * (2.0 * np.random.random() - 1.0)
            left = path[(j - 1) % P]
            right = path[(j + 1) % P]
            dS = spring * ((x_new - left) ** 2 + (right - x_new) ** 2 - (x_old - left) ** 2 - (right - x_old) ** 2)
            dS += tau * (_potential(x_new, omega, lam) - _potential(x_old, omega, lam))
            if dS <= 0.0 or np.random.random() < np.exp(-dS):
                path[j] = x_new
                accepted += 1
        # a rigid shift of the whole polymer moves the centroid, which bead moves do slowly
        shift = shift_delta * (2.0 * np.random.random() - 1.0)
        dS = 0.0
        for j in range(P):
            dS += _potential(path[j] + shift, omega, lam) - _potential(path[j], omega, lam)
        if np.random.random() < np.exp(-tau * dS):
            for j in range(P):
                path[j] += shift
        x2 = 0.0
        e = 0.0
        for j in range(P):
            x = path[j]
            x2 += x * x
            # virial estimator: E = <V + x V'/2>
            e += _potential(x, omega, lam) + 0.5 * x * (omega * omega * x + 4.0 * lam * x**3)
        x2_out[sweep] = x2 / P
        energy_out[sweep] = e / P
    return accepted / (n_sweeps * P)


def harmonic_x2_exact(beta, omega=1.0):
    """Exact thermal :math:`\\langle x^2 \\rangle` of a quantum harmonic oscillator.

    .. math::

        \\langle x^2 \\rangle = \\frac{1}{2\\omega} \\coth\\frac{\\beta\\omega}{2}
        \\quad (\\hbar = m = 1).

    Parameters
    ----------
    beta : float
        Inverse temperature.
    omega : float, default=1.0
        Oscillator frequency.

    Returns
    -------
    float

    Examples
    --------
    >>> round(float(harmonic_x2_exact(100.0)), 6)  # ground state: 1/(2 omega)
    0.5
    """
    return 1.0 / (2.0 * omega * np.tanh(beta * omega / 2.0))


def harmonic_x2_primitive(beta, n_beads, omega=1.0):
    """Exact :math:`\\langle x^2 \\rangle` of the :math:`P`-bead primitive-action ring polymer.

    For a harmonic well the discretized action is a Gaussian
    :math:`\\tfrac12 x^T A x` with :math:`A = \\tfrac{P}{\\beta}(2I - S -
    S^T) + \\tfrac{\\beta\\omega^2}{P} I` (:math:`S` the cyclic shift), so
    :math:`\\langle x^2 \\rangle_P = \\operatorname{tr}(A^{-1})/P`. This is
    the exact target of a :class:`PathIntegralParticle` run with
    ``lam=0``, and tends to :func:`harmonic_x2_exact` as :math:`P \\to
    \\infty`; the gap is the :math:`O(\\tau^2)` Trotter error.

    Parameters
    ----------
    beta : float
        Inverse temperature.
    n_beads : int
        Number of imaginary-time slices :math:`P`.
    omega : float, default=1.0
        Oscillator frequency.

    Returns
    -------
    float

    Examples
    --------
    >>> exact = harmonic_x2_exact(5.0)
    >>> bool(abs(harmonic_x2_primitive(5.0, 200) - exact) < 1e-3 * exact)
    True
    """
    P = n_beads
    # eigenvalues of the circulant A: P/beta * (2 - 2 cos(2 pi k / P)) + beta omega^2 / P
    k = np.arange(P)
    eig = P / beta * (2.0 - 2.0 * np.cos(2.0 * np.pi * k / P)) + beta * omega**2 / P
    return float(np.sum(1.0 / eig) / P)


class PathIntegralParticle:
    """Ring-polymer path-integral Monte Carlo for :math:`V(x) = \\tfrac12\\omega^2x^2 + \\lambda x^4`.

    Parameters
    ----------
    beta : float
        Inverse temperature (:math:`\\hbar = m = k_B = 1`).
    n_beads : int, default=32
        Number of imaginary-time slices :math:`P`.
    omega : float, default=1.0
        Harmonic frequency.
    lam : float, default=0.0
        Quartic anharmonicity :math:`\\lambda \\ge 0`.
    seed : int, optional
        Seed for reproducible sampling.

    Attributes
    ----------
    path : ndarray of shape (n_beads,)
        Current bead positions.

    Examples
    --------
    >>> pimc = PathIntegralParticle(beta=2.0, n_beads=16, seed=0)
    >>> out = pimc.run(n_equil=500, n_measure=4000)
    >>> abs(out["x2"] - harmonic_x2_primitive(2.0, 16)) < 0.05
    True
    """

    def __init__(self, beta, n_beads=32, omega=1.0, lam=0.0, seed=None):
        if lam < 0:
            raise ValueError("lam must be non-negative for a bound well")
        self.beta = float(beta)
        self.n_beads = int(n_beads)
        self.omega = float(omega)
        self.lam = float(lam)
        if seed is not None:
            seed_numba_random(seed)
        self.path = np.zeros(self.n_beads)
        # a bead step comparable to the free-particle spread over one slice
        self.delta = 2.0 * np.sqrt(self.beta / self.n_beads)
        # a rigid-shift step comparable to the classical thermal spread 1/sqrt(beta omega^2)
        self.shift_delta = 2.0 / (self.omega * np.sqrt(self.beta))

    def run(self, n_equil=1000, n_measure=10000):
        """Equilibrate, then sample the ring polymer.

        Parameters
        ----------
        n_equil : int, default=1000
            Discarded sweeps.
        n_measure : int, default=10000
            Measured sweeps (one sample per sweep of :math:`P` bead moves).

        Returns
        -------
        dict
            ``"x2"`` and ``"energy"`` (means of :math:`\\langle x^2\\rangle`
            and of the virial energy estimator), their per-sweep series
            ``"x2_series"`` and ``"energy_series"``, and the bead-move
            ``"acceptance"`` rate.
        """
        scratch = np.empty(n_equil)
        _pimc_sweeps(self.path, self.beta, self.omega, self.lam, self.delta, self.shift_delta, n_equil, scratch, scratch.copy())
        x2 = np.empty(n_measure)
        energy = np.empty(n_measure)
        acc = _pimc_sweeps(self.path, self.beta, self.omega, self.lam, self.delta, self.shift_delta, n_measure, x2, energy)
        return {
            "x2": float(x2.mean()),
            "energy": float(energy.mean()),
            "x2_series": x2,
            "energy_series": energy,
            "acceptance": acc,
        }
