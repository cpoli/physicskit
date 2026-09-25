"""Coupled-oscillator and soliton lattice chains.

All three chains are 1D lattices of N particles with fixed
(wall-anchored) boundary conditions, i.e. q_0 = q_{N+1} = 0. Each is a
separable Hamiltonian system H = sum_i p_i^2/(2m) + V(q), with the
force computed by a hand-written ``@njit`` loop over lattice sites (the
"lattice loops" the project brief calls out for mandatory Numba
acceleration) rather than through the symbolic engine, since the
physics is fixed and the loop structure is identical for every N.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numba import njit

from physicskit.classical.core.base_system import HamiltonianSystem

__all__ = ["HarmonicChain", "FPUTChain", "SineGordonChain"]


# ---------------------------------------------------------------------------
# Harmonic chain: V = sum_i k/2 (q_{i+1} - q_i)^2, fixed ends
# ---------------------------------------------------------------------------


@njit(cache=True)
def _harmonic_force(q, k):
    n = q.shape[0]
    out = np.empty(n)
    for i in range(n):
        left = q[i - 1] if i > 0 else 0.0
        right = q[i + 1] if i < n - 1 else 0.0
        out[i] = k * (left - 2.0 * q[i] + right)
    return out


@lru_cache(maxsize=64)
def _make_harmonic_force(k: float):
    @njit(cache=True)
    def force(q, t):
        return _harmonic_force(q, k)

    return force


class _ChainBase(HamiltonianSystem):
    separable = True

    def __init__(self, q0, p0, m=1.0, k=1.0):
        self.m, self.k = m, k
        self.mass_inv = 1.0 / m
        super().__init__(q0, p0)

    def kinetic_energy(self, p: np.ndarray) -> float:
        return 0.5 * float(np.sum(p * p)) / self.m

    def normal_mode_frequencies(self) -> np.ndarray:
        """Angular frequencies omega_k of the N linear normal modes of a fixed-fixed harmonic chain.

        Returns
        -------
        ndarray, shape (n,)
            omega_k for k = 1..N.
        """
        n = self.ndof
        modes = np.arange(1, n + 1)
        return 2.0 * np.sqrt(self.k / self.m) * np.sin(modes * np.pi / (2.0 * (n + 1)))

    def normal_mode_coords(self, q: np.ndarray = None, p: np.ndarray = None):
        """Transform (q, p) to normal-mode coordinates (Q_k, P_k).

        Via the discrete sine transform appropriate for fixed-fixed
        boundaries: ``Q_k = sqrt(2/(N+1)) * sum_i q_i sin(i*k*pi/(N+1))``,
        ``i, k = 1..N``.

        Parameters
        ----------
        q, p : ndarray, optional
            State to transform; defaults to the current state.

        Returns
        -------
        Q, P : ndarray, shape (n,)
        """
        q = self.q if q is None else q
        p = self.p if p is None else p
        n = self.ndof
        i = np.arange(1, n + 1)
        k = np.arange(1, n + 1)
        S = np.sqrt(2.0 / (n + 1)) * np.sin(np.outer(i, k) * np.pi / (n + 1))
        Q = S.T @ q
        P = S.T @ p
        return Q, P

    def modal_energies(self, q: np.ndarray = None, p: np.ndarray = None) -> np.ndarray:
        """Energy stored in each normal mode.

        ``E_k = 0.5*(P_k^2/m + m*omega_k^2*Q_k^2)`` -- the diagnostic
        used to show non-ergodic recurrence (the FPUT paradox) in
        :class:`FPUTChain`.

        Parameters
        ----------
        q, p : ndarray, optional
            State to evaluate; defaults to the current state.

        Returns
        -------
        ndarray, shape (n,)
            E_k for k = 1..N.
        """
        Q, P = self.normal_mode_coords(q, p)
        omega = self.normal_mode_frequencies()
        return 0.5 * (P**2) / self.m + 0.5 * self.m * (omega**2) * (Q**2)


class HarmonicChain(_ChainBase):
    """Linear chain of N masses coupled by identical springs k, both
    ends fixed to walls -- the exactly-solvable reference lattice whose
    normal modes are the sine waves used to analyze the non-linear
    :class:`FPUTChain` below.

    Parameters
    ----------
    q0, p0 : array-like, shape (n,)
        Initial displacements and momenta.
    m, k : float
        Mass and spring constant (uniform across the chain).
    """

    def __init__(self, q0, p0, m=1.0, k=1.0):
        super().__init__(q0, p0, m=m, k=k)
        self._force_njit = _make_harmonic_force(k)

    def potential_energy(self, q: np.ndarray) -> float:
        n = q.shape[0]
        ext = np.empty(n + 2)
        ext[0] = 0.0
        ext[-1] = 0.0
        ext[1:-1] = q
        return 0.5 * self.k * float(np.sum(np.diff(ext) ** 2))


# ---------------------------------------------------------------------------
# FPUT beta-lattice: V(r) = k/2 r^2 + beta/4 r^4
# ---------------------------------------------------------------------------


@njit(cache=True)
def _fput_force(q, k, beta):
    n = q.shape[0]
    out = np.empty(n)
    for i in range(n):
        left = q[i - 1] if i > 0 else 0.0
        right = q[i + 1] if i < n - 1 else 0.0
        r_right = right - q[i]
        r_left = q[i] - left
        f_right = k * r_right + beta * r_right**3
        f_left = k * r_left + beta * r_left**3
        out[i] = f_right - f_left
    return out


@lru_cache(maxsize=64)
def _make_fput_force(k: float, beta: float):
    @njit(cache=True)
    def force(q, t):
        return _fput_force(q, k, beta)

    return force


class FPUTChain(_ChainBase):
    """Fermi-Pasta-Ulam-Tsingou beta-lattice: N masses (default 32) with
    the quartic non-linear coupling potential
    V(r) = (k/2) r^2 + (beta/4) r^4, r = q_{i+1} - q_i.

    Famous for its "paradox": exciting a single low-order normal mode
    does *not* thermalize energy across all modes as naive ergodic
    reasoning predicts; instead the energy in :meth:`modal_energies`
    returns almost exactly to the initially excited mode after a
    characteristic recurrence time (the FPUT recurrence).

    Parameters
    ----------
    n : int
        Number of masses.
    m, k : float
        Mass and (linear) spring constant.
    beta : float
        Quartic non-linearity coefficient.
    mode : int
        Which normal mode to excite initially.
    amplitude : float
        Initial amplitude of that mode.
    """

    def __init__(self, n=32, m=1.0, k=1.0, beta=0.0, mode=1, amplitude=1.0):
        self.beta = beta
        q0 = amplitude * np.sin(np.arange(1, n + 1) * mode * np.pi / (n + 1))
        p0 = np.zeros(n)
        super().__init__(q0, p0, m=m, k=k)
        self._force_njit = _make_fput_force(k, beta)

    def potential_energy(self, q: np.ndarray) -> float:
        n = q.shape[0]
        ext = np.empty(n + 2)
        ext[0] = 0.0
        ext[-1] = 0.0
        ext[1:-1] = q
        r = np.diff(ext)
        return float(np.sum(0.5 * self.k * r**2 + 0.25 * self.beta * r**4))


# ---------------------------------------------------------------------------
# Discrete Sine-Gordon chain (Frenkel-Kontorova): kink/antikink solitons
# ---------------------------------------------------------------------------


@njit(cache=True)
def _sine_gordon_force(q, k):
    n = q.shape[0]
    out = np.empty(n)
    for i in range(n):
        left = q[i - 1] if i > 0 else q[i]
        right = q[i + 1] if i < n - 1 else q[i]
        out[i] = k * (left - 2.0 * q[i] + right) - np.sin(q[i])
    return out


@lru_cache(maxsize=64)
def _make_sine_gordon_force(k: float):
    @njit(cache=True)
    def force(q, t):
        return _sine_gordon_force(q, k)

    return force


class SineGordonChain(_ChainBase):
    """Discretized pendulum chain (Frenkel-Kontorova model): N coupled
    pendulums, V = sum_i [k/2 (q_{i+1}-q_i)^2 + (1 - cos q_i)], with
    free (Neumann-like) boundary conditions so a topological kink can
    propagate off either end without an artificial restoring wall.

    Continuum-limit kink initial data from :meth:`kink`
    (q_i = 4*atan(exp((i - i0)/width)), width=1 for the default m=1,
    k=1) launches a soliton that propagates and scatters off other
    kinks/antikinks -- the discrete analogue of the exact sine-Gordon
    soliton solutions.

    Parameters
    ----------
    n : int
        Number of pendulums.
    m, k : float
        Mass and coupling constant.
    q0, p0 : array-like, shape (n,), optional
        Initial displacements and momenta; defaults to all zeros. Use
        :meth:`kink` to build soliton initial data instead.
    """

    def __init__(self, n=200, m=1.0, k=1.0, q0=None, p0=None):
        if q0 is None:
            q0 = np.zeros(n)
        if p0 is None:
            p0 = np.zeros(n)
        super().__init__(q0, p0, m=m, k=k)
        self._force_njit = _make_sine_gordon_force(k)

    def potential_energy(self, q: np.ndarray) -> float:
        r = np.diff(q)  # free boundaries: no coupling to phantom exterior sites
        return float(0.5 * self.k * np.sum(r**2) + np.sum(1.0 - np.cos(q)))

    @staticmethod
    def kink(n: int, center: float, width: float = 1.0, velocity: float = 0.0, polarity: int = 1) -> tuple:
        """Continuum-limit single-kink (or antikink, polarity=-1) initial
        condition, optionally boosted to move at ``velocity`` (sites per
        unit time; to the right for positive ``velocity``, assuming the
        default ``m=1, k=1`` -- i.e. a natural wave/"light" speed
        ``c = sqrt(k/m) = 1``, the sine-Gordon equation's exact
        Lorentz-invariance scale, so ``|velocity| < 1`` is required).

        ``width`` is *not* a free shape knob: q_xx = sin(q) (the
        continuum equation this chain approximates, with the default
        m=1, k=1) is solved by q=4*atan(exp(x)) only for exactly unit
        width. Passing a different ``width`` here without also scaling
        ``k`` to match (the static kink of ``m q_tt = k q_xx - sin(q)`` has
        ``width = sqrt(k)``; ``m`` only sets the wave speed
        ``c = sqrt(k/m)``) gives a profile
        that visibly relaxes/radiates under the true dynamics instead
        of propagating as a clean, stable soliton.

        The exact traveling-wave solution of the continuum sine-Gordon
        equation is the *Lorentz-contracted* profile
        ``q(x, t) = polarity * 4*atan(exp(gamma * (x - center - v*t) / width))``,
        ``gamma = 1 / sqrt(1 - v**2)`` -- both the spatial profile at t=0
        and its time derivative must be boosted together (not just the
        momentum on top of the unboosted static profile, which is only
        a small-v approximation and becomes badly wrong as ``|v|`` approaches 1).

        Parameters
        ----------
        n : int
            Chain length.
        center : float
            Lattice site the kink is centered on at t=0.
        width : float
            Soliton width; must equal ``sqrt(k)`` (1, for the default
            k=1) for a genuine soliton solution.
        velocity : float
            Boost velocity, ``|velocity| < 1``.
        polarity : {1, -1}
            +1 for a kink, -1 for an antikink.

        Returns
        -------
        q0, p0 : ndarray, shape (n,)
            Initial displacements and momenta. ``p0`` is ``dq/dt`` at t=0,
            i.e. the momentum for the default ``m=1``; multiply by ``m``
            for another mass.

        Raises
        ------
        ValueError
            If ``abs(velocity) >= 1``.
        """
        if abs(velocity) >= 1.0:
            raise ValueError("|velocity| must be < 1 (the natural wave speed sqrt(k/m), default 1) for a valid moving kink")
        gamma = 1.0 / np.sqrt(1.0 - velocity**2)
        i = np.arange(n)
        u = gamma * (i - center) / width
        q0 = polarity * 4.0 * np.arctan(np.exp(u))
        p0 = -polarity * velocity * (2.0 * gamma / width) / np.cosh(u)
        return q0, p0
