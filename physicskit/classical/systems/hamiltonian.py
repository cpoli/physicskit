"""Hamiltonian phase-space systems illustrating Liouville mechanics.

:class:`HenonHeilesSystem` is the classic 2-DOF non-linear oscillator
whose Poincare sections show the breakdown of KAM tori as energy
increases. :class:`PendulumSwarm` evolves an ensemble of independent
phase points under identical 1-DOF pendulum dynamics to visually
demonstrate Liouville's theorem (phase-space volume/area is conserved
by Hamiltonian flow, even though the occupied region shears and
filaments). :func:`pendulum_action_angle` computes the action J(E) and
period of a simple pendulum via elliptic integrals, the archetypal
worked example of action-angle variables.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numba import njit
from scipy import special

from physicskit.classical.core.base_system import HamiltonianSystem

__all__ = ["HenonHeilesSystem", "PendulumSwarm", "pendulum_action_angle"]


@njit(cache=True)
def _henon_heiles_force(q, t):
    x, y = q[0], q[1]
    out = np.empty(2)
    out[0] = -(x + 2.0 * x * y)
    out[1] = -(y + x * x - y * y)
    return out


class HenonHeilesSystem(HamiltonianSystem):
    """H = (px^2 + py^2)/2 + (x^2 + y^2)/2 + x^2*y - y^3/3.

    A non-integrable 2-DOF oscillator (originally a model of stellar
    orbits in an axisymmetric galactic potential). Below E ~ 1/6 the
    Poincare section (x, px) at y=0 is dominated by smooth KAM tori;
    as E approaches and exceeds 1/6 the tori progressively break up
    into chaotic seas.

    Parameters
    ----------
    q0, p0 : array-like, shape (2,)
        Initial position ``(x, y)`` and momentum ``(px, py)``.
    """

    separable = True
    mass_inv = 1.0

    def __init__(self, q0, p0):
        self._force_njit = _henon_heiles_force
        super().__init__(q0, p0)

    def kinetic_energy(self, p: np.ndarray) -> float:
        return 0.5 * float(np.dot(p, p))

    def potential_energy(self, q: np.ndarray) -> float:
        x, y = q[0], q[1]
        return 0.5 * (x * x + y * y) + x * x * y - (y**3) / 3.0


@lru_cache(maxsize=64)
def _make_pendulum_force(g_over_l: float):
    @njit(cache=True)
    def force(q, t):
        return -g_over_l * np.sin(q)

    return force


class PendulumSwarm(HamiltonianSystem):
    """An ensemble of N independent, identical simple pendulums,
    H_i = p_i^2 / 2 - (g/l) cos(q_i) for each i = 1..N.

    The pendulums are mutually decoupled (each is its own 1-DOF
    Hamiltonian system) but are integrated together as one N-DOF
    separable Hamiltonian so a whole ensemble can be evolved with a
    single Yoshida4/Verlet call. Seeding N points inside a small
    (dq, dp) box and watching the occupied phase-space patch shear
    over time -- its area invariant even as its shape stretches into a
    filament -- is the standard visual proof of Liouville's theorem.

    Parameters
    ----------
    q0, p0 : array-like, shape (n,)
        Initial angles and momenta, one pair per pendulum in the ensemble.
    g_over_l : float
        Ratio g/l shared by every pendulum.
    """

    separable = True

    def __init__(self, q0, p0, g_over_l: float = 1.0):
        self.g_over_l = g_over_l
        self.mass_inv = 1.0
        self._force_njit = _make_pendulum_force(g_over_l)
        super().__init__(q0, p0)

    @classmethod
    def from_box(cls, q_center: float, p_center: float, dq: float, dp: float, n: int = 1000, g_over_l: float = 1.0, seed: int = 0):
        """Seed N phase points uniformly inside a (dq x dp) box centered at (q_center, p_center).

        Parameters
        ----------
        q_center, p_center : float
            Box center.
        dq, dp : float
            Box width and height.
        n : int
            Number of phase points to seed.
        g_over_l : float
            Ratio g/l shared by every pendulum.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        PendulumSwarm
        """
        rng = np.random.default_rng(seed)
        q0 = q_center + (rng.random(n) - 0.5) * dq
        p0 = p_center + (rng.random(n) - 0.5) * dp
        return cls(q0, p0, g_over_l=g_over_l)

    def kinetic_energy(self, p: np.ndarray) -> float:
        return 0.5 * float(np.sum(p * p))

    def potential_energy(self, q: np.ndarray) -> float:
        return -self.g_over_l * float(np.sum(np.cos(q)))

    def per_particle_energy(self, q: np.ndarray = None, p: np.ndarray = None) -> np.ndarray:
        """Energy of each individual pendulum, used to color/track the swarm.

        Conserved per-particle since the ensemble is decoupled.

        Parameters
        ----------
        q, p : ndarray, optional
            State to evaluate; defaults to the current state.

        Returns
        -------
        ndarray, shape (n,)
        """
        q = self.q if q is None else q
        p = self.p if p is None else p
        return 0.5 * p * p - self.g_over_l * np.cos(q)

    def phase_space_area(self, q: np.ndarray = None, p: np.ndarray = None) -> float:
        """Convex-hull area of the current swarm's (q, p) point cloud.

        A practical proxy for the occupied phase-space volume.

        Parameters
        ----------
        q, p : ndarray, optional
            State to evaluate; defaults to the current state.

        Returns
        -------
        float
        """
        from scipy.spatial import ConvexHull

        q = self.q if q is None else q
        p = self.p if p is None else p
        pts = np.column_stack([q, p])
        return float(ConvexHull(pts).volume)  # 'volume' is the 2D area for a 2D hull


def pendulum_action_angle(E: float, g_over_l: float = 1.0):
    """Action J(E) and period T(E) for a simple pendulum H = p^2/2 - (g/l)cos(q).

    For librating orbits (-g/l < E < g/l), uses the standard elliptic-integral
    closed form: with k^2 = (E + g/l) / (2 g/l),

        J(E) = (8/pi) * sqrt(g/l) * [E_ellip(k) - (1 - k^2) * K_ellip(k)]
        T(E) = 4 * K_ellip(k) / sqrt(g/l)

    where K, E are the complete elliptic integrals of the first/second
    kind (SciPy's ``special.ellipk``/``ellipe`` take the parameter
    m = k^2, not the modulus k).

    Parameters
    ----------
    E : float
        Energy, with ``-g_over_l < E < g_over_l`` (librating orbits only).
    g_over_l : float
        Ratio g/l.

    Returns
    -------
    J : float
        Action.
    T : float
        Orbital period.

    Raises
    ------
    ValueError
        If ``E`` is outside the librating range.
    """
    w0 = np.sqrt(g_over_l)
    if -g_over_l >= E or g_over_l <= E:
        raise ValueError("pendulum_action_angle only supports librating orbits (-g/l < E < g/l)")
    m = (E + g_over_l) / (2.0 * g_over_l)  # = k^2
    K = special.ellipk(m)
    Eei = special.ellipe(m)
    J = (8.0 / np.pi) * w0 * (Eei - (1.0 - m) * K)
    T = 4.0 * K / w0
    return float(J), float(T)
