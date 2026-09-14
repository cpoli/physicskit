"""Lagrangian-mechanics systems, derived symbolically via
:class:`physicskit.classical.utils.symbolic.LagrangianEngine`.

Each system here writes down L(q, qdot) with SymPy, hands it to the
engine to auto-derive the Euler-Lagrange accelerations and the
Legendre-transformed Hamiltonian, and exposes the resulting JIT
equations of motion through :class:`physicskit.classical.core.base_system.LagrangianSystem`.
Because the Hamiltonian form is derived even when the mass matrix M(q)
is configuration-dependent (i.e. the system is non-separable), the
default integration method is ``implicit_midpoint`` -- the general
symplectic integrator from :mod:`physicskit.classical.core.integrators`.

Deriving equations of motion (SymPy differentiation, matrix inversion,
simplification, then Numba compilation) costs on the order of a second
even for a 2-DOF system -- so each system's engine-building step is
wrapped in ``functools.lru_cache``, keyed on the physical parameters
(masses, lengths, ...). Constructing many instances of the *same*
physical system (e.g. an ensemble of initial conditions to show
sensitivity to initial conditions, as in the chaos example) reuses the
cached engine instead of re-deriving and re-compiling from scratch.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import sympy as sp

from physicskit.classical.core.base_system import LagrangianSystem
from physicskit.classical.utils.symbolic import LagrangianEngine

__all__ = ["DoublePendulum", "BeadOnRotatingHoop", "CoupledOscillators", "ElasticPendulum"]


@lru_cache(maxsize=32)
def _double_pendulum_engine(m1: float, m2: float, l1: float, l2: float, g: float) -> LagrangianEngine:
    th1, th2, w1, w2 = sp.symbols("th1 th2 w1 w2")
    m1s, m2s, l1s, l2s, gs = sp.symbols("m1 m2 l1 l2 g")

    x1 = l1s * sp.sin(th1)
    y1 = -l1s * sp.cos(th1)
    x2 = x1 + l2s * sp.sin(th2)
    y2 = y1 - l2s * sp.cos(th2)
    x1d = sp.diff(x1, th1) * w1
    y1d = sp.diff(y1, th1) * w1
    x2d = sp.diff(x2, th1) * w1 + sp.diff(x2, th2) * w2
    y2d = sp.diff(y2, th1) * w1 + sp.diff(y2, th2) * w2

    T = sp.Rational(1, 2) * m1s * (x1d**2 + y1d**2) + sp.Rational(1, 2) * m2s * (x2d**2 + y2d**2)
    V = m1s * gs * y1 + m2s * gs * y2
    L = sp.simplify(T - V)

    params = {m1s: m1, m2s: m2, l1s: l1, l2s: l2, gs: g}
    return LagrangianEngine([th1, th2], [w1, w2], L, params=params)


class DoublePendulum(LagrangianSystem):
    """Planar double pendulum: two point masses on massless rods.

    Generalized coordinates: q = (theta1, theta2), angles from the
    downward vertical. A classic showcase of deterministic chaos and of
    the non-separable-Hamiltonian case for symplectic integration (the
    mass matrix depends on ``theta2 - theta1``).

    Parameters
    ----------
    theta0, thetadot0 : array-like, shape (2,)
        Initial angles and angular velocities.
    m1, m2 : float
        Bob masses.
    l1, l2 : float
        Rod lengths.
    g : float
        Gravitational acceleration.
    """

    def __init__(self, theta0, thetadot0, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81):
        self.m1, self.m2, self.l1, self.l2, self.g = m1, m2, l1, l2, g
        self.engine = _double_pendulum_engine(m1, m2, l1, l2, g)
        self._accel_njit = self.engine.acceleration_njit
        self._momentum_njit = self.engine.momentum_njit
        self._canonical_deriv_njit = self.engine.canonical_deriv_njit
        self._velocity_njit = self.engine.velocity_njit
        self._deriv_njit = self.engine.full_deriv_njit
        self._hamiltonian_njit = self.engine.hamiltonian_njit

        super().__init__(theta0, thetadot0)

    def positions(self, q: np.ndarray = None):
        """Cartesian (x1, y1, x2, y2) bob positions for a given (theta1, theta2).

        Parameters
        ----------
        q : ndarray, optional
            ``(theta1, theta2)``; defaults to the current state.

        Returns
        -------
        ndarray, shape (4,)
            ``(x1, y1, x2, y2)``.
        """
        q = self.q if q is None else q
        th1, th2 = q
        x1 = self.l1 * np.sin(th1)
        y1 = -self.l1 * np.cos(th1)
        x2 = x1 + self.l2 * np.sin(th2)
        y2 = y1 - self.l2 * np.cos(th2)
        return np.array([x1, y1, x2, y2])

    def energy(self, state: np.ndarray = None) -> float:
        state = self.state if state is None else state
        q, qdot = self.split(state, self.ndof)
        p = self._momentum_njit(q, qdot)
        return float(self._hamiltonian_njit(q, p, 0.0))


@lru_cache(maxsize=32)
def _bead_on_hoop_engine(R: float, omega: float, g: float) -> LagrangianEngine:
    th, w = sp.symbols("th w")
    Rs, Om, gs = sp.symbols("R Omega g")

    L = sp.Rational(1, 2) * Rs**2 * (w**2 + Om**2 * sp.sin(th) ** 2) - gs * Rs * (1 - sp.cos(th))
    params = {Rs: R, Om: omega, gs: g}
    return LagrangianEngine([th], [w], L, params=params)


class BeadOnRotatingHoop(LagrangianSystem):
    """A bead sliding without friction on a hoop of radius R rotating at
    fixed angular speed Omega about the vertical diameter.

    Generalized coordinate: q = (theta,), the bead's polar angle on the
    hoop. In the rotating frame the effective Lagrangian is autonomous
    (no explicit time dependence), so the system is a genuine 1-DOF
    conservative Hamiltonian; above the critical speed
    ``Omega_c = sqrt(g / R)`` the theta=0 equilibrium becomes unstable
    and two symmetric stable equilibria appear (a pitchfork bifurcation).

    Parameters
    ----------
    theta0, thetadot0 : float
        Initial angle and angular velocity.
    R : float
        Hoop radius.
    omega : float
        Fixed rotation rate ``Omega`` about the vertical diameter.
    g : float
        Gravitational acceleration.
    """

    def __init__(self, theta0, thetadot0, R=1.0, omega=2.0, g=9.81):
        self.R, self.omega, self.g = R, omega, g
        self.engine = _bead_on_hoop_engine(R, omega, g)
        self._accel_njit = self.engine.acceleration_njit
        self._momentum_njit = self.engine.momentum_njit
        self._canonical_deriv_njit = self.engine.canonical_deriv_njit
        self._velocity_njit = self.engine.velocity_njit
        self._deriv_njit = self.engine.full_deriv_njit
        self._hamiltonian_njit = self.engine.hamiltonian_njit

        super().__init__(np.atleast_1d(theta0), np.atleast_1d(thetadot0))

    def energy(self, state: np.ndarray = None) -> float:
        state = self.state if state is None else state
        q, qdot = self.split(state, self.ndof)
        p = self._momentum_njit(q, qdot)
        return float(self._hamiltonian_njit(q, p, 0.0))


@lru_cache(maxsize=32)
def _elastic_pendulum_engine(m: float, k: float, L0: float, g: float) -> LagrangianEngine:
    s, th, sd, wd = sp.symbols("s th sd wd")
    ms, ks, L0s, gs = sp.symbols("m k L0 g")

    r = L0s + s
    T = sp.Rational(1, 2) * ms * (sd**2 + r**2 * wd**2)
    V = sp.Rational(1, 2) * ks * s**2 - ms * gs * r * sp.cos(th)
    L = sp.simplify(T - V)

    params = {ms: m, ks: k, L0s: L0, gs: g}
    return LagrangianEngine([s, th], [sd, wd], L, params=params)


class ElasticPendulum(LagrangianSystem):
    """Spring pendulum: a mass on a Hookean spring that both stretches and
    swings, in 2 DOF.

    Generalized coordinates q = (s, theta): `s` is the spring's stretch
    beyond its natural length `L0` (so the pivot-to-mass distance is
    ``r = L0 + s``), `theta` is the swing angle from the downward
    vertical. Lagrangian:

        L = (m/2)(sdot^2 + (L0+s)^2 thetadot^2)
            - (k/2) s^2 + m g (L0+s) cos(theta)

    The ``r^2 thetadot^2`` term is what couples the (otherwise linear)
    stretch and swing motions: whenever the mass swings, its varying
    distance from the pivot pumps the spring, and vice versa. The coupling
    is strongest, and famously resonant, in *1:2 autoparametric resonance*
    -- when the natural stretch frequency ``omega_s = sqrt(k/m)`` is twice
    the natural swing frequency ``omega_theta = sqrt(g/L0)`` -- where
    energy started as pure vertical stretching oscillation periodically
    leaks into, and back out of, swinging. The default parameters below
    satisfy that condition exactly (``k = 4*m*g/L0``).

    Parameters
    ----------
    q0, qdot0 : array-like, shape (2,)
        Initial ``(s, theta)`` and ``(sdot, thetadot)``.
    m : float, default 1.0
        Mass.
    k : float, default 39.24
        Spring constant; the default is ``4*m*g/L0`` for the default
        `m`, `g`, `L0`, i.e. the 1:2 autoparametric resonance condition.
    L0 : float, default 1.0
        Spring natural length.
    g : float, default 9.81
        Gravitational acceleration.
    """

    def __init__(self, q0, qdot0, m=1.0, k=39.24, L0=1.0, g=9.81):
        self.m, self.k, self.L0, self.g = m, k, L0, g
        self.engine = _elastic_pendulum_engine(m, k, L0, g)
        self._accel_njit = self.engine.acceleration_njit
        self._momentum_njit = self.engine.momentum_njit
        self._canonical_deriv_njit = self.engine.canonical_deriv_njit
        self._velocity_njit = self.engine.velocity_njit
        self._deriv_njit = self.engine.full_deriv_njit
        self._hamiltonian_njit = self.engine.hamiltonian_njit

        super().__init__(q0, qdot0)

    def positions(self, q: np.ndarray = None):
        """Cartesian (x, y) mass position for a given (s, theta).

        Parameters
        ----------
        q : ndarray, optional
            ``(s, theta)``; defaults to the current state.

        Returns
        -------
        ndarray, shape (2,)
            ``(x, y)``.
        """
        q = self.q if q is None else q
        s, th = q
        r = self.L0 + s
        return np.array([r * np.sin(th), -r * np.cos(th)])

    def energy(self, state: np.ndarray = None) -> float:
        state = self.state if state is None else state
        q, qdot = self.split(state, self.ndof)
        p = self._momentum_njit(q, qdot)
        return float(self._hamiltonian_njit(q, p, 0.0))


@lru_cache(maxsize=32)
def _coupled_oscillators_engine(n: int, m: float, k: float) -> LagrangianEngine:
    qs = sp.symbols(f"q0:{n}")
    ws = sp.symbols(f"w0:{n}")
    ms, ks = sp.symbols("m k")

    T = sum(sp.Rational(1, 2) * ms * w**2 for w in ws)
    chain = (0,) + qs + (0,)
    V = sum(sp.Rational(1, 2) * ks * (chain[i + 1] - chain[i]) ** 2 for i in range(n + 1))
    L = sp.simplify(T - V)

    params = {ms: m, ks: k}
    return LagrangianEngine(list(qs), list(ws), L, params=params)


class CoupledOscillators(LagrangianSystem):
    """A chain of N masses connected by linear springs, both ends fixed
    to immovable walls -- the small-N, harmonic prototype of the larger
    lattice chains in :mod:`physicskit.classical.systems.chains`.

    Generalized coordinates: q_i = displacement of mass i from
    equilibrium, i = 1..N.

    Parameters
    ----------
    q0, qdot0 : array-like, shape (n,)
        Initial displacements and velocities.
    n : int
        Number of masses.
    m, k : float
        Mass and spring constant (uniform across the chain).
    """

    def __init__(self, q0, qdot0, n=3, m=1.0, k=1.0):
        self.n_masses = n
        self.m, self.k = m, k
        self.engine = _coupled_oscillators_engine(n, m, k)
        self._accel_njit = self.engine.acceleration_njit
        self._momentum_njit = self.engine.momentum_njit
        self._canonical_deriv_njit = self.engine.canonical_deriv_njit
        self._velocity_njit = self.engine.velocity_njit
        self._deriv_njit = self.engine.full_deriv_njit
        self._hamiltonian_njit = self.engine.hamiltonian_njit

        super().__init__(q0, qdot0)

    def energy(self, state: np.ndarray = None) -> float:
        state = self.state if state is None else state
        q, qdot = self.split(state, self.ndof)
        p = self._momentum_njit(q, qdot)
        return float(self._hamiltonian_njit(q, p, 0.0))
