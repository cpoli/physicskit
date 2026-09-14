"""Rigid-body rotation systems: free tops and heavy symmetric tops.

:class:`EulerTop` integrates the torque-free rigid body (Euler's
equations for the body-frame angular velocity) together with the
orientation quaternion, and demonstrates the Intermediate Axis Theorem
(the "Dzhanibekov effect" / tennis-racket effect): rotation about the
axis of intermediate moment of inertia is unstable, while rotation
about the axes of largest and smallest moment is stable.

:class:`HeavySymmetricTop` is a symmetric top with one point fixed,
spinning under gravity -- derived symbolically (like the Lagrangian
systems) in the three Euler angles (phi, theta, psi), since its
Hamiltonian is intrinsically non-separable (the kinetic term's
coefficients depend on theta).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import sympy as sp
from numba import njit
from scipy.optimize import minimize_scalar

from physicskit.classical.core.base_system import LagrangianSystem, ODESystem
from physicskit.classical.utils.symbolic import LagrangianEngine

__all__ = [
    "EulerTop",
    "HeavySymmetricTop",
    "EulersDisk",
    "Rattleback",
    "effective_potential_symmetric_top",
    "find_theta_equilibrium",
    "nutation_frequency",
    "precession_frequency",
    "eulers_disk_theta_analytic",
]


# ---------------------------------------------------------------------------
# Free asymmetric rigid body (Euler top)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=64)
def _make_euler_top_deriv(I1: float, I2: float, I3: float):
    @njit(cache=False)
    def deriv(t, y):
        w1, w2, w3 = y[0], y[1], y[2]
        qw, qx, qy, qz = y[3], y[4], y[5], y[6]

        dw1 = (I2 - I3) / I1 * w2 * w3
        dw2 = (I3 - I1) / I2 * w3 * w1
        dw3 = (I1 - I2) / I3 * w1 * w2

        # dq/dt = 0.5 * q (x) (0, w1, w2, w3)  (Hamilton product, body-frame omega)
        dqw = -0.5 * (qx * w1 + qy * w2 + qz * w3)
        dqx = 0.5 * (qw * w1 + qy * w3 - qz * w2)
        dqy = 0.5 * (qw * w2 + qz * w1 - qx * w3)
        dqz = 0.5 * (qw * w3 + qx * w2 - qy * w1)

        out = np.empty(7)
        out[0], out[1], out[2] = dw1, dw2, dw3
        out[3], out[4], out[5], out[6] = dqw, dqx, dqy, dqz
        return out

    return deriv


class EulerTop(ODESystem):
    """Torque-free rigid body: Euler's equations for omega = (w1, w2, w3)
    in the body frame, coupled to the orientation quaternion (qw, qx, qy, qz).

    State: ``[w1, w2, w3, qw, qx, qy, qz]``. The default integrator,
    ``implicit_midpoint``, exactly conserves both the rotational kinetic
    energy and the (squared) angular momentum magnitude and quaternion
    norm -- all quadratic invariants of this quadratic ODE (a classical
    result: the implicit midpoint / average-vector-field map preserves
    every quadratic invariant of any ODE, not only linear ones).

    Excite a spin dominantly about the intermediate-inertia axis (e.g.
    ``omega0 = [0.01, 1.0, 0.01]`` with ``I1 < I2 < I3``) to see the
    periodic tumbling of the Intermediate Axis Theorem.

    Parameters
    ----------
    omega0 : array-like, shape (3,)
        Initial body-frame angular velocity.
    I1, I2, I3 : float
        Principal moments of inertia.
    quat0 : array-like, shape (4,), optional
        Initial orientation quaternion ``(w, x, y, z)``; defaults to
        the identity orientation. Normalized automatically.
    """

    def __init__(self, omega0, I1=1.0, I2=2.0, I3=3.0, quat0=None):
        self.I1, self.I2, self.I3 = I1, I2, I3
        omega0 = np.asarray(omega0, dtype=np.float64)
        quat0 = np.array([1.0, 0.0, 0.0, 0.0]) if quat0 is None else np.asarray(quat0, dtype=np.float64)
        quat0 = quat0 / np.linalg.norm(quat0)
        self._deriv_njit = _make_euler_top_deriv(I1, I2, I3)
        super().__init__(np.concatenate([omega0, quat0]))

    @property
    def omega(self) -> np.ndarray:
        return self.state[:3]

    @property
    def quaternion(self) -> np.ndarray:
        return self.state[3:]

    def energy(self, state: np.ndarray = None) -> float:
        """Rotational kinetic energy.

        Parameters
        ----------
        state : ndarray, optional
            State to evaluate; defaults to the current state.

        Returns
        -------
        float
        """
        state = self.state if state is None else state
        w1, w2, w3 = state[0], state[1], state[2]
        return 0.5 * (self.I1 * w1**2 + self.I2 * w2**2 + self.I3 * w3**2)

    def angular_momentum_squared(self, state: np.ndarray = None) -> float:
        """``|L|^2`` in the body frame.

        Parameters
        ----------
        state : ndarray, optional
            State to evaluate; defaults to the current state.

        Returns
        -------
        float
        """
        state = self.state if state is None else state
        w1, w2, w3 = state[0], state[1], state[2]
        return (self.I1 * w1) ** 2 + (self.I2 * w2) ** 2 + (self.I3 * w3) ** 2

    def rotation_matrix(self, state: np.ndarray = None) -> np.ndarray:
        """3x3 body-to-world rotation matrix from the current quaternion.

        Parameters
        ----------
        state : ndarray, optional
            State to evaluate; defaults to the current state.

        Returns
        -------
        ndarray, shape (3, 3)
        """
        state = self.state if state is None else state
        qw, qx, qy, qz = state[3], state[4], state[5], state[6]
        return np.array(
            [
                [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
            ]
        )


# ---------------------------------------------------------------------------
# Heavy symmetric top (Lagrange top): fixed point under gravity
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _heavy_symmetric_top_engine(I1: float, I3: float, M: float, l: float, g: float) -> LagrangianEngine:
    phi, theta, psi, phid, thd, psid = sp.symbols("phi theta psi phid thd psid")
    I1s, I3s, Ms, ls, gs = sp.symbols("I1 I3 M l g")

    L = (
        sp.Rational(1, 2) * I1s * (thd**2 + phid**2 * sp.sin(theta) ** 2)
        + sp.Rational(1, 2) * I3s * (psid + phid * sp.cos(theta)) ** 2
        - Ms * gs * ls * sp.cos(theta)
    )
    params = {I1s: I1, I3s: I3, Ms: M, ls: l, gs: g}
    return LagrangianEngine([phi, theta, psi], [phid, thd, psid], L, params=params)


class HeavySymmetricTop(LagrangianSystem):
    """Symmetric top with one point fixed, spinning under gravity.

    Generalized coordinates q = (phi, theta, psi) -- the standard
    z-x-z Euler angles (precession, nutation, spin). Lagrangian:

        L = (I1/2)(thetadot^2 + phidot^2 sin^2(theta))
            + (I3/2)(psidot + phidot cos(theta))^2 - M*g*l*cos(theta)

    where I1 is the transverse moment of inertia about the fixed point,
    I3 the axial moment, M the mass, and l the distance from the pivot
    to the center of mass. Because M(q) depends on theta, this is a
    genuinely non-separable Hamiltonian system -- exactly the case
    :class:`physicskit.classical.utils.symbolic.LagrangianEngine` and
    ``implicit_midpoint`` were built to handle.

    Parameters
    ----------
    angles0, angledots0 : array-like, shape (3,)
        Initial ``(phi, theta, psi)`` and their time derivatives.
    I1 : float
        Transverse moment of inertia about the fixed point.
    I3 : float
        Axial moment of inertia.
    M : float
        Mass.
    l : float
        Distance from the pivot to the center of mass.
    g : float
        Gravitational acceleration.
    """

    def __init__(self, angles0, angledots0, I1=1.0, I3=0.5, M=1.0, l=1.0, g=9.81):
        self.I1, self.I3, self.M, self.l, self.g = I1, I3, M, l, g
        self.engine = _heavy_symmetric_top_engine(I1, I3, M, l, g)
        self._accel_njit = self.engine.acceleration_njit
        self._momentum_njit = self.engine.momentum_njit
        self._canonical_deriv_njit = self.engine.canonical_deriv_njit
        self._velocity_njit = self.engine.velocity_njit
        self._deriv_njit = self.engine.full_deriv_njit
        self._hamiltonian_njit = self.engine.hamiltonian_njit

        super().__init__(angles0, angledots0)

    def energy(self, state: np.ndarray = None) -> float:
        state = self.state if state is None else state
        q, qdot = self.split(state, self.ndof)
        p = self._momentum_njit(q, qdot)
        return float(self._hamiltonian_njit(q, p, 0.0))

    def precession_nutation_rate(self, state: np.ndarray = None):
        """Instantaneous (phidot, thetadot) -- precession and nutation rates.

        Parameters
        ----------
        state : ndarray, optional
            State to evaluate; defaults to the current state.

        Returns
        -------
        phidot, thetadot : float
        """
        state = self.state if state is None else state
        _, qdot = self.split(state, self.ndof)
        return float(qdot[0]), float(qdot[1])


def effective_potential_symmetric_top(theta, p_phi: float, p_psi: float, I1: float, I3: float, Mgl: float):
    """V_eff(theta) for the heavy symmetric top, given the two conserved
    (cyclic-coordinate) momenta p_phi and p_psi:

        V_eff(theta) = (p_phi - p_psi cos(theta))^2 / (2 I1 sin^2(theta))
                       + p_psi^2 / (2 I3) + Mgl * cos(theta)

    Motion in theta is confined to where the total energy E >= V_eff(theta);
    the turning points bound the nutation range.

    Parameters
    ----------
    theta : array-like
        Angle(s) to evaluate at.
    p_phi, p_psi : float
        Conserved (cyclic-coordinate) momenta.
    I1, I3 : float
        Transverse and axial moments of inertia.
    Mgl : float
        ``M * g * l`` (mass times gravity times pivot-to-CM distance).

    Returns
    -------
    ndarray
    """
    theta = np.asarray(theta, dtype=np.float64)
    return (p_phi - p_psi * np.cos(theta)) ** 2 / (2.0 * I1 * np.sin(theta) ** 2) + p_psi**2 / (2.0 * I3) + Mgl * np.cos(theta)


def find_theta_equilibrium(p_phi: float, p_psi: float, I1: float, I3: float, Mgl: float, bracket=(1e-3, np.pi - 1e-3)) -> float:
    """The nutation angle theta_eq at which V_eff(theta) is minimized.

    The angle steady (non-nutating) precession would hold at, and the
    center about which a real top's nutation oscillates.

    Parameters
    ----------
    p_phi, p_psi : float
        Conserved (cyclic-coordinate) momenta.
    I1, I3 : float
        Transverse and axial moments of inertia.
    Mgl : float
        ``M * g * l``.
    bracket : tuple of float
        Search interval for the minimization.

    Returns
    -------
    float
    """
    result = minimize_scalar(
        lambda th: effective_potential_symmetric_top(th, p_phi, p_psi, I1, I3, Mgl),
        bounds=bracket,
        method="bounded",
    )
    return float(result.x)


def nutation_frequency(p_phi: float, p_psi: float, I1: float, I3: float, Mgl: float, theta_eq: float = None, dtheta: float = 1e-5) -> float:
    """Small-oscillation nutation frequency about ``theta_eq``.

    Treating theta as a 1-DOF particle of "mass" I1 moving in the
    effective potential V_eff (see :func:`effective_potential_symmetric_top`),
    the harmonic-approximation frequency about its minimum is
    ``sqrt(V_eff''(theta_eq) / I1)``, found here by a central finite
    difference. If ``theta_eq`` is not supplied it is located with
    :func:`find_theta_equilibrium`.

    Parameters
    ----------
    p_phi, p_psi : float
        Conserved (cyclic-coordinate) momenta.
    I1, I3 : float
        Transverse and axial moments of inertia.
    Mgl : float
        ``M * g * l``.
    theta_eq : float, optional
        Equilibrium angle; computed via :func:`find_theta_equilibrium`
        if omitted.
    dtheta : float
        Finite-difference step.

    Returns
    -------
    float
    """
    if theta_eq is None:
        theta_eq = find_theta_equilibrium(p_phi, p_psi, I1, I3, Mgl)
    V = lambda th: effective_potential_symmetric_top(th, p_phi, p_psi, I1, I3, Mgl)  # noqa: E731
    d2V = (V(theta_eq + dtheta) - 2.0 * V(theta_eq) + V(theta_eq - dtheta)) / dtheta**2
    return float(np.sqrt(d2V / I1))


# ---------------------------------------------------------------------------
# Euler's disk: a rolling, spinning disk running down to a finite-time stop
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _make_eulers_disk_deriv(decay_rate: float, precession_const: float, theta_floor: float):
    @njit(cache=False)
    def deriv(t, y):
        theta = y[0]
        theta_c = theta if theta > theta_floor else theta_floor
        out = np.empty(2)
        out[0] = -decay_rate / theta_c
        out[1] = precession_const / theta_c
        return out

    return deriv


class EulersDisk(ODESystem):
    """Simplified rolling-disk-with-dissipation model of "Euler's disk".

    A coin or disk spun and set rolling on a table settles into a
    near-steady state described, at each instant, by just its inclination
    angle ``theta`` (between the disk plane and the table) and a precession
    angle ``phi`` (the azimuthal angle of the rolling contact point around
    the disk's resting position). As real Euler's-disk demonstrations
    strikingly show, the contact point's precession visibly speeds up
    (and the characteristic rattling sound rises in pitch) right up until
    the disk suddenly stops -- a *finite-time singularity*, famously
    analyzed by Moffatt (2000, Nature 404, 833) for an idealized
    viscous-dissipation mechanism. The exact dissipation mechanism (viscous
    air drag vs. rolling friction vs. contact-point slipping) and even the
    precise power-law exponent are still debated in the literature (e.g.
    the response by Van den Engh et al., 2004); rather than committing to
    one first-principles derivation, this class uses a standard *reduced*
    phenomenological pair of ODEs that reproduces the same qualitative
    finite-time-collapse behavior:

    .. math::

        \\dot\\theta = -\\text{decay\\_rate} / \\theta, \\qquad
        \\dot\\phi = \\text{precession\\_const} / \\theta

    i.e. both the precession rate and (by assumption) the dissipation rate
    diverge as ``theta -> 0``, which is exactly what drives theta to zero
    in finite time. Ignoring the regularizing `theta_floor`, the first
    equation integrates in closed form to
    ``theta(t) = sqrt(theta0**2 - 2*decay_rate*t)`` (see
    :func:`eulers_disk_theta_analytic`), vanishing at the finite collapse
    time ``t_f = theta0**2 / (2*decay_rate)``. `theta_floor` clips both
    right-hand sides near ``theta = 0`` so the (unphysical) literal
    singularity never actually has to be integrated through.

    State: ``[theta, phi]``.

    Parameters
    ----------
    theta0 : float
        Initial inclination angle (radians), measured from horizontal
        (``theta = pi/2`` would be an upright, unspun disk; realistic
        "rattling" motion is at small `theta0`, e.g. 0.2-0.6 rad).
    phi0 : float, default 0.0
        Initial precession angle.
    decay_rate : float, default 0.05
        Rate constant governing how fast theta collapses; sets the
        collapse time ``t_f = theta0**2 / (2*decay_rate)``.
    precession_const : float, default 1.0
        Rate constant governing the precession rate at a given theta.
    theta_floor : float, default 1e-3
        Regularizing floor for `theta` in both right-hand sides.

    Attributes
    ----------
    decay_rate, precession_const, theta_floor : float
        System parameters.
    """

    def __init__(self, theta0: float, phi0: float = 0.0, decay_rate: float = 0.05, precession_const: float = 1.0, theta_floor: float = 1e-3):
        self.decay_rate, self.precession_const, self.theta_floor = decay_rate, precession_const, theta_floor
        self._deriv_njit = _make_eulers_disk_deriv(decay_rate, precession_const, theta_floor)
        super().__init__(np.array([theta0, phi0], dtype=np.float64))

    @property
    def theta(self) -> float:
        return self.state[0]

    @property
    def phi(self) -> float:
        return self.state[1]

    def energy(self, state: np.ndarray = None) -> float:
        """Not a true conserved energy -- the whole point of this model is
        dissipation. Returns `theta` itself, a monotonically decreasing
        proxy that (like the disk's actual energy) vanishes at the
        finite-time collapse, just so this fits the common ``energy()``
        diagnostic slot shared by every :class:`~physicskit.classical.core.base_system.ODESystem`.

        Parameters
        ----------
        state : ndarray, optional
            State to evaluate; defaults to the current state.

        Returns
        -------
        float
        """
        state = self.state if state is None else state
        return float(state[0])


def eulers_disk_theta_analytic(t, theta0: float, decay_rate: float) -> np.ndarray:
    """Closed-form ``theta(t)`` for :class:`EulersDisk`, ignoring the
    regularizing floor: solves ``dtheta/dt = -decay_rate/theta`` exactly as
    ``theta(t) = sqrt(theta0**2 - 2*decay_rate*t)``, valid up to the finite
    collapse time ``t_f = theta0**2 / (2*decay_rate)`` (clipped to 0
    thereafter).

    Parameters
    ----------
    t : array-like
        Time(s) to evaluate at.
    theta0 : float
        Initial inclination angle.
    decay_rate : float
        Decay-rate constant; see :class:`EulersDisk`.

    Returns
    -------
    ndarray
    """
    t = np.asarray(t, dtype=np.float64)
    return np.sqrt(np.clip(theta0**2 - 2.0 * decay_rate * t, 0.0, None))


# ---------------------------------------------------------------------------
# Rattleback (celt stone): a toy model of one-way spin instability/reversal
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _make_rattleback_deriv(I1: float, I2: float, I3: float, gamma: float, mu: float, eta: float):
    @njit(cache=False)
    def deriv(t, y):
        n1, n2, n3 = y[0], y[1], y[2]
        dn1 = (I2 - I3) / I1 * n2 * n3 + gamma * n3 * n3 - mu * n1 - eta * n1**3
        dn2 = (I3 - I1) / I2 * n3 * n1 - mu * n2 - eta * n2**3
        dn3 = (I1 - I2) / I3 * n1 * n2 - mu * n3
        out = np.empty(3)
        out[0], out[1], out[2] = dn1, dn2, dn3
        return out

    return deriv


class Rattleback(ODESystem):
    """Toy reduced model of rattleback (Celt stone) spin reversal.

    A rattleback -- a boat-shaped top whose principal axes of inertia are
    slightly misaligned from its geometric (contact-ellipsoid) axes --
    famously spins stably in one sense but, when spun the other way,
    wobbles with growing amplitude until it stops and reverses into the
    stable sense. The true mechanism is the nonholonomic rolling-without-
    slipping contact-point dynamics of a rigid ellipsoidal body on a plane
    (see Garcia & Hubbard, "Spin reversal of the rattleback: theory and
    experiment," Proc. R. Soc. Lond. A 418, 1988), which requires tracking
    the body's full orientation, the moving contact point, and the no-slip
    constraint.

    **This class does not do that.** It is a deliberately simplified toy
    model built directly on :class:`EulerTop`'s free-rigid-body equations
    for reduced state variables ``(n1, n2, n3)`` -- ``n3`` playing the role
    of spin about the (roughly vertical) axis, ``n1, n2`` the two
    "rocking"/tipping-mode rates -- with two modifications standing in for
    the real contact-point physics:

    1. A term ``gamma * n3**2``, added to the ``n1`` equation only. The
       real rolling constraint on a rattleback whose principal-inertia axes
       are misaligned (by some small angle) from its geometric axes
       couples the vertical spin asymmetrically into the two rocking
       modes; adding a term that is even in ``n3`` to only *one* of the two
       symmetric rocking equations is the minimal way to break the
       ``n3 -> -n3`` symmetry an ordinary (non-reversing) Euler top has --
       exactly the qualitative signature of a rattleback: spin one way
       pumps energy into rocking, spin the other way does not.
    2. Linear damping (`mu`) on all three components plus cubic,
       Landau-type self-saturation (`eta`) on ``n1, n2``, so the rocking
       amplitude the instability pumps up saturates rather than diverging,
       and the whole system eventually settles to rest -- as a real
       rattleback, slowed by genuine friction, does.

    With `gamma` = 0 this reduces to a damped :class:`EulerTop` and never
    reverses; for `gamma` large enough relative to the damping, a spin
    started predominantly in ``n3`` visibly collapses, overshoots into
    negative ``n3`` (the reversal), and then decays -- reproducing the
    qualitative phenomenology (decelerate, wobble, reverse, decay to rest)
    without claiming quantitative accuracy for any specific physical body.

    State: ``[n1, n2, n3]``.

    Parameters
    ----------
    n0 : array-like, shape (3,)
        Initial ``(n1, n2, n3)``; e.g. ``[0.01, 0.01, 3.0]`` models a
        rattleback spun up hard about its ``n3`` axis with only a tiny
        rocking perturbation.
    I1, I2, I3 : float
        Principal moments of inertia (same role as in :class:`EulerTop`).
    gamma : float, default 1.0
        Strength of the symmetry-breaking rolling-constraint correction.
    mu : float, default 0.01
        Linear (frictional) damping rate on all three components.
    eta : float, default 0.01
        Cubic self-saturation coefficient on ``n1, n2``.

    Attributes
    ----------
    I1, I2, I3, gamma, mu, eta : float
        System parameters.
    """

    def __init__(self, n0, I1: float = 1.0, I2: float = 1.5, I3: float = 2.0, gamma: float = 1.0, mu: float = 0.01, eta: float = 0.01):
        self.I1, self.I2, self.I3 = I1, I2, I3
        self.gamma, self.mu, self.eta = gamma, mu, eta
        self._deriv_njit = _make_rattleback_deriv(I1, I2, I3, gamma, mu, eta)
        super().__init__(np.asarray(n0, dtype=np.float64))

    def energy(self, state: np.ndarray = None) -> float:
        """``0.5*(I1*n1^2 + I2*n2^2 + I3*n3^2)``: not conserved (this
        system is dissipative and, transiently, unstable by construction),
        but the natural quadratic diagnostic for watching the overall
        motion decay toward rest.

        Parameters
        ----------
        state : ndarray, optional
            State to evaluate; defaults to the current state.

        Returns
        -------
        float
        """
        state = self.state if state is None else state
        n1, n2, n3 = state[0], state[1], state[2]
        return 0.5 * (self.I1 * n1**2 + self.I2 * n2**2 + self.I3 * n3**2)


def precession_frequency(p_phi: float, p_psi: float, I1: float, theta_eq: float) -> float:
    """Mean precession rate phidot at nutation angle ``theta_eq``:
    ``(p_phi - p_psi*cos(theta_eq)) / (I1*sin(theta_eq)**2)`` -- the
    standard steady-precession-rate approximation, exact when the top
    is not nutating at all (theta held fixed at theta_eq).

    Parameters
    ----------
    p_phi, p_psi : float
        Conserved (cyclic-coordinate) momenta.
    I1 : float
        Transverse moment of inertia.
    theta_eq : float
        Nutation angle to evaluate at.

    Returns
    -------
    float
    """
    return float((p_phi - p_psi * np.cos(theta_eq)) / (I1 * np.sin(theta_eq) ** 2))
