"""Newtonian vector-dynamics systems.

:class:`ProjectileMotion` is the classic "cannonball" problem -- free
fall under constant gravity with an oblique (angled) initial velocity,
the trajectory-is-a-parabola result at the historical root of
Newtonian dynamics -- with an optional quadratic air-drag term.

:class:`KeplerSystem` integrates a reduced 2-body problem in the
center-of-mass frame as a planar, separable Hamiltonian system in
Cartesian coordinates q = (x, y), p = (px, py), with::

    H(q, p) = |p|^2 / (2 mu) + V(r),  r = |q|
    V(r) = -k / r + c_eps / r**(1 + eps) + c_pn / r**3

The ``c_eps`` term is an adjustable power-law perturbation to the pure
1/r potential; ``c_pn`` is an effective post-Newtonian correction that
reproduces the qualitative apsidal (perihelion) precession General
Relativity predicts for Mercury's orbit. Both terms are still purely
radial (velocity-independent), so the system remains conservative and
separable -- exactly integrable with the symplectic Verlet/Yoshida4
backends -- while the Laplace-Runge-Lenz vector, exactly conserved for
the pure Kepler problem, slowly precesses once either perturbation is
turned on.

:class:`FoucaultPendulum` is a small-oscillation pendulum viewed from
the rotating Earth: Newton's second law in a non-inertial frame picks
up a velocity-dependent Coriolis term that precesses the swing plane
without doing any work, so energy stays exactly conserved even as the
plane of swing rotates -- the effect Leon Foucault used in 1851 to
demonstrate Earth's rotation mechanically.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numba import njit

from physicskit.classical.core.base_system import HamiltonianSystem, ODESystem

__all__ = ["ProjectileMotion", "KeplerSystem", "FoucaultPendulum"]


@lru_cache(maxsize=64)
def _make_projectile_force(m: float, g: float):
    @njit(cache=True)
    def force(q, t):
        out = np.empty(2)
        out[0] = 0.0
        out[1] = -m * g
        return out

    return force


@lru_cache(maxsize=64)
def _make_projectile_deriv_with_drag(m: float, g: float, drag_coeff: float):
    """Full (non-conservative) [dq, dp] derivative including quadratic
    air drag, ``F_drag = -drag_coeff * |v| * v``, used by the 'rk4' and
    'implicit_midpoint' fallback methods when drag_coeff != 0 (drag
    depends on velocity/momentum, so it cannot enter a separable
    force(q, t) and is invisible to the Verlet/Yoshida4 backends)."""

    @njit(cache=False)
    def deriv(t, y):
        px, py = y[2], y[3]
        vx, vy = px / m, py / m
        speed = np.sqrt(vx * vx + vy * vy)
        out = np.empty(4)
        out[0] = vx
        out[1] = vy
        out[2] = -drag_coeff * speed * vx
        out[3] = -m * g - drag_coeff * speed * vy
        return out

    return deriv


class ProjectileMotion(HamiltonianSystem):
    """Free fall under constant gravity with an oblique initial velocity
    -- the "cannonball" problem -- with optional quadratic air drag.

    Without drag this is the separable Hamiltonian
    ``H(q, p) = |p|^2/(2m) + m*g*y``, q = (x, y), whose exact solution is
    Galileo's parabolic trajectory (see :meth:`analytic_trajectory`);
    because the force is constant, the symplectic Verlet/Yoshida4
    backends reproduce it to machine precision at any step size.

    With ``drag_coeff > 0`` a quadratic air-resistance force
    ``F = -drag_coeff * |v| * v`` is added. This makes the system
    genuinely dissipative (energy decreases monotonically, there is no
    closed-form trajectory), so it is only available through the
    ``rk4``/``implicit_midpoint`` methods -- ``verlet``/``yoshida4``
    would silently ignore it (their force callback only sees the
    conservative gravity term) and are therefore refused outright.

    Parameters
    ----------
    q0, p0 : array-like, shape (2,)
        Initial position and momentum, q = (x, y), p = (px, py).
    m, g : float
        Projectile mass and gravitational acceleration.
    drag_coeff : float
        Quadratic air-drag coefficient (0 for the vacuum/exact case).
    """

    separable = True

    def __init__(self, q0, p0, m: float = 1.0, g: float = 9.81, drag_coeff: float = 0.0):
        self.m, self.g, self.drag_coeff = m, g, drag_coeff
        self.mass_inv = 1.0 / m
        self._force_njit = _make_projectile_force(m, g)
        if drag_coeff:
            self._deriv_njit = _make_projectile_deriv_with_drag(m, g, drag_coeff)
        super().__init__(q0, p0)

    def integrate(self, t_span, dt: float, method: str = "yoshida4"):
        if self.drag_coeff and method in ("verlet", "yoshida4"):
            raise ValueError(
                f"method='{method}' only sees the conservative force and would silently ignore drag_coeff != 0; use method='rk4' or 'implicit_midpoint' instead"
            )
        return super().integrate(t_span, dt, method=method)

    def kinetic_energy(self, p: np.ndarray) -> float:
        return float(np.dot(p, p)) / (2.0 * self.m)

    def potential_energy(self, q: np.ndarray) -> float:
        return self.m * self.g * q[1]

    @classmethod
    def from_launch(cls, speed: float, angle_deg: float, m: float = 1.0, g: float = 9.81, drag_coeff: float = 0.0, height: float = 0.0):
        """Construct a system launched from ``(0, height)`` -- the natural "aim the cannon" entry point.

        Parameters
        ----------
        speed : float
            Launch speed.
        angle_deg : float
            Launch angle above the horizontal, in degrees.
        m, g : float
            Projectile mass and gravitational acceleration.
        drag_coeff : float
            Quadratic air-drag coefficient (0 for the vacuum/exact case).
        height : float
            Initial height.

        Returns
        -------
        ProjectileMotion
        """
        theta = np.deg2rad(angle_deg)
        q0 = np.array([0.0, height])
        p0 = m * speed * np.array([np.cos(theta), np.sin(theta)])
        return cls(q0, p0, m=m, g=g, drag_coeff=drag_coeff)

    @staticmethod
    def analytic_trajectory(speed: float, angle_deg: float, g: float, t, height: float = 0.0):
        """Exact vacuum trajectory x(t), y(t) -- Galileo's parabola.

        Used to validate the numerical integration (``drag_coeff=0`` only).

        Parameters
        ----------
        speed, angle_deg : float
            Launch speed and angle above the horizontal, in degrees.
        g : float
            Gravitational acceleration.
        t : array-like
            Times at which to evaluate the trajectory.
        height : float
            Initial height.

        Returns
        -------
        x, y : ndarray
        """
        theta = np.deg2rad(angle_deg)
        t = np.asarray(t, dtype=np.float64)
        vx0, vy0 = speed * np.cos(theta), speed * np.sin(theta)
        x = vx0 * t
        y = height + vy0 * t - 0.5 * g * t**2
        return x, y

    @staticmethod
    def range_and_max_height(speed: float, angle_deg: float, g: float, height: float = 0.0):
        """Closed-form (vacuum) horizontal range and maximum height.

        Parameters
        ----------
        speed, angle_deg : float
            Launch speed and angle above the horizontal, in degrees.
        g : float
            Gravitational acceleration.
        height : float
            Initial height.

        Returns
        -------
        range, max_height : float
        """
        theta = np.deg2rad(angle_deg)
        vx0, vy0 = speed * np.cos(theta), speed * np.sin(theta)
        t_apex = vy0 / g
        max_height = height + vy0 * t_apex - 0.5 * g * t_apex**2
        t_flight = (vy0 + np.sqrt(vy0**2 + 2.0 * g * height)) / g
        rng = vx0 * t_flight
        return float(rng), float(max_height)


@lru_cache(maxsize=64)
def _make_force_njit(k: float, eps: float, c_eps: float, c_pn: float, softening: float):
    @njit(cache=False)
    def force(q, t):
        x, y = q[0], q[1]
        r2 = x * x + y * y + softening * softening
        r = np.sqrt(r2)
        # dV/dr for V(r) = -k/r + c_eps * r**(-(1+eps)) + c_pn * r**(-3)
        dVdr = k / r2 - c_eps * (1.0 + eps) * r ** (-(2.0 + eps)) - 3.0 * c_pn * r ** (-4.0)
        # force = -dV/dq = -dVdr * (q / r)
        fmag = -dVdr / r
        out = np.empty(2)
        out[0] = fmag * x
        out[1] = fmag * y
        return out

    return force


class KeplerSystem(HamiltonianSystem):
    """Planar 2-body Kepler problem with optional non-Newtonian perturbations.

    Parameters
    ----------
    q0, p0 : array-like, shape (2,)
        Initial position and (reduced-mass) momentum.
    k : float
        Gravitational coupling ``G * M * mu`` (standard Kepler strength).
    mu : float
        Reduced mass (sets kinetic energy ``T = |p|^2 / (2 mu)``).
    eps, c_eps : float
        Exponent and coefficient of an additional ``c_eps / r**(1+eps)``
        perturbing potential (eps=0 reduces to an extra inverse-square term).
    c_pn : float
        Coefficient of the effective post-Newtonian ``c_pn / r**3``
        correction driving perihelion precession.
    softening : float
        Small Plummer-style softening length to avoid a singular force
        at r=0 (0 for the exact point-mass limit).
    """

    separable = True

    def __init__(self, q0, p0, k=1.0, mu=1.0, eps=0.0, c_eps=0.0, c_pn=0.0, softening=0.0):
        self.k = k
        self.mu = mu
        self.eps = eps
        self.c_eps = c_eps
        self.c_pn = c_pn
        self.softening = softening
        self.mass_inv = 1.0 / mu
        self._force_njit = _make_force_njit(k, eps, c_eps, c_pn, softening)
        super().__init__(q0, p0)

    def kinetic_energy(self, p: np.ndarray) -> float:
        return float(np.dot(p, p)) / (2.0 * self.mu)

    def potential_energy(self, q: np.ndarray) -> float:
        r = np.sqrt(q[0] ** 2 + q[1] ** 2 + self.softening**2)
        return -self.k / r + self.c_eps * r ** (-(1.0 + self.eps)) + self.c_pn * r ** (-3.0)

    def angular_momentum(self, q: np.ndarray = None, p: np.ndarray = None) -> float:
        """Out-of-plane angular momentum ``L = x*py - y*px``.

        Parameters
        ----------
        q, p : ndarray, optional
            State to evaluate; defaults to the current state.

        Returns
        -------
        float
        """
        q = self.q if q is None else q
        p = self.p if p is None else p
        return float(q[0] * p[1] - q[1] * p[0])

    def lrl_vector(self, q: np.ndarray = None, p: np.ndarray = None) -> np.ndarray:
        """Laplace-Runge-Lenz vector ``A = p x L - mu*k*r_hat`` (planar form).

        Exactly conserved (a fixed vector pointing at perihelion) for the
        unperturbed 1/r potential; its slow rotation once ``c_eps`` or
        ``c_pn`` is nonzero is the precession signature (e.g. Mercury's
        perihelion advance).

        Parameters
        ----------
        q, p : ndarray, optional
            State to evaluate; defaults to the current state.

        Returns
        -------
        ndarray, shape (2,)
        """
        q = self.q if q is None else q
        p = self.p if p is None else p
        L = self.angular_momentum(q, p)
        r = np.linalg.norm(q)
        # p x L in 2D (L is the out-of-plane scalar): (py*L, -px*L)
        px_L = np.array([p[1] * L, -p[0] * L])
        return px_L - self.mu * self.k * q / r

    @classmethod
    def from_orbital_elements(cls, a: float, e: float, k: float = 1.0, mu: float = 1.0, **kwargs):
        """Construct a system starting at perihelion (pure two-body reference orbit).

        Parameters
        ----------
        a : float
            Semi-major axis.
        e : float
            Eccentricity.
        k, mu : float
            Gravitational coupling and reduced mass, as in ``__init__``.
        **kwargs
            Forwarded to ``__init__`` (e.g. ``eps``, ``c_eps``, ``c_pn``, ``softening``).

        Returns
        -------
        KeplerSystem
        """
        r_peri = a * (1.0 - e)
        v_peri = np.sqrt(k / mu * (2.0 / r_peri - 1.0 / a))
        q0 = np.array([r_peri, 0.0])
        p0 = np.array([0.0, mu * v_peri])
        return cls(q0, p0, k=k, mu=mu, **kwargs)


_SIDEREAL_DAY = 86164.0905
"""float: Earth's rotation period relative to the stars, in seconds."""


@lru_cache(maxsize=64)
def _make_foucault_deriv(omega0: float, omega_z: float):
    @njit(cache=True)
    def deriv(t, state):
        x, y, vx, vy = state[0], state[1], state[2], state[3]
        out = np.empty(4)
        out[0] = vx
        out[1] = vy
        out[2] = -omega0 * omega0 * x + 2.0 * omega_z * vy
        out[3] = -omega0 * omega0 * y - 2.0 * omega_z * vx
        return out

    return deriv


class FoucaultPendulum(ODESystem):
    """Small-oscillation pendulum viewed from Earth's rotating frame.

    Swinging near a fixed vertical, a pendulum's horizontal displacement
    q = (x, y) (x = East, y = North) obeys the isotropic harmonic
    equation ``qddot = -omega0**2 * q`` in a non-rotating (inertial)
    frame. Viewed instead from the surface of the rotating Earth, that
    frame itself turns about the local vertical at
    ``omega_z = omega_earth * sin(latitude)`` -- the projection of
    Earth's angular velocity onto the local vertical -- which adds a
    velocity-dependent Coriolis term::

        xddot = -omega0**2 * x + 2 * omega_z * ydot
        yddot = -omega0**2 * y - 2 * omega_z * xdot

    Because the Coriolis force is always perpendicular to the velocity
    it does no work, so the (non-rotating-frame) mechanical energy
    ``T + V = |v|**2 / 2 + omega0**2 * |q|**2 / 2`` stays exactly
    conserved even as the plane of swing itself steadily precesses --
    exactly the effect Leon Foucault used in 1851 to demonstrate
    Earth's rotation mechanically, without reference to the sky.

    Parameters
    ----------
    q0, v0 : array-like, shape (2,)
        Initial horizontal displacement (x, y) and velocity (vx, vy) of
        the bob relative to the vertical through the pivot.
    length, g : float
        Pendulum length and gravitational acceleration, setting the
        non-rotating swing frequency ``omega0 = sqrt(g / length)``.
    latitude_deg : float
        Geographic latitude in degrees (default 48.85, the Paris
        Pantheon's latitude for Foucault's original 1851 demonstration).
    omega_earth : float
        Earth's sidereal rotation rate, ``2*pi / 86164.0905`` rad/s by default.
    """

    def __init__(
        self,
        q0,
        v0,
        length: float = 67.0,
        g: float = 9.81,
        latitude_deg: float = 48.85,
        omega_earth: float = 2.0 * np.pi / _SIDEREAL_DAY,
    ):
        self.length, self.g = length, g
        self.omega0 = float(np.sqrt(g / length))
        self.latitude_deg = latitude_deg
        self.omega_earth = omega_earth
        self.omega_z = float(omega_earth * np.sin(np.deg2rad(latitude_deg)))
        self._deriv_njit = _make_foucault_deriv(self.omega0, self.omega_z)
        state0 = np.concatenate([np.asarray(q0, dtype=np.float64), np.asarray(v0, dtype=np.float64)])
        super().__init__(state0)

    @property
    def position(self) -> np.ndarray:
        """ndarray, shape (2,): Current horizontal displacement (x, y)."""
        return self.state[:2]

    @property
    def velocity(self) -> np.ndarray:
        """ndarray, shape (2,): Current horizontal velocity (vx, vy)."""
        return self.state[2:]

    def energy(self, state: np.ndarray = None) -> float:
        state = self.state if state is None else state
        x, y, vx, vy = state[0], state[1], state[2], state[3]
        return 0.5 * (vx * vx + vy * vy) + 0.5 * self.omega0**2 * (x * x + y * y)

    @classmethod
    def from_deflection(
        cls,
        amplitude: float,
        angle_deg: float = 0.0,
        length: float = 67.0,
        g: float = 9.81,
        latitude_deg: float = 48.85,
        omega_earth: float = 2.0 * np.pi / _SIDEREAL_DAY,
    ):
        """Construct a system released from rest at a fixed deflection.

        This is how a real Foucault pendulum is started: pulled aside
        and tied off, then released from rest.

        Parameters
        ----------
        amplitude : float
            Initial horizontal displacement magnitude.
        angle_deg : float
            Direction of the initial displacement, in degrees
            counterclockwise from East.
        length, g, latitude_deg, omega_earth : float
            As in ``__init__``.

        Returns
        -------
        FoucaultPendulum
        """
        theta = np.deg2rad(angle_deg)
        q0 = amplitude * np.array([np.cos(theta), np.sin(theta)])
        v0 = np.zeros(2)
        return cls(q0, v0, length=length, g=g, latitude_deg=latitude_deg, omega_earth=omega_earth)

    def precession_period(self) -> float:
        """Time for the swing plane to complete one full precession.

        Equal to the sidereal day divided by ``sin(latitude)`` -- a
        "Foucault day" -- e.g. about 32 hours at Paris's latitude.

        Returns
        -------
        float
        """
        return 2.0 * np.pi / abs(self.omega_z)

    @staticmethod
    def analytic_solution(q0, v0, omega0: float, omega_z: float, t):
        """Exact horizontal trajectory x(t), y(t) in the small-oscillation approximation.

        The complex displacement ``w = x + i*y`` satisfies the linear
        equation ``wddot + 2i*omega_z*wdot + omega0**2*w = 0``.
        Substituting ``w = exp(-i*omega_z*t) * u(t)`` reduces this to
        plain simple harmonic motion for ``u`` at frequency
        ``Omega = sqrt(omega0**2 + omega_z**2)``: the solution is an
        ordinary (non-rotating) harmonic swing pattern ``u(t)``, rotated
        rigidly as a whole at the constant rate ``omega_z`` -- exactly
        the precession Foucault observed.

        Parameters
        ----------
        q0, v0 : array-like, shape (2,)
            Initial displacement and velocity.
        omega0, omega_z : float
            Non-rotating swing frequency and the local-vertical
            component of Earth's angular velocity.
        t : array-like
            Times at which to evaluate the trajectory.

        Returns
        -------
        x, y : ndarray
        """
        t = np.asarray(t, dtype=np.float64)
        Omega = np.sqrt(omega0**2 + omega_z**2)
        w0 = complex(q0[0], q0[1])
        wdot0 = complex(v0[0], v0[1])
        u0 = w0
        udot0 = wdot0 + 1j * omega_z * w0
        u = u0 * np.cos(Omega * t) + (udot0 / Omega) * np.sin(Omega * t)
        w = np.exp(-1j * omega_z * t) * u
        return w.real, w.imag

    @staticmethod
    def to_corotating_frame(x, y, t, omega_z: float):
        """Rotate (x, y) by ``+omega_z * t``, undoing the precession.

        In this frame the pendulum swings in a single fixed plane, just
        like an ordinary non-rotating pendulum -- the frame in which
        Foucault's precession is "unwound".

        Parameters
        ----------
        x, y : array-like
            Trajectory in the lab (ground) frame.
        t : array-like
            Corresponding times.
        omega_z : float
            Local-vertical component of Earth's angular velocity.

        Returns
        -------
        x_rot, y_rot : ndarray
        """
        x, y, t = np.asarray(x), np.asarray(y), np.asarray(t)
        c, s = np.cos(omega_z * t), np.sin(omega_z * t)
        x_rot = c * x - s * y
        y_rot = s * x + c * y
        return x_rot, y_rot
