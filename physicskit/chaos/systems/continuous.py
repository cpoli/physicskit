"""Continuous-time chaotic dynamical systems: Lorenz, Rossler, Double Pendulum, Duffing, forced Van der Pol, Chua.

Each system exposes a plain-Python :meth:`rhs` (for interactive use / plotting)
plus a module-level Numba ``@njit`` right-hand-side function with the signature
``rhs(state, t, params) -> ndarray`` compatible with
:func:`physicskit.chaos.core.integrators.rk4_integrate`, and a convenience
:meth:`trajectory` method that drives that integrator.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from physicskit.chaos.core.base_system import DynamicalSystem
from physicskit.chaos.core.integrators import rk4_integrate

# ---------------------------------------------------------------------------
# Lorenz system
# ---------------------------------------------------------------------------


@njit(cache=True)
def _lorenz_rhs(state: NDArray[np.float64], t: float, params: NDArray[np.float64]) -> NDArray[np.float64]:
    """Lorenz vector field ``dx/dt = f(x, t; sigma, rho, beta)``.

    Parameters
    ----------
    state : ndarray of float, shape (3,)
        State vector ``(x, y, z)``.
    t : float
        Current time (unused; the system is autonomous).
    params : ndarray of float, shape (3,)
        Parameters ``(sigma, rho, beta)``.

    Returns
    -------
    ndarray of float, shape (3,)
        Time derivative ``(dx/dt, dy/dt, dz/dt)``.
    """
    sigma, rho, beta = params[0], params[1], params[2]
    x, y, z = state[0], state[1], state[2]
    out = np.empty(3)
    out[0] = sigma * (y - x)
    out[1] = x * (rho - z) - y
    out[2] = x * y - beta * z
    return out


class Lorenz(DynamicalSystem):
    """The Lorenz attractor.

    Parameters
    ----------
    sigma : float, default 10.0
        Prandtl-number-like parameter.
    rho : float, default 28.0
        Rayleigh-number-like parameter.
    beta : float, default 8.0 / 3.0
        Geometric parameter.

    Attributes
    ----------
    sigma, rho, beta : float
        System parameters.
    """

    #: State dimension, always 3.
    dim = 3

    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0):
        self.sigma = float(sigma)
        self.rho = float(rho)
        self.beta = float(beta)

    @property
    def params(self) -> NDArray[np.float64]:
        """Parameter vector ``(sigma, rho, beta)``.

        Returns
        -------
        ndarray of float, shape (3,)
        """
        return np.array([self.sigma, self.rho, self.beta])

    def rhs(self, state: NDArray[np.float64], t: float) -> NDArray[np.float64]:
        """Evaluate the Lorenz vector field.

        Parameters
        ----------
        state : ndarray of float, shape (3,)
            State vector ``(x, y, z)``.
        t : float
            Current time (unused; the system is autonomous).

        Returns
        -------
        ndarray of float, shape (3,)
            Time derivative ``(dx/dt, dy/dt, dz/dt)``.
        """
        return np.asarray(_lorenz_rhs(np.asarray(state, dtype=np.float64), t, self.params))

    def initial_state(self) -> NDArray[np.float64]:
        """Default initial condition ``(1, 1, 1)``.

        Returns
        -------
        ndarray of float, shape (3,)
        """
        return np.array([1.0, 1.0, 1.0])

    def trajectory(
        self,
        state0: NDArray[np.float64] | None = None,
        t0: float = 0.0,
        dt: float = 0.01,
        n_steps: int = 10000,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Integrate a trajectory with the Numba-accelerated RK4 integrator.

        Parameters
        ----------
        state0 : array_like of float, shape (3,), optional
            Initial state; defaults to :meth:`initial_state`.
        t0 : float, default 0.0
            Initial time.
        dt : float, default 0.01
            Integration step size.
        n_steps : int, default 10000
            Number of integration steps.

        Returns
        -------
        times : ndarray of float, shape (n_steps + 1,)
        states : ndarray of float, shape (n_steps + 1, 3)
        """
        state0 = self.initial_state() if state0 is None else np.asarray(state0, dtype=np.float64)
        return rk4_integrate(_lorenz_rhs, state0, t0, dt, n_steps, self.params)


# ---------------------------------------------------------------------------
# Rossler system
# ---------------------------------------------------------------------------


@njit(cache=True)
def _rossler_rhs(state: NDArray[np.float64], t: float, params: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rossler vector field ``dx/dt = f(x, t; a, b, c)``.

    Parameters
    ----------
    state : ndarray of float, shape (3,)
        State vector ``(x, y, z)``.
    t : float
        Current time (unused; the system is autonomous).
    params : ndarray of float, shape (3,)
        Parameters ``(a, b, c)``.

    Returns
    -------
    ndarray of float, shape (3,)
        Time derivative ``(dx/dt, dy/dt, dz/dt)``.
    """
    a, b, c = params[0], params[1], params[2]
    x, y, z = state[0], state[1], state[2]
    out = np.empty(3)
    out[0] = -y - z
    out[1] = x + a * y
    out[2] = b + z * (x - c)
    return out


class Rossler(DynamicalSystem):
    """The Rossler attractor.

    Parameters
    ----------
    a : float, default 0.2
        System parameter.
    b : float, default 0.2
        System parameter.
    c : float, default 5.7
        System parameter.

    Attributes
    ----------
    a, b, c : float
        System parameters.
    """

    #: State dimension, always 3.
    dim = 3

    def __init__(self, a: float = 0.2, b: float = 0.2, c: float = 5.7):
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)

    @property
    def params(self) -> NDArray[np.float64]:
        """Parameter vector ``(a, b, c)``.

        Returns
        -------
        ndarray of float, shape (3,)
        """
        return np.array([self.a, self.b, self.c])

    def rhs(self, state: NDArray[np.float64], t: float) -> NDArray[np.float64]:
        """Evaluate the Rossler vector field.

        Parameters
        ----------
        state : ndarray of float, shape (3,)
            State vector ``(x, y, z)``.
        t : float
            Current time (unused; the system is autonomous).

        Returns
        -------
        ndarray of float, shape (3,)
            Time derivative ``(dx/dt, dy/dt, dz/dt)``.
        """
        return np.asarray(_rossler_rhs(np.asarray(state, dtype=np.float64), t, self.params))

    def initial_state(self) -> NDArray[np.float64]:
        """Default initial condition ``(1, 1, 1)``.

        Returns
        -------
        ndarray of float, shape (3,)
        """
        return np.array([1.0, 1.0, 1.0])

    def trajectory(
        self,
        state0: NDArray[np.float64] | None = None,
        t0: float = 0.0,
        dt: float = 0.01,
        n_steps: int = 10000,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Integrate a trajectory with the Numba-accelerated RK4 integrator.

        Parameters
        ----------
        state0 : array_like of float, shape (3,), optional
            Initial state; defaults to :meth:`initial_state`.
        t0 : float, default 0.0
            Initial time.
        dt : float, default 0.01
            Integration step size.
        n_steps : int, default 10000
            Number of integration steps.

        Returns
        -------
        times : ndarray of float, shape (n_steps + 1,)
        states : ndarray of float, shape (n_steps + 1, 3)
        """
        state0 = self.initial_state() if state0 is None else np.asarray(state0, dtype=np.float64)
        return rk4_integrate(_rossler_rhs, state0, t0, dt, n_steps, self.params)


# ---------------------------------------------------------------------------
# Double pendulum
# ---------------------------------------------------------------------------


@njit(cache=True)
def _double_pendulum_rhs(state: NDArray[np.float64], t: float, params: NDArray[np.float64]) -> NDArray[np.float64]:
    """Double-pendulum vector field ``dx/dt = f(x, t; m1, m2, l1, l2, g)``.

    Parameters
    ----------
    state : ndarray of float, shape (4,)
        State vector ``(theta1, theta2, omega1, omega2)``.
    t : float
        Current time (unused; the system is autonomous).
    params : ndarray of float, shape (5,)
        Parameters ``(m1, m2, l1, l2, g)``.

    Returns
    -------
    ndarray of float, shape (4,)
        Time derivative ``(dtheta1/dt, dtheta2/dt, domega1/dt, domega2/dt)``.
    """
    m1, m2, l1, l2, g = params[0], params[1], params[2], params[3], params[4]
    th1, th2, w1, w2 = state[0], state[1], state[2], state[3]
    delta = th1 - th2
    den1 = l1 * (2.0 * m1 + m2 - m2 * np.cos(2.0 * delta))
    den2 = l2 * (2.0 * m1 + m2 - m2 * np.cos(2.0 * delta))

    dw1 = (
        -g * (2.0 * m1 + m2) * np.sin(th1) - m2 * g * np.sin(th1 - 2.0 * th2) - 2.0 * np.sin(delta) * m2 * (w2 * w2 * l2 + w1 * w1 * l1 * np.cos(delta))
    ) / den1
    dw2 = (2.0 * np.sin(delta) * (w1 * w1 * l1 * (m1 + m2) + g * (m1 + m2) * np.cos(th1) + w2 * w2 * l2 * m2 * np.cos(delta))) / den2

    out = np.empty(4)
    out[0] = w1
    out[1] = w2
    out[2] = dw1
    out[3] = dw2
    return out


class DoublePendulum(DynamicalSystem):
    """Planar double pendulum: point masses on massless rods.

    Parameters
    ----------
    m1 : float, default 1.0
        Mass of the first (inner) bob.
    m2 : float, default 1.0
        Mass of the second (outer) bob.
    l1 : float, default 1.0
        Length of the first (inner) rod.
    l2 : float, default 1.0
        Length of the second (outer) rod.
    g : float, default 9.81
        Gravitational acceleration.

    Attributes
    ----------
    m1, m2, l1, l2, g : float
        System parameters.
    """

    #: State dimension, always 4. State is ``(theta1, theta2, omega1, omega2)``.
    dim = 4

    def __init__(self, m1: float = 1.0, m2: float = 1.0, l1: float = 1.0, l2: float = 1.0, g: float = 9.81):
        self.m1, self.m2, self.l1, self.l2, self.g = (
            float(m1),
            float(m2),
            float(l1),
            float(l2),
            float(g),
        )

    @property
    def params(self) -> NDArray[np.float64]:
        """Parameter vector ``(m1, m2, l1, l2, g)``.

        Returns
        -------
        ndarray of float, shape (5,)
        """
        return np.array([self.m1, self.m2, self.l1, self.l2, self.g])

    def rhs(self, state: NDArray[np.float64], t: float) -> NDArray[np.float64]:
        """Evaluate the double-pendulum vector field.

        Parameters
        ----------
        state : ndarray of float, shape (4,)
            State vector ``(theta1, theta2, omega1, omega2)``.
        t : float
            Current time (unused; the system is autonomous).

        Returns
        -------
        ndarray of float, shape (4,)
            Time derivative ``(dtheta1/dt, dtheta2/dt, domega1/dt, domega2/dt)``.
        """
        return np.asarray(_double_pendulum_rhs(np.asarray(state, dtype=np.float64), t, self.params))

    def initial_state(self) -> NDArray[np.float64]:
        """Default initial condition: both rods horizontal, at rest.

        Returns
        -------
        ndarray of float, shape (4,)
        """
        return np.array([np.pi / 2.0, np.pi / 2.0, 0.0, 0.0])

    def trajectory(
        self,
        state0: NDArray[np.float64] | None = None,
        t0: float = 0.0,
        dt: float = 0.005,
        n_steps: int = 10000,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Integrate a trajectory with the Numba-accelerated RK4 integrator.

        Parameters
        ----------
        state0 : array_like of float, shape (4,), optional
            Initial state; defaults to :meth:`initial_state`.
        t0 : float, default 0.0
            Initial time.
        dt : float, default 0.005
            Integration step size.
        n_steps : int, default 10000
            Number of integration steps.

        Returns
        -------
        times : ndarray of float, shape (n_steps + 1,)
        states : ndarray of float, shape (n_steps + 1, 4)
        """
        state0 = self.initial_state() if state0 is None else np.asarray(state0, dtype=np.float64)
        return rk4_integrate(_double_pendulum_rhs, state0, t0, dt, n_steps, self.params)

    def energy(self, state: NDArray[np.float64]) -> float:
        """Total mechanical energy, useful for checking integrator drift.

        Parameters
        ----------
        state : array_like of float, shape (4,)
            State vector ``(theta1, theta2, omega1, omega2)``.

        Returns
        -------
        float
            Total (kinetic + potential) mechanical energy.
        """
        th1, th2, w1, w2 = state
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g
        kinetic = 0.5 * m1 * (l1 * w1) ** 2 + 0.5 * m2 * ((l1 * w1) ** 2 + (l2 * w2) ** 2 + 2 * l1 * l2 * w1 * w2 * np.cos(th1 - th2))
        potential = -(m1 + m2) * g * l1 * np.cos(th1) - m2 * g * l2 * np.cos(th2)
        return float(kinetic + potential)


# ---------------------------------------------------------------------------
# Duffing oscillator
# ---------------------------------------------------------------------------


@njit(cache=True)
def _duffing_rhs(state: NDArray[np.float64], t: float, params: NDArray[np.float64]) -> NDArray[np.float64]:
    """Duffing vector field ``dx/dt = f(x, t; delta, alpha, beta, gamma, omega)``.

    Parameters
    ----------
    state : ndarray of float, shape (2,)
        State vector ``(x, v)``.
    t : float
        Current time (the system is non-autonomous: forcing depends on `t`).
    params : ndarray of float, shape (5,)
        Parameters ``(delta, alpha, beta, gamma, omega)``.

    Returns
    -------
    ndarray of float, shape (2,)
        Time derivative ``(dx/dt, dv/dt)``.
    """
    delta, alpha, beta, gamma, omega = params[0], params[1], params[2], params[3], params[4]
    x, v = state[0], state[1]
    out = np.empty(2)
    out[0] = v
    out[1] = -delta * v - alpha * x - beta * x**3 + gamma * np.cos(omega * t)
    return out


class Duffing(DynamicalSystem):
    """The forced, damped Duffing oscillator.

    Governed by ``x'' + delta*x' + alpha*x + beta*x^3 = gamma*cos(omega*t)``.

    Parameters
    ----------
    delta : float, default 0.3
        Damping coefficient.
    alpha : float, default -1.0
        Linear stiffness.
    beta : float, default 1.0
        Cubic (nonlinear) stiffness.
    gamma : float, default 0.37
        Forcing amplitude.
    omega : float, default 1.2
        Forcing angular frequency.

    Attributes
    ----------
    delta, alpha, beta, gamma, omega : float
        System parameters.
    """

    #: State dimension, always 2. State is ``(x, v)``.
    dim = 2

    def __init__(
        self,
        delta: float = 0.3,
        alpha: float = -1.0,
        beta: float = 1.0,
        gamma: float = 0.37,
        omega: float = 1.2,
    ):
        self.delta, self.alpha, self.beta, self.gamma, self.omega = (
            float(delta),
            float(alpha),
            float(beta),
            float(gamma),
            float(omega),
        )

    @property
    def params(self) -> NDArray[np.float64]:
        """Parameter vector ``(delta, alpha, beta, gamma, omega)``.

        Returns
        -------
        ndarray of float, shape (5,)
        """
        return np.array([self.delta, self.alpha, self.beta, self.gamma, self.omega])

    def rhs(self, state: NDArray[np.float64], t: float) -> NDArray[np.float64]:
        """Evaluate the Duffing vector field.

        Parameters
        ----------
        state : ndarray of float, shape (2,)
            State vector ``(x, v)``.
        t : float
            Current time (the forcing term depends on `t`).

        Returns
        -------
        ndarray of float, shape (2,)
            Time derivative ``(dx/dt, dv/dt)``.
        """
        return np.asarray(_duffing_rhs(np.asarray(state, dtype=np.float64), t, self.params))

    def initial_state(self) -> NDArray[np.float64]:
        """Default initial condition ``(1, 0)``.

        Returns
        -------
        ndarray of float, shape (2,)
        """
        return np.array([1.0, 0.0])

    def trajectory(
        self,
        state0: NDArray[np.float64] | None = None,
        t0: float = 0.0,
        dt: float = 0.01,
        n_steps: int = 10000,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Integrate a trajectory with the Numba-accelerated RK4 integrator.

        Parameters
        ----------
        state0 : array_like of float, shape (2,), optional
            Initial state; defaults to :meth:`initial_state`.
        t0 : float, default 0.0
            Initial time.
        dt : float, default 0.01
            Integration step size.
        n_steps : int, default 10000
            Number of integration steps.

        Returns
        -------
        times : ndarray of float, shape (n_steps + 1,)
        states : ndarray of float, shape (n_steps + 1, 2)
        """
        state0 = self.initial_state() if state0 is None else np.asarray(state0, dtype=np.float64)
        return rk4_integrate(_duffing_rhs, state0, t0, dt, n_steps, self.params)


# ---------------------------------------------------------------------------
# Forced Van der Pol oscillator
# ---------------------------------------------------------------------------


@njit(cache=True, inline="always")
def _forced_van_der_pol_accel(x: float, v: float, t: float, mu: float, A: float, omega: float) -> float:
    return mu * (1.0 - x * x) * v - x + A * np.cos(omega * t)


@njit(cache=True)
def _forced_van_der_pol_rhs(state: NDArray[np.float64], t: float, params: NDArray[np.float64]) -> NDArray[np.float64]:
    """Forced Van der Pol vector field ``dx/dt = f(x, t; mu, A, omega)``.

    Parameters
    ----------
    state : ndarray of float, shape (2,)
        State vector ``(x, v)``.
    t : float
        Current time (the forcing term depends on `t`).
    params : ndarray of float, shape (3,)
        Parameters ``(mu, A, omega)``.

    Returns
    -------
    ndarray of float, shape (2,)
        Time derivative ``(dx/dt, dv/dt)``.
    """
    out = np.empty(2)
    out[0] = state[1]
    out[1] = _forced_van_der_pol_accel(state[0], state[1], t, params[0], params[1], params[2])
    return out


@njit(cache=True, parallel=True)
def _forced_van_der_pol_strobe(states0: NDArray[np.float64], n_periods: int, steps_per_period: int, params: NDArray[np.float64]) -> NDArray[np.float64]:
    # Scalar RK4 (no per-step array allocation, which would serialize the
    # parallel threads on the allocator).
    mu, A, omega = params[0], params[1], params[2]
    n = states0.shape[0]
    dt = 2.0 * np.pi / omega / steps_per_period
    h = 0.5 * dt
    out = np.empty((n, n_periods + 1, 2))
    for i in prange(n):  # type: ignore[attr-defined]  # numba lacks type stubs for prange
        x, v = states0[i, 0], states0[i, 1]
        out[i, 0, 0], out[i, 0, 1] = x, v
        for p in range(n_periods):
            for s in range(steps_per_period):
                # Time measured from the start of the current forcing period
                # (the forcing is T-periodic), so it never accumulates error.
                t = s * dt
                k1x, k1v = v, _forced_van_der_pol_accel(x, v, t, mu, A, omega)
                k2x, k2v = v + h * k1v, _forced_van_der_pol_accel(x + h * k1x, v + h * k1v, t + h, mu, A, omega)
                k3x, k3v = v + h * k2v, _forced_van_der_pol_accel(x + h * k2x, v + h * k2v, t + h, mu, A, omega)
                k4x, k4v = v + dt * k3v, _forced_van_der_pol_accel(x + dt * k3x, v + dt * k3v, t + dt, mu, A, omega)
                x += dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
                v += dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
            out[i, p + 1, 0], out[i, p + 1, 1] = x, v
    return out


class ForcedVanDerPol(DynamicalSystem):
    """The sinusoidally forced Van der Pol oscillator.

    Governed by ``x'' - mu*(1 - x^2)*x' + x = A*cos(omega*t)``: a
    self-excited (negatively damped at small amplitude) valve-circuit
    oscillator driven by a periodic signal. With ``A = 0`` it is the plain
    Van der Pol oscillator, whose limit cycle has amplitude close to 2 for
    small `mu` and becomes a relaxation oscillation of period roughly
    ``(3 - 2 ln 2) mu`` for large `mu`.

    Cartwright & Littlewood (1945) proved, in the scaling
    ``A = b * omega * mu`` with `mu` large, that for a range of `b` the
    forced oscillator has two coexisting stable periodic motions whose
    periods are different odd multiples of the forcing period
    ``2*pi/omega``, plus an invariant "bad" set of infinitely many periodic
    and uncountably many non-periodic orbits -- the first proof of chaos in
    an equation from physics. The defaults below are one such parameter set
    (``b = 0.58``), where stable subharmonics of period ``3T`` and ``5T``
    coexist.

    Parameters
    ----------
    mu : float, default 10.0
        Nonlinear damping strength (Cartwright and Littlewood's `k`).
    A : float, default 14.5
        Forcing amplitude.
    omega : float, default 2.5
        Forcing angular frequency (Cartwright and Littlewood's `lambda`).

    Attributes
    ----------
    mu, A, omega : float
        System parameters.

    References
    ----------
    M. L. Cartwright and J. E. Littlewood, "On Non-Linear Differential
    Equations of the Second Order: I. The Equation
    y'' - k(1 - y^2)y' + y = b lambda k cos(lambda t + a), k Large,"
    *J. London Math. Soc.* **20**, 180-189 (1945).

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.chaos.systems.continuous import ForcedVanDerPol
    >>> system = ForcedVanDerPol(mu=1.0, A=0.5, omega=2.0)
    >>> system.rhs(np.array([0.0, 1.0]), 0.0).tolist()
    [1.0, 1.5]
    >>> system.forcing_period == 2 * np.pi / 2.0
    True
    """

    #: State dimension, always 2. State is ``(x, v)``.
    dim = 2

    def __init__(self, mu: float = 10.0, A: float = 14.5, omega: float = 2.5):
        self.mu = float(mu)
        self.A = float(A)
        self.omega = float(omega)

    @property
    def params(self) -> NDArray[np.float64]:
        """Parameter vector ``(mu, A, omega)``.

        Returns
        -------
        ndarray of float, shape (3,)
        """
        return np.array([self.mu, self.A, self.omega])

    @property
    def forcing_period(self) -> float:
        """Period ``T = 2*pi/omega`` of the forcing.

        Returns
        -------
        float
        """
        return 2.0 * np.pi / self.omega

    def rhs(self, state: NDArray[np.float64], t: float) -> NDArray[np.float64]:
        """Evaluate the forced Van der Pol vector field.

        Parameters
        ----------
        state : ndarray of float, shape (2,)
            State vector ``(x, v)``.
        t : float
            Current time (the forcing term depends on `t`).

        Returns
        -------
        ndarray of float, shape (2,)
            Time derivative ``(dx/dt, dv/dt)``.
        """
        return np.asarray(_forced_van_der_pol_rhs(np.asarray(state, dtype=np.float64), t, self.params))

    def initial_state(self) -> NDArray[np.float64]:
        """Default initial condition ``(1, 0)``.

        Returns
        -------
        ndarray of float, shape (2,)
        """
        return np.array([1.0, 0.0])

    def trajectory(
        self,
        state0: NDArray[np.float64] | None = None,
        t0: float = 0.0,
        dt: float = 0.005,
        n_steps: int = 20000,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Integrate a trajectory with the Numba-accelerated RK4 integrator.

        Parameters
        ----------
        state0 : array_like of float, shape (2,), optional
            Initial state; defaults to :meth:`initial_state`.
        t0 : float, default 0.0
            Initial time.
        dt : float, default 0.005
            Integration step size (the relaxation jumps at large `mu` need
            ``dt`` well below ``1/mu``).
        n_steps : int, default 20000
            Number of integration steps.

        Returns
        -------
        times : ndarray of float, shape (n_steps + 1,)
        states : ndarray of float, shape (n_steps + 1, 2)
        """
        state0 = self.initial_state() if state0 is None else np.asarray(state0, dtype=np.float64)
        return rk4_integrate(_forced_van_der_pol_rhs, state0, t0, dt, n_steps, self.params)

    def stroboscopic_map(
        self,
        states0: ArrayLike,
        n_periods: int,
        steps_per_period: int = 400,
    ) -> NDArray[np.float64]:
        """Sample many trajectories once per forcing period, in parallel.

        Every trajectory starts at ``t = 0`` and is integrated with RK4 at a
        fixed step ``forcing_period / steps_per_period``, recording the state
        at ``t = 0, T, 2T, ...``. This stroboscopic (Poincare) map turns the
        periodic orbits of the flow into finite cycles of points -- a
        subharmonic of period ``nT`` becomes an `n`-cycle -- which is how
        Cartwright and Littlewood's coexisting periodic motions are told
        apart. Many initial conditions (e.g. a basin-of-attraction grid)
        are integrated at once on all cores.

        Parameters
        ----------
        states0 : array_like of float, shape (n, 2) or (2,)
            Initial states ``(x, v)`` at ``t = 0``.
        n_periods : int
            Number of forcing periods to integrate.
        steps_per_period : int, default 400
            RK4 steps per forcing period.

        Returns
        -------
        ndarray of float, shape (n, n_periods + 1, 2)
            ``out[i, k]`` is the state of trajectory `i` at ``t = k*T``
            (``out[i, 0]`` is its initial state). A single ``(2,)`` input
            still returns a leading axis of length 1.

        Examples
        --------
        >>> import numpy as np
        >>> from physicskit.chaos.systems.continuous import ForcedVanDerPol
        >>> linear = ForcedVanDerPol(mu=0.0, A=0.0, omega=1.0)
        >>> out = linear.stroboscopic_map([1.0, 0.0], n_periods=3)
        >>> out.shape
        (1, 4, 2)
        >>> bool(np.allclose(out[0, -1], [1.0, 0.0], atol=1e-8))  # x = cos(t) has period T
        True
        """
        states = np.atleast_2d(np.asarray(states0, dtype=np.float64))
        if states.ndim != 2 or states.shape[1] != 2:
            raise ValueError(f"states0 must have shape (n, 2) or (2,), got {np.shape(states0)}")
        return np.asarray(_forced_van_der_pol_strobe(np.ascontiguousarray(states), int(n_periods), int(steps_per_period), self.params))


# ---------------------------------------------------------------------------
# Chua's circuit
# ---------------------------------------------------------------------------


@njit(cache=True)
def _chua_rhs(state: NDArray[np.float64], t: float, params: NDArray[np.float64]) -> NDArray[np.float64]:
    """Chua's circuit vector field ``dx/dt = f(x, t; alpha, beta, m0, m1)``.

    Parameters
    ----------
    state : ndarray of float, shape (3,)
        State vector ``(x, y, z)``.
    t : float
        Current time (unused; the system is autonomous).
    params : ndarray of float, shape (4,)
        Parameters ``(alpha, beta, m0, m1)``.

    Returns
    -------
    ndarray of float, shape (3,)
        Time derivative ``(dx/dt, dy/dt, dz/dt)``.
    """
    alpha, beta, m0, m1 = params[0], params[1], params[2], params[3]
    x, y, z = state[0], state[1], state[2]
    h = m1 * x + 0.5 * (m0 - m1) * (abs(x + 1.0) - abs(x - 1.0))
    out = np.empty(3)
    out[0] = alpha * (y - x - h)
    out[1] = x - y + z
    out[2] = -beta * y
    return out


class Chua(DynamicalSystem):
    """Chua's circuit: a simple chaotic electronic oscillator.

    Built from just a resistor, two capacitors, an inductor, and one
    piecewise-linear nonlinear resistor (the "Chua diode", ``h`` below), this
    is one of the simplest physical systems known to be chaotic, and the
    first to have its chaos confirmed experimentally in real hardware. For
    the classic parameters below it produces the famous *double-scroll*
    attractor: two spiral lobes, with the trajectory unpredictably switching
    between them.

    .. math::

        \\dot{x} &= \\alpha (y - x - h(x)) \\\\
        \\dot{y} &= x - y + z \\\\
        \\dot{z} &= -\\beta y \\\\
        h(x) &= m_1 x + \\tfrac{1}{2}(m_0 - m_1)(|x + 1| - |x - 1|)

    Parameters
    ----------
    alpha : float, default 15.6
        Ratio of the two capacitances.
    beta : float, default 28.0
        Ratio involving the inductance and second capacitance.
    m0 : float, default -8/7
        Inner (small-``|x|``) slope of the Chua diode's piecewise-linear
        characteristic.
    m1 : float, default -5/7
        Outer (large-``|x|``) slope of the Chua diode's piecewise-linear
        characteristic.

    Attributes
    ----------
    alpha, beta, m0, m1 : float
        System parameters.
    """

    #: State dimension, always 3.
    dim = 3

    def __init__(
        self,
        alpha: float = 15.6,
        beta: float = 28.0,
        m0: float = -8.0 / 7.0,
        m1: float = -5.0 / 7.0,
    ):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.m0 = float(m0)
        self.m1 = float(m1)

    @property
    def params(self) -> NDArray[np.float64]:
        """Parameter vector ``(alpha, beta, m0, m1)``.

        Returns
        -------
        ndarray of float, shape (4,)
        """
        return np.array([self.alpha, self.beta, self.m0, self.m1])

    def rhs(self, state: NDArray[np.float64], t: float) -> NDArray[np.float64]:
        """Evaluate Chua's circuit vector field.

        Parameters
        ----------
        state : ndarray of float, shape (3,)
            State vector ``(x, y, z)``.
        t : float
            Current time (unused; the system is autonomous).

        Returns
        -------
        ndarray of float, shape (3,)
            Time derivative ``(dx/dt, dy/dt, dz/dt)``.
        """
        return np.asarray(_chua_rhs(np.asarray(state, dtype=np.float64), t, self.params))

    def initial_state(self) -> NDArray[np.float64]:
        """Default initial condition, slightly off the unstable origin.

        Returns
        -------
        ndarray of float, shape (3,)
        """
        return np.array([0.7, 0.0, 0.0])

    def trajectory(
        self,
        state0: NDArray[np.float64] | None = None,
        t0: float = 0.0,
        dt: float = 0.01,
        n_steps: int = 20000,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Integrate a trajectory with the Numba-accelerated RK4 integrator.

        Parameters
        ----------
        state0 : array_like of float, shape (3,), optional
            Initial state; defaults to :meth:`initial_state`.
        t0 : float, default 0.0
            Initial time.
        dt : float, default 0.01
            Integration step size.
        n_steps : int, default 20000
            Number of integration steps.

        Returns
        -------
        times : ndarray of float, shape (n_steps + 1,)
        states : ndarray of float, shape (n_steps + 1, 3)
        """
        state0 = self.initial_state() if state0 is None else np.asarray(state0, dtype=np.float64)
        return rk4_integrate(_chua_rhs, state0, t0, dt, n_steps, self.params)


# ---------------------------------------------------------------------------
# Restricted three-body problem
# ---------------------------------------------------------------------------


@njit(cache=True)
def _restricted_three_body_rhs(state: NDArray[np.float64], t: float, params: NDArray[np.float64]) -> NDArray[np.float64]:
    """CR3BP vector field (planar, rotating frame) ``dx/dt = f(x, t; mu)``.

    Parameters
    ----------
    state : ndarray of float, shape (4,)
        State vector ``(x, y, vx, vy)`` in the rotating frame.
    t : float
        Current time (unused; the system is autonomous in the rotating frame).
    params : ndarray of float, shape (1,)
        Parameters ``(mu,)``.

    Returns
    -------
    ndarray of float, shape (4,)
        Time derivative ``(dx/dt, dy/dt, dvx/dt, dvy/dt)``.
    """
    mu = params[0]
    x, y, vx, vy = state[0], state[1], state[2], state[3]
    r1 = np.sqrt((x + mu) ** 2 + y * y)
    r2 = np.sqrt((x - 1.0 + mu) ** 2 + y * y)
    out = np.empty(4)
    out[0] = vx
    out[1] = vy
    out[2] = 2.0 * vy + x - (1.0 - mu) * (x + mu) / r1**3 - mu * (x - 1.0 + mu) / r2**3
    out[3] = -2.0 * vx + y - (1.0 - mu) * y / r1**3 - mu * y / r2**3
    return out


class RestrictedThreeBody(DynamicalSystem):
    """The planar circular restricted three-body problem (CR3BP).

    A massless third body moves under the gravity of two massive primaries
    (masses ``1 - mu`` and ``mu``, in normalized units) that are themselves in
    a fixed circular orbit about their common center of mass. In the
    rotating (co-precessing) reference frame, the primaries sit fixed at
    ``(-mu, 0)`` and ``(1 - mu, 0)``, and the third body's motion picks up
    centrifugal and Coriolis terms alongside the two gravitational pulls.

    This system is what led Poincare to the first discovery of deterministic
    chaos: he found that, unlike the exactly solvable two-body problem, CR3BP
    trajectories can depend on initial conditions in an essentially
    unpredictable way. It also genuinely coexists with regular
    (quasi-periodic, KAM-stable) motion -- the default initial condition
    below is the classic *Arenstorf orbit*, a stable *periodic* orbit famous
    in the numerical-methods literature as an ODE-solver stress test (it
    passes very close to the smaller primary); see the example gallery for a
    nearby, only slightly perturbed initial condition that is chaotic instead.

    Because of the velocity-dependent Coriolis terms, this system is *not* a
    separable Hamiltonian of the form ``pos'' = force(pos, t)``, so it must
    be integrated with :func:`physicskit.chaos.core.integrators.rk4_integrate`
    (the symplectic :func:`~physicskit.chaos.core.integrators.leapfrog_integrate` /
    :func:`~physicskit.chaos.core.integrators.yoshida4_integrate` do not apply).

    Parameters
    ----------
    mu : float, default 0.012277471
        Mass parameter (mass of the smaller primary, in units where the
        total mass is 1); the default is the Earth-Moon-like value used in
        the classic Arenstorf orbit.

    Attributes
    ----------
    mu : float
        Mass parameter.
    """

    #: State dimension, always 4. State is ``(x, y, vx, vy)``.
    dim = 4

    def __init__(self, mu: float = 0.012277471):
        self.mu = float(mu)

    @property
    def params(self) -> NDArray[np.float64]:
        """Parameter vector ``(mu,)``.

        Returns
        -------
        ndarray of float, shape (1,)
        """
        return np.array([self.mu])

    def rhs(self, state: NDArray[np.float64], t: float) -> NDArray[np.float64]:
        """Evaluate the CR3BP vector field.

        Parameters
        ----------
        state : ndarray of float, shape (4,)
            State vector ``(x, y, vx, vy)``.
        t : float
            Current time (unused; the system is autonomous in the rotating
            frame).

        Returns
        -------
        ndarray of float, shape (4,)
            Time derivative ``(dx/dt, dy/dt, dvx/dt, dvy/dt)``.
        """
        return np.asarray(_restricted_three_body_rhs(np.asarray(state, dtype=np.float64), t, self.params))

    def initial_state(self) -> NDArray[np.float64]:
        """The classic Arenstorf periodic-orbit initial condition.

        Returns
        -------
        ndarray of float, shape (4,)
        """
        return np.array([0.994, 0.0, 0.0, -2.00158510637908252240537862224])

    def trajectory(
        self,
        state0: NDArray[np.float64] | None = None,
        t0: float = 0.0,
        dt: float = 0.0005,
        n_steps: int = 40000,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Integrate a trajectory with the Numba-accelerated RK4 integrator.

        Parameters
        ----------
        state0 : array_like of float, shape (4,), optional
            Initial state; defaults to :meth:`initial_state`.
        t0 : float, default 0.0
            Initial time.
        dt : float, default 0.0005
            Integration step size (small, since the Arenstorf orbit's default
            initial condition passes very close to the smaller primary).
        n_steps : int, default 40000
            Number of integration steps.

        Returns
        -------
        times : ndarray of float, shape (n_steps + 1,)
        states : ndarray of float, shape (n_steps + 1, 4)
        """
        state0 = self.initial_state() if state0 is None else np.asarray(state0, dtype=np.float64)
        return rk4_integrate(_restricted_three_body_rhs, state0, t0, dt, n_steps, self.params)

    def jacobi_constant(self, state: NDArray[np.float64]) -> float:
        """The Jacobi constant: the CR3BP's conserved rotating-frame energy analog.

        Parameters
        ----------
        state : array_like of float, shape (4,)
            State vector ``(x, y, vx, vy)``.

        Returns
        -------
        float
            The Jacobi constant ``C = 2*Omega(x, y) - (vx^2 + vy^2)``, where
            ``Omega`` is the effective (gravitational + centrifugal)
            potential. Conserved along any trajectory; useful for checking
            integrator fidelity (e.g. with
            :func:`physicskit.chaos.utils.metrics.energy_drift`).
        """
        mu = self.mu
        x, y, vx, vy = state
        omega = effective_potential(x, y, mu)
        return float(2.0 * omega - (vx * vx + vy * vy))


def effective_potential(x: NDArray[np.float64] | float, y: NDArray[np.float64] | float, mu: float) -> NDArray[np.float64]:
    """The CR3BP's effective (gravitational + centrifugal) potential ``Omega(x, y)``.

    Vectorized over `x`/`y` (works directly on ``np.meshgrid`` output), so it
    is the building block for both
    :meth:`RestrictedThreeBody.jacobi_constant` and that orbit's *zero-velocity
    curve*: since ``C = 2*Omega(x, y) - speed**2``, a trajectory with Jacobi
    constant `C` can only reach points where ``Omega(x, y) >= C / 2`` (speed
    would otherwise be imaginary) -- the boundary ``Omega(x, y) == C / 2`` is
    a curve the body can never cross, carving the plane into allowed and
    forbidden ("Hill") regions.

    Parameters
    ----------
    x, y : float or ndarray of float
        Rotating-frame position(s).
    mu : float
        CR3BP mass parameter.

    Returns
    -------
    ndarray of float
        ``Omega(x, y) = 0.5*(x^2 + y^2) + (1 - mu)/r1 + mu/r2``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    r1 = np.sqrt((x + mu) ** 2 + y * y)
    r2 = np.sqrt((x - 1.0 + mu) ** 2 + y * y)
    return 0.5 * (x * x + y * y) + (1.0 - mu) / r1 + mu / r2


def lagrange_points(mu: float) -> NDArray[np.float64]:
    """The five Lagrange (libration) equilibrium points of the CR3BP, in
    the rotating frame, for mass parameter `mu`.

    L1, L2, L3 are the collinear equilibria on the ``y = 0`` axis: points
    where a body held at rest (``vx = vy = 0``) in the rotating frame feels
    zero net acceleration, found here by root-finding
    :func:`_restricted_three_body_rhs`'s ``ax`` output directly (rather
    than re-deriving the quintic equations by hand) in the three intervals
    it is known to have exactly one root each: between the primaries
    (L1), beyond the smaller primary (L2), and beyond the larger primary,
    on the opposite side (L3). L4 and L5 form equilateral triangles with
    the two primaries and have the closed-form rotating-frame position
    ``(0.5 - mu, +-sqrt(3)/2)``.

    Parameters
    ----------
    mu : float
        CR3BP mass parameter.

    Returns
    -------
    ndarray of float, shape (5, 2)
        ``(x, y)`` of L1 through L5, in that order.
    """

    def ax_residual(x: float) -> float:
        state = np.array([x, 0.0, 0.0, 0.0])
        return float(_restricted_three_body_rhs(state, 0.0, np.array([mu]))[2])

    eps = 1e-8
    x_l1 = brentq(ax_residual, -mu + eps, 1.0 - mu - eps)
    x_l2 = brentq(ax_residual, 1.0 - mu + eps, 1.0 - mu + 2.0)
    x_l3 = brentq(ax_residual, -mu - 2.0, -mu - eps)
    return np.array(
        [
            [x_l1, 0.0],
            [x_l2, 0.0],
            [x_l3, 0.0],
            [0.5 - mu, np.sqrt(3.0) / 2.0],
            [0.5 - mu, -np.sqrt(3.0) / 2.0],
        ]
    )


# ---------------------------------------------------------------------------
# Magnetic pendulum
# ---------------------------------------------------------------------------


@njit(cache=True)
def magnetic_pendulum_rhs(state: NDArray[np.float64], t: float, params: NDArray[np.float64]) -> NDArray[np.float64]:
    """Magnetic pendulum vector field ``dx/dt = f(x, t; params)``.

    Public (not underscore-prefixed) so it can be passed directly to the
    low-level, Numba-accelerated integrators in
    :mod:`physicskit.chaos.core.integrators` for high-throughput grid computations
    such as :func:`physicskit.chaos.visualizers.basins.basin_of_attraction`, which
    needs to integrate one trajectory per pixel of a basin map.

    Parameters
    ----------
    state : ndarray of float, shape (4,)
        State vector ``(x, y, vx, vy)``.
    t : float
        Current time (unused; the system is autonomous).
    params : ndarray of float, shape (4 + 2*n_magnets,)
        ``(friction, spring, height, strength, mx_0, my_0, mx_1, my_1, ...)``:
        see :class:`MagneticPendulum` for the meaning of each.

    Returns
    -------
    ndarray of float, shape (4,)
        Time derivative ``(dx/dt, dy/dt, dvx/dt, dvy/dt)``.
    """
    friction, spring, height, strength = params[0], params[1], params[2], params[3]
    x, y, vx, vy = state[0], state[1], state[2], state[3]
    n_magnets = (params.shape[0] - 4) // 2

    ax = -friction * vx - spring * x
    ay = -friction * vy - spring * y
    for k in range(n_magnets):
        mx = params[4 + 2 * k]
        my = params[4 + 2 * k + 1]
        dx = x - mx
        dy = y - my
        r3 = (dx * dx + dy * dy + height * height) ** 1.5
        ax -= strength * dx / r3
        ay -= strength * dy / r3

    out = np.empty(4)
    out[0] = vx
    out[1] = vy
    out[2] = ax
    out[3] = ay
    return out


class MagneticPendulum(DynamicalSystem):
    """A pendulum bob swinging over several fixed magnets: a multistable, chaotic system.

    A damped pendulum bob, modeled in the small-swing (flat, 2D) limit, is
    pulled down toward the origin by a linear restoring force and attracted
    toward each of several fixed magnets by an inverse-square-like force
    (softened by a "height" offset `height`, the bob's height above the
    magnet plane, which avoids a force singularity directly above a magnet).
    Friction eventually settles the bob at rest near whichever magnet "won" --
    but *which* magnet wins depends on the starting position with famously
    fractal sensitivity, making this the classic system for visualizing
    fractal basin boundaries; see
    :func:`physicskit.chaos.visualizers.basins.plot_basin_of_attraction`.

    Parameters
    ----------
    magnet_positions : array_like of float, shape (n_magnets, 2), optional
        Magnet ``(x, y)`` positions; defaults to 3 magnets at the vertices of
        an equilateral triangle inscribed in the unit circle.
    friction : float, default 0.2
        Damping coefficient.
    spring : float, default 0.2
        Linear restoring-force coefficient (pulling the bob back toward the
        origin, as in the small-swing limit of gravity).
    height : float, default 0.2
        The bob's height above the magnet plane; softens the force near a
        magnet (larger values give a gentler, less singular pull).
    strength : float, default 1.0
        Magnet attraction strength.

    Attributes
    ----------
    magnet_positions : ndarray of float, shape (n_magnets, 2)
        Magnet positions.
    friction, spring, height, strength : float
        System parameters.
    """

    #: State dimension, always 4. State is ``(x, y, vx, vy)``.
    dim = 4

    def __init__(
        self,
        magnet_positions: NDArray[np.float64] | None = None,
        friction: float = 0.2,
        spring: float = 0.2,
        height: float = 0.2,
        strength: float = 1.0,
    ):
        if magnet_positions is None:
            angles = np.array([90.0, 210.0, 330.0]) * np.pi / 180.0
            magnet_positions = np.column_stack([np.cos(angles), np.sin(angles)])
        self.magnet_positions = np.asarray(magnet_positions, dtype=np.float64)
        self.friction = float(friction)
        self.spring = float(spring)
        self.height = float(height)
        self.strength = float(strength)

    @property
    def params(self) -> NDArray[np.float64]:
        """Parameter vector ``(friction, spring, height, strength, mx_0, my_0, ...)``.

        Returns
        -------
        ndarray of float, shape (4 + 2*n_magnets,)
        """
        return np.concatenate(
            [
                [self.friction, self.spring, self.height, self.strength],
                self.magnet_positions.ravel(),
            ]
        )

    def rhs(self, state: NDArray[np.float64], t: float) -> NDArray[np.float64]:
        """Evaluate the magnetic pendulum vector field.

        Parameters
        ----------
        state : ndarray of float, shape (4,)
            State vector ``(x, y, vx, vy)``.
        t : float
            Current time (unused; the system is autonomous).

        Returns
        -------
        ndarray of float, shape (4,)
            Time derivative ``(dx/dt, dy/dt, dvx/dt, dvy/dt)``.
        """
        return np.asarray(magnetic_pendulum_rhs(np.asarray(state, dtype=np.float64), t, self.params))

    def initial_state(self) -> NDArray[np.float64]:
        """Default initial condition: released from rest, off-center.

        Returns
        -------
        ndarray of float, shape (4,)
        """
        return np.array([0.5, 0.5, 0.0, 0.0])

    def trajectory(
        self,
        state0: NDArray[np.float64] | None = None,
        t0: float = 0.0,
        dt: float = 0.02,
        n_steps: int = 5000,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Integrate a trajectory with the Numba-accelerated RK4 integrator.

        Parameters
        ----------
        state0 : array_like of float, shape (4,), optional
            Initial state; defaults to :meth:`initial_state`.
        t0 : float, default 0.0
            Initial time.
        dt : float, default 0.02
            Integration step size.
        n_steps : int, default 5000
            Number of integration steps.

        Returns
        -------
        times : ndarray of float, shape (n_steps + 1,)
        states : ndarray of float, shape (n_steps + 1, 4)
        """
        state0 = self.initial_state() if state0 is None else np.asarray(state0, dtype=np.float64)
        return rk4_integrate(magnetic_pendulum_rhs, state0, t0, dt, n_steps, self.params)


# ---------------------------------------------------------------------------
# Driven, damped pendulum
# ---------------------------------------------------------------------------


@njit(cache=True)
def _driven_pendulum_rhs(state: NDArray[np.float64], t: float, params: NDArray[np.float64]) -> NDArray[np.float64]:
    """Driven-pendulum vector field ``dx/dt = f(x, t; damping, g_over_l, A, omega_d)``.

    Parameters
    ----------
    state : ndarray of float, shape (2,)
        State vector ``(theta, omega)``.
    t : float
        Current time (the forcing term depends on `t`).
    params : ndarray of float, shape (4,)
        Parameters ``(damping, g_over_l, A, omega_d)``.

    Returns
    -------
    ndarray of float, shape (2,)
        Time derivative ``(dtheta/dt, domega/dt)``.
    """
    damping, g_over_l, A, omega_d = params[0], params[1], params[2], params[3]
    theta, omega = state[0], state[1]
    out = np.empty(2)
    out[0] = omega
    out[1] = -damping * omega - g_over_l * np.sin(theta) + A * np.cos(omega_d * t)
    return out


class DrivenPendulum(DynamicalSystem):
    """The damped, sinusoidally forced pendulum.

    Governed by ``theta'' + damping*theta' + (g/l)*sin(theta) = A*cos(omega_d*t)``
    -- the classic mechanical system for the period-doubling route to chaos
    (Baker & Gollub, *Chaotic Dynamics: An Introduction*): for fixed
    damping and forcing frequency, sweeping the forcing amplitude `A`
    produces the same period-doubling cascade as the logistic map. Feed
    :meth:`rhs` to
    :func:`physicskit.chaos.visualizers.bifurcation.stroboscopic_bifurcation_sampler`
    (sampling `theta` once per forcing period ``2*pi/omega_d``) to build
    that bifurcation diagram directly with
    :func:`physicskit.chaos.visualizers.bifurcation.bifurcation_diagram`.

    Parameters
    ----------
    damping : float, default 0.5
        Viscous damping coefficient.
    g_over_l : float, default 1.0
        ``g/l``, the undamped/undriven pendulum's small-angle frequency
        squared.
    A : float, default 1.5
        Forcing torque amplitude; the default, with the other defaults
        below, is deep in the chaotic regime.
    omega_d : float, default 2.0/3.0
        Forcing angular frequency.

    Attributes
    ----------
    damping, g_over_l, A, omega_d : float
        System parameters.
    """

    #: State dimension, always 2. State is ``(theta, omega)``.
    dim = 2

    def __init__(self, damping: float = 0.5, g_over_l: float = 1.0, A: float = 1.5, omega_d: float = 2.0 / 3.0):
        self.damping = float(damping)
        self.g_over_l = float(g_over_l)
        self.A = float(A)
        self.omega_d = float(omega_d)

    @property
    def params(self) -> NDArray[np.float64]:
        """Parameter vector ``(damping, g_over_l, A, omega_d)``.

        Returns
        -------
        ndarray of float, shape (4,)
        """
        return np.array([self.damping, self.g_over_l, self.A, self.omega_d])

    def rhs(self, state: NDArray[np.float64], t: float) -> NDArray[np.float64]:
        """Evaluate the driven-pendulum vector field.

        Parameters
        ----------
        state : ndarray of float, shape (2,)
            State vector ``(theta, omega)``.
        t : float
            Current time (the forcing term depends on `t`).

        Returns
        -------
        ndarray of float, shape (2,)
            Time derivative ``(dtheta/dt, domega/dt)``.
        """
        return np.asarray(_driven_pendulum_rhs(np.asarray(state, dtype=np.float64), t, self.params))

    def initial_state(self) -> NDArray[np.float64]:
        """Default initial condition: small angle, at rest.

        Returns
        -------
        ndarray of float, shape (2,)
        """
        return np.array([0.2, 0.0])

    def trajectory(
        self,
        state0: NDArray[np.float64] | None = None,
        t0: float = 0.0,
        dt: float = 0.02,
        n_steps: int = 5000,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Integrate a trajectory with the Numba-accelerated RK4 integrator.

        Parameters
        ----------
        state0 : array_like of float, shape (2,), optional
            Initial state; defaults to :meth:`initial_state`.
        t0 : float, default 0.0
            Initial time.
        dt : float, default 0.02
            Integration step size.
        n_steps : int, default 5000
            Number of integration steps.

        Returns
        -------
        times : ndarray of float, shape (n_steps + 1,)
        states : ndarray of float, shape (n_steps + 1, 2)
        """
        state0 = self.initial_state() if state0 is None else np.asarray(state0, dtype=np.float64)
        return rk4_integrate(_driven_pendulum_rhs, state0, t0, dt, n_steps, self.params)
