"""Langevin dynamics: Brownian motion, the Einstein relation, and the Ornstein-Uhlenbeck process.

Langevin (1908) wrote Newton's law for a colloidal particle with the
fluid's effect split into a friction :math:`-\\gamma v` and a white-noise
force whose strength the same friction fixes:

.. math::

    m\\,\\dot v = F(x) - \\gamma v + \\sqrt{2\\gamma k_B T}\\,\\xi(t), \\qquad
    \\langle \\xi(t)\\xi(t')\\rangle = \\delta(t - t').

Dropping inertia gives the overdamped (Brownian) limit
:math:`\\dot x = F/\\gamma + \\sqrt{2D}\\,\\xi`, whose mean-squared
displacement grows as :math:`2Dt` per dimension with the Einstein relation
:math:`D = \\mu k_B T = k_B T/\\gamma` linking diffusion to the mobility
:math:`\\mu = 1/\\gamma` (Einstein 1905). The velocity of the underdamped
particle, or the position in a harmonic trap, is an Ornstein-Uhlenbeck
process (Uhlenbeck and Ornstein 1930).

Units follow the rest of :mod:`physicskit.statphys`: :math:`k_B = 1`, so
``kT`` is the temperature in energy units. All three classes integrate
with :mod:`physicskit.integrators.stochastic`, using module-level
``@njit`` drift/diffusion callbacks and a ``float64`` parameter array, and
draw their noise from a NumPy ``Generator`` seeded by ``seed``.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from physicskit.integrators.stochastic import baoab_integrate, euler_maruyama_integrate

__all__ = [
    "BrownianMotion",
    "LangevinDynamics",
    "OrnsteinUhlenbeck",
    "einstein_diffusion_coefficient",
    "stokes_einstein_diffusion_coefficient",
]


def einstein_diffusion_coefficient(kT: float, gamma: float) -> float:
    """Einstein relation :math:`D = \\mu k_B T = k_B T / \\gamma`.

    (A. Einstein, Ann. Phys. 322, 549 (1905), eq. for :math:`D` in §4.)

    Parameters
    ----------
    kT : float
        Temperature in energy units (:math:`k_B = 1`).
    gamma : float
        Friction coefficient, the inverse of the mobility :math:`\\mu`.

    Returns
    -------
    float
        Diffusion coefficient.

    Examples
    --------
    >>> einstein_diffusion_coefficient(kT=2.0, gamma=4.0)
    0.5
    """
    return kT / gamma


def stokes_einstein_diffusion_coefficient(kT: float, viscosity: float, radius: float) -> float:
    """Stokes-Einstein diffusion coefficient of a sphere, :math:`D = k_B T / (6\\pi\\eta a)`.

    Combines the Einstein relation with Stokes' drag
    :math:`\\gamma = 6\\pi\\eta a` on a sphere of radius :math:`a` in a
    fluid of viscosity :math:`\\eta`.

    Parameters
    ----------
    kT : float
        Thermal energy :math:`k_B T` (in joules for SI inputs).
    viscosity : float
        Dynamic viscosity :math:`\\eta`.
    radius : float
        Sphere radius :math:`a`.

    Returns
    -------
    float
        Diffusion coefficient.

    Examples
    --------
    A 0.5 micron bead in room-temperature water diffuses about 0.43
    square microns per second:

    >>> D = stokes_einstein_diffusion_coefficient(1.380649e-23 * 293.0, 1.0e-3, 0.5e-6)
    >>> round(D * 1e12, 2)
    0.43
    """
    return kT / (6.0 * np.pi * viscosity * radius)


# params = [gamma, kT, stiffness, dim, F_0, ..., F_{dim-1}]
@njit(cache=True)
def _overdamped_drift(x, t, p):
    gamma = p[0]
    k = p[2]
    dim = int(p[3])
    out = -k * x / gamma
    for i in range(x.shape[0]):
        out[i] += p[4 + i % dim] / gamma
    return out


@njit(cache=True)
def _overdamped_diffusion(x, t, p):
    return np.sqrt(2.0 * p[1] / p[0]) * np.ones_like(x)


# state = [x (n), v (n)]; params = [mass, gamma, kT, stiffness, dim, F_0, ..., F_{dim-1}]
@njit(cache=True)
def _underdamped_drift(s, t, p):
    m, gamma, k = p[0], p[1], p[3]
    dim = int(p[4])
    n = s.shape[0] // 2
    out = np.empty_like(s)
    for i in range(n):
        out[i] = s[n + i]
        out[n + i] = (p[5 + i % dim] - k * s[i] - gamma * s[n + i]) / m
    return out


@njit(cache=True)
def _underdamped_force(x, t, p):
    k = p[3]
    dim = int(p[4])
    out = -k * x
    for i in range(x.shape[0]):
        out[i] += p[5 + i % dim]
    return out


@njit(cache=True)
def _underdamped_diffusion(s, t, p):
    n = s.shape[0] // 2
    out = np.zeros_like(s)
    out[n:] = np.sqrt(2.0 * p[1] * p[2]) / p[0]
    return out


# params = [theta, mu, sigma]
@njit(cache=True)
def _ou_drift(x, t, p):
    return p[0] * (p[1] - x)


@njit(cache=True)
def _ou_diffusion(x, t, p):
    return p[2] * np.ones_like(x)


def _force_vector(force, dim: int) -> np.ndarray:
    f = np.broadcast_to(np.asarray(force, dtype=np.float64), (dim,))
    return np.array(f)


def _run_frames(drift, diffusion, state0, params, t_max, dt, n_frames, rng):
    """Integrate in ``n_frames`` chunks, drawing noise per chunk, keeping one state per frame."""
    stride = max(1, int(round(t_max / (n_frames * dt))))
    ndof = state0.shape[0]
    states = np.empty((n_frames + 1, ndof))
    states[0] = state0
    state = state0.copy()
    sqdt = np.sqrt(dt)
    for f in range(n_frames):
        dW = sqdt * rng.standard_normal((stride, ndof))
        _, traj = euler_maruyama_integrate(drift, diffusion, state, f * stride * dt, dt, dW, params, stride)
        state = traj[-1]
        states[f + 1] = state
    times = np.arange(n_frames + 1) * stride * dt
    return times, states


class BrownianMotion:
    """Overdamped Langevin (Brownian) particles, optionally driven and trapped.

    Each coordinate obeys

    .. math::

        dx = \\frac{F - kx}{\\gamma}\\,dt + \\sqrt{\\frac{2k_BT}{\\gamma}}\\,dW,

    with constant force :math:`F`, trap stiffness :math:`k`, and friction
    :math:`\\gamma` (the inverse mobility). Free particles
    (:math:`F = k = 0`) diffuse with :math:`\\langle \\Delta x^2\\rangle =
    2Dt` per dimension and :math:`D = k_BT/\\gamma` (Einstein 1905); a
    constant force adds a drift :math:`F/\\gamma` without changing the
    spread. Euler-Maruyama is exact in distribution for :math:`k = 0`,
    since the drift is then constant.

    Parameters
    ----------
    n_particles : int, default=1000
        Number of independent particles.
    dim : int, default=1
        Spatial dimension.
    gamma : float, default=1.0
        Friction coefficient.
    kT : float, default=1.0
        Temperature (:math:`k_B = 1`).
    force : float or array_like of shape (dim,), default=0.0
        Constant external force.
    stiffness : float, default=0.0
        Harmonic trap stiffness :math:`k` (trap centered at the origin).
    seed : int, optional
        Seed for the noise generator.

    Attributes
    ----------
    times : ndarray or None
        Recorded times, set by :meth:`run`.
    positions : ndarray of shape (n_frames + 1, n_particles, dim) or None
        Recorded positions, set by :meth:`run`.

    Examples
    --------
    >>> bm = BrownianMotion(n_particles=4000, dim=2, gamma=2.0, kT=1.0, seed=0)
    >>> t, x = bm.run(t_max=5.0, dt=0.01, n_frames=50)
    >>> x.shape
    (51, 4000, 2)
    >>> round(bm.diffusion_coefficient, 3), round(bm.measured_diffusion_coefficient(), 1)
    (0.5, 0.5)
    """

    def __init__(self, n_particles=1000, dim=1, gamma=1.0, kT=1.0, force=0.0, stiffness=0.0, seed=None):
        if gamma <= 0 or kT < 0:
            raise ValueError("gamma must be positive and kT non-negative")
        self.n_particles = int(n_particles)
        self.dim = int(dim)
        self.gamma = float(gamma)
        self.kT = float(kT)
        self.force = _force_vector(force, self.dim)
        self.stiffness = float(stiffness)
        self.rng = np.random.default_rng(seed)
        self.params = np.concatenate([[self.gamma, self.kT, self.stiffness, self.dim], self.force])
        self._drift_njit = _overdamped_drift
        self._diffusion_njit = _overdamped_diffusion
        self.times = None
        self.positions = None

    @property
    def diffusion_coefficient(self) -> float:
        """Einstein's :math:`D = k_BT/\\gamma`."""
        return einstein_diffusion_coefficient(self.kT, self.gamma)

    @property
    def mobility(self) -> float:
        """Mobility :math:`\\mu = 1/\\gamma`: drift velocity per unit force."""
        return 1.0 / self.gamma

    def run(self, t_max, dt, n_frames=200, x0=None):
        """Integrate every particle from ``x0`` to ``t_max``.

        Parameters
        ----------
        t_max : float
            Total time (rounded to a whole number of steps per frame).
        dt : float
            Time step.
        n_frames : int, default=200
            Number of recorded frames after the initial one.
        x0 : array_like of shape (n_particles, dim), optional
            Initial positions; defaults to the origin.

        Returns
        -------
        times : ndarray of shape (n_frames + 1,)
        positions : ndarray of shape (n_frames + 1, n_particles, dim)
        """
        start = np.zeros((self.n_particles, self.dim)) if x0 is None else np.array(x0, dtype=np.float64).reshape(self.n_particles, self.dim)
        times, states = _run_frames(self._drift_njit, self._diffusion_njit, start.ravel(), self.params, t_max, dt, n_frames, self.rng)
        self.times = times
        self.positions = states.reshape(n_frames + 1, self.n_particles, self.dim)
        return self.times, self.positions

    def _require_run(self) -> tuple[np.ndarray, np.ndarray]:
        if self.positions is None or self.times is None:
            raise RuntimeError("call run() first")
        return self.times, self.positions

    def mean_squared_displacement(self):
        """Ensemble mean-squared displacement from the initial positions, summed over dimensions.

        Returns
        -------
        times : ndarray
        msd : ndarray
            :math:`\\langle |\\mathbf x(t) - \\mathbf x(0)|^2\\rangle`; for free
            particles, :math:`2\\,d\\,D t`.
        """
        times, positions = self._require_run()
        disp = positions - positions[0]
        return times, np.mean(np.sum(disp**2, axis=-1), axis=1)

    def measured_diffusion_coefficient(self) -> float:
        """Diffusion coefficient from the growth of the displacement variance.

        Fits :math:`\\mathrm{Var}[\\Delta x] = 2Dt` (per dimension, through
        the origin) so that a constant-force drift does not bias it.

        Returns
        -------
        float
        """
        t, positions = self._require_run()
        disp = positions - positions[0]
        var = np.mean(np.var(disp, axis=1), axis=-1)
        return float(np.sum(var * t) / (2.0 * np.sum(t * t)))

    def measured_mobility(self) -> float:
        """Mobility from the mean drift velocity along the applied force.

        Returns
        -------
        float
            :math:`\\langle \\Delta\\mathbf x\\rangle\\cdot\\hat{\\mathbf F} /
            (|\\mathbf F|\\, t)` at the last frame; ``nan`` without a force.
        """
        times, positions = self._require_run()
        fnorm = np.linalg.norm(self.force)
        if fnorm == 0:
            return float("nan")
        drift = np.mean(positions[-1] - positions[0], axis=0) @ (self.force / fnorm)
        return float(drift / (fnorm * times[-1]))


class LangevinDynamics:
    """Underdamped Langevin particles: inertia, friction, and thermal noise.

    .. math::

        dx = v\\,dt, \\qquad
        m\\,dv = (F - kx - \\gamma v)\\,dt + \\sqrt{2\\gamma k_BT}\\,dW

    (Langevin 1908). The noise amplitude is tied to the friction by the
    fluctuation-dissipation relation, so every velocity component
    thermalizes to :math:`\\langle v^2\\rangle = k_BT/m` on the time scale
    :math:`m/\\gamma`, and at times long compared with it the particles
    diffuse with Einstein's :math:`D = k_BT/\\gamma`.

    Integrated by default with the BAOAB splitting
    (:func:`~physicskit.integrators.stochastic.baoab_integrate`), which
    stays at the bath temperature for weak friction. Plain Euler-Maruyama
    (``method="euler_maruyama"``) heats a trapped particle of frequency
    :math:`\\omega` to roughly :math:`k_BT\\,\\gamma/(\\gamma - m\\omega^2\\Delta t)`,
    a large error once :math:`\\gamma/m` is comparable to
    :math:`\\omega^2\\Delta t`.

    Parameters
    ----------
    n_particles : int, default=1000
        Number of independent particles.
    dim : int, default=1
        Spatial dimension.
    mass : float, default=1.0
        Particle mass :math:`m`.
    gamma : float, default=1.0
        Friction coefficient.
    kT : float, default=1.0
        Bath temperature (:math:`k_B = 1`).
    force : float or array_like of shape (dim,), default=0.0
        Constant external force.
    stiffness : float, default=0.0
        Harmonic trap stiffness.
    seed : int, optional
        Seed for the noise generator.

    Examples
    --------
    >>> ld = LangevinDynamics(n_particles=5000, mass=1.0, gamma=1.0, kT=0.5, seed=1)
    >>> t, x, v = ld.run(t_max=10.0, dt=0.005, n_frames=20)
    >>> round(ld.kinetic_temperature(), 1)
    0.5
    """

    def __init__(self, n_particles=1000, dim=1, mass=1.0, gamma=1.0, kT=1.0, force=0.0, stiffness=0.0, seed=None):
        if mass <= 0 or gamma <= 0 or kT < 0:
            raise ValueError("mass and gamma must be positive and kT non-negative")
        self.n_particles = int(n_particles)
        self.dim = int(dim)
        self.mass = float(mass)
        self.gamma = float(gamma)
        self.kT = float(kT)
        self.force = _force_vector(force, self.dim)
        self.stiffness = float(stiffness)
        self.rng = np.random.default_rng(seed)
        self.params = np.concatenate([[self.mass, self.gamma, self.kT, self.stiffness, self.dim], self.force])
        self._drift_njit = _underdamped_drift
        self._diffusion_njit = _underdamped_diffusion
        self._force_njit = _underdamped_force
        self.times = None
        self.positions = None
        self.velocities = None

    @property
    def diffusion_coefficient(self) -> float:
        """Long-time Einstein diffusion coefficient :math:`k_BT/\\gamma` (free particles)."""
        return einstein_diffusion_coefficient(self.kT, self.gamma)

    @property
    def velocity_relaxation_time(self) -> float:
        """Momentum relaxation time :math:`m/\\gamma`."""
        return self.mass / self.gamma

    def run(self, t_max, dt, n_frames=200, x0=None, v0=None, method="baoab"):
        """Integrate every particle to ``t_max``.

        Parameters
        ----------
        t_max : float
            Total time.
        dt : float
            Time step, which should be small compared with :math:`m/\\gamma`.
        n_frames : int, default=200
            Number of recorded frames after the initial one.
        x0, v0 : array_like of shape (n_particles, dim), optional
            Initial positions and velocities; default zero (at rest at the
            origin).
        method : {"baoab", "euler_maruyama"}, default="baoab"
            Integration scheme.

        Returns
        -------
        times : ndarray of shape (n_frames + 1,)
        positions, velocities : ndarray of shape (n_frames + 1, n_particles, dim)
        """
        shape = (self.n_particles, self.dim)
        x = np.zeros(shape) if x0 is None else np.array(x0, dtype=np.float64).reshape(shape)
        v = np.zeros(shape) if v0 is None else np.array(v0, dtype=np.float64).reshape(shape)
        n = self.n_particles * self.dim
        if method == "euler_maruyama":
            state0 = np.concatenate([x.ravel(), v.ravel()])
            times, states = _run_frames(self._drift_njit, self._diffusion_njit, state0, self.params, t_max, dt, n_frames, self.rng)
            xs, vs = states[:, :n], states[:, n:]
        elif method == "baoab":
            stride = max(1, int(round(t_max / (n_frames * dt))))
            xs = np.empty((n_frames + 1, n))
            vs = np.empty((n_frames + 1, n))
            xs[0], vs[0] = x.ravel(), v.ravel()
            for f in range(n_frames):
                noise = self.rng.standard_normal((stride, n))
                _, px, pv = baoab_integrate(self._force_njit, xs[f], vs[f], f * stride * dt, dt, self.mass, self.gamma, self.kT, noise, self.params, stride)
                xs[f + 1], vs[f + 1] = px[-1], pv[-1]
            times = np.arange(n_frames + 1) * stride * dt
        else:
            raise ValueError("method must be 'baoab' or 'euler_maruyama'")
        self.times = times
        self.positions = xs.reshape((n_frames + 1,) + shape)
        self.velocities = vs.reshape((n_frames + 1,) + shape)
        return self.times, self.positions, self.velocities

    def kinetic_temperature(self, frame: int = -1) -> float:
        """Kinetic temperature :math:`m\\langle v^2\\rangle` per component at one frame.

        Parameters
        ----------
        frame : int, default=-1
            Frame index.

        Returns
        -------
        float
        """
        if self.velocities is None:
            raise RuntimeError("call run() first")
        return float(self.mass * np.mean(self.velocities[frame] ** 2))

    def mean_squared_displacement(self):
        """Ensemble mean-squared displacement, summed over dimensions.

        Returns
        -------
        times : ndarray
        msd : ndarray
        """
        if self.positions is None:
            raise RuntimeError("call run() first")
        disp = self.positions - self.positions[0]
        return self.times, np.mean(np.sum(disp**2, axis=-1), axis=1)


class OrnsteinUhlenbeck:
    """An ensemble of Ornstein-Uhlenbeck processes, :math:`dX = \\theta(\\mu - X)\\,dt + \\sigma\\,dW`.

    The unique stationary, Gaussian, Markov process (Doob 1942). Started
    at :math:`x_0` it has

    .. math::

        \\langle X_t\\rangle = \\mu + (x_0 - \\mu)e^{-\\theta t}, \\qquad
        \\mathrm{Var}\\,X_t = \\frac{\\sigma^2}{2\\theta}(1 - e^{-2\\theta t}),

    and in the stationary state the autocovariance is
    :math:`\\frac{\\sigma^2}{2\\theta}e^{-\\theta|\\tau|}` (G. E. Uhlenbeck and
    L. S. Ornstein, Phys. Rev. 36, 823 (1930)). It is the velocity of a
    free Langevin particle (:math:`\\theta = \\gamma/m`,
    :math:`\\sigma^2/2\\theta = k_BT/m`) and the position of an overdamped
    particle in a harmonic trap.

    Parameters
    ----------
    theta : float, default=1.0
        Relaxation rate :math:`\\theta > 0`.
    mu : float, default=0.0
        Long-time mean.
    sigma : float, default=1.0
        Noise amplitude.
    n_paths : int, default=1000
        Number of independent realizations.
    seed : int, optional
        Seed for the noise generator.

    Examples
    --------
    >>> ou = OrnsteinUhlenbeck(theta=2.0, sigma=1.0, n_paths=20000, seed=0)
    >>> t, X = ou.run(t_max=5.0, dt=0.01, x0=3.0)
    >>> round(ou.stationary_variance, 3), round(float(X[-1].var()), 2)
    (0.25, 0.25)
    """

    def __init__(self, theta=1.0, mu=0.0, sigma=1.0, n_paths=1000, seed=None):
        if theta <= 0:
            raise ValueError("theta must be positive")
        self.theta = float(theta)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.n_paths = int(n_paths)
        self.rng = np.random.default_rng(seed)
        self.params = np.array([self.theta, self.mu, self.sigma])
        self._drift_njit = _ou_drift
        self._diffusion_njit = _ou_diffusion
        self.times = None
        self.paths = None

    @property
    def stationary_variance(self) -> float:
        """:math:`\\sigma^2 / (2\\theta)`."""
        return self.sigma**2 / (2.0 * self.theta)

    def mean(self, t, x0):
        """Exact ensemble mean at time ``t`` from a common start ``x0``.

        Parameters
        ----------
        t : array_like
        x0 : float

        Returns
        -------
        ndarray
        """
        return self.mu + (x0 - self.mu) * np.exp(-self.theta * np.asarray(t, dtype=np.float64))

    def variance(self, t):
        """Exact ensemble variance at time ``t`` from a deterministic start.

        Parameters
        ----------
        t : array_like

        Returns
        -------
        ndarray
        """
        return self.stationary_variance * (1.0 - np.exp(-2.0 * self.theta * np.asarray(t, dtype=np.float64)))

    def autocovariance(self, tau):
        """Stationary autocovariance :math:`\\frac{\\sigma^2}{2\\theta}e^{-\\theta|\\tau|}`.

        Parameters
        ----------
        tau : array_like

        Returns
        -------
        ndarray
        """
        return self.stationary_variance * np.exp(-self.theta * np.abs(np.asarray(tau, dtype=np.float64)))

    def run(self, t_max, dt, n_frames=200, x0=None, method="exact"):
        """Sample every path from ``x0`` to ``t_max``.

        Parameters
        ----------
        t_max : float
            Total time.
        dt : float
            Time step.
        n_frames : int, default=200
            Number of recorded frames after the initial one.
        x0 : float or array_like of shape (n_paths,), optional
            Initial values; by default drawn from the stationary distribution.
        method : {"exact", "euler_maruyama"}, default="exact"
            ``"exact"`` uses the exact Gaussian transition
            :math:`X_{t+\\Delta t} = \\mu + (X_t-\\mu)e^{-\\theta\\Delta t} +
            \\sigma\\sqrt{(1-e^{-2\\theta\\Delta t})/2\\theta}\\;\\mathcal N(0,1)`
            (Gillespie, Phys. Rev. E 54, 2084 (1996)), free of time-step
            error; ``"euler_maruyama"`` integrates the SDE, with stationary
            variance :math:`\\sigma^2/(\\theta(2 - \\theta\\Delta t))`.

        Returns
        -------
        times : ndarray of shape (n_frames + 1,)
        paths : ndarray of shape (n_frames + 1, n_paths)
        """
        if x0 is None:
            start = self.mu + np.sqrt(self.stationary_variance) * self.rng.standard_normal(self.n_paths)
        else:
            start = np.array(np.broadcast_to(np.asarray(x0, dtype=np.float64), (self.n_paths,)))
        if method == "euler_maruyama":
            times, paths = _run_frames(self._drift_njit, self._diffusion_njit, start, self.params, t_max, dt, n_frames, self.rng)
        elif method == "exact":
            stride = max(1, int(round(t_max / (n_frames * dt))))
            h = stride * dt
            decay = np.exp(-self.theta * h)
            kick = np.sqrt(self.stationary_variance * (1.0 - decay**2))
            paths = np.empty((n_frames + 1, self.n_paths))
            paths[0] = start
            for f in range(n_frames):
                paths[f + 1] = self.mu + (paths[f] - self.mu) * decay + kick * self.rng.standard_normal(self.n_paths)
            times = np.arange(n_frames + 1) * h
        else:
            raise ValueError("method must be 'exact' or 'euler_maruyama'")
        self.times, self.paths = times, paths
        return times, paths
