"""Kuramoto synchronization of coupled phase oscillators.

Kuramoto (1975) reduced a population of weakly coupled limit-cycle
oscillators to their phases :math:`\\theta_i`, with natural frequencies
:math:`\\omega_i` drawn from a distribution :math:`g(\\omega)` and all-to-all
coupling of strength :math:`K`,

.. math::

    \\dot\\theta_i = \\omega_i + \\frac{K}{N}\\sum_{j=1}^N \\sin(\\theta_j - \\theta_i)
                  = \\omega_i + K r \\sin(\\psi - \\theta_i),
    \\qquad r e^{i\\psi} = \\frac{1}{N}\\sum_j e^{i\\theta_j}.

As :math:`N \\to \\infty` the incoherent state :math:`r = 0` loses stability
at the critical coupling :math:`K_c = 2 / (\\pi g(0))`. For a Lorentzian
:math:`g` of half-width :math:`\\gamma`, the order parameter above onset
is exactly :math:`r = \\sqrt{1 - K_c/K}` with :math:`K_c = 2\\gamma`
(Kuramoto, *Chemical Oscillations, Waves, and Turbulence*, 1984; Strogatz,
Physica D 143, 1 (2000)).
"""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from physicskit.chaos.core.base_system import DynamicalSystem
from physicskit.chaos.core.integrators import rk4_integrate
from physicskit.chaos.exceptions import InvalidParameterError

__all__ = ["Kuramoto", "kuramoto_order_parameter_lorentzian"]


@njit(cache=True)
def _kuramoto_rhs(state: NDArray[np.float64], t: float, params: NDArray[np.float64]) -> NDArray[np.float64]:
    """Kuramoto vector field; ``params = (K, omega_1, ..., omega_N)``."""
    K = params[0]
    re = np.mean(np.cos(state))
    im = np.mean(np.sin(state))
    # K r sin(psi - theta) = K (im cos(theta) - re sin(theta))
    return params[1:] + K * (im * np.cos(state) - re * np.sin(state))


@njit(cache=True)
def _kuramoto_order_parameter_run(theta, omega, K, dt, n_steps, record_every):
    """RK4 integration that stores only r(t), so N can be large."""
    params = np.empty(omega.size + 1)
    params[0] = K
    params[1:] = omega
    n_rec = n_steps // record_every + 1
    r = np.empty(n_rec)
    r[0] = np.abs(np.mean(np.exp(1j * theta)))
    t = 0.0
    for n in range(1, n_steps + 1):
        k1 = _kuramoto_rhs(theta, t, params)
        k2 = _kuramoto_rhs(theta + 0.5 * dt * k1, t, params)
        k3 = _kuramoto_rhs(theta + 0.5 * dt * k2, t, params)
        k4 = _kuramoto_rhs(theta + dt * k3, t, params)
        theta = theta + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += dt
        if n % record_every == 0:
            r[n // record_every] = np.abs(np.mean(np.exp(1j * theta)))
    return r, theta


def kuramoto_order_parameter_lorentzian(K: ArrayLike, gamma: float = 1.0) -> NDArray[np.float64]:
    """Exact steady-state order parameter for Lorentzian natural frequencies.

    .. math::

        r(K) = \\begin{cases} 0, & K \\le 2\\gamma, \\\\ \\sqrt{1 - 2\\gamma/K}, & K > 2\\gamma. \\end{cases}

    Parameters
    ----------
    K : array_like
        Coupling strength(s).
    gamma : float, default 1.0
        Half-width at half-maximum of the Lorentzian :math:`g(\\omega)`.

    Returns
    -------
    ndarray of float

    Examples
    --------
    >>> kuramoto_order_parameter_lorentzian([1.0, 4.0], gamma=1.0).tolist()
    [0.0, 0.7071067811865476]
    """
    K = np.asarray(K, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(K > 2.0 * gamma, np.sqrt(np.clip(1.0 - 2.0 * gamma / K, 0.0, None)), 0.0)


class Kuramoto(DynamicalSystem):
    """The Kuramoto model of :math:`N` all-to-all coupled phase oscillators.

    Natural frequencies are placed deterministically at the quantiles of
    :math:`g(\\omega)`, :math:`\\omega_i = G^{-1}((i - 1/2)/N)`, which removes
    the sample-to-sample scatter of a random draw and makes finite-:math:`N`
    runs follow the :math:`N \\to \\infty` theory closely.

    Parameters
    ----------
    n_oscillators : int, default 500
        Number of oscillators :math:`N`.
    K : float, default 1.0
        Coupling strength.
    gamma : float, default 0.5
        Width of :math:`g(\\omega)`: the half-width for ``"lorentzian"``, the
        standard deviation for ``"gaussian"``.
    distribution : {"lorentzian", "gaussian"}, default "lorentzian"
        Natural-frequency distribution, centred on zero.

    Attributes
    ----------
    omega : ndarray of float, shape (n_oscillators,)
        Natural frequencies.

    Examples
    --------
    >>> model = Kuramoto(n_oscillators=400, K=3.0, gamma=0.5)
    >>> model.critical_coupling
    1.0
    >>> r = model.order_parameter_series(t_max=60.0, seed=0)
    >>> bool(abs(r[-200:].mean() - kuramoto_order_parameter_lorentzian(3.0, 0.5)) < 0.03)
    True
    """

    def __init__(self, n_oscillators: int = 500, K: float = 1.0, gamma: float = 0.5, distribution: str = "lorentzian"):
        if distribution not in ("lorentzian", "gaussian"):
            raise InvalidParameterError("distribution must be 'lorentzian' or 'gaussian'")
        if gamma <= 0:
            raise InvalidParameterError("gamma must be positive")
        self.n_oscillators = int(n_oscillators)
        self.K = float(K)
        self.gamma = float(gamma)
        self.distribution = distribution
        self.dim = self.n_oscillators
        u = (np.arange(self.n_oscillators) + 0.5) / self.n_oscillators
        if distribution == "lorentzian":
            self.omega = self.gamma * np.tan(np.pi * (u - 0.5))
        else:
            from scipy.special import ndtri

            self.omega = self.gamma * ndtri(u)

    @property
    def critical_coupling(self) -> float:
        """Kuramoto's onset of synchronization, :math:`K_c = 2/(\\pi g(0))`.

        :math:`2\\gamma` for a Lorentzian and :math:`\\sqrt{8/\\pi}\\,\\sigma` for a
        Gaussian of standard deviation :math:`\\sigma`.
        """
        if self.distribution == "lorentzian":
            return 2.0 * self.gamma
        return float(np.sqrt(8.0 / np.pi) * self.gamma)

    @property
    def params(self) -> NDArray[np.float64]:
        """Parameter vector ``(K, omega_1, ..., omega_N)``."""
        return np.concatenate(([self.K], self.omega))

    def rhs(self, state: NDArray[np.float64], t: float) -> NDArray[np.float64]:
        """Evaluate :math:`\\dot\\theta_i = \\omega_i + K r \\sin(\\psi - \\theta_i)`.

        Parameters
        ----------
        state : ndarray of float, shape (N,)
            Phases.
        t : float
            Current time (unused).

        Returns
        -------
        ndarray of float, shape (N,)
        """
        return np.asarray(_kuramoto_rhs(np.asarray(state, dtype=np.float64), t, self.params))

    def initial_state(self, seed: int | None = 0) -> NDArray[np.float64]:
        """Incoherent initial phases, uniform on :math:`[0, 2\\pi)`.

        Parameters
        ----------
        seed : int or None, default 0
            Seed for the phases.

        Returns
        -------
        ndarray of float, shape (N,)
        """
        return np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, self.n_oscillators)

    @staticmethod
    def order_parameter(states: ArrayLike) -> NDArray[np.float64]:
        """Order parameter :math:`r = |\\langle e^{i\\theta_j} \\rangle_j|` of each row of phases.

        Parameters
        ----------
        states : array_like, shape (..., N)

        Returns
        -------
        ndarray of float, shape (...)
        """
        return np.abs(np.mean(np.exp(1j * np.asarray(states)), axis=-1))

    def trajectory(
        self,
        state0: NDArray[np.float64] | None = None,
        t0: float = 0.0,
        dt: float = 0.05,
        n_steps: int = 2000,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Integrate the phases with RK4, storing every step.

        Parameters
        ----------
        state0 : array_like of float, shape (N,), optional
            Initial phases; defaults to :meth:`initial_state`.
        t0 : float, default 0.0
        dt : float, default 0.05
        n_steps : int, default 2000

        Returns
        -------
        times : ndarray of float, shape (n_steps + 1,)
        states : ndarray of float, shape (n_steps + 1, N)
        """
        state0 = self.initial_state() if state0 is None else np.asarray(state0, dtype=np.float64)
        return rk4_integrate(_kuramoto_rhs, state0, t0, dt, n_steps, self.params)

    def order_parameter_series(
        self,
        t_max: float = 100.0,
        dt: float = 0.05,
        state0: NDArray[np.float64] | None = None,
        seed: int | None = 0,
        record_every: int = 1,
    ) -> NDArray[np.float64]:
        """Integrate and return only :math:`r(t)`, which keeps memory flat in :math:`N`.

        Parameters
        ----------
        t_max : float, default 100.0
            Integration time.
        dt : float, default 0.05
            RK4 step.
        state0 : array_like, optional
            Initial phases; defaults to :meth:`initial_state` with ``seed``.
        seed : int or None, default 0
            Seed for the default initial phases.
        record_every : int, default 1
            Store :math:`r` every this many steps.

        Returns
        -------
        ndarray of float
            :math:`r` at times ``0, record_every*dt, ...``.
        """
        theta = self.initial_state(seed) if state0 is None else np.asarray(state0, dtype=np.float64)
        r, _ = _kuramoto_order_parameter_run(theta.copy(), self.omega, self.K, dt, int(round(t_max / dt)), record_every)
        return r
