r"""Laser dynamics: rate equations, the lasing threshold, and the Maxwell-Bloch (Haken-Lorenz) equations.

Both models are integrated with the shared numba RK4 integrator
(:func:`physicskit.integrators.rk4_integrate`). Following the package's
first-class-function pattern, each right-hand side is a module-level
``@njit`` function with the shared ``rhs(state, t, params)`` convention,
held by each instance as ``self._deriv_njit`` and never passed as a bound
method. Rates and times are in arbitrary (consistent) units.

- :class:`LaserRateEquations` -- the single-mode rate equations for the
  population inversion and the photon number (Statz and deMars 1960;
  Siegman, *Lasers*, 1986, Ch. 24-25): threshold, steady state, and
  relaxation oscillations.
- :class:`MaxwellBloch` -- the semiclassical single-mode laser of Haken
  (Phys. Lett. A 53, 77 (1975)), isomorphic to the Lorenz equations, with
  its lasing threshold and second (chaotic) instability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from physicskit.integrators import rk4_integrate

__all__ = ["LaserRateEquations", "MaxwellBloch"]


@njit
def _rate_equation_rhs(state, t, params):
    """(N, q)' for params = (pump, tau, tau_c, B, beta)."""
    N, q = state[0], state[1]
    pump, tau, tau_c, B, beta = params[0], params[1], params[2], params[3], params[4]
    out = np.empty(2)
    out[0] = pump - N / tau - B * N * q
    out[1] = B * N * q - q / tau_c + beta * N / tau
    return out


@njit
def _maxwell_bloch_rhs(state, t, params):
    """(E, P, D)' for params = (kappa, gamma_perp, gamma_par, r)."""
    E, P, D = state[0], state[1], state[2]
    kappa, g_perp, g_par, r = params[0], params[1], params[2], params[3]
    out = np.empty(3)
    out[0] = kappa * (P - E)
    out[1] = g_perp * (E * D - P)
    out[2] = g_par * (r - D - E * P)
    return out


@dataclass
class LaserRateEquations:
    r"""Single-mode laser rate equations for the inversion :math:`N` and photon number :math:`q`.

    .. math::

        \frac{dN}{dt} = P - \frac{N}{\tau} - BNq, \qquad
        \frac{dq}{dt} = BNq - \frac{q}{\tau_c} + \beta\frac{N}{\tau},

    with pump rate :math:`P`, upper-level lifetime :math:`\tau`, cavity
    photon lifetime :math:`\tau_c`, stimulated-emission coefficient
    :math:`B`, and fraction :math:`\beta` of spontaneous emission going into
    the lasing mode (Siegman, *Lasers*, Eqs. 24.8-24.9; Milonni and Eberly,
    *Laser Physics*, 2010, Sec. 6.3). Gain equals loss at the threshold
    inversion :math:`N_\mathrm{th} = 1/(B\tau_c)`, reached at pump
    :math:`P_\mathrm{th} = N_\mathrm{th}/\tau` -- the condition first
    formulated by Schawlow and Townes (Phys. Rev. 112, 1940 (1958)). For
    :math:`\beta = 0`, above threshold the inversion clamps at
    :math:`N_\mathrm{th}` and every extra pumped excitation becomes a
    photon: :math:`q = \tau_c(P - P_\mathrm{th})`.

    Parameters
    ----------
    pump : float, default=2.0
        Pump rate :math:`P`.
    tau : float, default=1.0
        Upper-level (spontaneous) lifetime.
    tau_c : float, default=0.01
        Cavity photon lifetime.
    B : float, default=1.0
        Stimulated-emission coupling.
    beta : float, default=0.0
        Spontaneous-emission coupling into the mode, :math:`0 \le \beta \le 1`.

    Examples
    --------
    >>> laser = LaserRateEquations(pump=300.0, tau=1.0, tau_c=0.01, B=1.0)
    >>> laser.threshold_pump
    100.0
    >>> N, q = laser.steady_state()
    >>> round(N, 10), round(q, 10)
    (100.0, 2.0)
    """

    pump: float = 2.0
    tau: float = 1.0
    tau_c: float = 0.01
    B: float = 1.0
    beta: float = 0.0

    def __post_init__(self) -> None:
        # an instance attribute, so the njit dispatcher is never bound to self
        self._deriv_njit = _rate_equation_rhs

    @property
    def params(self) -> np.ndarray:
        """Parameter vector passed to the njit right-hand side."""
        return np.array([self.pump, self.tau, self.tau_c, self.B, self.beta], dtype=float)

    @property
    def threshold_inversion(self) -> float:
        r""":math:`N_\mathrm{th} = 1/(B\tau_c)`."""
        return 1.0 / (self.B * self.tau_c)

    @property
    def threshold_pump(self) -> float:
        r""":math:`P_\mathrm{th} = N_\mathrm{th}/\tau`."""
        return self.threshold_inversion / self.tau

    def steady_state(self) -> tuple[float, float]:
        r"""Stationary inversion and photon number :math:`(N_s, q_s)`.

        Setting both derivatives to zero gives :math:`q = \tau_c[P - (1-\beta)N/\tau]`
        and the quadratic
        :math:`\frac{(1-\beta)B\tau_c}{\tau}N^2 - \left(PB\tau_c + \frac1\tau\right)N + P = 0`,
        whose smaller root is the physical one. For :math:`\beta = 0` it is
        :math:`(P\tau, 0)` below threshold and
        :math:`(N_\mathrm{th}, \tau_c(P - P_\mathrm{th}))` above.

        Returns
        -------
        N, q : float
        """
        a = (1 - self.beta) * self.B * self.tau_c / self.tau
        b = self.pump * self.B * self.tau_c + 1 / self.tau
        if a == 0:
            N = self.pump / b
        else:
            # smaller root, written to avoid cancellation: 2P / (b + sqrt(b^2 - 4aP))
            N = 2 * self.pump / (b + np.sqrt(max(b**2 - 4 * a * self.pump, 0.0)))
        q = self.tau_c * (self.pump - (1 - self.beta) * N / self.tau)
        return float(N), float(max(q, 0.0))

    def relaxation_oscillation(self) -> tuple[float, float]:
        r"""Frequency and damping rate of the relaxation oscillations (:math:`\beta = 0`, above threshold).

        Linearizing about the lasing steady state, with pump ratio
        :math:`r = P/P_\mathrm{th}`, perturbations evolve as
        :math:`e^{-\Gamma t}\cos\Omega t` with

        .. math::

            \Gamma = \frac{r}{2\tau}, \qquad
            \Omega = \sqrt{\frac{r - 1}{\tau\tau_c} - \Gamma^2}

        (Siegman, Eq. 25.14).

        Returns
        -------
        Omega, Gamma : float

        Raises
        ------
        ValueError
            Below threshold, or if the oscillations are overdamped.
        """
        r = self.pump / self.threshold_pump
        if r <= 1:
            raise ValueError("relaxation oscillations need pumping above threshold")
        Gamma = r / (2 * self.tau)
        omega2 = (r - 1) / (self.tau * self.tau_c) - Gamma**2
        if omega2 <= 0:
            raise ValueError("relaxation oscillations are overdamped for these parameters")
        return float(np.sqrt(omega2)), float(Gamma)

    def integrate(self, t_max: float, dt: float, N0: float = 0.0, q0: float = 1.0) -> tuple:
        r"""Integrate the rate equations with RK4 from :math:`(N_0, q_0)` at :math:`t = 0`.

        A small seed :math:`q_0 > 0` stands in for spontaneous emission when
        :math:`\beta = 0` (with no photons, stimulated emission never starts).

        Parameters
        ----------
        t_max : float
            Final time.
        dt : float
            Step size (must resolve :math:`\tau_c`).
        N0, q0 : float, default=0.0, 1.0
            Initial inversion and photon number.

        Returns
        -------
        t, N, q : numpy.ndarray
        """
        n_steps = int(np.ceil(t_max / dt))
        t, y = rk4_integrate(self._deriv_njit, np.array([N0, q0], dtype=float), 0.0, dt, n_steps, self.params)
        return t, y[:, 0], y[:, 1]


@dataclass
class MaxwellBloch:
    r"""Semiclassical single-mode laser: the resonant Maxwell-Bloch equations in Haken's form.

    For the (real, resonant) field :math:`E`, atomic polarization
    :math:`P`, and inversion :math:`D`, normalized so that lasing starts at
    pump parameter :math:`r = 1`,

    .. math::

        \dot E = \kappa(P - E), \qquad
        \dot P = \gamma_\perp(ED - P), \qquad
        \dot D = \gamma_\parallel(r - D - EP)

    (Haken 1975; Narducci and Abraham, *Laser Physics and Laser
    Instabilities*, 1988, Ch. 7). With :math:`\sigma = \kappa/\gamma_\perp`,
    :math:`b = \gamma_\parallel/\gamma_\perp`, time :math:`\gamma_\perp t`,
    and :math:`x = \sqrt b\,E`, :math:`y = \sqrt b\,P`, :math:`z = r - D`,
    these are exactly the Lorenz equations
    :math:`\dot x = \sigma(y - x)`, :math:`\dot y = x(r - z) - y`,
    :math:`\dot z = xy - bz`. Below :math:`r = 1` the field decays;
    above it the stable lasing state is :math:`E = P = \pm\sqrt{r-1}`,
    :math:`D = 1`. In the "bad cavity" regime :math:`\sigma > b + 1` that
    state loses stability at the second threshold

    .. math::

        r_H = \frac{\sigma(\sigma + b + 3)}{\sigma - b - 1},

    beyond which the laser output is chaotic.

    Parameters
    ----------
    kappa : float, default=3.0
        Cavity field decay rate.
    gamma_perp : float, default=1.0
        Polarization (transverse) decay rate.
    gamma_par : float, default=0.25
        Inversion (longitudinal) decay rate.
    r : float, default=2.0
        Pump parameter (1 at threshold).

    Examples
    --------
    >>> mb = MaxwellBloch(kappa=10.0, gamma_perp=1.0, gamma_par=8 / 3, r=28.0)
    >>> round(mb.second_threshold, 4)  # the Lorenz value for sigma=10, b=8/3
    24.7368
    """

    kappa: float = 3.0
    gamma_perp: float = 1.0
    gamma_par: float = 0.25
    r: float = 2.0

    def __post_init__(self) -> None:
        # an instance attribute, so the njit dispatcher is never bound to self
        self._deriv_njit = _maxwell_bloch_rhs

    @property
    def params(self) -> np.ndarray:
        """Parameter vector passed to the njit right-hand side."""
        return np.array([self.kappa, self.gamma_perp, self.gamma_par, self.r], dtype=float)

    @property
    def sigma(self) -> float:
        r"""Lorenz :math:`\sigma = \kappa/\gamma_\perp`."""
        return self.kappa / self.gamma_perp

    @property
    def b(self) -> float:
        r"""Lorenz :math:`b = \gamma_\parallel/\gamma_\perp`."""
        return self.gamma_par / self.gamma_perp

    @property
    def second_threshold(self) -> float:
        r"""Haken's second threshold :math:`r_H`; ``inf`` in the good-cavity regime :math:`\sigma \le b + 1`."""
        s, b = self.sigma, self.b
        return float(s * (s + b + 3) / (s - b - 1)) if s > b + 1 else float("inf")

    def steady_state(self) -> np.ndarray:
        r"""The stable-branch fixed point: :math:`(0, 0, r)` below threshold, :math:`(\sqrt{r-1}, \sqrt{r-1}, 1)` above.

        Returns
        -------
        numpy.ndarray
            :math:`(E, P, D)`.
        """
        if self.r <= 1:
            return np.array([0.0, 0.0, self.r])
        e = np.sqrt(self.r - 1)
        return np.array([e, e, 1.0])

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        r"""Jacobian :math:`\partial\dot{\mathbf y}/\partial\mathbf y` at ``state = (E, P, D)``.

        Parameters
        ----------
        state : numpy.ndarray

        Returns
        -------
        numpy.ndarray
            Shape ``(3, 3)``.
        """
        E, P, D = state
        k, gp, gl = self.kappa, self.gamma_perp, self.gamma_par
        return np.array([[-k, k, 0.0], [gp * D, -gp, gp * E], [-gl * P, -gl * E, -gl]])

    def integrate(self, state0: np.ndarray, t_max: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
        r"""Integrate with RK4 from ``state0 = (E, P, D)`` at :math:`t = 0`.

        Parameters
        ----------
        state0 : array_like
            Initial :math:`(E, P, D)`; a small :math:`E \ne 0` seeds lasing.
        t_max : float
            Final time.
        dt : float
            Step size.

        Returns
        -------
        t : numpy.ndarray
            Shape ``(n + 1,)``.
        states : numpy.ndarray
            Shape ``(n + 1, 3)``.
        """
        n_steps = int(np.ceil(t_max / dt))
        return rk4_integrate(self._deriv_njit, np.asarray(state0, dtype=float), 0.0, dt, n_steps, self.params)
