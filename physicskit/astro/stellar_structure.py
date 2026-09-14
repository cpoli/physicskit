r"""Polytropic stellar models via the Lane-Emden equation.

Uses **gravitational units with** :math:`G=1` by default -- every function
that needs it accepts ``G`` as a keyword argument, the same convention
:mod:`physicskit.relativity` uses for its own geometrized units.

- :func:`lane_emden` -- the dimensionless hydrostatic-equilibrium profile
  of a self-gravitating polytropic sphere.
- :class:`PolytropicStar` -- a physical star (radius, mass) built from one
  Lane-Emden solution.
- :func:`chandrasekhar_mass` -- the white-dwarf mass limit.
- :func:`main_sequence_luminosity` -- the empirical mass-luminosity relation.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

__all__ = ["lane_emden", "PolytropicStar", "chandrasekhar_mass", "main_sequence_luminosity"]


def lane_emden(n, xi_max=10.0, n_points=2000):
    r"""Integrate the Lane-Emden equation for a polytrope of index ``n``.

    .. math::

        \frac{1}{\xi^2}\frac{d}{d\xi}\left(\xi^2\frac{d\theta}{d\xi}\right)
        + \theta^n = 0, \qquad \theta(0)=1,\ \theta'(0)=0

    the dimensionless hydrostatic-equilibrium profile of a self-gravitating
    sphere with equation of state :math:`P=K\rho^{1+1/n}`.

    Parameters
    ----------
    n : float
        Polytropic index.
    xi_max : float, default=10.0
        Upper integration bound, used only as a fallback for indices
        (e.g. ``n=5``) where :math:`\theta` never reaches zero.
    n_points : int, default=2000
        Number of points to sample the returned arrays at.

    Returns
    -------
    xi : ndarray
        Dimensionless radius, from just above 0 to the surface (the first
        zero of :math:`\theta`) or to ``xi_max``.
    theta : ndarray
        The Lane-Emden function :math:`\theta(\xi)`.

    Examples
    --------
    >>> xi, theta = lane_emden(0.0)
    >>> round(float(xi[-1]), 3)  # exact analytic surface is sqrt(6)
    2.449
    """
    xi0 = 1e-6
    theta0 = 1.0 - xi0**2 / 6.0
    dtheta0 = -xi0 / 3.0

    def rhs(xi, y):
        theta, dtheta = y
        theta_pos = max(theta, 0.0)
        d2theta = -(theta_pos**n) - (2.0 / xi) * dtheta
        return [dtheta, d2theta]

    def surface_event(xi, y):
        return y[0]

    surface_event.terminal = True
    surface_event.direction = -1

    sol = solve_ivp(
        rhs,
        [xi0, xi_max],
        [theta0, dtheta0],
        events=surface_event,
        dense_output=True,
        rtol=1e-9,
        atol=1e-11,
        max_step=xi_max / 2000,
    )
    xi_surface = sol.t[-1]
    xi = np.linspace(xi0, xi_surface, n_points)
    theta = sol.sol(xi)[0]
    theta = np.clip(theta, 0.0, None)
    return xi, theta


class PolytropicStar:
    r"""A physical polytropic star built from one Lane-Emden solution.

    Parameters
    ----------
    n : float
        Polytropic index, :math:`P=K\rho^{1+1/n}`.
    K : float
        Polytropic constant.
    rho_c : float
        Central density.
    G : float, default=1.0
        Gravitational constant.

    Examples
    --------
    >>> star = PolytropicStar(n=0.0, K=1.0, rho_c=1.0)
    >>> round(star.mass / (4 / 3 * np.pi * star.radius**3 * star.rho_c), 3)
    1.0
    """

    def __init__(self, n, K, rho_c, G=1.0):
        self.n = n
        self.K = K
        self.rho_c = rho_c
        self.G = G
        self._xi, self._theta = lane_emden(n)

    @property
    def xi1(self):
        """The dimensionless surface radius, the last point of the Lane-Emden solution."""
        return float(self._xi[-1])

    @property
    def alpha(self):
        r"""Physical length scale, :math:`\alpha=\sqrt{(n+1)K\rho_c^{1/n-1}/(4\pi G)}`."""
        if self.n == 0:
            exponent = -1.0
        else:
            exponent = 1.0 / self.n - 1.0
        return float(np.sqrt((self.n + 1) * self.K * self.rho_c**exponent / (4.0 * np.pi * self.G)))

    @property
    def radius(self):
        r"""Physical stellar radius, :math:`R=\alpha\xi_1`."""
        return self.alpha * self.xi1

    @property
    def mass(self):
        r"""Physical stellar mass, :math:`M=4\pi\alpha^3\rho_c\left[-\xi_1^2\theta'(\xi_1)\right]`."""
        dtheta_surface = (self._theta[-1] - self._theta[-2]) / (self._xi[-1] - self._xi[-2])
        return float(4.0 * np.pi * self.alpha**3 * self.rho_c * (-(self.xi1**2) * dtheta_surface))


def chandrasekhar_mass(mu_e=2.0):
    r"""The Chandrasekhar mass limit, in solar masses.

    .. math::

        M_{\rm Ch} \approx \frac{5.83}{\mu_e^2}\ M_\odot

    (the standard numerical prefactor from the n=3 polytrope /
    relativistic-electron-degeneracy treatment; see e.g. Kippenhahn,
    Weigert & Weiss, *Stellar Structure and Evolution*).

    Parameters
    ----------
    mu_e : float, default=2.0
        Mean molecular weight per electron (2.0 for carbon/oxygen white
        dwarfs).

    Returns
    -------
    float
        Mass limit, in solar masses.

    Examples
    --------
    >>> M = chandrasekhar_mass(2.0)
    >>> 1.2 < M < 1.5
    True
    """
    return 5.83 / mu_e**2


def main_sequence_luminosity(mass_solar):
    r"""Empirical main-sequence mass-luminosity relation, :math:`L/L_\odot\approx(M/M_\odot)^{3.5}`.

    Valid roughly for main-sequence stars in the 0.5-10 solar mass range;
    this is a pure empirical power law, not derived from first-principles
    stellar structure.

    Parameters
    ----------
    mass_solar : float
        Mass, in solar masses.

    Returns
    -------
    float
        Luminosity, in solar luminosities.

    Examples
    --------
    >>> round(main_sequence_luminosity(1.0), 6)
    1.0
    """
    return mass_solar**3.5
