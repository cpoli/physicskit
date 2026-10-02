"""Grey radiative transfer in a plane-parallel stellar atmosphere.

Radiation crossing a stellar atmosphere obeys, at optical depth :math:`\\tau`
and direction cosine :math:`\\mu`,

.. math::

    \\mu\\frac{dI}{d\\tau} = I - S.

For a *grey* opacity (independent of frequency) in radiative equilibrium the
source function equals the mean intensity, :math:`S = J`, and the flux
:math:`F = \\sigma T_{\\text{eff}}^4` is constant with depth. The solution
is (Milne 1921; Hopf 1930)

.. math::

    T^4(\\tau) = \\tfrac34 T_{\\text{eff}}^4\\,[\\tau + q(\\tau)],

where the Hopf function :math:`q` rises from :math:`1/\\sqrt3` at the
surface to :math:`0.7104` deep inside. Eddington's (1926) approximation,
closing the moment equations with :math:`K = J/3`, gives the constant
:math:`q = 2/3`, and with it the limb darkening :math:`I(0, \\mu)/I(0, 1) =
(2 + 3\\mu)/5`.

:class:`GreyAtmosphere` solves the problem exactly in Chandrasekhar's
discrete-ordinates method (Astrophys. J. 100, 76 (1944); *Radiative
Transfer*, 1950, ch. III), which converges to the exact solution as the
number of streams grows and keeps :math:`q(0) = 1/\\sqrt3` exactly in every
order.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

__all__ = ["GreyAtmosphere", "eddington_limb_darkening", "eddington_temperature", "emergent_intensity"]


def eddington_temperature(tau: ArrayLike, T_eff: float = 1.0) -> NDArray[np.float64]:
    """Eddington's grey temperature law :math:`T^4 = \\tfrac34 T_{\\text{eff}}^4(\\tau + 2/3)`.

    Parameters
    ----------
    tau : array_like
        Optical depth.
    T_eff : float, default 1.0

    Returns
    -------
    ndarray

    Examples
    --------
    >>> float(eddington_temperature(2 / 3, T_eff=5772.0))  # T = T_eff at tau = 2/3
    5772.0
    """
    tau = np.asarray(tau, dtype=float)
    return T_eff * (0.75 * (tau + 2.0 / 3.0)) ** 0.25


def eddington_limb_darkening(mu: ArrayLike) -> NDArray[np.float64]:
    """Eddington limb darkening :math:`I(0,\\mu)/I(0,1) = (2 + 3\\mu)/5`.

    Parameters
    ----------
    mu : array_like
        Cosine of the angle from the disc normal (1 at disc centre, 0 at the limb).

    Returns
    -------
    ndarray
    """
    return (2.0 + 3.0 * np.asarray(mu, dtype=float)) / 5.0


def emergent_intensity(source, mu: ArrayLike, n: int = 64) -> NDArray[np.float64]:
    """Formal solution :math:`I(0, \\mu) = \\int_0^\\infty S(\\tau)e^{-\\tau/\\mu}\\,d\\tau/\\mu`.

    Evaluated by :math:`n`-point Gauss-Laguerre quadrature in :math:`t = \\tau/\\mu`.

    Parameters
    ----------
    source : callable
        Vectorized source function :math:`S(\\tau)`.
    mu : array_like
        Direction cosines in :math:`(0, 1]`.
    n : int, default 64
        Quadrature points.

    Returns
    -------
    ndarray

    Examples
    --------
    A linear source function :math:`S = a + b\\tau` emerges as :math:`a + b\\mu`
    (the Eddington-Barbier relation):

    >>> round(float(emergent_intensity(lambda t: 1 + 2 * t, 0.5)), 10)
    2.0
    """
    mu_arr = np.asarray(mu, dtype=float)
    t, w = np.polynomial.laguerre.laggauss(n)
    out = np.array([np.sum(w * source(m * t)) for m in mu_arr.ravel()])
    return out.reshape(mu_arr.shape)


class GreyAtmosphere:
    """Exact grey atmosphere in Chandrasekhar's :math:`n`-stream discrete-ordinates approximation.

    With Gauss-Legendre nodes :math:`\\pm\\mu_i` and weights :math:`a_i`
    (:math:`i = 1..n`), the intensities are

    .. math::

        I(\\tau, \\mu_i) = \\tfrac34 F\\left[\\tau + \\mu_i + Q + \\sum_{\\alpha=1}^{n-1}
                          \\frac{L_\\alpha e^{-k_\\alpha\\tau}}{1 + k_\\alpha\\mu_i}\\right],

    where the :math:`k_\\alpha` are the positive roots of
    :math:`\\sum_i a_i/(1 - k^2\\mu_i^2) = 1` and :math:`Q`, :math:`L_\\alpha`
    follow from no incoming radiation at :math:`\\tau = 0`. The Hopf function
    is :math:`q(\\tau) = Q + \\sum_\\alpha L_\\alpha e^{-k_\\alpha\\tau}`.

    Parameters
    ----------
    n_streams : int, default 32
        Number :math:`n` of directions per hemisphere.

    Attributes
    ----------
    Q : float
        :math:`q(\\infty)`, which tends to Hopf's 0.710446 as :math:`n` grows.
    k, L : ndarray
        Decay constants and amplitudes of the transient terms.

    Examples
    --------
    >>> atm = GreyAtmosphere()
    >>> round(float(atm.hopf(0.0)), 6)  # 1/sqrt(3) in every approximation
    0.57735
    >>> round(atm.Q, 4)
    0.7104
    """

    def __init__(self, n_streams: int = 32):
        if n_streams < 1:
            raise ValueError("n_streams must be at least 1")
        nodes, weights = np.polynomial.legendre.leggauss(2 * n_streams)
        pos = nodes > 0
        self.mu = nodes[pos]
        self.a = weights[pos]  # full-range weights sum to 2, so these sum to 1
        self.n = n_streams

        def characteristic(k):
            return np.sum(self.a / (1.0 - k**2 * self.mu**2)) - 1.0

        # one root between consecutive poles 1/mu_i, ordered by decreasing mu
        poles = np.sort(1.0 / self.mu)
        roots = [brentq(characteristic, p1 * (1 + 1e-12), p2 * (1 - 1e-12)) for p1, p2 in zip(poles[:-1], poles[1:])]
        self.k = np.array(roots)
        # boundary condition I(0, -mu_i) = 0:  Q + sum_a L_a / (1 - k_a mu_i) = mu_i
        A = np.ones((self.n, self.n))
        if self.n > 1:
            A[:, 1:] = 1.0 / (1.0 - np.outer(self.mu, self.k))
        sol = np.linalg.solve(A, self.mu)
        self.Q = float(sol[0])
        self.L = sol[1:]

    def hopf(self, tau: ArrayLike) -> NDArray[np.float64]:
        """Hopf function :math:`q(\\tau)`.

        Parameters
        ----------
        tau : array_like

        Returns
        -------
        ndarray
        """
        tau = np.asarray(tau, dtype=float)
        return self.Q + np.sum(self.L * np.exp(-np.multiply.outer(tau, self.k)), axis=-1)

    def temperature(self, tau: ArrayLike, T_eff: float = 1.0) -> NDArray[np.float64]:
        """:math:`T(\\tau) = T_{\\text{eff}}\\,[\\tfrac34(\\tau + q(\\tau))]^{1/4}`."""
        tau = np.asarray(tau, dtype=float)
        return T_eff * (0.75 * (tau + self.hopf(tau))) ** 0.25

    def emergent_intensity(self, mu: ArrayLike) -> NDArray[np.float64]:
        """Emergent intensity in units of :math:`F/\\pi = \\sigma T_{\\text{eff}}^4/\\pi`.

        The Laplace transform of :math:`S = \\tfrac34 F(\\tau + q)/\\pi` gives
        :math:`I(0, \\mu) = \\tfrac34\\left[\\mu + Q + \\sum_\\alpha L_\\alpha/(1 + k_\\alpha\\mu)\\right]`.

        Parameters
        ----------
        mu : array_like

        Returns
        -------
        ndarray
        """
        mu = np.asarray(mu, dtype=float)
        return 0.75 * (mu + self.Q + np.sum(self.L / (1.0 + np.multiply.outer(mu, self.k)), axis=-1))

    def limb_darkening(self, mu: ArrayLike) -> NDArray[np.float64]:
        """Normalized limb darkening :math:`I(0, \\mu)/I(0, 1)`."""
        return self.emergent_intensity(mu) / self.emergent_intensity(1.0)
