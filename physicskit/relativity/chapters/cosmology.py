"""FLRW cosmology: the Friedmann equations, cosmic expansion, and redshift.

The Friedmann-Lemaitre-Robertson-Walker (FLRW) metric is the unique
spacetime describing a homogeneous, isotropic universe -- an excellent
approximation to our own on large scales. Plugging it into Einstein's field
equations yields the Friedmann equation, which governs how the universe's
scale factor :math:`a(t)` evolves under the combined gravity of radiation,
matter, curvature, and dark energy (:math:`\\Lambda`). This single equation
explains Hubble's 1929 discovery of cosmic expansion, predicts the
universe's age, and -- through the distance-redshift relation -- underlies
the 1998 discovery (via observations of distant supernovae) that this
expansion is accelerating.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from physicskit.relativity.utils.constants import C_SI, PARSEC_M

__all__ = ["FLRWCosmology"]

_MPC_M = PARSEC_M * 1.0e6
_GYR_S = 365.25 * 24.0 * 3600.0 * 1.0e9


class FLRWCosmology:
    """A flat or curved FLRW universe with radiation, matter, curvature, and dark energy.

    Parameters
    ----------
    H0 : float, default=70.0
        Hubble constant today, in km/s/Mpc.
    Omega_m : float, default=0.3
        Matter density parameter today.
    Omega_r : float, default=1e-4
        Radiation density parameter today.
    Omega_Lambda : float, default=0.7
        Dark energy (cosmological constant) density parameter today.

    Attributes
    ----------
    Omega_k : float
        Curvature density parameter, :math:`1 - \\Omega_m - \\Omega_r - \\Omega_\\Lambda`
        (computed so all density parameters sum to 1, as required by the
        Friedmann equation at :math:`a=1`).
    """

    def __init__(self, H0=70.0, Omega_m=0.3, Omega_r=1.0e-4, Omega_Lambda=0.7):
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_r = Omega_r
        self.Omega_Lambda = Omega_Lambda
        self.Omega_k = 1.0 - Omega_m - Omega_r - Omega_Lambda

    def _H0_si(self):
        return self.H0 * 1.0e3 / _MPC_M  # km/s/Mpc -> 1/s

    def E(self, a):
        """Dimensionless expansion rate :math:`E(a) = H(a)/H_0`.

        .. math::

            E(a) = \\sqrt{\\Omega_r a^{-4} + \\Omega_m a^{-3} + \\Omega_k a^{-2} + \\Omega_\\Lambda}

        Parameters
        ----------
        a : float or array_like
            Scale factor (:math:`a=1` today).

        Returns
        -------
        float or ndarray
        """
        a = np.asarray(a, dtype=np.float64)
        return np.sqrt(self.Omega_r * a**-4 + self.Omega_m * a**-3 + self.Omega_k * a**-2 + self.Omega_Lambda)

    def hubble_parameter(self, a):
        """Hubble parameter :math:`H(a) = H_0 E(a)`, in km/s/Mpc.

        Parameters
        ----------
        a : float or array_like
            Scale factor.

        Returns
        -------
        float or ndarray
        """
        return self.H0 * self.E(a)

    def redshift_to_scale_factor(self, z):
        """Convert redshift to scale factor, :math:`a = 1/(1+z)`."""
        return 1.0 / (1.0 + np.asarray(z, dtype=np.float64))

    def scale_factor_to_redshift(self, a):
        """Convert scale factor to redshift, :math:`z = 1/a - 1`."""
        return 1.0 / np.asarray(a, dtype=np.float64) - 1.0

    def comoving_distance_mpc(self, z):
        """Comoving distance to redshift ``z``, in Mpc.

        .. math::

            D_C(z) = c \\int_0^z \\frac{dz'}{H(z')}

        Parameters
        ----------
        z : float
            Redshift.

        Returns
        -------
        float
        """
        H0_si = self._H0_si()

        def integrand(zp):
            a = 1.0 / (1.0 + zp)
            return 1.0 / (H0_si * self.E(a))

        distance_m = C_SI * quad(integrand, 0.0, z)[0]
        return distance_m / _MPC_M

    def luminosity_distance_mpc(self, z):
        """Luminosity distance to redshift ``z``, in Mpc.

        .. math::

            D_L = (1+z) D_M, \\qquad
            D_M = \\begin{cases}
                \\frac{D_H}{\\sqrt{\\Omega_k}} \\sinh\\left(\\sqrt{\\Omega_k}\\, D_C/D_H\\right), & \\Omega_k > 0, \\\\
                D_C, & \\Omega_k = 0, \\\\
                \\frac{D_H}{\\sqrt{|\\Omega_k|}} \\sin\\left(\\sqrt{|\\Omega_k|}\\, D_C/D_H\\right), & \\Omega_k < 0,
            \\end{cases}

        where :math:`D_H = c/H_0` is the Hubble distance and :math:`D_M` the
        transverse comoving distance, which differs from the line-of-sight
        :meth:`comoving_distance_mpc` :math:`D_C` only in a curved universe
        (Hogg 1999, astro-ph/9905116, eqs. 16 and 21).

        Parameters
        ----------
        z : float
            Redshift.

        Returns
        -------
        float

        Examples
        --------
        An open, matter-only universe matches Mattig's closed form,
        :math:`D_L = \\frac{2 D_H}{\\Omega_m^2}\\left[\\Omega_m z + (\\Omega_m - 2)(\\sqrt{1+\\Omega_m z} - 1)\\right]`:

        >>> cosmo = FLRWCosmology(H0=70.0, Omega_m=0.3, Omega_r=0.0, Omega_Lambda=0.0)
        >>> round(cosmo.luminosity_distance_mpc(1.0), 1)
        5872.3
        """
        d_c = self.comoving_distance_mpc(z)
        d_h = C_SI / self._H0_si() / _MPC_M
        if self.Omega_k > 0.0:
            sqrt_ok = np.sqrt(self.Omega_k)
            d_m = d_h / sqrt_ok * np.sinh(sqrt_ok * d_c / d_h)
        elif self.Omega_k < 0.0:
            sqrt_ok = np.sqrt(-self.Omega_k)
            d_m = d_h / sqrt_ok * np.sin(sqrt_ok * d_c / d_h)
        else:
            d_m = d_c
        return float((1.0 + z) * d_m)

    def age_gyr(self, a=1.0):
        """Cosmic age at scale factor ``a``, in Gyr.

        .. math::

            t(a) = \\int_0^a \\frac{da'}{a' H(a')}

        Parameters
        ----------
        a : float, default=1.0
            Scale factor at which to evaluate the age (``a=1`` gives the
            current age of the universe).

        Returns
        -------
        float

        Examples
        --------
        >>> cosmo = FLRWCosmology(H0=70.0, Omega_m=0.3, Omega_r=0.0, Omega_Lambda=0.7)
        >>> 12.0 < cosmo.age_gyr() < 15.0
        True
        """
        H0_si = self._H0_si()

        def integrand(ap):
            return 1.0 / (ap * H0_si * self.E(ap))

        age_s = quad(integrand, 1.0e-8, a, limit=200)[0]
        return age_s / _GYR_S

    def scale_factor_history(self, n_points=200, a_min=1.0e-3, a_max=3.0):
        """Tabulate the age of the universe at a range of scale factors, for plotting :math:`a(t)`.

        Parameters
        ----------
        n_points : int, default=200
            Number of scale-factor grid points.
        a_min, a_max : float
            Range of scale factors to tabulate.

        Returns
        -------
        t_gyr : ndarray of shape (n_points,)
            Cosmic age at each scale factor, in Gyr.
        a : ndarray of shape (n_points,)
            The scale factor grid.
        """
        a_grid = np.linspace(a_min, a_max, n_points)
        t_gyr = np.array([self.age_gyr(a) for a in a_grid])
        return t_gyr, a_grid
