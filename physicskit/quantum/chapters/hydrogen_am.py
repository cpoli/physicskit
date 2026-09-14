r"""3D spherical harmonics and hydrogen-atom orbitals :math:`\lvert n,l,m\rangle`."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import genlaguerre

from .._compat import sph_harm_y, trapz

__all__ = [
    "spherical_harmonic",
    "radial_wavefunction",
    "HydrogenOrbital",
    "orbital_superposition_psi",
    "orbital_superposition_density",
]

A0 = 1.0
"""float: Bohr radius, in atomic units (length measured in units of :math:`a_0`)."""


def spherical_harmonic(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    r"""The spherical harmonic :math:`Y_l^m(\theta,\phi)`.

    Parameters
    ----------
    l : int
        Orbital angular momentum quantum number.
    m : int
        Magnetic quantum number (:math:`\lvert m\rvert \le l`).
    theta : numpy.ndarray
        Polar angle, in :math:`[0, \pi]`.
    phi : numpy.ndarray
        Azimuthal angle, in :math:`[0, 2\pi)`.

    Returns
    -------
    numpy.ndarray
        Complex-valued :math:`Y_l^m(\theta,\phi)`.
    """
    return sph_harm_y(l, m, theta, phi)


def radial_wavefunction(n: int, l: int, r: np.ndarray, Z: int = 1, a0: float = A0) -> np.ndarray:
    r"""The hydrogenic radial wavefunction :math:`R_{nl}(r)`.

    Normalized so that :math:`\int_0^\infty \lvert R_{nl}(r)\rvert^2 r^2\,dr = 1`.

    Parameters
    ----------
    n : int
        Principal quantum number.
    l : int
        Orbital angular momentum quantum number (:math:`0 \le l < n`).
    r : numpy.ndarray
        Radii to evaluate at.
    Z : int, default=1
        Nuclear charge.
    a0 : float, default=1.0
        Bohr radius.

    Returns
    -------
    numpy.ndarray

    Raises
    ------
    ValueError
        If ``l >= n``.
    """
    if l >= n:
        raise ValueError("Require l < n.")
    rho = 2 * Z * r / (n * a0)
    L = genlaguerre(n - l - 1, 2 * l + 1)(rho)
    from scipy.special import gammaln

    log_norm = 1.5 * np.log(2 * Z / (n * a0)) + 0.5 * (gammaln(n - l) - np.log(2 * n) - gammaln(n + l + 1))
    norm = np.exp(log_norm)
    return norm * np.exp(-rho / 2) * rho**l * L


@dataclass
class HydrogenOrbital:
    r"""The hydrogenic wavefunction
    :math:`\psi_{nlm}(r,\theta,\phi) = R_{nl}(r)\, Y_l^m(\theta,\phi)`.

    Parameters
    ----------
    n : int
        Principal quantum number.
    l : int
        Orbital angular momentum quantum number (:math:`0 \le l < n`).
    m : int
        Magnetic quantum number (:math:`\lvert m\rvert \le l`).
    Z : int, default=1
        Nuclear charge.
    a0 : float, default=1.0
        Bohr radius.

    Raises
    ------
    ValueError
        If the quantum numbers violate :math:`0 \le l < n` or
        :math:`\lvert m\rvert \le l`.
    """

    n: int
    l: int
    m: int
    Z: int = 1
    a0: float = A0

    def __post_init__(self):
        if not (0 <= self.l < self.n):
            raise ValueError("Require 0 <= l < n.")
        if abs(self.m) > self.l:
            raise ValueError("Require |m| <= l.")

    @property
    def energy(self) -> float:
        r"""float: Energy :math:`E_n = -Z^2/(2n^2)` Hartree (atomic units,
        :math:`\hbar = m_e = e = 1`)."""
        return -(self.Z**2) / (2 * self.n**2)

    def radial(self, r: np.ndarray) -> np.ndarray:
        """The radial part :math:`R_{nl}(r)`; see :func:`radial_wavefunction`.

        Parameters
        ----------
        r : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        return radial_wavefunction(self.n, self.l, r, self.Z, self.a0)

    def angular(self, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """The angular part :math:`Y_l^m(\\theta,\\phi)`; see :func:`spherical_harmonic`.

        Parameters
        ----------
        theta, phi : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        return spherical_harmonic(self.l, self.m, theta, phi)

    def psi(self, r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        r"""The full wavefunction :math:`\psi_{nlm}(r,\theta,\phi)`.

        Parameters
        ----------
        r, theta, phi : numpy.ndarray

        Returns
        -------
        numpy.ndarray
            Complex-valued.
        """
        return self.radial(r) * self.angular(theta, phi)

    def density(self, r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        r"""The probability density :math:`\lvert\psi_{nlm}\rvert^2`.

        Parameters
        ----------
        r, theta, phi : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        return np.abs(self.psi(r, theta, phi)) ** 2

    def radial_density(self, r: np.ndarray) -> np.ndarray:
        r"""The radial probability density :math:`P(r) = r^2 \lvert R_{nl}(r)\rvert^2`.

        The probability of finding the electron in a shell
        :math:`[r, r+dr]`, already integrated over solid angle.

        Parameters
        ----------
        r : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        return r**2 * np.abs(self.radial(r)) ** 2

    def most_probable_radius(self, r_max: float | None = None, n_points: int = 20000) -> float:
        """The radius at which :meth:`radial_density` is maximal.

        Parameters
        ----------
        r_max : float or None, optional
            Upper bound of the search range; defaults to a value scaled to
            the orbital's expected extent.
        n_points : int, default=20000
            Number of grid points used for the search.

        Returns
        -------
        float
        """
        if r_max is None:
            r_max = 4 * self.n**2 * self.a0 / self.Z + 10
        r = np.linspace(1e-6, r_max, n_points)
        P = self.radial_density(r)
        return float(r[np.argmax(P)])

    def check_normalization(self, r_max: float | None = None, n_points: int = 20000) -> float:
        r"""Integrate :meth:`radial_density` over :math:`[0, r_\text{max}]` (should be 1).

        Parameters
        ----------
        r_max : float or None, optional
            Upper integration limit; defaults to a value scaled to the
            orbital's expected extent.
        n_points : int, default=20000
            Number of quadrature points.

        Returns
        -------
        float
        """
        if r_max is None:
            r_max = 6 * self.n**2 * self.a0 / self.Z + 20
        r = np.linspace(1e-8, r_max, n_points)
        return float(trapz(self.radial_density(r), r))


def orbital_superposition_psi(
    orb_a: HydrogenOrbital, orb_b: HydrogenOrbital, r: np.ndarray, theta: np.ndarray, phi: np.ndarray, t: float, ca: complex = 2**-0.5, cb: complex = 2**-0.5
) -> np.ndarray:
    r"""A coherent superposition of two hydrogen eigenstates at time :math:`t`.

    .. math::

        \psi(t) = c_a\,\psi_a\, e^{-iE_a t} + c_b\,\psi_b\, e^{-iE_b t},

    beating at the Bohr frequency :math:`\omega_{ab}=E_a-E_b` between the two
    stationary densities -- genuine dynamics, unlike either eigenstate alone
    (whose density is static).

    Parameters
    ----------
    orb_a, orb_b : HydrogenOrbital
        The two eigenstates in the superposition (should differ in energy
        for the beating to be visible).
    r, theta, phi : numpy.ndarray
        Coordinates to evaluate at (e.g. from a Cartesian grid, as in
        :func:`~physicskit.quantum.visualizers.orbitals.orbital_density_grid`).
    t : float
        Time.
    ca, cb : complex, default=1/sqrt(2) each
        Superposition amplitudes (renormalized internally).

    Returns
    -------
    numpy.ndarray
        Complex-valued :math:`\psi(r,\theta,\phi,t)`.
    """
    norm = np.sqrt(np.abs(ca) ** 2 + np.abs(cb) ** 2)
    ca, cb = ca / norm, cb / norm
    phase_a = np.exp(-1j * orb_a.energy * t)
    phase_b = np.exp(-1j * orb_b.energy * t)
    return ca * phase_a * orb_a.psi(r, theta, phi) + cb * phase_b * orb_b.psi(r, theta, phi)


def orbital_superposition_density(
    orb_a: HydrogenOrbital, orb_b: HydrogenOrbital, r: np.ndarray, theta: np.ndarray, phi: np.ndarray, t: float, ca: complex = 2**-0.5, cb: complex = 2**-0.5
) -> np.ndarray:
    r"""The probability density of :func:`orbital_superposition_psi`.

    Parameters
    ----------
    orb_a, orb_b : HydrogenOrbital
    r, theta, phi : numpy.ndarray
    t : float
    ca, cb : complex, default=1/sqrt(2) each

    Returns
    -------
    numpy.ndarray
    """
    return np.abs(orbital_superposition_psi(orb_a, orb_b, r, theta, phi, t, ca, cb)) ** 2
