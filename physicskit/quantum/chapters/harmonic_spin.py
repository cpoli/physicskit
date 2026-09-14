r"""The harmonic oscillator suite: Fock states, Glauber coherent states,
squeezed states, and thermal (mixed) states.

Wavefunctions are built from the numerically stable three-term recurrence
for normalized Hermite-Gaussian functions, which avoids the overflow of
evaluating raw Hermite polynomials :math:`H_n(x)` and :math:`n!` separately
at large :math:`n`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln

from .._compat import trapz

__all__ = ["HarmonicOscillator", "ThermalState"]


def _eigenfunctions_up_to(n_max: int, xi: np.ndarray) -> np.ndarray:
    r"""Normalized :math:`\psi_n(\xi)` for :math:`n=0..n_\text{max}` on the
    dimensionless coordinate :math:`\xi = \sqrt{m\omega/\hbar}\,x`, via the
    stable recurrence

    .. math::

        \psi_n = \sqrt{2/n}\,\xi\,\psi_{n-1} - \sqrt{(n-1)/n}\,\psi_{n-2}.
    """
    psi = np.empty((n_max + 1, xi.shape[0]))
    psi[0] = (1.0 / np.pi) ** 0.25 * np.exp(-(xi**2) / 2)
    if n_max >= 1:
        psi[1] = np.sqrt(2.0) * xi * psi[0]
    for n in range(2, n_max + 1):
        psi[n] = np.sqrt(2.0 / n) * xi * psi[n - 1] - np.sqrt((n - 1) / n) * psi[n - 2]
    return psi


@dataclass
class HarmonicOscillator:
    r"""The 1D quantum harmonic oscillator,

    .. math::

        \hat H = \frac{\hat p^2}{2m} + \frac{1}{2} m \omega^2 \hat x^2.

    Parameters
    ----------
    m : float, default=1.0
        Particle mass.
    omega : float, default=1.0
        Angular frequency :math:`\omega`.
    hbar : float, default=1.0
        Value of :math:`\hbar` to use.
    """

    m: float = 1.0
    omega: float = 1.0
    hbar: float = 1.0

    @property
    def x0(self) -> float:
        r"""float: Natural length scale :math:`\sqrt{\hbar/(m\omega)}`."""
        return np.sqrt(self.hbar / (self.m * self.omega))

    def energy(self, n) -> np.ndarray:
        r"""Eigenenergy :math:`E_n = \hbar\omega(n+1/2)`.

        Parameters
        ----------
        n : int or array_like
            Quantum number(s).

        Returns
        -------
        numpy.ndarray
        """
        n = np.asarray(n)
        return self.hbar * self.omega * (n + 0.5)

    def _xi(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x) / self.x0

    def eigenfunction(self, n: int, x: np.ndarray) -> np.ndarray:
        r"""The Fock (energy) eigenstate :math:`\phi_n(x)`.

        Normalized so that :math:`\int \lvert\phi_n\rvert^2\,dx = 1`.

        Parameters
        ----------
        n : int
            Quantum number.
        x : numpy.ndarray
            Positions to evaluate at.

        Returns
        -------
        numpy.ndarray
        """
        xi = self._xi(x)
        psi = _eigenfunctions_up_to(n, xi)[n]
        return psi / np.sqrt(self.x0)

    def eigenfunctions(self, n_max: int, x: np.ndarray) -> np.ndarray:
        r"""Stack of eigenstates :math:`\phi_0, \dots, \phi_{n_\text{max}}(x)`.

        Parameters
        ----------
        n_max : int
            Highest quantum number to include.
        x : numpy.ndarray
            Positions to evaluate at.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_max+1, len(x))``.
        """
        xi = self._xi(x)
        return _eigenfunctions_up_to(n_max, xi) / np.sqrt(self.x0)

    def eigenfunction_t(self, n: int, x: np.ndarray, t: float) -> np.ndarray:
        r"""The time-evolved stationary state
        :math:`\psi_n(x,t) = \phi_n(x)\, e^{-i E_n t/\hbar}`.

        A pure phase-rotating stationary state:
        :math:`\lvert\psi_n(x,t)\rvert^2` is time-independent.

        Parameters
        ----------
        n : int
            Quantum number.
        x : numpy.ndarray
            Positions to evaluate at.
        t : float
            Time.

        Returns
        -------
        numpy.ndarray
            Complex-valued :math:`\psi_n(x,t)`.
        """
        phase = np.exp(-1j * self.energy(n) * t / self.hbar)
        return self.eigenfunction(n, x).astype(complex) * phase

    def superposition_wavefunction(self, n_values, coeffs, x: np.ndarray, t: float) -> np.ndarray:
        r"""A general time-evolved superposition of Fock eigenstates.

        .. math::

            \psi(x,t) = \sum_k c_k\, \phi_{n_k}(x)\, e^{-i E_{n_k} t/\hbar},

        with beats at every pairwise frequency :math:`(E_{n_k}-E_{n_j})/\hbar`
        -- unlike a single stationary state (:meth:`eigenfunction_t`) or a
        coherent state (:meth:`coherent_wavefunction`, a rigidly oscillating
        Gaussian), a generic superposition's density genuinely reshapes over
        time.

        Parameters
        ----------
        n_values : array_like of int
            Quantum numbers included in the superposition.
        coeffs : array_like of complex
            Amplitudes :math:`c_k` (renormalized internally so
            :math:`\sum_k \lvert c_k\rvert^2=1`).
        x : numpy.ndarray
            Positions to evaluate at.
        t : float
            Time.

        Returns
        -------
        numpy.ndarray
            Complex-valued :math:`\psi(x,t)`.
        """
        n_values = np.asarray(n_values)
        coeffs = np.asarray(coeffs, dtype=complex)
        coeffs = coeffs / np.sqrt(np.sum(np.abs(coeffs) ** 2))
        phis = self.eigenfunctions(int(n_values.max()), x)[n_values]
        phases = np.exp(-1j * self.energy(n_values) * t / self.hbar)
        return (coeffs * phases) @ phis

    def superposition_trajectory(self, n_values, coeffs, x: np.ndarray, t_values: np.ndarray) -> np.ndarray:
        """Stack of :meth:`superposition_wavefunction` snapshots over a time grid.

        Shaped for :func:`~physicskit.quantum.visualizers.wavefunctions.animate_density`.

        Parameters
        ----------
        n_values : array_like of int
            Quantum numbers included in the superposition.
        coeffs : array_like of complex
            Amplitudes :math:`c_k`.
        x : numpy.ndarray
            Positions to evaluate at.
        t_values : numpy.ndarray
            Times to evaluate at.

        Returns
        -------
        numpy.ndarray
            Complex-valued, shape ``(len(t_values), len(x))``.
        """
        return np.array([self.superposition_wavefunction(n_values, coeffs, x, t) for t in t_values])

    # --- Glauber coherent states -------------------------------------

    def _coherent_coefficients(self, alpha: complex, n_max: int) -> np.ndarray:
        n = np.arange(n_max + 1)
        log_mag = -0.5 * abs(alpha) ** 2 + n * np.log(abs(alpha) + 1e-300) - 0.5 * gammaln(n + 1)
        phase = n * np.angle(alpha)
        return np.exp(log_mag) * np.exp(1j * phase)

    def coherent_n_max(self, alpha: complex, margin: float = 8.0) -> int:
        r"""Fock-basis truncation order for a coherent state :math:`\lvert\alpha\rangle`.

        Captures the Poisson-distributed photon number
        :math:`n \sim \lvert\alpha\rvert^2` out to several standard
        deviations :math:`\sqrt{\lvert\alpha\rvert^2}`.

        Parameters
        ----------
        alpha : complex
            Coherent-state amplitude.
        margin : float, default=8.0
            Number of standard deviations of safety margin.

        Returns
        -------
        int
        """
        nbar = abs(alpha) ** 2
        return int(nbar + margin * np.sqrt(nbar + 1) + 20)

    def coherent_wavefunction(self, alpha: complex, x: np.ndarray, t: float = 0.0, n_max: int | None = None) -> np.ndarray:
        r"""The Glauber coherent state :math:`\lvert\alpha\rangle` in position space.

        .. math::

            \psi_\alpha(x,t) = \sum_n c_n\, \phi_n(x)\, e^{-i E_n t/\hbar},
            \qquad c_n = e^{-\lvert\alpha\rvert^2/2}\frac{\alpha^n}{\sqrt{n!}},

        the Poissonian Fock coefficients of :math:`\lvert\alpha\rangle`. The
        packet is a Gaussian of fixed shape that oscillates rigidly,
        matching the classical trajectory (see :meth:`coherent_trajectory`).

        Parameters
        ----------
        alpha : complex
            Coherent-state amplitude.
        x : numpy.ndarray
            Positions to evaluate at.
        t : float, default=0.0
            Time.
        n_max : int or None, optional
            Fock-basis truncation order; defaults to :meth:`coherent_n_max`.

        Returns
        -------
        numpy.ndarray
            Complex-valued :math:`\psi_\alpha(x,t)`.
        """
        if n_max is None:
            n_max = self.coherent_n_max(alpha)
        c = self._coherent_coefficients(alpha, n_max)
        phis = self.eigenfunctions(n_max, x)
        phases = np.exp(-1j * self.energy(np.arange(n_max + 1)) * t / self.hbar)
        return (c * phases) @ phis

    def coherent_trajectory(self, alpha: complex, t: np.ndarray):
        r"""Classical mean position/momentum of a coherent state.

        .. math::

            \langle x\rangle(t) = x_0\sqrt2\,\mathrm{Re}\!\left[\alpha e^{-i\omega t}\right],
            \qquad
            \langle p\rangle(t) = \sqrt{2\hbar m \omega}\,\mathrm{Im}\!\left[\alpha e^{-i\omega t}\right].

        Parameters
        ----------
        alpha : complex
            Coherent-state amplitude.
        t : numpy.ndarray
            Times to evaluate at.

        Returns
        -------
        x_t, p_t : numpy.ndarray
        """
        t = np.asarray(t)
        a_t = alpha * np.exp(-1j * self.omega * t)
        x_t = self.x0 * np.sqrt(2.0) * a_t.real
        p_t = np.sqrt(2.0 * self.hbar * self.m * self.omega) * a_t.imag
        return x_t, p_t

    # --- Squeezed states -------------------------------------------------

    def squeezed_vacuum_wavefunction(self, x: np.ndarray, r: float, phi: float = 0.0) -> np.ndarray:
        r"""Position-space squeezed vacuum :math:`\lvert r,\phi\rangle`.

        A Gaussian with quadrature variance

        .. math::

            \Delta x^2 = \frac{\hbar}{2 m \omega}\, e^{-2r}

        (for :math:`\phi=0`, squeezed below the zero-point value when
        :math:`r>0`), at the expense of
        :math:`\Delta p^2 = (\hbar m \omega/2)\, e^{2r}`.

        Parameters
        ----------
        x : numpy.ndarray
            Positions to evaluate at.
        r : float
            Squeezing parameter.
        phi : float, default=0.0
            Squeezing angle (rotates the quadrature ellipse).

        Returns
        -------
        numpy.ndarray
            Complex-valued (real when ``phi=0``) squeezed-vacuum wavefunction.
        """
        x = np.asarray(x)
        var_x = (self.hbar / (2 * self.m * self.omega)) * np.exp(-2 * r)
        norm = (1.0 / (2 * np.pi * var_x)) ** 0.25
        gaussian = norm * np.exp(-(x**2) / (4 * var_x))
        # A squeezing-angle phi rotates the quadrature ellipse; encode it as
        # a quadratic chirp phase, exact for the Gaussian squeezed state.
        chirp = np.exp(1j * np.tan(phi) * x**2 / (4 * var_x)) if phi != 0.0 else 1.0
        return gaussian * chirp

    def squeezed_uncertainties(self, r: float):
        r"""Position/momentum uncertainties of the :math:`r`-squeezed vacuum
        (:math:`\phi=0` axis).

        Parameters
        ----------
        r : float
            Squeezing parameter.

        Returns
        -------
        dx, dp : float
            :math:`\Delta x`, :math:`\Delta p`; their product stays at the
            minimum :math:`\hbar/2` for all :math:`r`.
        """
        dx = np.sqrt(self.hbar / (2 * self.m * self.omega) * np.exp(-2 * r))
        dp = np.sqrt(self.hbar * self.m * self.omega / 2 * np.exp(2 * r))
        return dx, dp

    def zero_point_uncertainties(self):
        """Position/momentum uncertainties of the ordinary ground state.

        Equivalent to ``squeezed_uncertainties(r=0)``, provided for comparison.

        Returns
        -------
        dx, dp : float
        """
        return self.squeezed_uncertainties(r=0.0)

    # --- Thermal (mixed) states -------------------------------------------

    def thermal_populations(self, T: float, n_max: int, k_B: float = 1.0) -> np.ndarray:
        r"""Diagonal Boltzmann populations of the thermal density matrix.

        .. math::

            \rho_{nn} = (1-e^{-x})\, e^{-nx}, \qquad x = \frac{\hbar\omega}{k_B T},

        for :math:`\hat\rho = e^{-\beta\hat H}/Z` (``T=0`` returns the pure
        ground state).

        Parameters
        ----------
        T : float
            Temperature (``T<=0`` returns the ground state).
        n_max : int
            Highest Fock level to include.
        k_B : float, default=1.0
            Boltzmann constant.

        Returns
        -------
        numpy.ndarray
            Populations :math:`\rho_{nn}`, shape ``(n_max+1,)``, summing to 1.
        """
        n = np.arange(n_max + 1)
        if T <= 0:
            p = np.zeros(n_max + 1)
            p[0] = 1.0
            return p
        x = self.hbar * self.omega / (k_B * T)
        p = (1 - np.exp(-x)) * np.exp(-n * x)
        return p / p.sum()

    def thermal_position_distribution(self, x: np.ndarray, T: float, k_B: float = 1.0, n_max: int = 200) -> np.ndarray:
        r"""The (temperature-broadened) position-space probability density.

        .. math::

            P(x) = \sum_n \rho_{nn}\, \lvert\phi_n(x)\rvert^2.

        Parameters
        ----------
        x : numpy.ndarray
            Positions to evaluate at.
        T : float
            Temperature.
        k_B : float, default=1.0
            Boltzmann constant.
        n_max : int, default=200
            Highest Fock level to include.

        Returns
        -------
        numpy.ndarray
        """
        pops = self.thermal_populations(T, n_max, k_B)
        phis = self.eigenfunctions(n_max, x)
        return pops @ (phis**2)

    def thermal_position_variance_analytic(self, T: float, k_B: float = 1.0) -> float:
        r"""Closed-form position variance of the thermal state.

        .. math::

            \langle x^2\rangle = \frac{\hbar}{2m\omega}
                \coth\!\left(\frac{\hbar\omega}{2 k_B T}\right),

        reducing to the zero-point value :math:`\hbar/2m\omega` as
        :math:`T\to 0`.

        Parameters
        ----------
        T : float
            Temperature.
        k_B : float, default=1.0
            Boltzmann constant.

        Returns
        -------
        float
        """
        if T <= 0:
            return self.hbar / (2 * self.m * self.omega)
        x = self.hbar * self.omega / (2 * k_B * T)
        return (self.hbar / (2 * self.m * self.omega)) / np.tanh(x)


@dataclass
class ThermalState:
    """Convenience wrapper bundling an oscillator with a fixed temperature.

    Parameters
    ----------
    oscillator : HarmonicOscillator
        The underlying oscillator.
    T : float
        Temperature.
    k_B : float, default=1.0
        Boltzmann constant.
    n_max : int, default=200
        Fock-basis truncation order used for the thermal sum.
    """

    oscillator: HarmonicOscillator
    T: float
    k_B: float = 1.0
    n_max: int = 200

    def populations(self) -> np.ndarray:
        """See :meth:`HarmonicOscillator.thermal_populations`.

        Returns
        -------
        numpy.ndarray
        """
        return self.oscillator.thermal_populations(self.T, self.n_max, self.k_B)

    def position_distribution(self, x: np.ndarray) -> np.ndarray:
        """See :meth:`HarmonicOscillator.thermal_position_distribution`.

        Parameters
        ----------
        x : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        return self.oscillator.thermal_position_distribution(x, self.T, self.k_B, self.n_max)

    def variance(self) -> float:
        """See :meth:`HarmonicOscillator.thermal_position_variance_analytic`.

        Returns
        -------
        float
        """
        return self.oscillator.thermal_position_variance_analytic(self.T, self.k_B)

    def check_normalization(self, x: np.ndarray) -> float:
        """Integrate :meth:`position_distribution` over ``x`` (should be 1).

        Parameters
        ----------
        x : numpy.ndarray

        Returns
        -------
        float
        """
        return float(trapz(self.position_distribution(x), x))
