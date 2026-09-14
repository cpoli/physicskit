r"""Spin dynamics: the Stern-Gerlach experiment and driven two-level (Rabi) systems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.operators import identity2, sigma_x, sigma_z
from .wave_packets import GaussianDispersion

__all__ = ["SternGerlach", "RabiProblem"]


# --- Stern-Gerlach: semiclassical spin-splitting ----------------------------


@dataclass
class SternGerlach:
    r"""Semiclassical Stern-Gerlach beam splitting.

    A spin-1/2 particle beam enters an inhomogeneous field
    :math:`B_z(y) \approx B_0 + y\,\partial_y B_z`; the spin-dependent force
    :math:`F_y = \pm\mu\,\partial_y B_z` (the sign set by the :math:`z`-spin
    eigenvalue) pushes the two spin branches apart transversely while they
    drift along :math:`x`.

    This implements the standard textbook *simplified* treatment: the two
    spin branches are modeled as two decoupled Gaussian wavepackets (rather
    than solving the fully entangled spin-position Hamiltonian), each in a
    uniform transverse force -- exactly solvable, since a wavepacket under a
    constant force keeps the free-particle spreading law
    (:class:`~physicskit.quantum.chapters.wave_packets.GaussianDispersion`)
    while its center follows the classical trajectory
    :math:`y(t) = y_0 + v_{y0}t + \tfrac12 a t^2` (Kennard's theorem /
    the Ehrenfest theorem being exact for a linear potential), picking up an
    extra "accelerated frame" phase
    :math:`\varphi(y,t) = m a t\, y/\hbar - m a^2 t^3/(6\hbar)`.

    Parameters
    ----------
    mu : float, default=1.0
        Magnetic-moment scale coupling the spin to the field gradient.
    grad_B : float, default=1.0
        Field gradient :math:`\partial_y B_z`.
    sigma0 : float, default=1.0
        Initial transverse (:math:`y`) width of the beam.
    sigma_x : float, default=2.0
        Initial longitudinal (:math:`x`, beam-propagation) width.
    k0 : float, default=5.0
        Central longitudinal wavenumber (beam velocity :math:`\hbar k_0/m`).
    hbar : float, default=1.0
        Value of :math:`\hbar` to use.
    m : float, default=1.0
        Particle mass.
    spin_weights : tuple of float, default=(0.5, 0.5)
        Probabilities :math:`(P_{+z}, P_{-z})` of the incoming spin state in
        the measurement (:math:`z`) basis -- ``(0.5, 0.5)`` is the standard
        demo of an unpolarized or :math:`x`-polarized incoming beam, split
        50/50 by the apparatus.
    """

    mu: float = 1.0
    grad_B: float = 1.0
    sigma0: float = 1.0
    sigma_x: float = 2.0
    k0: float = 5.0
    hbar: float = 1.0
    m: float = 1.0
    spin_weights: tuple = (0.5, 0.5)

    def acceleration(self, spin_sign: float) -> float:
        r"""Transverse acceleration :math:`a = \text{spin\_sign}\cdot\mu\,\partial_y B_z/m`.

        Parameters
        ----------
        spin_sign : float
            :math:`+1` for spin-up, :math:`-1` for spin-down.

        Returns
        -------
        float
        """
        return spin_sign * self.mu * self.grad_B / self.m

    def transverse_wavefunction(self, y: np.ndarray, t: float, spin_sign: float) -> np.ndarray:
        r"""The transverse packet :math:`\psi_{\pm}(y,t)` for one spin branch.

        Parameters
        ----------
        y : numpy.ndarray
            Transverse positions.
        t : float
            Time.
        spin_sign : float
            :math:`+1` for spin-up, :math:`-1` for spin-down.

        Returns
        -------
        numpy.ndarray
            Complex-valued.
        """
        a = self.acceleration(spin_sign)
        shift = 0.5 * a * t**2
        free = GaussianDispersion(x0=0.0, sigma0=self.sigma0, k0=0.0, hbar=self.hbar, m=self.m)
        psi_free = free.psi(y - shift, t)
        accel_phase = np.exp(1j * (self.m * a * t / self.hbar) * y - 1j * (self.m * a**2 * t**3) / (6 * self.hbar))
        return psi_free * accel_phase

    def longitudinal_wavefunction(self, x: np.ndarray, t: float) -> np.ndarray:
        r"""The (force-free) longitudinal packet :math:`\psi(x,t)`.

        Parameters
        ----------
        x : numpy.ndarray
            Positions along the beam-propagation axis.
        t : float
            Time.

        Returns
        -------
        numpy.ndarray
            Complex-valued.
        """
        free = GaussianDispersion(x0=0.0, sigma0=self.sigma_x, k0=self.k0, hbar=self.hbar, m=self.m)
        return free.psi(x, t)

    def joint_density(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        r"""The joint spatial density on the :math:`(x,y)` plane at time :math:`t`.

        .. math::

            P(x,y,t) = P_{+z}\,\lvert\psi(x,t)\rvert^2 \lvert\psi_+(y,t)\rvert^2
                     + P_{-z}\,\lvert\psi(x,t)\rvert^2 \lvert\psi_-(y,t)\rvert^2,

        the incoherent (classically correlated, not entangled) mixture of
        the two spin branches -- splitting into two lobes as :math:`t` grows.

        Parameters
        ----------
        x, y : numpy.ndarray
            1D position grids; combined into a meshgrid internally.
        t : float
            Time.

        Returns
        -------
        numpy.ndarray
            Real-valued, shape ``(len(x), len(y))``.
        """
        X, Y = np.meshgrid(x, y, indexing="ij")
        density_x = np.abs(self.longitudinal_wavefunction(X, t)) ** 2
        density_up = np.abs(self.transverse_wavefunction(Y, t, +1.0)) ** 2
        density_down = np.abs(self.transverse_wavefunction(Y, t, -1.0)) ** 2
        p_up, p_down = self.spin_weights
        return density_x * (p_up * density_up + p_down * density_down)

    def joint_density_stack(self, x: np.ndarray, y: np.ndarray, t_values: np.ndarray) -> np.ndarray:
        """Stack of :meth:`joint_density` snapshots over a time grid.

        Shaped for :func:`~physicskit.quantum.visualizers.wavefunctions.animate_density_2d`.

        Parameters
        ----------
        x, y : numpy.ndarray
            1D position grids.
        t_values : numpy.ndarray
            Times to evaluate at.

        Returns
        -------
        numpy.ndarray
            Real-valued, shape ``(len(t_values), len(x), len(y))``.
        """
        return np.array([self.joint_density(x, y, t) for t in t_values])


# --- Driven two-level system: Rabi oscillations -----------------------------


@dataclass
class RabiProblem:
    r"""A driven two-level system in the rotating-wave approximation (RWA).

    In the frame rotating at the drive frequency :math:`\omega_d`, the RWA
    Hamiltonian is time-independent,

    .. math::

        \hat H_\text{RWA} = \frac{\hbar}{2}\left(\Delta\,\sigma_z + \Omega\,\sigma_x\right),
        \qquad \Delta = \omega_d - \omega_0,

    giving the exact closed-form propagator (the Rabi formula)

    .. math::

        U(t) = \cos\!\left(\frac{\Omega_R t}{2}\right) I
             - i\sin\!\left(\frac{\Omega_R t}{2}\right)
               \frac{\Delta\,\sigma_z + \Omega\,\sigma_x}{\Omega_R},
        \qquad \Omega_R = \sqrt{\Delta^2+\Omega^2},

    with :math:`\Omega_R` the generalized Rabi frequency. On resonance
    (:math:`\Delta=0`) the population fully cycles between the two levels;
    detuning caps the cycling amplitude at :math:`\Omega^2/\Omega_R^2`.

    Parameters
    ----------
    omega0 : float, default=1.0
        Bare qubit splitting :math:`\omega_0`.
    omega_d : float, default=1.0
        Drive frequency :math:`\omega_d`.
    Omega : float, default=0.2
        Drive (Rabi) coupling strength :math:`\Omega`.
    hbar : float, default=1.0
        Value of :math:`\hbar` to use.
    """

    omega0: float = 1.0
    omega_d: float = 1.0
    Omega: float = 0.2
    hbar: float = 1.0

    @property
    def detuning(self) -> float:
        r"""float: The detuning :math:`\Delta = \omega_d - \omega_0`."""
        return self.omega_d - self.omega0

    @property
    def generalized_rabi_frequency(self) -> float:
        r"""float: :math:`\Omega_R = \sqrt{\Delta^2+\Omega^2}`."""
        return float(np.sqrt(self.detuning**2 + self.Omega**2))

    def propagator(self, t: float) -> np.ndarray:
        """The RWA time-evolution operator :math:`U(t)`.

        Parameters
        ----------
        t : float
            Time.

        Returns
        -------
        numpy.ndarray
            The :math:`2\\times2` unitary propagator.
        """
        Delta, Omega, Omega_R = self.detuning, self.Omega, self.generalized_rabi_frequency
        if Omega_R == 0.0:
            return identity2.copy()
        n_dot_sigma = Delta * sigma_z + Omega * sigma_x
        return np.cos(Omega_R * t / 2) * identity2 - 1j * np.sin(Omega_R * t / 2) * (n_dot_sigma / Omega_R)

    def state(self, t: float, psi0: np.ndarray | None = None) -> np.ndarray:
        r"""The evolved qubit state :math:`\psi(t) = U(t)\,\psi_0`.

        Parameters
        ----------
        t : float
            Time.
        psi0 : numpy.ndarray or None, optional
            Initial state; defaults to the ground state :math:`\lvert0\rangle`.

        Returns
        -------
        numpy.ndarray
            The 2-component qubit state.
        """
        if psi0 is None:
            psi0 = np.array([1.0, 0.0], dtype=complex)
        return self.propagator(t) @ np.asarray(psi0, dtype=complex)

    def state_trajectory(self, t_values: np.ndarray, psi0: np.ndarray | None = None) -> np.ndarray:
        """Stack of :meth:`state` snapshots over a time grid.

        Shaped for
        :func:`~physicskit.quantum.visualizers.bloch_sphere.state_to_bloch_trajectory`.

        Parameters
        ----------
        t_values : numpy.ndarray
            Times to evaluate at.
        psi0 : numpy.ndarray or None, optional
            Initial state; defaults to the ground state.

        Returns
        -------
        numpy.ndarray
            Complex-valued, shape ``(len(t_values), 2)``.
        """
        return np.array([self.state(t, psi0) for t in t_values])

    def excited_state_population(self, t: np.ndarray) -> np.ndarray:
        r"""The Rabi formula :math:`P_e(t) = (\Omega^2/\Omega_R^2)\sin^2(\Omega_R t/2)`.

        Parameters
        ----------
        t : numpy.ndarray
            Times to evaluate at.

        Returns
        -------
        numpy.ndarray
        """
        Omega_R = self.generalized_rabi_frequency
        if Omega_R == 0.0:
            return np.zeros_like(np.asarray(t, dtype=float))
        return (self.Omega**2 / Omega_R**2) * np.sin(Omega_R * np.asarray(t) / 2) ** 2
