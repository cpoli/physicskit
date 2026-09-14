r"""Non-stationary wave dynamics: free dispersion, the twin-slit experiment,
and quantum revivals in the infinite square well.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from .._compat import trapz
from ..core.solvers import SplitOperatorSolver2D

__all__ = [
    "free_gaussian_wavepacket",
    "GaussianDispersion",
    "TwinSlit",
    "QuantumRevival",
    "double_slit_potential",
    "propagate_double_slit",
]


def free_gaussian_wavepacket(x: np.ndarray, x0: float, sigma0: float, k0: float) -> np.ndarray:
    r"""A minimum-uncertainty Gaussian wavepacket at :math:`t=0`.

    .. math::

        \psi(x,0) = (2\pi\sigma_0^2)^{-1/4}\,
            \exp\!\left[-\frac{(x-x_0)^2}{4\sigma_0^2}\right] e^{ik_0 x}.

    Parameters
    ----------
    x : numpy.ndarray
        Positions to evaluate at.
    x0 : float
        Center of the packet.
    sigma0 : float
        Initial width (:math:`\Delta x = \sigma_0`).
    k0 : float
        Central wavenumber.

    Returns
    -------
    numpy.ndarray
    """
    norm = (2 * np.pi * sigma0**2) ** (-0.25)
    return norm * np.exp(-((x - x0) ** 2) / (4 * sigma0**2)) * np.exp(1j * k0 * x)


@dataclass
class GaussianDispersion:
    r"""Free-particle spreading of a Gaussian wavepacket.

    The exact analytic solution of the free-particle Schrodinger equation
    for a Gaussian initial condition; used to benchmark the FFT
    split-operator propagator.

    Parameters
    ----------
    x0 : float, default=0.0
        Initial center of the packet.
    sigma0 : float, default=1.0
        Initial width.
    k0 : float, default=5.0
        Central wavenumber.
    hbar : float, default=1.0
        Value of :math:`\hbar` to use.
    m : float, default=1.0
        Particle mass.
    """

    x0: float = 0.0
    sigma0: float = 1.0
    k0: float = 5.0
    hbar: float = 1.0
    m: float = 1.0

    def width(self, t: np.ndarray) -> np.ndarray:
        r"""The spreading width :math:`\sigma(t)`.

        .. math::

            \sigma(t) = \sigma_0\sqrt{1 + \left(\frac{\hbar t}{2m\sigma_0^2}\right)^2}.

        Parameters
        ----------
        t : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        t = np.asarray(t)
        return self.sigma0 * np.sqrt(1 + (self.hbar * t / (2 * self.m * self.sigma0**2)) ** 2)

    def center(self, t: np.ndarray) -> np.ndarray:
        r"""The ballistic center :math:`x_0 + (\hbar k_0/m)\,t`.

        Parameters
        ----------
        t : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        return self.x0 + (self.hbar * self.k0 / self.m) * np.asarray(t)

    def psi(self, x: np.ndarray, t: float) -> np.ndarray:
        """The exact free-particle wavefunction :math:`\\psi(x,t)`.

        Parameters
        ----------
        x : numpy.ndarray
            Positions to evaluate at.
        t : float
            Time.

        Returns
        -------
        numpy.ndarray
            Complex-valued.
        """
        m, hbar, sigma0, k0, x0 = self.m, self.hbar, self.sigma0, self.k0, self.x0
        a = sigma0**2 + 1j * hbar * t / (2 * m)
        norm = (2 * np.pi * sigma0**2) ** (-0.25) * np.sqrt(sigma0**2 / a)
        envelope = np.exp(-((x - x0 - hbar * k0 * t / m) ** 2) / (4 * a))
        plane = np.exp(1j * k0 * (x - x0) - 1j * hbar * k0**2 * t / (2 * m))
        return norm * envelope * plane

    def density(self, x: np.ndarray, t: float) -> np.ndarray:
        r"""The probability density :math:`\lvert\psi(x,t)\rvert^2`.

        Parameters
        ----------
        x : numpy.ndarray
            Positions to evaluate at.
        t : float
            Time.

        Returns
        -------
        numpy.ndarray
        """
        return np.abs(self.psi(x, t)) ** 2

    def trajectory(self, x: np.ndarray, t_values: np.ndarray) -> np.ndarray:
        r"""Stack of :math:`\psi(x,t)` snapshots over a time grid.

        Shaped for :func:`~physicskit.quantum.visualizers.wavefunctions.animate_density`.

        Parameters
        ----------
        x : numpy.ndarray
            Positions to evaluate at.
        t_values : numpy.ndarray
            Times to evaluate at.

        Returns
        -------
        numpy.ndarray
            Complex-valued, shape ``(len(t_values), len(x))``.
        """
        return np.array([self.psi(x, t) for t in t_values])


@dataclass
class TwinSlit:
    r"""Young's double-slit for a massive particle.

    Built from two coherent Gaussian point sources at the slit plane,
    propagated freely to a downstream screen -- the quantum interference
    pattern is the same :math:`\lvert\psi_1+\psi_2\rvert^2` mechanism as
    light, now for massive-particle probability amplitude.

    Parameters
    ----------
    slit_separation : float, default=4.0
        Distance between the two slits.
    slit_width : float, default=0.4
        Width of each slit (sets the Gaussian source envelope).
    k0 : float, default=10.0
        Central wavenumber of the incident beam.
    hbar : float, default=1.0
        Value of :math:`\hbar` to use.
    m : float, default=1.0
        Particle mass.
    """

    slit_separation: float = 4.0
    slit_width: float = 0.4
    k0: float = 10.0
    hbar: float = 1.0
    m: float = 1.0

    def amplitude(self, x: np.ndarray, screen_distance: float, t: float | None = None) -> np.ndarray:
        r"""Fraunhofer-style amplitude on a screen at distance :math:`L`.

        Sums the two slit contributions with their propagation phase
        :math:`kL + kx^2/2L` (paraxial/Fresnel approximation), :math:`x`
        measured along the screen.

        Parameters
        ----------
        x : numpy.ndarray
            Screen positions.
        screen_distance : float
            Distance :math:`L` from the slit plane to the screen.
        t : float or None, optional
            Unused; retained for API symmetry with time-dependent amplitudes.

        Returns
        -------
        numpy.ndarray
            Complex-valued amplitude.
        """
        k = self.k0
        L = screen_distance
        d = self.slit_separation
        w = self.slit_width

        def slit_amp(y0):
            # each slit acts as a Gaussian source of width w
            r = np.sqrt(L**2 + (x - y0) ** 2)
            phase = k * r
            sinc_envelope = np.exp(-((x - y0) ** 2) * (w**2) / (4 * L**2 + 1e-12))
            return sinc_envelope * np.exp(1j * phase) / np.sqrt(r)

        return slit_amp(d / 2) + slit_amp(-d / 2)

    def intensity(self, x: np.ndarray, screen_distance: float) -> np.ndarray:
        r"""Normalized interference intensity :math:`\lvert\psi_1+\psi_2\rvert^2`.

        Parameters
        ----------
        x : numpy.ndarray
            Screen positions.
        screen_distance : float
            Distance from the slit plane to the screen.

        Returns
        -------
        numpy.ndarray
            Normalized so that :math:`\int I\,dx = 1`.
        """
        amp = self.amplitude(x, screen_distance)
        I = np.abs(amp) ** 2
        return I / trapz(I, x)


def double_slit_potential(
    X: np.ndarray,
    Y: np.ndarray,
    wall_x: float = 0.0,
    wall_thickness: float = 0.5,
    slit_separation: float = 3.0,
    slit_width: float = 0.8,
    wall_height: float = 400.0,
) -> np.ndarray:
    r"""An opaque wall at :math:`x=\text{wall\_x}` pierced by two gaps.

    :math:`V=\text{wall\_height}` inside the wall slab except within two
    strips of width ``slit_width`` centered at
    :math:`y = \pm\text{slit\_separation}/2`, where :math:`V=0`.

    Parameters
    ----------
    X, Y : numpy.ndarray
        Coordinate meshgrid (e.g. ``SplitOperatorSolver2D.X/.Y``).
    wall_x : float, default=0.0
        Position of the wall along :math:`x`.
    wall_thickness : float, default=0.5
        Thickness of the wall along :math:`x`.
    slit_separation : float, default=3.0
        Center-to-center distance between the two slits.
    slit_width : float, default=0.8
        Width of each slit (gap in the wall).
    wall_height : float, default=400.0
        Potential inside the opaque part of the wall.

    Returns
    -------
    numpy.ndarray
        :math:`V(x,y)` on the grid.
    """
    in_wall = np.abs(X - wall_x) <= wall_thickness / 2
    in_slit = (np.abs(Y - slit_separation / 2) <= slit_width / 2) | (np.abs(Y + slit_separation / 2) <= slit_width / 2)
    return np.where(in_wall & ~in_slit, wall_height, 0.0)


def propagate_double_slit(
    x: np.ndarray,
    y: np.ndarray,
    x0: float = -8.0,
    k0: float = 8.0,
    sigma_x: float = 0.7,
    sigma_y: float = 4.0,
    wall_x: float = 0.0,
    wall_thickness: float = 0.5,
    slit_separation: float = 3.0,
    slit_width: float = 0.8,
    wall_height: float = 400.0,
    hbar: float = 1.0,
    m: float = 1.0,
    dt: float = 1e-4,
    n_steps: int = 4000,
    save_every: int = 40,
):
    r"""Propagate a wavepacket through :func:`double_slit_potential`.

    A genuinely time-propagated double-slit: a Gaussian packet, elongated
    along :math:`y` (to illuminate both slits coherently) and moving along
    :math:`+x` with wavenumber ``k0``, is evolved with
    :class:`~physicskit.quantum.core.solvers.SplitOperatorSolver2D` through the
    two-slit wall, producing the interference pattern building up downstream
    from genuine wave dynamics rather than the far-field Fraunhofer formula
    (:class:`TwinSlit`).

    Parameters
    ----------
    x, y : numpy.ndarray
        Uniform spatial grids.
    x0 : float, default=-8.0
        Initial center of the packet along :math:`x` (should sit well
        upstream of ``wall_x``).
    k0 : float, default=8.0
        Central wavenumber along :math:`x`.
    sigma_x, sigma_y : float, default=0.7, 4.0
        Initial packet widths (``sigma_y`` should span both slits).
    wall_x, wall_thickness, slit_separation, slit_width, wall_height : float
        Passed to :func:`double_slit_potential`.
    hbar, m, dt : float
        Propagation parameters, passed to
        :class:`~physicskit.quantum.core.solvers.SplitOperatorSolver2D`.
    n_steps : int, default=4000
        Total number of time steps.
    save_every : int, default=40
        Save a snapshot every this many steps.

    Returns
    -------
    X, Y : numpy.ndarray
        The coordinate meshgrid.
    frames : numpy.ndarray
        Complex wavefunction snapshots, shape ``(n_saved, len(x), len(y))``.
    times : numpy.ndarray
        The time of each snapshot.
    """

    def V(xx, yy):
        return double_slit_potential(xx, yy, wall_x, wall_thickness, slit_separation, slit_width, wall_height)

    solver = SplitOperatorSolver2D(x, y, V, hbar=hbar, m=m, dt=dt)
    X, Y = solver.X, solver.Y
    psi0 = np.exp(-((X - x0) ** 2) / (4 * sigma_x**2) - Y**2 / (4 * sigma_y**2)) * np.exp(1j * k0 * X)
    frames, times = solver.propagate(psi0.astype(complex), n_steps, save_every=save_every)
    return X, Y, frames, times


@njit(cache=True)
def _revival_sum(coeffs_re, coeffs_im, sin_table, n_values, energies_over_hbar, t):
    """Sum_n c_n phi_n(x) exp(-i E_n t/hbar) evaluated on the box eigenbasis,
    vectorized manually for numba (coeffs and sin_table are precomputed)."""
    n_states, n_x = sin_table.shape
    out_re = np.zeros(n_x)
    out_im = np.zeros(n_x)
    for k in range(n_states):
        phase = -energies_over_hbar[k] * t
        cos_p = np.cos(phase)
        sin_p = np.sin(phase)
        # (c_re + i c_im)(cos_p + i sin_p) = (c_re cos_p - c_im sin_p) + i(c_re sin_p + c_im cos_p)
        a = coeffs_re[k] * cos_p - coeffs_im[k] * sin_p
        b = coeffs_re[k] * sin_p + coeffs_im[k] * cos_p
        for j in range(n_x):
            s = sin_table[k, j]
            out_re[j] += a * s
            out_im[j] += b * s
    return out_re, out_im


@dataclass
class QuantumRevival:
    r"""A localized wavepacket in an infinite square well of width :math:`L`.

    Because :math:`E_n = n^2\pi^2\hbar^2/(2mL^2)` is quadratic in :math:`n`,
    the packet disperses into apparent chaos yet exactly reassembles into
    replicas of :math:`\psi(x,0)` at fractional multiples of the revival time

    .. math::

        t_\text{rev} = \frac{4mL^2}{\pi\hbar}

    (since :math:`E_n = n^2\cdot(\hbar\cdot 2\pi/t_\text{rev})/2`), with a
    mirror-image replica at :math:`t_\text{rev}/2`, and reduced-scale
    "fractional revivals" (clones/anti-clones) at rational fractions
    :math:`t_\text{rev}\cdot(p/q)`.

    Parameters
    ----------
    L : float, default=1.0
        Well width.
    hbar : float, default=1.0
        Value of :math:`\hbar` to use.
    m : float, default=1.0
        Particle mass.
    n_max : int, default=400
        Number of box eigenstates used in the expansion.
    """

    L: float = 1.0
    hbar: float = 1.0
    m: float = 1.0
    n_max: int = 400

    def energy(self, n: np.ndarray) -> np.ndarray:
        r"""Box eigenenergy :math:`E_n = n^2\pi^2\hbar^2/(2mL^2)`.

        Parameters
        ----------
        n : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        return (n**2 * np.pi**2 * self.hbar**2) / (2 * self.m * self.L**2)

    @property
    def revival_time(self) -> float:
        r"""float: The revival time :math:`t_\text{rev} = 4mL^2/(\pi\hbar)`."""
        return 4 * self.m * self.L**2 / (np.pi * self.hbar)

    def eigenbasis_coefficients(self, psi0_func, n_grid: int = 4000) -> np.ndarray:
        r"""Box-eigenbasis expansion coefficients of an initial state.

        .. math::

            c_n = \langle\phi_n\rvert\psi_0\rangle, \qquad
            \phi_n(x) = \sqrt{2/L}\,\sin(n\pi x/L).

        Parameters
        ----------
        psi0_func : callable
            Function ``psi0_func(x) -> array`` giving the initial state.
        n_grid : int, default=4000
            Number of quadrature points used to evaluate the overlap integral.

        Returns
        -------
        numpy.ndarray
            Coefficients :math:`c_n`, shape ``(n_max,)``.
        """
        x = np.linspace(0, self.L, n_grid)
        psi0 = psi0_func(x)
        n = np.arange(1, self.n_max + 1)
        sin_table = np.sqrt(2.0 / self.L) * np.sin(np.outer(n, np.pi * x / self.L))
        c = trapz(sin_table * psi0[None, :], x, axis=1)
        return c

    def wavefunction(self, x: np.ndarray, t: float, coeffs: np.ndarray) -> np.ndarray:
        r"""Evaluate the evolved state :math:`\psi(x,t) = \sum_n c_n \phi_n(x)
        e^{-iE_n t/\hbar}`.

        Parameters
        ----------
        x : numpy.ndarray
            Positions to evaluate at.
        t : float
            Time.
        coeffs : numpy.ndarray
            Expansion coefficients from :meth:`eigenbasis_coefficients`.

        Returns
        -------
        numpy.ndarray
            Complex-valued :math:`\psi(x,t)`.
        """
        n = np.arange(1, self.n_max + 1)
        sin_table = np.sqrt(2.0 / self.L) * np.sin(np.outer(n, np.pi * x / self.L))
        energies_over_hbar = self.energy(n) / self.hbar
        out_re, out_im = _revival_sum(
            np.ascontiguousarray(coeffs.real),
            np.ascontiguousarray(coeffs.imag),
            np.ascontiguousarray(sin_table),
            n,
            energies_over_hbar,
            t,
        )
        return out_re + 1j * out_im

    def gaussian_initial_state(self, x0: float, sigma: float):
        """Build a normalized Gaussian initial-state function for the box.

        Parameters
        ----------
        x0 : float
            Center of the packet (should lie well inside ``(0, L)``).
        sigma : float
            Width of the packet.

        Returns
        -------
        callable
            A function ``psi0(x) -> array`` (zero outside ``[0, L]``).
        """

        def psi0(x):
            packet = np.exp(-((x - x0) ** 2) / (4 * sigma**2))
            packet[x <= 0] = 0.0
            packet[x >= self.L] = 0.0
            norm = np.sqrt(trapz(np.abs(packet) ** 2, np.linspace(0, self.L, x.shape[0])))
            return packet / norm

        return psi0

    def fidelity_to_initial(self, x: np.ndarray, t: float, coeffs: np.ndarray, psi0: np.ndarray) -> float:
        r"""Fidelity :math:`\lvert\langle\psi_0\rvert\psi(t)\rangle\rvert^2`
        of the evolved state to the initial state.

        Approaches 1 at (fractional) revival times.

        Parameters
        ----------
        x : numpy.ndarray
            Positions to evaluate at.
        t : float
            Time.
        coeffs : numpy.ndarray
            Expansion coefficients from :meth:`eigenbasis_coefficients`.
        psi0 : numpy.ndarray
            The initial state sampled on ``x``.

        Returns
        -------
        float
        """
        psi_t = self.wavefunction(x, t, coeffs)
        overlap = trapz(np.conj(psi0) * psi_t, x)
        return float(np.abs(overlap) ** 2)
