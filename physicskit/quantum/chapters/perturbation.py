r"""Stark and Zeeman splitting, and Floquet analysis of periodically driven
quantum boxes (multiphoton resonance, photon-assisted tunneling).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._compat import trapz
from ..core.solvers import SplitOperatorSolver1D

__all__ = ["zeeman_splitting", "zeeman_spectrum", "linear_stark_shift", "stark_n2_quartet", "FloquetDrivenBox"]


# --- Zeeman effect -----------------------------------------------------------


def zeeman_splitting(m_l: int, m_s: float, B: float, mu_B: float = 1.0, g_s: float = 2.0, hbar: float = 1.0) -> float:
    r"""First-order (weak-field, no spin-orbit) Zeeman energy shift.

    .. math::

        \Delta E = \mu_B B\,(m_l + g_s m_s),

    in the :math:`\lvert l, m_l; s, m_s\rangle` basis.

    Parameters
    ----------
    m_l : int
        Orbital magnetic quantum number.
    m_s : float
        Spin magnetic quantum number.
    B : float
        Magnetic field strength.
    mu_B : float, default=1.0
        Bohr magneton.
    g_s : float, default=2.0
        Electron spin g-factor.
    hbar : float, default=1.0
        Value of :math:`\hbar` (unused in this first-order formula; kept
        for API symmetry with other perturbation functions).

    Returns
    -------
    float
    """
    return mu_B * B * (m_l + g_s * m_s)


def zeeman_spectrum(l: int, B: float, mu_B: float = 1.0, g_s: float = 2.0) -> np.ndarray:
    r"""All :math:`(m_l, m_s)` sublevel shifts for orbital number :math:`l`, spin-1/2.

    Parameters
    ----------
    l : int
        Orbital angular momentum quantum number.
    B : float
        Magnetic field strength.
    mu_B : float, default=1.0
        Bohr magneton.
    g_s : float, default=2.0
        Electron spin g-factor.

    Returns
    -------
    numpy.ndarray
        Sorted array of :math:`2(2l+1)` energy shifts.
    """
    m_l_values = np.arange(-l, l + 1)
    m_s_values = np.array([-0.5, 0.5])
    shifts = np.array([zeeman_splitting(ml, ms, B, mu_B, g_s) for ml in m_l_values for ms in m_s_values])
    return np.sort(shifts)


# --- Linear Stark effect (hydrogen, degenerate perturbation theory) --------


def linear_stark_shift(n: int, n1: int, n2: int, m: int, F: float, a0: float = 1.0) -> float:
    r"""First-order Stark shift of a hydrogenic parabolic state.

    For :math:`\lvert n, n_1, n_2, m\rangle` under a uniform field :math:`F`
    along :math:`z`,

    .. math::

        \Delta E = \frac{3}{2}\, n\, (n_1 - n_2)\, a_0 F,

    valid when :math:`n_1 + n_2 + \lvert m\rvert + 1 = n`. This linear-in-
    :math:`F` splitting is a hallmark of hydrogen's "accidental"
    :math:`l`-degeneracy (removed by any core penetration).

    Parameters
    ----------
    n : int
        Principal quantum number.
    n1, n2 : int
        Parabolic quantum numbers.
    m : int
        Magnetic quantum number.
    F : float
        Electric field strength.
    a0 : float, default=1.0
        Bohr radius.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If ``n1 + n2 + abs(m) + 1 != n``.
    """
    if n1 + n2 + abs(m) + 1 != n:
        raise ValueError("Parabolic quantum numbers must satisfy n1+n2+|m|+1=n.")
    return 1.5 * n * (n1 - n2) * a0 * F


@dataclass
class StarkLevel:
    """One sublevel returned by :func:`stark_n2_quartet`."""

    n1: int
    """int: Parabolic quantum number."""

    n2: int
    """int: Parabolic quantum number."""

    m: int
    """int: Magnetic quantum number."""

    shift: float
    """float: The first-order Stark energy shift."""


def stark_n2_quartet(F: float, a0: float = 1.0) -> list:
    r"""The linear Stark splitting of hydrogen's :math:`n=2` shell.

    The 4-fold degenerate :math:`n=2` level (2s, 2p0, :math:`2p_{\pm1}`)
    splits under a field :math:`F` into three levels with shifts
    :math:`\{-3a_0F, 0, 0, +3a_0F\}` -- the classic linear Stark quartet,
    since only the :math:`m=0` pair (2s/2p0) is coupled by the perturbation.

    Parameters
    ----------
    F : float
        Electric field strength.
    a0 : float, default=1.0
        Bohr radius.

    Returns
    -------
    list of StarkLevel
        Sorted by energy shift.
    """
    levels = []
    for n1 in range(2):
        for n2 in range(2):
            for m in (-1, 0, 1):
                if n1 + n2 + abs(m) + 1 == 2:
                    levels.append(StarkLevel(n1, n2, m, linear_stark_shift(2, n1, n2, m, F, a0)))
    return sorted(levels, key=lambda lv: lv.shift)


# --- Floquet-driven infinite box --------------------------------------------


class FloquetDrivenBox:
    r"""An infinite square well :math:`[0,L]` driven by an AC dipole field.

    .. math::

        V(x,t) = V_0 (x - L/2) \cos(\omega t),

    simulated with the FFT split-operator propagator. The box walls are
    emulated with a steep confining potential outside :math:`[0,L]` padded
    into a larger FFT domain.

    Provides both time-domain propagation (to see multiphoton Rabi
    oscillations / photon-assisted tunneling between box eigenstates) and
    the Floquet quasi-energy spectrum from the one-period propagator
    :math:`\hat U(T)`.

    Parameters
    ----------
    L : float, default=1.0
        Well width.
    V0 : float, default=5.0
        Driving field amplitude.
    omega : float, default=20.0
        Driving angular frequency :math:`\omega`.
    hbar : float, default=1.0
        Value of :math:`\hbar` to use.
    m : float, default=1.0
        Particle mass.
    wall_height : float, default=5000.0
        Height of the confining walls outside :math:`[0,L]`.
    pad : float, default=0.4
        Fractional padding of the FFT domain beyond :math:`[0,L]`.
    n_grid : int, default=2048
        Number of FFT grid points.
    """

    def __init__(
        self,
        L: float = 1.0,
        V0: float = 5.0,
        omega: float = 20.0,
        hbar: float = 1.0,
        m: float = 1.0,
        wall_height: float = 5000.0,
        pad: float = 0.4,
        n_grid: int = 2048,
    ):
        self.L, self.V0, self.omega, self.hbar, self.m = L, V0, omega, hbar, m
        self.x = np.linspace(-pad * L, (1 + pad) * L, n_grid)

        def wall(x):
            return np.where((x >= 0) & (x <= L), 0.0, wall_height)

        self._wall = wall(self.x)

        def V(x, t):
            return wall(x) + V0 * (x - L / 2) * np.cos(omega * t)

        self.V = V

    def box_eigenstate(self, n: int) -> np.ndarray:
        """The :math:`n`-th unperturbed infinite-well eigenstate on the grid.

        Parameters
        ----------
        n : int
            Quantum number (:math:`n=1,2,\\dots`).

        Returns
        -------
        numpy.ndarray
        """
        psi = np.zeros_like(self.x)
        inside = (self.x >= 0) & (self.x <= self.L)
        psi[inside] = np.sqrt(2 / self.L) * np.sin(n * np.pi * self.x[inside] / self.L)
        norm = np.sqrt(trapz(psi**2, self.x))
        return psi / norm

    def box_energy(self, n: int) -> float:
        r"""Unperturbed box eigenenergy :math:`E_n = n^2\pi^2\hbar^2/(2mL^2)`.

        Parameters
        ----------
        n : int

        Returns
        -------
        float
        """
        return (n**2 * np.pi**2 * self.hbar**2) / (2 * self.m * self.L**2)

    def make_solver(self, dt: float = 1e-4) -> SplitOperatorSolver1D:
        """Build a :class:`~physicskit.quantum.core.solvers.SplitOperatorSolver1D` for this drive.

        Parameters
        ----------
        dt : float, default=1e-4
            Time step.

        Returns
        -------
        physicskit.quantum.core.solvers.SplitOperatorSolver1D
        """
        return SplitOperatorSolver1D(self.x, self.V, hbar=self.hbar, m=self.m, dt=dt)

    def transition_probability(self, n_initial: int, n_final: int, t_max: float, dt: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
        r"""Multiphoton transition probability :math:`P_{i\to f}(t)`.

        Propagates the ``n_initial`` box eigenstate under the AC drive and
        returns its overlap probability with the ``n_final`` unperturbed
        box eigenstate -- shows resonant multiphoton transitions when
        :math:`\hbar\omega` (or a multiple) bridges the level gap.

        Parameters
        ----------
        n_initial : int
            Initial box quantum number.
        n_final : int
            Final box quantum number to project onto.
        t_max : float
            Total propagation time.
        dt : float, default=1e-4
            Time step.

        Returns
        -------
        times : numpy.ndarray
        probs : numpy.ndarray
            :math:`P_{n_\text{initial}\to n_\text{final}}(t)`.
        """
        solver = self.make_solver(dt)
        psi0 = self.box_eigenstate(n_initial).astype(complex)
        n_steps = int(t_max / dt)
        save_every = max(n_steps // 400, 1)
        frames, times = solver.propagate(psi0, n_steps, save_every=save_every)

        target = self.box_eigenstate(n_final)
        probs = np.array([np.abs(trapz(np.conj(target) * f, self.x)) ** 2 for f in frames])
        return times, probs

    def floquet_quasienergies(self, n_levels: int = 4, dt: float = 1e-4) -> np.ndarray:
        r"""Floquet quasi-energy spectrum from the one-period propagator.

        Diagonalizes :math:`\hat U(T)` (:math:`T=2\pi/\omega`), restricted
        to the span of the lowest ``n_levels`` box eigenstates, to obtain
        quasi-energies

        .. math::

            \epsilon_k = -\frac{\hbar}{T}\arg(\lambda_k),

        where :math:`\lambda_k` are the eigenvalues of :math:`\hat U(T)`.

        Parameters
        ----------
        n_levels : int, default=4
            Size of the truncated box-eigenstate basis.
        dt : float, default=1e-4
            Time step used to build :math:`\hat U(T)`.

        Returns
        -------
        numpy.ndarray
            Sorted quasi-energies, shape ``(n_levels,)``.
        """
        T = 2 * np.pi / self.omega
        solver = self.make_solver(dt)
        n_steps = int(round(T / dt))

        basis = [self.box_eigenstate(n) for n in range(1, n_levels + 1)]
        U = np.zeros((n_levels, n_levels), dtype=complex)
        for j, psi0 in enumerate(basis):
            psi = psi0.astype(complex)
            for _ in range(n_steps):
                psi = solver.step(psi)
            for i, target in enumerate(basis):
                U[i, j] = trapz(np.conj(target) * psi, self.x)

        eigvals = np.linalg.eigvals(U)
        quasi_energies = -(self.hbar / T) * np.angle(eigvals)
        return np.sort(quasi_energies)
