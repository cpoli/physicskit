r"""Quantum boxes and potential wells beyond the flat infinite box.

Covers asymmetric/step wells, the V-shaped 'quantum bouncer', the symmetric
double well (tunneling doublets), the finite square well (bound + continuum
states), and 2D quantum boxes (rectangle, circular dot, triangular/stadium
billiards -- the latter bridging to ``physicskit.chaos`` via quantum scars).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import airy, jn_zeros, jv

from .._compat import trapz
from ..core.eigensolvers import (
    EigenResult,
    NumerovSolver,
    asymmetric_step_well,
    double_well,
    linear_gravitational_well,
)
from ..core.solvers import SplitOperatorSolver1D
from .wave_packets import free_gaussian_wavepacket

__all__ = [
    "asymmetric_well_states",
    "gravitational_bouncer_states",
    "airy_bouncer_energies",
    "airy_wavefunction",
    "DoubleWellSimulator",
    "DoubleWellResult",
    "FiniteSquareWell",
    "ScatteringResult",
    "RectangularBox2D",
    "Box2DEigenstate",
    "CircularBox2D",
    "CircularEigenstate",
    "StadiumBilliard2D",
]


# --- Asymmetric / step wells ------------------------------------------------


def asymmetric_well_states(
    width: float = 2.0, V_left: float = 40.0, V_right: float = 15.0, x_extent: float = 6.0, n_points: int = 1200, n_states: int = 6
) -> EigenResult:
    """Bound states of a well with unequal wall heights.

    The wavefunctions decay at different rates into the left vs. right
    forbidden regions (asymmetric evanescent tails), unlike the symmetric
    finite well.

    Parameters
    ----------
    width : float, default=2.0
        Full width of the zero-potential region.
    V_left : float, default=40.0
        Wall height on the left side.
    V_right : float, default=15.0
        Wall height on the right side.
    x_extent : float, default=6.0
        Half-width of the numerical domain, ``x in [-x_extent, x_extent]``.
    n_points : int, default=1200
        Number of grid points.
    n_states : int, default=6
        Number of bound states to compute.

    Returns
    -------
    physicskit.quantum.core.eigensolvers.EigenResult
        The energies and wavefunctions.
    """
    x = np.linspace(-x_extent, x_extent, n_points)
    V = asymmetric_step_well(width, V_left, V_right)
    return NumerovSolver(x, V).solve(n_states)


# --- V-shaped (linear, gravitational) well ---------------------------------


def gravitational_bouncer_states(alpha: float = 1.0, x_extent: float = 15.0, n_points: int = 2000, n_states: int = 8) -> EigenResult:
    r"""The 'quantum bouncer': a particle falling under uniform gravity.

    Solves for :math:`V(x) = \alpha x` (:math:`x>0`) with a hard floor at
    :math:`x=0`. The grid's lower boundary supplies the floor (:math:`\psi`
    is forced to vanish there), so the domain spans only
    :math:`x \in [0, x_\text{extent}]`. Eigenstates are Airy functions
    :math:`\mathrm{Ai}(z)`; the energies returned here should closely match
    :func:`airy_bouncer_energies`.

    Parameters
    ----------
    alpha : float, default=1.0
        Slope of the linear potential (e.g. :math:`mg` for gravity).
    x_extent : float, default=15.0
        Extent of the numerical domain, ``x in [0, x_extent]``.
    n_points : int, default=2000
        Number of grid points.
    n_states : int, default=8
        Number of bound states to compute.

    Returns
    -------
    physicskit.quantum.core.eigensolvers.EigenResult
        The energies and wavefunctions.
    """
    x = np.linspace(0.0, x_extent, n_points)
    V = linear_gravitational_well(alpha)
    return NumerovSolver(x, V).solve(n_states)


def airy_bouncer_energies(alpha: float = 1.0, n_states: int = 8, hbar: float = 1.0, m: float = 1.0) -> np.ndarray:
    r"""Analytic bound-state energies for the half-line gravitational bouncer.

    For :math:`V(x) = \alpha x` (:math:`x>0`, infinite floor at :math:`x=0`),

    .. math::

        E_n = -\alpha \left(\frac{\hbar^2}{2 m \alpha^2}\right)^{1/3} a_n,

    where :math:`a_n < 0` are the zeros of the Airy function,
    :math:`\mathrm{Ai}(a_n) = 0`.

    Parameters
    ----------
    alpha : float, default=1.0
        Slope of the linear potential.
    n_states : int, default=8
        Number of energy levels to return.
    hbar : float, default=1.0
        Value of :math:`\hbar` to use.
    m : float, default=1.0
        Particle mass.

    Returns
    -------
    numpy.ndarray
        The lowest ``n_states`` analytic energies, ascending.
    """
    from scipy.special import ai_zeros

    a_zeros, _, _, _ = ai_zeros(n_states)
    scale = (hbar**2 * alpha**2 / (2 * m)) ** (1 / 3)
    return -a_zeros * scale


def airy_wavefunction(x: np.ndarray, alpha: float = 1.0, n: int = 0, hbar: float = 1.0, m: float = 1.0) -> np.ndarray:
    r"""Normalized Airy-function eigenstate for the half-line bouncer.

    Parameters
    ----------
    x : numpy.ndarray
        Grid to evaluate the wavefunction on (values at :math:`x<0` are
        forced to zero).
    alpha : float, default=1.0
        Slope of the linear potential :math:`V(x) = \alpha x`.
    n : int, default=0
        Quantum number (0 = ground state).
    hbar : float, default=1.0
        Value of :math:`\hbar` to use.
    m : float, default=1.0
        Particle mass.

    Returns
    -------
    numpy.ndarray
        The normalized wavefunction :math:`\psi_n(x)`.
    """
    from scipy.special import ai_zeros

    a_zeros, _, _, _ = ai_zeros(n + 1)
    a_n = a_zeros[n]
    length = (hbar**2 / (2 * m * alpha)) ** (1 / 3)
    # z(x) = x/length + a_n, so that z(0) = a_n (a zero of Ai), satisfying
    # the psi(0)=0 hard-floor boundary condition; x/length - a_n would put
    # z(0) = -a_n > 0, the wrong (decaying) branch, and psi(0) != 0.
    Ai, _, _, _ = airy(x / length + a_n)
    psi = np.where(x >= 0, Ai, 0.0)
    norm = np.sqrt(trapz(psi**2, x))
    return psi / norm


# --- Double well: tunneling doublets ---------------------------------------


@dataclass
class DoubleWellResult:
    """Container for the output of :meth:`DoubleWellSimulator.solve`."""

    x: np.ndarray
    """numpy.ndarray: The spatial grid."""

    energies: np.ndarray
    r"""numpy.ndarray: Eigenenergies :math:`E_n`, ascending."""

    wavefunctions: np.ndarray
    """numpy.ndarray: Normalized eigenfunctions, shape ``(n_states, len(x))``."""

    splitting: float
    r"""float: The symmetric/antisymmetric doublet gap :math:`\Delta E = E_1 - E_0`."""

    tunneling_period: float
    r"""float: The tunneling oscillation period :math:`T = 2\pi\hbar/\Delta E`."""


class DoubleWellSimulator:
    r"""Symmetric double well :math:`V(x) = \lambda (x^2 - a^2)^2`.

    The two lowest eigenstates form a nearly degenerate symmetric/
    antisymmetric doublet whose splitting :math:`\Delta E = E_1 - E_0`
    sets the period :math:`T=2\pi\hbar/\Delta E` of left-right tunneling
    oscillations of a wavepacket initially localized in one well.

    Parameters
    ----------
    lam : float, default=0.5
        Overall strength :math:`\lambda` of the quartic potential.
    a : float, default=2.5
        Half-separation of the two minima.
    x_extent : float, default=6.0
        Half-width of the numerical domain, ``x in [-x_extent, x_extent]``.
    n_points : int, default=1600
        Number of grid points.
    hbar : float, default=1.0
        Value of :math:`\hbar` to use.
    m : float, default=1.0
        Particle mass.
    """

    def __init__(self, lam: float = 0.5, a: float = 2.5, x_extent: float = 6.0, n_points: int = 1600, hbar: float = 1.0, m: float = 1.0):
        self.lam = lam
        self.a = a
        self.hbar = hbar
        self.m = m
        self.x = np.linspace(-x_extent, x_extent, n_points)
        self.V = double_well(lam, a)
        self._solver = NumerovSolver(self.x, self.V, hbar=hbar, m=m)

    def solve(self, n_states: int = 6) -> DoubleWellResult:
        """Compute the eigenstates and the tunneling splitting/period.

        Parameters
        ----------
        n_states : int, default=6
            Number of bound states to compute (at least 2, for the doublet).

        Returns
        -------
        DoubleWellResult
        """
        res = self._solver.solve(n_states)
        splitting = res.energies[1] - res.energies[0]
        period = 2 * np.pi * self.hbar / splitting
        return DoubleWellResult(res.x, res.energies, res.wavefunctions, splitting, period)

    def localized_state(self, result: DoubleWellResult, side: str = "left") -> np.ndarray:
        r"""Build a state localized in one well from the doublet.

        .. math::

            \psi(x) = \frac{\psi_0(x) \pm \psi_1(x)}{\sqrt 2},

        with ``+`` for the left well and ``-`` for the right well.

        Parameters
        ----------
        result : DoubleWellResult
            Output of :meth:`solve` (must include at least 2 states).
        side : {'left', 'right'}, default='left'
            Which well to localize the state in.

        Returns
        -------
        numpy.ndarray
            The localized wavefunction at :math:`t=0`.
        """
        psi0, psi1 = result.wavefunctions[0], result.wavefunctions[1]
        sign = 1.0 if side == "left" else -1.0
        psi = (psi0 + sign * psi1) / np.sqrt(2)
        return psi

    def tunneling_wavefunction(self, result: DoubleWellResult, t: np.ndarray, side: str = "left") -> np.ndarray:
        r"""Time-dependent wavefunction of a state localized on ``side``.

        Evolves :math:`\psi(x,0) = (\psi_0 \pm \psi_1)/\sqrt2` under the two
        doublet energies; :math:`\lvert\psi(x,t)\rvert^2`
        (:meth:`tunneling_oscillation`) periodically tunnels to the other
        well with period :math:`T=2\pi\hbar/\Delta E`. Shaped for
        :func:`~physicskit.quantum.visualizers.wavefunctions.animate_density`.

        Parameters
        ----------
        result : DoubleWellResult
            Output of :meth:`solve`.
        t : numpy.ndarray
            Times to evaluate at.
        side : {'left', 'right'}, default='left'
            Which well the state starts localized in.

        Returns
        -------
        numpy.ndarray
            Complex-valued :math:`\psi(x,t)`, shape ``(len(t), len(x))``.
        """
        psi0, psi1 = result.wavefunctions[0], result.wavefunctions[1]
        E0, E1 = result.energies[0], result.energies[1]
        sign = 1.0 if side == "left" else -1.0
        t = np.asarray(t)
        phase0 = np.exp(-1j * E0 * t / self.hbar)[:, None]
        phase1 = np.exp(-1j * E1 * t / self.hbar)[:, None]
        return (psi0[None, :] * phase0 + sign * psi1[None, :] * phase1) / np.sqrt(2)

    def tunneling_oscillation(self, result: DoubleWellResult, t: np.ndarray, side: str = "left") -> np.ndarray:
        r"""Time-dependent probability density of a state localized on ``side``.

        See :meth:`tunneling_wavefunction`.

        Parameters
        ----------
        result : DoubleWellResult
            Output of :meth:`solve`.
        t : numpy.ndarray
            Times to evaluate at.
        side : {'left', 'right'}, default='left'
            Which well the state starts localized in.

        Returns
        -------
        numpy.ndarray
            :math:`\lvert\psi(x,t)\rvert^2`, shape ``(len(t), len(x))``.
        """
        return np.abs(self.tunneling_wavefunction(result, t, side=side)) ** 2

    def left_well_probability(self, result: DoubleWellResult, t: np.ndarray, side: str = "left") -> np.ndarray:
        r"""Probability of finding the particle in the left well (:math:`x<0`) vs. time.

        .. math::

            P(t) = \int_{-\infty}^0 \lvert\psi(x,t)\rvert^2\, dx,

        oscillating with the tunneling period :math:`T=2\pi\hbar/\Delta E`.

        Parameters
        ----------
        result : DoubleWellResult
            Output of :meth:`solve`.
        t : numpy.ndarray
            Times to evaluate at.
        side : {'left', 'right'}, default='left'
            Which well the state starts localized in.

        Returns
        -------
        numpy.ndarray
            :math:`P(t)`, shape ``(len(t),)``.
        """
        density_t = self.tunneling_oscillation(result, t, side=side)
        mask = self.x < 0
        return trapz(density_t[:, mask], self.x[mask], axis=1)


# --- Finite square well: bound states + continuum scattering ---------------


@dataclass
class ScatteringResult:
    """Container for the output of :meth:`FiniteSquareWell.scattering`."""

    E: float
    """float: Incident energy."""

    k1: float
    """float: Wavenumber outside the well."""

    k2: complex
    """complex: Wavenumber inside the well/barrier. Purely imaginary in a
    classically forbidden interior (sub-barrier tunneling), real otherwise."""

    T: float
    """float: Transmission probability."""

    R: float
    r"""float: Reflection probability (:math:`R = 1 - T`)."""


class FiniteSquareWell:
    r"""A finite square well of depth :math:`V_0` and width :math:`a`,

    .. math::

        V(x) = \begin{cases} -V_0 & \lvert x\rvert < a/2 \\ 0 & \text{otherwise} \end{cases}.

    Provides bound states (:math:`E<0`) via the Numerov solver and exact
    analytic transmission/reflection coefficients for scattering states
    (:math:`E>0`).

    Parameters
    ----------
    V0 : float, default=20.0
        Well depth (:math:`V_0 > 0`; the floor sits at :math:`-V_0`).
    width : float, default=2.0
        Full width :math:`a` of the well.
    hbar : float, default=1.0
        Value of :math:`\hbar` to use.
    m : float, default=1.0
        Particle mass.
    """

    def __init__(self, V0: float = 20.0, width: float = 2.0, hbar: float = 1.0, m: float = 1.0):
        self.V0 = V0
        self.width = width
        self.hbar = hbar
        self.m = m

    def potential(self, x: np.ndarray) -> np.ndarray:
        r"""Evaluate :math:`V(x)` (a barrier if ``V0 < 0``).

        Parameters
        ----------
        x : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        return np.where(np.abs(x) <= self.width / 2, -self.V0, 0.0)

    def bound_states(self, x_extent: float = 6.0, n_points: int = 1600, n_states: int = 6) -> EigenResult:
        """Compute the bound states (:math:`E<0`).

        Parameters
        ----------
        x_extent : float, default=6.0
            Half-width of the numerical domain.
        n_points : int, default=1600
            Number of grid points.
        n_states : int, default=6
            Number of Numerov eigenstates to request (some may be discarded
            if they lie above the well rim; see Notes).

        Returns
        -------
        physicskit.quantum.core.eigensolvers.EigenResult
            Only the states with :math:`E<0` are kept -- the solver also
            returns box-quantized "continuum" states above the well rim,
            which are artifacts of the finite numerical domain and are
            filtered out here.
        """
        x = np.linspace(-x_extent, x_extent, n_points)
        res = NumerovSolver(x, self.potential, hbar=self.hbar, m=self.m).solve(n_states)
        bound_mask = res.energies < 0
        return EigenResult(res.x, res.energies[bound_mask], res.wavefunctions[bound_mask])

    def scattering(self, E: float) -> ScatteringResult:
        r"""Exact 1D transmission/reflection coefficients for :math:`E>0`.

        .. math::

            T(E) = \left[1 + \frac{V_0^2 \sin^2(k_2 a)}{4 E (E+V_0)}\right]^{-1},
            \qquad R = 1 - T,

        with :math:`k_1=\sqrt{2mE}/\hbar` outside the well and
        :math:`k_2=\sqrt{2m(E+V_0)}/\hbar` inside it. :math:`T(E)=1`
        exactly whenever :math:`k_2 a` is a multiple of :math:`\pi`
        (the Ramsauer-Townsend resonances).

        The same formula analytically continues correctly into a
        classically forbidden interior (:math:`E+V_0<0`, e.g. a *barrier*
        built by passing a negative ``V0``): :math:`k_2` becomes purely
        imaginary, :math:`\sin` becomes :math:`i\sinh` of a real argument,
        and :math:`T` reduces to the standard sub-barrier tunneling
        formula. This method evaluates :math:`k_2` with complex arithmetic
        so that regime is handled automatically.

        Parameters
        ----------
        E : float
            Incident kinetic energy (:math:`E>0`).

        Returns
        -------
        ScatteringResult
        """
        hbar, m, V0, a = self.hbar, self.m, self.V0, self.width
        k1 = np.sqrt(2 * m * E) / hbar
        k2 = np.sqrt(2 * m * (E + V0) + 0j) / hbar
        T = 1.0 / (1.0 + (V0**2 * np.sin(k2 * a) ** 2) / (4 * E * (E + V0)))
        T = float(np.real_if_close(T))
        R = 1.0 - T
        return ScatteringResult(E=E, k1=k1, k2=k2, T=T, R=R)

    def transmission_spectrum(self, E_values: np.ndarray) -> np.ndarray:
        """Transmission probability :math:`T(E)` over an array of energies.

        Parameters
        ----------
        E_values : numpy.ndarray
            Incident energies (:math:`E>0`).

        Returns
        -------
        numpy.ndarray
            :math:`T(E)` for each value in ``E_values``.
        """
        return np.array([self.scattering(E).T for E in E_values])

    def evanescent_wavefunction(self, result: EigenResult, n: int = 0) -> np.ndarray:
        r"""The :math:`n`-th bound-state wavefunction.

        Its exponential tails outside :math:`\lvert x\rvert < a/2` are
        evanescent by construction of the Numerov solver.

        Parameters
        ----------
        result : physicskit.quantum.core.eigensolvers.EigenResult
            Output of :meth:`bound_states`.
        n : int, default=0
            Which bound state to return (0 = ground state).

        Returns
        -------
        numpy.ndarray
        """
        return result.wavefunctions[n]

    def wavepacket_scattering(
        self,
        x_extent: float = 25.0,
        n_points: int = 2048,
        x0: float = -10.0,
        sigma0: float = 1.5,
        k0: float | None = None,
        dt: float = 2e-4,
        n_steps: int = 6000,
        save_every: int = 30,
    ):
        r"""Propagate a Gaussian wavepacket incident on this well/barrier.

        A single FFT split-operator run
        (:class:`~physicskit.quantum.core.solvers.SplitOperatorSolver1D`) shared
        by the tunnelling (``V0 < 0``, a barrier) and the resonant-scattering
        (``V0 > 0``, a well) demos: the packet splits into a reflected and a
        transmitted piece as it crosses :math:`\lvert x\rvert < \text{width}/2`.

        Parameters
        ----------
        x_extent : float, default=25.0
            Half-width of the numerical domain.
        n_points : int, default=2048
            Number of grid points.
        x0 : float, default=-10.0
            Initial center of the packet (upstream of the well/barrier).
        sigma0 : float, default=1.5
            Initial packet width.
        k0 : float or None, optional
            Central wavenumber; defaults to a value giving mean kinetic
            energy comparable to :math:`\lvert V_0\rvert`, so both partial
            reflection and partial transmission occur.
        dt : float, default=2e-4
            Time step.
        n_steps : int, default=6000
            Total number of time steps.
        save_every : int, default=30
            Save a snapshot every this many steps.

        Returns
        -------
        x : numpy.ndarray
            The spatial grid.
        frames : numpy.ndarray
            Complex wavefunction snapshots, shape ``(n_saved, len(x))``.
        times : numpy.ndarray
            The time of each snapshot.
        """
        if k0 is None:
            k0 = np.sqrt(2 * self.m * max(abs(self.V0), 1.0)) / self.hbar

        x = np.linspace(-x_extent, x_extent, n_points)
        psi0 = free_gaussian_wavepacket(x, x0, sigma0, k0)
        solver = SplitOperatorSolver1D(x, self.potential, hbar=self.hbar, m=self.m, dt=dt)
        frames, times = solver.propagate(psi0, n_steps, save_every=save_every)
        return x, frames, times


# --- 2D rectangular box ------------------------------------------------------


@dataclass
class Box2DEigenstate:
    """One eigenstate of :class:`RectangularBox2D`."""

    nx: int
    r"""int: Quantum number along :math:`x` (:math:`\ge 1`)."""

    ny: int
    r"""int: Quantum number along :math:`y` (:math:`\ge 1`)."""

    energy: float
    r"""float: The eigenenergy :math:`E_{n_x,n_y}`."""

    def psi(self, X: np.ndarray, Y: np.ndarray, Lx: float, Ly: float) -> np.ndarray:
        r"""Evaluate the eigenfunction on a grid.

        .. math::

            \psi_{n_x,n_y}(x,y) = \sqrt{\frac{4}{L_x L_y}}
                \sin\!\left(\frac{n_x \pi x}{L_x}\right)
                \sin\!\left(\frac{n_y \pi y}{L_y}\right).

        Parameters
        ----------
        X, Y : numpy.ndarray
            Coordinate meshgrid (e.g. from :meth:`RectangularBox2D.grid`).
        Lx, Ly : float
            Box dimensions.

        Returns
        -------
        numpy.ndarray
            :math:`\psi_{n_x,n_y}(x,y)` on the grid.
        """
        return np.sqrt(4 / (Lx * Ly)) * np.sin(self.nx * np.pi * X / Lx) * np.sin(self.ny * np.pi * Y / Ly)


class RectangularBox2D:
    r"""Infinite 2D box of size :math:`L_x \times L_y`.

    .. math::

        E_{n_x,n_y} = \frac{\pi^2\hbar^2}{2m}
            \left(\frac{n_x^2}{L_x^2} + \frac{n_y^2}{L_y^2}\right).

    Rational ratios :math:`L_x/L_y` (e.g. the square box :math:`1{:}1`)
    produce accidental degeneracies between distinct :math:`(n_x,n_y)` pairs.

    Parameters
    ----------
    Lx, Ly : float, default=1.0
        Box dimensions.
    hbar : float, default=1.0
        Value of :math:`\hbar` to use.
    m : float, default=1.0
        Particle mass.
    """

    def __init__(self, Lx: float = 1.0, Ly: float = 1.0, hbar: float = 1.0, m: float = 1.0):
        self.Lx, self.Ly, self.hbar, self.m = Lx, Ly, hbar, m

    def energy(self, nx: int, ny: int) -> float:
        r"""Eigenenergy :math:`E_{n_x,n_y}`.

        Parameters
        ----------
        nx, ny : int
            Quantum numbers (each :math:`\ge 1`).

        Returns
        -------
        float
        """
        return (np.pi**2 * self.hbar**2 / (2 * self.m)) * ((nx / self.Lx) ** 2 + (ny / self.Ly) ** 2)

    def spectrum(self, n_max: int = 6) -> list:
        """All eigenstates with :math:`1 \\le n_x, n_y \\le n_\\text{max}`, sorted by energy.

        Parameters
        ----------
        n_max : int, default=6
            Maximum quantum number along each axis.

        Returns
        -------
        list of Box2DEigenstate
        """
        states = [Box2DEigenstate(nx, ny, self.energy(nx, ny)) for nx in range(1, n_max + 1) for ny in range(1, n_max + 1)]
        return sorted(states, key=lambda s: s.energy)

    def degeneracies(self, n_max: int = 6, tol: float = 1e-9) -> dict:
        """Group states by (rounded) energy to reveal degenerate levels.

        Parameters
        ----------
        n_max : int, default=6
            Maximum quantum number along each axis.
        tol : float, default=1e-9
            Energy values within ``tol`` are treated as degenerate.

        Returns
        -------
        dict
            Maps each degenerate energy to its list of ``(nx, ny)`` pairs;
            non-degenerate levels are omitted.
        """
        groups: dict = {}
        for s in self.spectrum(n_max):
            key = round(s.energy / tol) * tol
            groups.setdefault(key, []).append((s.nx, s.ny))
        return {E: pairs for E, pairs in groups.items() if len(pairs) > 1}

    def grid(self, n_points: int = 200):
        """A Cartesian meshgrid spanning the box.

        Parameters
        ----------
        n_points : int, default=200
            Number of points along each axis.

        Returns
        -------
        X, Y : numpy.ndarray
            Meshgrid arrays (``indexing='ij'``).
        """
        x = np.linspace(0, self.Lx, n_points)
        y = np.linspace(0, self.Ly, n_points)
        return np.meshgrid(x, y, indexing="ij")


# --- 2D circular quantum dot -------------------------------------------------


@dataclass
class CircularEigenstate:
    """One eigenstate of :class:`CircularBox2D`."""

    m: int
    """int: Angular momentum quantum number."""

    n: int
    r"""int: Radial quantum number (the state uses the :math:`n`-th positive
    zero of :math:`J_{\lvert m\rvert}`)."""

    k: float
    r"""float: Wavenumber :math:`k_{mn} = z_{mn}/R`."""

    energy: float
    r"""float: The eigenenergy :math:`E_{mn} = \hbar^2 k_{mn}^2 / 2m`."""

    def psi(self, r: np.ndarray, phi: np.ndarray, R: float) -> np.ndarray:
        r"""Evaluate the eigenfunction on a polar grid.

        .. math::

            \psi_{mn}(r,\phi) = N_{mn}\, J_m(k_{mn} r)\, e^{i m \phi}.

        Parameters
        ----------
        r, phi : numpy.ndarray
            Polar coordinate meshgrid (e.g. from :meth:`CircularBox2D.polar_grid`).
        R : float
            Dot radius (used to normalize :math:`N_{mn}`).

        Returns
        -------
        numpy.ndarray
            Complex-valued :math:`\psi_{mn}(r,\phi)`.
        """
        radial = jv(self.m, self.k * r)
        norm = 1.0 / np.sqrt(np.pi * R**2 * jv(self.m + 1, self.k * R) ** 2)
        return norm * radial * np.exp(1j * self.m * phi)


class CircularBox2D:
    r"""Circular infinite well (quantum dot) of radius :math:`R`.

    Eigenstates are Bessel functions
    :math:`\psi_{mn}(r,\phi) \propto J_m(k_{mn} r)\, e^{im\phi}`, vanishing
    at :math:`r=R`, showing nodal rings (radial quantum number :math:`n`)
    and angular momentum quantization (:math:`m`).

    Parameters
    ----------
    R : float, default=1.0
        Dot radius.
    hbar : float, default=1.0
        Value of :math:`\hbar` to use.
    m_mass : float, default=1.0
        Particle mass (named to avoid clashing with the angular momentum
        quantum number ``m``).
    """

    def __init__(self, R: float = 1.0, hbar: float = 1.0, m_mass: float = 1.0):
        self.R = R
        self.hbar = hbar
        self.m_mass = m_mass

    def eigenstate(self, m: int, n: int) -> CircularEigenstate:
        r"""Build the :math:`(m,n)` eigenstate.

        Parameters
        ----------
        m : int
            Angular momentum quantum number.
        n : int
            Radial index (:math:`n=1,2,3,\dots`); the state uses the
            :math:`n`-th positive zero of the order-:math:`\lvert m\rvert`
            Bessel function.

        Returns
        -------
        CircularEigenstate
        """
        zero = jn_zeros(abs(m), n)[-1]
        k = zero / self.R
        energy = self.hbar**2 * k**2 / (2 * self.m_mass)
        return CircularEigenstate(m, n, k, energy)

    def spectrum(self, m_max: int = 4, n_max: int = 4) -> list:
        """All eigenstates with :math:`\\lvert m\\rvert \\le m_\\text{max}`,
        :math:`1 \\le n \\le n_\\text{max}`, sorted by energy.

        Parameters
        ----------
        m_max : int, default=4
            Maximum :math:`\\lvert m \\rvert`.
        n_max : int, default=4
            Maximum radial index.

        Returns
        -------
        list of CircularEigenstate
        """
        states = [self.eigenstate(m, n) for m in range(-m_max, m_max + 1) for n in range(1, n_max + 1)]
        return sorted(states, key=lambda s: s.energy)

    def polar_grid(self, n_r: int = 150, n_phi: int = 150):
        """A polar meshgrid spanning the dot.

        Parameters
        ----------
        n_r : int, default=150
            Number of radial points.
        n_phi : int, default=150
            Number of angular points.

        Returns
        -------
        r, phi : numpy.ndarray
            Meshgrid arrays (``indexing='ij'``).
        """
        r = np.linspace(0, self.R, n_r)
        phi = np.linspace(0, 2 * np.pi, n_phi)
        return np.meshgrid(r, phi, indexing="ij")


# --- Triangular / stadium billiard (quantum scars, bridge to physicskit.chaos) -----


class StadiumBilliard2D:
    """Bunimovich stadium billiard: a rectangle of length ``L`` capped by two
    semicircles of radius ``R``.

    The classical dynamics is chaotic, and the quantum eigenstates of this
    box can exhibit *scars* -- probability density concentrated along
    unstable classical periodic orbits -- making this the natural bridge
    between ``physicskit.quantum`` and ``physicskit.chaos``.

    Eigenstates are obtained by direct diagonalization of the 2D
    finite-difference Laplacian on a grid, with the wavefunction forced to
    zero outside the stadium boundary (infinite-wall billiard).

    Parameters
    ----------
    L : float, default=1.0
        Length of the central rectangular section.
    R : float, default=0.5
        Radius of the semicircular end-caps (also the half-height).
    hbar : float, default=1.0
        Value of :math:`\\hbar` to use.
    m : float, default=1.0
        Particle mass.
    """

    def __init__(self, L: float = 1.0, R: float = 0.5, hbar: float = 1.0, m: float = 1.0):
        self.L = L
        self.R = R
        self.hbar = hbar
        self.m = m

    def mask(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Boolean mask, ``True`` inside the stadium.

        The stadium is a central :math:`L \\times 2R` rectangle with two
        semicircular end-caps of radius :math:`R`.

        Parameters
        ----------
        X, Y : numpy.ndarray
            Coordinate meshgrid (e.g. from :meth:`grid`).

        Returns
        -------
        numpy.ndarray of bool
        """
        inside_rect = (np.abs(X) <= self.L / 2) & (np.abs(Y) <= self.R)
        left_cap = (X < -self.L / 2) & ((X + self.L / 2) ** 2 + Y**2 <= self.R**2)
        right_cap = (X > self.L / 2) & ((X - self.L / 2) ** 2 + Y**2 <= self.R**2)
        return inside_rect | left_cap | right_cap

    def grid(self, n_points: int = 120):
        """A Cartesian meshgrid enclosing the stadium.

        Parameters
        ----------
        n_points : int, default=120
            Number of points along the (longer) x-axis.

        Returns
        -------
        (X, Y) : tuple of numpy.ndarray
            Meshgrid arrays (``indexing='ij'``).
        x, y : numpy.ndarray
            The 1D coordinate arrays used to build the meshgrid.
        """
        half_x = self.L / 2 + self.R
        half_y = self.R * 1.05
        x = np.linspace(-half_x, half_x, n_points)
        dx = x[1] - x[0]
        # Force dy == dx (an isotropic grid): with mismatched spacing the
        # semicircular caps are staircase-approximated as ellipses rather
        # than circles, which was causing non-monotonic, resolution-
        # dependent errors in the eigenvalues.
        n_y = max(int(round(2 * half_y / dx)) + 1, 20)
        y = np.linspace(-half_y, half_y, n_y)
        return np.meshgrid(x, y, indexing="ij"), x, y

    def solve(self, n_points: int = 160, n_states: int = 6):
        r"""Diagonalize :math:`-\hbar^2/2m\,\nabla^2` on the masked grid
        (Dirichlet boundary).

        The curved end-caps are approximated by a "staircased" boundary
        (grid points are simply included or excluded by :meth:`mask`),
        which converges only as :math:`O(h)`: low resolutions give
        eigenvalues with a few percent of resolution-dependent error and
        visibly blocky eigenstate edges. :meth:`grid` keeps the grid
        isotropic (:math:`dx=dy`) so the caps are at least staircased as
        circles rather than ellipses; beyond that, accuracy is improved
        simply by raising ``n_points`` -- the sparse eigensolver used here
        stays fast (well under a second) even at ``n_points=260``.

        Parameters
        ----------
        n_points : int, default=160
            Number of grid points along the x-axis (see :meth:`grid`).
        n_states : int, default=6
            Number of lowest-energy eigenstates to compute.

        Returns
        -------
        energies : numpy.ndarray
            Eigenenergies, ascending, shape ``(n_states,)``.
        wavefunctions : numpy.ndarray
            Normalized eigenfunctions, shape ``(n_states, *X.shape)``, zero
            outside the stadium.
        X, Y : numpy.ndarray
            The coordinate meshgrid.
        mask : numpy.ndarray of bool
            ``True`` inside the stadium (see :meth:`mask`).
        """
        (X, Y), x, y = self.grid(n_points)
        inside = self.mask(X, Y)
        idx = -np.ones(X.shape, dtype=int)
        interior_points = np.argwhere(inside)
        for order, (i, j) in enumerate(interior_points):
            idx[i, j] = order
        n_interior = interior_points.shape[0]

        hx = x[1] - x[0]
        hy = y[1] - y[0]
        from scipy.sparse import lil_matrix
        from scipy.sparse.linalg import eigsh

        H = lil_matrix((n_interior, n_interior))
        coeff = self.hbar**2 / (2 * self.m)
        for order, (i, j) in enumerate(interior_points):
            H[order, order] = coeff * (2 / hx**2 + 2 / hy**2)
            for di, dj, h in ((1, 0, hx), (-1, 0, hx), (0, 1, hy), (0, -1, hy)):
                ni, nj = i + di, j + dj
                if 0 <= ni < X.shape[0] and 0 <= nj < X.shape[1] and idx[ni, nj] >= 0:
                    H[order, idx[ni, nj]] = -coeff / h**2

        energies, vecs = eigsh(H.tocsr(), k=n_states, which="SM")
        order = np.argsort(energies)
        energies = energies[order]
        vecs = vecs[:, order]

        wavefunctions = np.zeros((n_states, *X.shape))
        for k in range(n_states):
            psi = np.zeros(X.shape)
            for order_idx, (i, j) in enumerate(interior_points):
                psi[i, j] = vecs[order_idx, k]
            norm = np.sqrt(trapz(trapz(psi**2, y, axis=1), x))
            wavefunctions[k] = psi / norm

        return energies, wavefunctions, X, Y, inside
