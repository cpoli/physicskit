r"""The Gutzwiller trace formula: reconstructing a spectrum from classical periodic orbits alone.

Gutzwiller's (1971) trace formula expresses the density of states
:math:`g(E)=\sum_n\delta(E-E_n)` as a sum over every classical periodic
orbit of the system, with no reference to quantum eigenstates at all --
each orbit contributes an oscillatory term whose frequency (in
:math:`E`) is set by its classical action and whose amplitude is set by
its linear stability. For a bound, one-dimensional system this is not
merely an approximation: the single family of periodic orbits (one per
energy, since :math:`f=1` degree of freedom always has exactly one
oscillation) has an *exact* trace formula, obtained by Poisson-summing
the discrete Einstein-Brillouin-Keller (EBK) spectrum of
:mod:`physicskit.semiclassical.core.wkb` --
:func:`gutzwiller_density_of_states` implements exactly that sum,
reconstructing delta-function peaks at the Bohr-Sommerfeld energies
purely from the classical action :math:`S(E)` and period
:math:`T(E)=dS/dE`.

The general Gutzwiller formula (for a system with genuinely *isolated*,
unstable periodic orbits, as in chaotic scattering or 2D/3D billiards)
instead weights each orbit by :math:`1/\sqrt{|2-\operatorname{tr}M|}`,
its monodromy matrix's instability -- :func:`gutzwiller_amplitude_from_monodromy`
implements that general ingredient, for use with orbits found in
higher-dimensional chaotic systems (the stadium billiard scars analyzed
in :mod:`physicskit.semiclassical.systems.scarring` live on exactly this
kind of isolated unstable orbit).
"""

from __future__ import annotations

import numpy as np

from .wkb import turning_points, wkb_action

__all__ = [
    "classical_period",
    "gutzwiller_density_of_states",
    "gutzwiller_amplitude_from_monodromy",
]


def classical_period(E: float, V, m: float, x_min: float, x_max: float, hbar: float = 1.0, dE: float = 1e-4) -> float:
    r"""Classical (round-trip) oscillation period :math:`T(E)=2\,dS/dE` of a 1D bound orbit.

    The action :math:`S(E)` (:func:`~physicskit.semiclassical.core.wkb.wkb_action`)
    is the one-way traversal action, so its energy derivative :math:`dS/dE`
    is the time to cross the classically allowed region once; the full
    period (there and back) is twice that -- differentiated numerically
    here via a central difference.

    Parameters
    ----------
    E : float
        Energy at which to evaluate the period.
    V : callable
        Potential energy function ``V(x)``.
    m : float
        Particle mass.
    x_min, x_max : float
        Search domain for :func:`~physicskit.semiclassical.core.wkb.turning_points`.
    hbar : float, default=1.0
        Value of :math:`\hbar` to use (unused by the classical period
        itself; kept for a uniform signature with the rest of this module).
    dE : float, default=1e-4
        Step size for the central-difference derivative.

    Returns
    -------
    float
        The classical period :math:`T(E)`.

    See Also
    --------
    gutzwiller_density_of_states : Uses this as the trace formula's smooth prefactor.

    Examples
    --------
    The harmonic oscillator's period is famously independent of energy,
    :math:`T=2\pi/\omega`:

    >>> V = lambda x: 0.5 * x ** 2
    >>> round(classical_period(E=3.0, V=V, m=1.0, x_min=-20, x_max=20), 4)
    6.2832
    """

    def S(E_):
        tp = turning_points(E_, V, x_min, x_max)
        return wkb_action(E_, V, m, tp[0], tp[-1], hbar)

    return 2.0 * (S(E + dE) - S(E - dE)) / (2.0 * dE)


def gutzwiller_density_of_states(
    E_grid: np.ndarray,
    V,
    m: float,
    x_min: float,
    x_max: float,
    hbar: float = 1.0,
    r_max: int = 40,
    broadening: float = 0.05,
) -> np.ndarray:
    r"""Exact 1D trace-formula density of states, summed over repetitions of the single periodic orbit.

    For a bound one-dimensional system, EBK quantization places levels
    exactly where :math:`S(E_n)/\hbar=(n+\tfrac12)\pi`. Poisson-summing
    the resulting delta comb :math:`g(E)=\sum_n\delta(E-E_n)` over the
    integer :math:`n` converts it into a sum over an integer :math:`r`
    -- physically, the :math:`r`-fold repetition of the single primitive
    periodic orbit at each energy -- giving the *exact* identity

    .. math::

       g(E) = \frac{T(E)/2}{\pi\hbar}\left[1 + 2\sum_{r=1}^{\infty}
       \cos\!\left(\frac{2rS(E)}{\hbar} - r\pi\right)\right],

    the one-dimensional Gutzwiller trace formula: the :math:`r=0` term is
    the smooth Weyl density of states, and each :math:`r\ge1` term is one
    repetition of the orbit, carrying the Maslov phase :math:`-r\pi`
    (:math:`\sigma=2` soft turning points per traversal, repeated
    :math:`r` times). Truncating the sum at finite ``r_max`` with a
    convergence factor ``exp(-r*broadening)`` turns each delta function
    into a finite (Lorentzian-like) peak, suitable for numerical peak-finding.

    Parameters
    ----------
    E_grid : ndarray
        Energies at which to evaluate the density of states.
    V : callable
        Potential energy function ``V(x)``.
    m : float
        Particle mass.
    x_min, x_max : float
        Search domain for :func:`~physicskit.semiclassical.core.wkb.turning_points`.
    hbar : float, default=1.0
        Value of :math:`\hbar` to use.
    r_max : int, default=40
        Number of orbit repetitions to sum.
    broadening : float, default=0.05
        Per-repetition convergence factor; larger values broaden (and
        damp) each reconstructed peak.

    Returns
    -------
    ndarray
        :math:`g(E)`, same shape as ``E_grid``.

    See Also
    --------
    classical_period : Supplies :math:`T(E)`.
    physicskit.semiclassical.core.wkb.bohr_sommerfeld_energies :
        The exact peak locations this reconstructs.

    Examples
    --------
    The reconstructed peaks land exactly on the harmonic oscillator's
    Bohr-Sommerfeld spectrum:

    >>> import numpy as np
    >>> from scipy.signal import find_peaks
    >>> V = lambda x: 0.5 * x ** 2
    >>> E_grid = np.linspace(0.2, 4.5, 600)
    >>> dos = gutzwiller_density_of_states(E_grid, V, m=1.0, x_min=-20, x_max=20)
    >>> peak_idx, _ = find_peaks(dos, height=0.3 * dos.max())
    >>> np.round(E_grid[peak_idx], 1)
    array([0.5, 1.5, 2.5, 3.5])
    """
    E_grid = np.asarray(E_grid, dtype=float)
    dos = np.empty_like(E_grid)
    r = np.arange(1, r_max + 1)
    envelope = np.exp(-r * broadening)
    for i, E in enumerate(E_grid):
        tp = turning_points(E, V, x_min, x_max)
        x1, x2 = tp[0], tp[-1]
        S = wkb_action(E, V, m, x1, x2, hbar)
        T_one_way = classical_period(E, V, m, x_min, x_max, hbar) / 2.0
        oscillating = np.sum(envelope * np.cos(2.0 * r * S / hbar - r * np.pi))
        dos[i] = (T_one_way / (np.pi * hbar)) * (1.0 + 2.0 * oscillating)
    return dos


def gutzwiller_amplitude_from_monodromy(M: np.ndarray) -> float:
    r"""Gutzwiller stability amplitude :math:`1/\sqrt{|2-\operatorname{tr}M|}` for an isolated periodic orbit.

    In the full (multi-dimensional, generically chaotic) Gutzwiller trace
    formula, each isolated periodic orbit contributes with an amplitude
    set by how strongly nearby trajectories diverge from it over one
    period: an unstable (hyperbolic) orbit with monodromy eigenvalues
    :math:`\lambda,1/\lambda` (:math:`|\lambda|>1`) has
    :math:`\operatorname{tr}M=\lambda+1/\lambda`, so
    :math:`|2-\operatorname{tr}M|` grows with the instability and the
    orbit's contribution to the trace formula shrinks -- highly unstable
    orbits matter less for the exact spectrum, but (as in
    :mod:`physicskit.semiclassical.systems.scarring`) can still leave a
    visible imprint on individual eigenstates.

    Parameters
    ----------
    M : ndarray, shape (2, 2)
        Monodromy matrix of one period of the orbit, e.g. from
        :func:`physicskit.semiclassical.core.propagators.propagate_trajectory_monodromy_action`.

    Returns
    -------
    float
        The stability amplitude.

    Examples
    --------
    >>> import numpy as np
    >>> M = np.array([[2.0, 0.0], [0.0, 0.5]])
    >>> round(gutzwiller_amplitude_from_monodromy(M), 6)
    1.414214
    """
    return float(1.0 / np.sqrt(abs(2.0 - np.trace(M))))
