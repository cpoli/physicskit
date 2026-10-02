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

The smooth (:math:`r=0`) part of the trace formula is the Weyl term. For
a two-dimensional billiard, :func:`balian_bloch_counting_function` and
:func:`balian_bloch_level_density` give it with Balian and Bloch's
boundary, corner and curvature corrections.
"""

from __future__ import annotations

import numpy as np

from .wkb import turning_points, wkb_action

__all__ = [
    "classical_period",
    "gutzwiller_density_of_states",
    "gutzwiller_amplitude_from_monodromy",
    "balian_bloch_counting_function",
    "balian_bloch_level_density",
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


def _balian_bloch_constant(corner_angles, total_curvature: float) -> float:
    angles = np.asarray(corner_angles, dtype=float)
    corners = float(np.sum((np.pi**2 - angles**2) / (24.0 * np.pi * angles)))
    return corners + total_curvature / (12.0 * np.pi)


def balian_bloch_counting_function(
    k,
    area: float,
    perimeter: float,
    corner_angles=(),
    total_curvature: float = 0.0,
    boundary: str = "dirichlet",
):
    r"""Smoothed number of billiard levels below wavenumber :math:`k`, with boundary corrections.

    Balian and Bloch (1970) expanded the smoothed counting function of the
    Helmholtz equation :math:`(\nabla^2+k^2)\psi=0` in a 2D domain in
    powers of :math:`1/k`:

    .. math::

       \bar N(k) = \frac{A k^2}{4\pi} \mp \frac{L k}{4\pi} + C,
       \qquad
       C = \sum_i \frac{\pi^2-\theta_i^2}{24\pi\theta_i}
         + \frac{1}{12\pi}\oint\kappa\,ds.

    The first term is Weyl's law, the second is the boundary correction
    (minus for Dirichlet, plus for Neumann), and the constant comes from
    corners of interior angle :math:`\theta_i` and from the curvature
    :math:`\kappa` of the smooth parts of the boundary. Examples: a
    rectangle has :math:`C=4\times\tfrac1{16}=\tfrac14`, a disk has
    :math:`C=\tfrac16`.

    Parameters
    ----------
    k : float or ndarray
        Wavenumber(s) (:math:`E=\hbar^2k^2/2m`).
    area : float
        Billiard area :math:`A`.
    perimeter : float
        Boundary length :math:`L`.
    corner_angles : sequence of float, default=()
        Interior angles :math:`\theta_i` of the boundary's corners, in radians.
    total_curvature : float, default=0.0
        :math:`\oint\kappa\,ds` over the smooth parts of the boundary
        (:math:`2\pi` for a disk or a stadium, 0 for a polygon).
    boundary : {"dirichlet", "neumann"}, default="dirichlet"
        Boundary condition; sets the sign of the perimeter term.

    Returns
    -------
    float or ndarray
        :math:`\bar N(k)`, same shape as ``k``.

    See Also
    --------
    balian_bloch_level_density : Its derivative :math:`d\bar N/dk`.

    References
    ----------
    R. Balian and C. Bloch, "Distribution of eigenfrequencies for the wave
    equation in a finite domain. I," Ann. Phys. **60**, 401-447 (1970);
    "... III. Eigenfrequency density oscillations," Ann. Phys. **69**,
    76-160 (1972). H. P. Baltes and E. R. Hilf, *Spectra of Finite
    Systems* (Bibliographisches Institut, Mannheim, 1976).

    Examples
    --------
    The unit square with Dirichlet walls has levels
    :math:`k^2=\pi^2(n_x^2+n_y^2)`, :math:`n_x,n_y\ge1`. The exact
    staircase fluctuates about the expansion with zero mean:

    >>> import numpy as np
    >>> n = np.arange(1, 40)
    >>> k_exact = np.sort(np.pi * np.hypot(n[:, None], n[None, :]).ravel())
    >>> ks = np.linspace(10.0, 80.0, 2000)
    >>> corners = [np.pi / 2] * 4
    >>> residual = np.searchsorted(k_exact, ks) - balian_bloch_counting_function(ks, area=1.0, perimeter=4.0, corner_angles=corners)
    >>> bool(abs(residual.mean()) < 0.05)
    True
    """
    if boundary not in ("dirichlet", "neumann"):
        raise ValueError(f'boundary must be "dirichlet" or "neumann", got {boundary!r}.')
    sign = -1.0 if boundary == "dirichlet" else 1.0
    k = np.asarray(k, dtype=float)
    out = area * k**2 / (4.0 * np.pi) + sign * perimeter * k / (4.0 * np.pi) + _balian_bloch_constant(corner_angles, total_curvature)
    return out if out.ndim else float(out)


def balian_bloch_level_density(k, area: float, perimeter: float, boundary: str = "dirichlet"):
    r"""Smoothed density of billiard levels per unit wavenumber, :math:`d\bar N/dk`.

    .. math::

       \bar\rho(k) = \frac{A k}{2\pi} \mp \frac{L}{4\pi},

    the derivative of :func:`balian_bloch_counting_function` (the corner
    and curvature constant drops out). In energy, with
    :math:`E=k^2` (:math:`\hbar^2/2m=1`),
    :math:`\bar\rho(E)=A/4\pi\mp L/(8\pi\sqrt E)`.

    Parameters
    ----------
    k : float or ndarray
        Wavenumber(s).
    area : float
        Billiard area :math:`A`.
    perimeter : float
        Boundary length :math:`L`.
    boundary : {"dirichlet", "neumann"}, default="dirichlet"
        Boundary condition; sets the sign of the perimeter term.

    Returns
    -------
    float or ndarray
        :math:`\bar\rho(k)`, same shape as ``k``.

    Examples
    --------
    >>> round(balian_bloch_level_density(10.0, area=np.pi, perimeter=2 * np.pi), 6)
    4.5
    """
    if boundary not in ("dirichlet", "neumann"):
        raise ValueError(f'boundary must be "dirichlet" or "neumann", got {boundary!r}.')
    sign = -1.0 if boundary == "dirichlet" else 1.0
    k = np.asarray(k, dtype=float)
    out = area * k / (2.0 * np.pi) + sign * perimeter / (4.0 * np.pi)
    return out if out.ndim else float(out)
