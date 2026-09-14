r"""1D WKB wavefunctions, classical turning points, and Bohr-Sommerfeld (EBK) quantization.

The Wentzel-Kramers-Brillouin (WKB) approximation writes the wavefunction
as :math:`\psi(x) \sim p(x)^{-1/2}\exp(\pm i\int p\,dx/\hbar)` -- an
:math:`\hbar\to 0` asymptotic expansion valid wherever the local
de Broglie wavelength :math:`2\pi\hbar/p(x)` varies slowly compared to
itself. The prefactor :math:`p(x)^{-1/2}` is exactly the classical
probability density of a particle oscillating in the potential (it spends
more time, and so is more likely to be found, where it moves slowest);
this is the one-dimensional seed of every semiclassical idea in
:mod:`physicskit.semiclassical` -- the Van Vleck-Morette
determinant in :mod:`~physicskit.semiclassical.core.propagators` is
the same classical-probability prefactor for a propagator instead of a
stationary state, and the periodic-orbit sum in
:mod:`~physicskit.semiclassical.core.gutzwiller` is built from the
same action integral used here for Bohr-Sommerfeld quantization.

At each classical turning point the WKB approximation itself breaks down
(:math:`p(x)\to 0`); matching the oscillatory interior solution through
that breakdown region (via the Airy function connection formulas) costs
each *soft* (linear) turning point a phase of :math:`\pi/4`. For a bound
state with two such turning points, the round-trip quantization condition
:math:`\oint p\,dx = 2\pi\hbar(n+\tfrac12)` -- equivalently
:math:`\int_{x_1}^{x_2}p\,dx=(n+\tfrac12)\pi\hbar` -- is the
one-dimensional Einstein-Brillouin-Keller (EBK) rule implemented by
:func:`bohr_sommerfeld_energies`.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid, quad
from scipy.optimize import brentq

from .._compat import trapz

__all__ = [
    "classical_momentum",
    "turning_points",
    "wkb_action",
    "bohr_sommerfeld_energies",
    "wkb_wavefunction",
]


def classical_momentum(E: float, V, x: np.ndarray, m: float = 1.0) -> np.ndarray:
    r"""Classical momentum :math:`p(x) = \sqrt{2m(E-V(x))}` in the allowed region.

    Parameters
    ----------
    E : float
        Total energy.
    V : callable
        Potential energy function ``V(x)``.
    x : ndarray
        Positions at which to evaluate :math:`p(x)`.
    m : float, default=1.0
        Particle mass.

    Returns
    -------
    ndarray
        :math:`p(x)`, same shape as ``x``; ``nan`` wherever :math:`V(x)>E`
        (the classically forbidden region, where WKB oscillates into a
        real exponential instead).

    Examples
    --------
    >>> import numpy as np
    >>> V = lambda x: 0.5 * x ** 2
    >>> p = classical_momentum(E=2.0, V=V, x=np.array([0.0, 1.0, 3.0]))
    >>> np.round(p, 4)
    array([2.    , 1.7321,    nan])
    """
    x = np.asarray(x, dtype=float)
    diff = 2.0 * m * (E - V(x))
    return np.sqrt(np.where(diff >= 0, diff, np.nan))


def turning_points(E: float, V, x_min: float, x_max: float, n_search: int = 4000) -> list:
    r"""Locate the classical turning points (roots of :math:`E=V(x)`) in :math:`[x_{min},x_{max}]`.

    Scans a fine grid for sign changes of :math:`E-V(x)` and refines each
    with Brent's method; returns every turning point found, not just the
    innermost pair, so multi-well potentials are handled correctly.

    Parameters
    ----------
    E : float
        Total energy.
    V : callable
        Potential energy function ``V(x)``.
    x_min, x_max : float
        Search domain (should extend into the classically forbidden
        region on both sides of the state of interest).
    n_search : int, default=4000
        Number of grid points used to bracket the roots; increase for
        potentials with closely spaced turning points.

    Returns
    -------
    list of float
        Turning points, ascending.

    See Also
    --------
    wkb_action : Integrates :func:`classical_momentum` between two turning points.

    Examples
    --------
    >>> V = lambda x: 0.5 * x ** 2
    >>> [round(t, 4) for t in turning_points(E=2.0, V=V, x_min=-10, x_max=10)]
    [-2.0, 2.0]
    """
    xs = np.linspace(x_min, x_max, n_search)
    f = E - V(xs)
    roots = []
    for i in range(n_search - 1):
        if f[i] == 0:
            roots.append(float(xs[i]))
        elif f[i] * f[i + 1] < 0:
            roots.append(brentq(lambda x: E - V(x), xs[i], xs[i + 1]))
    return roots


def wkb_action(E: float, V, m: float, x1: float, x2: float, hbar: float = 1.0) -> float:
    r"""WKB action integral :math:`S(E)=\int_{x_1}^{x_2} p(x)\,dx` between two turning points.

    This is the classical action accumulated on one traversal of the
    classically allowed region -- half of the full round-trip action
    :math:`\oint p\,dx`, and (as a function of :math:`E`) exactly the
    Hamilton principal function whose energy derivative gives the
    classical period, :math:`T(E)=dS/dE`, used by
    :func:`physicskit.semiclassical.core.gutzwiller.classical_period`.

    Parameters
    ----------
    E : float
        Total energy.
    V : callable
        Potential energy function ``V(x)``.
    m : float
        Particle mass.
    x1, x2 : float
        The two turning points (e.g. from :func:`turning_points`).
    hbar : float, default=1.0
        Value of :math:`\hbar` to use (only needed by callers converting
        this to a phase; the integral itself is :math:`\hbar`-independent).

    Returns
    -------
    float
        The action :math:`S(E)`.

    Examples
    --------
    For the harmonic oscillator, :math:`S(E)=\pi E/\omega` exactly:

    >>> V = lambda x: 0.5 * x ** 2
    >>> x1, x2 = turning_points(E=2.0, V=V, x_min=-10, x_max=10)
    >>> round(wkb_action(E=2.0, V=V, m=1.0, x1=x1, x2=x2), 6)
    6.283185
    """

    def integrand(x):
        return float(np.nan_to_num(classical_momentum(E, V, np.array([x]), m)[0], nan=0.0))

    result, _ = quad(integrand, x1, x2, limit=200)
    return result


def bohr_sommerfeld_energies(V, m: float, x_min: float, x_max: float, n_max: int, hbar: float = 1.0, E_min: float = 1e-3, E_max: float = 100.0) -> np.ndarray:
    r"""Bohr-Sommerfeld (EBK) bound-state energies from the WKB quantization condition.

    Solves :math:`\int_{x_1(E_n)}^{x_2(E_n)}p(x)\,dx=(n+\tfrac12)\pi\hbar`
    for each :math:`n=0,\dots,n_{max}-1` by root-finding on :math:`E`,
    re-locating the turning points at every trial energy. For the
    harmonic oscillator this reproduces the exact spectrum
    :math:`E_n=(n+\tfrac12)\hbar\omega` to machine precision, since the
    WKB approximation is exact whenever the potential is exactly
    quadratic.

    Parameters
    ----------
    V : callable
        Potential energy function ``V(x)``, with a single classically
        allowed region containing its minimum for every energy searched.
    m : float
        Particle mass.
    x_min, x_max : float
        Search domain for :func:`turning_points`; must extend well into
        the classically forbidden region at ``E_max``.
    n_max : int
        Number of levels to compute (:math:`n=0,\dots,n_{max}-1`).
    hbar : float, default=1.0
        Value of :math:`\hbar` to use.
    E_min : float, default=1e-3
        Lower bracket for the ground-state root search; must be small
        enough that its turning points still resolve on the
        :func:`turning_points` search grid.
    E_max : float, default=100.0
        Upper bracket for the highest level's root search.

    Returns
    -------
    ndarray, shape (n_max,)
        Quantized energies, ascending.

    See Also
    --------
    wkb_action : The action integral this quantizes.
    wkb_wavefunction : The corresponding approximate wavefunctions.

    Examples
    --------
    >>> import numpy as np
    >>> V = lambda x: 0.5 * x ** 2
    >>> energies = bohr_sommerfeld_energies(V, m=1.0, x_min=-20, x_max=20, n_max=5)
    >>> np.round(energies, 6)
    array([0.5, 1.5, 2.5, 3.5, 4.5])
    """
    energies = np.empty(n_max)
    for n in range(n_max):

        def condition(E, n=n):
            tp = turning_points(E, V, x_min, x_max)
            x1, x2 = tp[0], tp[-1]
            return wkb_action(E, V, m, x1, x2, hbar) - (n + 0.5) * np.pi * hbar

        energies[n] = brentq(condition, E_min, E_max)
    return energies


def wkb_wavefunction(x: np.ndarray, E: float, V, m: float = 1.0, hbar: float = 1.0) -> np.ndarray:
    r"""The (real-valued, standing-wave) WKB wavefunction in the classically allowed region.

    .. math::

       \psi(x) \approx \frac{C}{\sqrt{p(x)}}
       \cos\!\left(\frac{1}{\hbar}\int_{x_1}^{x}p(x')\,dx' - \frac{\pi}{4}\right),

    with the :math:`\pi/4` phase set by the Airy connection formula at
    the left (soft) turning point :math:`x_1`, and :math:`C` fixed by
    numerical normalization over the allowed region. Diverges (is not
    valid) within a few de Broglie wavelengths of either turning point;
    zero is returned in the classically forbidden region rather than the
    (also invalid) naive imaginary-momentum continuation.

    Parameters
    ----------
    x : ndarray
        Positions at which to evaluate the wavefunction.
    E : float
        Energy (e.g. from :func:`bohr_sommerfeld_energies`, for a properly
        quantized bound state).
    V : callable
        Potential energy function ``V(x)``.
    m : float, default=1.0
        Particle mass.
    hbar : float, default=1.0
        Value of :math:`\hbar` to use.

    Returns
    -------
    ndarray
        :math:`\psi(x)`, same shape as ``x``, normalized so
        :math:`\int|\psi|^2\,dx=1` over ``x``'s range.

    See Also
    --------
    bohr_sommerfeld_energies : Quantized energies this is evaluated at.

    Examples
    --------
    The WKB wavefunction at the Bohr-Sommerfeld energy for quantum number
    ``n`` has exactly ``n`` nodes, the standard node-counting theorem:

    >>> import numpy as np
    >>> V = lambda x: 0.5 * x ** 2
    >>> E5 = bohr_sommerfeld_energies(V, m=1.0, x_min=-20, x_max=20, n_max=6)[5]
    >>> x = np.linspace(-10, 10, 4000)
    >>> psi = wkb_wavefunction(x, E5, V, m=1.0)
    >>> nonzero = psi[psi != 0]
    >>> int(np.sum(np.diff(np.sign(nonzero)) != 0))
    5
    """
    x = np.asarray(x, dtype=float)
    tp = turning_points(E, V, float(np.min(x)), float(np.max(x)))
    x1, x2 = tp[0], tp[-1]
    allowed = (x > x1) & (x < x2)

    psi = np.zeros_like(x)
    xa = x[allowed]
    p = classical_momentum(E, V, xa, m)
    order = np.argsort(xa)
    xa_sorted = xa[order]
    p_sorted = p[order]
    phase_sorted = cumulative_trapezoid(p_sorted, xa_sorted, initial=0.0) / hbar
    unsort = np.argsort(order)
    phase = phase_sorted[unsort]

    psi_allowed = np.cos(phase - np.pi / 4.0) / np.sqrt(p)
    psi[allowed] = psi_allowed

    norm = np.sqrt(trapz(psi**2, x))
    if norm > 0:
        psi = psi / norm
    return psi
