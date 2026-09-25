r"""The fractional quantum Hall effect (1982-1983): Laughlin's wavefunction.

Tsui, Stormer, and Gossard found that an ultraclean 2D electron gas in a
strong magnetic field develops Hall plateaus at simple fractional fillings,
:math:`\nu=1/3` foremost among them -- a result the single-particle Landau-
level picture behind the *integer* effect
(:mod:`physicskit.condensed.landau_levels`) cannot explain at all, since a
partially filled Landau level is macroscopically degenerate and gapless
without electron-electron interactions. Laughlin supplied the missing
many-body mechanism: a trial wavefunction for :math:`N` electrons confined
to the lowest Landau level of a disk,

.. math::

   \Psi_m(z_1,\dots,z_N) = \prod_{i<j}(z_i-z_j)^m\,
   \exp\!\left(-\sum_i \frac{|z_i|^2}{4\ell_B^2}\right), \qquad m \text{ odd},

with :math:`z_j=x_j+iy_j` (magnetic length :math:`\ell_B=1` throughout this
module) and filling :math:`\nu=1/m`. The prefactor's zeroes at coincident
particle positions are exactly what strong Coulomb repulsion demands, and
:math:`|\Psi_m|^2` doubles as the Boltzmann weight of a classical 2D
one-component plasma at an effective temperature set by :math:`m` (Laughlin's
"plasma analogy"), letting ordinary Metropolis Monte Carlo sample it
directly with no diagonalization of any kind.

The pair correlation function :math:`g(r)` extracted from those samples
shows the wavefunction's defining feature directly: a "correlation hole"
that drives :math:`g(r) \to 0` as :math:`r \to 0` -- particles actively
avoid each other far more strongly than Pauli exclusion alone would
enforce -- rising to the uncorrelated value :math:`g \approx 1` over a
few magnetic lengths.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = ["laughlin_metropolis_sweep", "laughlin_pair_correlation", "laughlin_radial_density"]


@njit(cache=True)
def _delta_log_prob(z, i, z_new, m):
    n = z.shape[0]
    dlog = 0.0
    for j in range(n):
        if j == i:
            continue
        dlog += 2.0 * m * (np.log(np.abs(z_new - z[j])) - np.log(np.abs(z[i] - z[j])))
    dlog += -0.5 * (np.abs(z_new) ** 2 - np.abs(z[i]) ** 2)
    return dlog


@njit(cache=True)
def laughlin_metropolis_sweep(z: np.ndarray, m: int, step: float = 1.0) -> np.ndarray:
    r"""Perform one Metropolis sweep sampling :math:`|\Psi_m|^2`, in place.

    Proposes a small random displacement of one particle at a time and
    accepts or rejects it with the exact Metropolis ratio of
    :math:`|\Psi_m|^2` -- Laughlin's "plasma analogy" Boltzmann weight,
    :math:`\prod_{i<j}|z_i-z_j|^{2m}\,e^{-\sum_i|z_i|^2/2}` -- with no
    explicit normalization needed since only the ratio between
    configurations ever enters.

    Parameters
    ----------
    z : ndarray of shape (N,), dtype complex128
        Particle positions :math:`z_j=x_j+iy_j`. Modified in place.
    m : int
        Laughlin exponent (filling :math:`\nu=1/m`); odd for fermions.
    step : float, default=1.0
        Maximum displacement of a trial move along each of :math:`x, y`,
        drawn uniformly from ``[-step, step]``.

    Returns
    -------
    ndarray of shape (N,)
        The same array passed in.

    Notes
    -----
    Trial moves draw from Numba's internal RNG, which neither
    ``np.random.seed`` nor a ``np.random.default_rng`` generator affects
    when called from ordinary Python. For a reproducible chain, seed it
    first with :func:`physicskit.statphys.core.monte_carlo.seed_numba_random`.

    Examples
    --------
    A two-particle system's Metropolis chain never proposes an *exactly*
    coincident configuration acceptable (:math:`|\Psi_m|^2 = 0` there), so
    repeated sweeps keep the two particles strictly apart:

    >>> import numpy as np
    >>> from physicskit.statphys.core.monte_carlo import seed_numba_random
    >>> seed_numba_random(0)
    >>> z = np.array([0.1 + 0.0j, -0.1 + 0.0j])
    >>> for _ in range(200):
    ...     _ = laughlin_metropolis_sweep(z, m=3, step=0.5)
    >>> bool(np.abs(z[0] - z[1]) > 1e-3)
    True
    """
    n = z.shape[0]
    for _ in range(n):
        i = np.random.randint(0, n)
        dx = (np.random.random() - 0.5) * 2.0 * step
        dy = (np.random.random() - 0.5) * 2.0 * step
        z_new = z[i] + dx + 1j * dy
        dlog = _delta_log_prob(z, i, z_new, m)
        if dlog >= 0.0 or np.random.random() < np.exp(dlog):
            z[i] = z_new
    return z


@njit(cache=True)
def _pair_distances(z):
    n = z.shape[0]
    out = np.empty(n * (n - 1) // 2)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            out[k] = np.abs(z[i] - z[j])
            k += 1
    return out


def laughlin_pair_correlation(z_samples: np.ndarray, m: int, r_max: float, n_bins: int = 60) -> tuple:
    r"""Pair correlation function :math:`g(r)` from sampled Laughlin configurations.

    Bins every pairwise separation across a stack of equilibrium samples
    (drawn with :func:`laughlin_metropolis_sweep`) and normalizes by the
    Laughlin state's exact bulk density :math:`\rho=1/(2\pi m \ell_B^2)`,
    so that :math:`g(r) \to 1` far from any correlation the wavefunction
    itself imposes.

    Parameters
    ----------
    z_samples : ndarray of shape (n_samples, N), dtype complex128
        Stack of particle-position snapshots, ideally decorrelated by
        several sweeps between samples.
    m : int
        Laughlin exponent used to generate the samples.
    r_max : float
        Largest separation to bin, in units of :math:`\ell_B`.
    n_bins : int, default=60
        Number of radial bins.

    Returns
    -------
    r : ndarray of shape (n_bins,)
        Bin-center separations.
    g : ndarray of shape (n_bins,)
        Pair correlation function; ``g[0] \approx 0`` is the correlation
        hole, ``g \to 1`` at large separation.

    See Also
    --------
    laughlin_metropolis_sweep : Generates the equilibrium samples this function bins.

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.statphys.core.monte_carlo import seed_numba_random
    >>> seed_numba_random(0)
    >>> rng = np.random.default_rng(0)
    >>> N, m = 12, 3
    >>> R0 = np.sqrt(2 * m * N)
    >>> r0 = R0 * np.sqrt(rng.random(N))
    >>> th0 = rng.uniform(0, 2 * np.pi, N)
    >>> z = (r0 * np.cos(th0) + 1j * r0 * np.sin(th0)).astype(np.complex128)
    >>> for _ in range(300):
    ...     _ = laughlin_metropolis_sweep(z, m=m, step=1.0)
    >>> samples = np.array([laughlin_metropolis_sweep(z, m=m, step=1.0).copy() for _ in range(200)])
    >>> r, g = laughlin_pair_correlation(samples, m=m, r_max=6.0, n_bins=30)
    >>> bool(g[0] < 0.1)
    True
    """
    n = z_samples.shape[1]
    rho = 1.0 / (2.0 * np.pi * m)
    all_dists = np.concatenate([_pair_distances(z) for z in z_samples])
    r_edges = np.linspace(0.0, r_max, n_bins + 1)
    hist, _ = np.histogram(all_dists, bins=r_edges)
    r_centers = 0.5 * (r_edges[1:] + r_edges[:-1])
    shell_area = np.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)
    n_samples = z_samples.shape[0]
    g = 2.0 * hist / (n_samples * n * rho * shell_area)
    return r_centers, g


def laughlin_radial_density(z_samples: np.ndarray, r_max: float, n_bins: int = 60) -> tuple:
    r"""Radially averaged particle density profile from sampled Laughlin configurations.

    Parameters
    ----------
    z_samples : ndarray of shape (n_samples, N), dtype complex128
        Stack of particle-position snapshots (see :func:`laughlin_pair_correlation`).
    r_max : float
        Largest radius to bin, in units of :math:`\ell_B`.
    n_bins : int, default=60
        Number of radial bins.

    Returns
    -------
    r : ndarray of shape (n_bins,)
        Bin-center radii.
    density : ndarray of shape (n_bins,)
        Mean particle density in each annulus, in units of :math:`\ell_B^{-2}`.

    Notes
    -----
    A large-:math:`N` Laughlin droplet is an incompressible liquid: this
    profile is approximately flat at the bulk density
    :math:`\rho=1/(2\pi m \ell_B^2)` out to the droplet radius, falling to
    zero over a few magnetic lengths at the edge -- the real-space
    counterpart of the sharp gap that makes the plateau itself possible.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> N, m = 12, 3
    >>> R0 = np.sqrt(2 * m * N)
    >>> r0 = R0 * np.sqrt(rng.random(N))
    >>> th0 = rng.uniform(0, 2 * np.pi, N)
    >>> z = (r0 * np.cos(th0) + 1j * r0 * np.sin(th0)).astype(np.complex128)
    >>> for _ in range(300):
    ...     _ = laughlin_metropolis_sweep(z, m=m, step=1.0)
    >>> samples = np.array([laughlin_metropolis_sweep(z, m=m, step=1.0).copy() for _ in range(200)])
    >>> r, density = laughlin_radial_density(samples, r_max=2 * R0, n_bins=20)
    >>> bool(density[-1] < density[2])
    True
    """
    r_edges = np.linspace(0.0, r_max, n_bins + 1)
    all_radii = np.abs(z_samples).ravel()
    hist, _ = np.histogram(all_radii, bins=r_edges)
    r_centers = 0.5 * (r_edges[1:] + r_edges[:-1])
    shell_area = np.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)
    n_samples = z_samples.shape[0]
    density = hist / (n_samples * shell_area)
    return r_centers, density
