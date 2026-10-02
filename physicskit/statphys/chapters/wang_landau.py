"""Wang-Landau flat-histogram sampling of the 2D Ising density of states.

Metropolis sampling draws configurations at one temperature. Wang and
Landau (2001) instead estimate the density of states :math:`g(E)` itself,
by a random walk in energy space that is biased against levels it has
already visited. Once :math:`g(E)` is known, the canonical partition
function

.. math::

    Z(\\beta) = \\sum_E g(E)\\, e^{-\\beta E}

and every thermodynamic average follow at *any* temperature from one run.

:class:`WangLandauIsing` runs the walk on the periodic :math:`L \\times L`
Ising lattice, and :func:`ising_density_of_states_exact` enumerates the
exact :math:`g(E)` of a small lattice to check it against.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from physicskit.statphys.core.monte_carlo import seed_numba_random, wang_landau_sweeps_ising

__all__ = ["WangLandauIsing", "canonical_from_density_of_states", "ising_density_of_states_exact"]


def ising_density_of_states_exact(L, J=1.0):
    """Exact density of states of a periodic :math:`L \\times L` Ising lattice, by enumeration.

    All :math:`2^{L^2}` configurations are enumerated, so this is only
    practical up to :math:`L = 4` (65536 states).

    Parameters
    ----------
    L : int
        Linear lattice size, at most 4.
    J : float, default=1.0
        Ferromagnetic coupling.

    Returns
    -------
    energies : ndarray of shape (L*L + 1,)
        Energy levels :math:`E_k = -2JN + 4Jk`.
    g : ndarray of shape (L*L + 1,)
        Number of configurations at each level (zero for the inaccessible
        levels :math:`E = \\pm(2JN - 4J)`).

    Examples
    --------
    >>> E, g = ising_density_of_states_exact(2)
    >>> int(g.sum()), int(g[0])
    (16, 2)
    """
    if L > 4:
        raise ValueError("exact enumeration is only practical for L <= 4")
    N = L * L
    configs = np.arange(2**N, dtype=np.int64)
    spins = (((configs[:, None] >> np.arange(N)) & 1) * 2 - 1).reshape(-1, L, L)
    bonds = spins * np.roll(spins, -1, axis=1) + spins * np.roll(spins, -1, axis=2)
    E = -J * bonds.sum(axis=(1, 2))
    k = np.rint((E + 2 * J * N) / (4 * J)).astype(np.int64)
    g = np.bincount(k, minlength=N + 1).astype(np.float64)
    return -2 * J * N + 4 * J * np.arange(N + 1), g


def canonical_from_density_of_states(energies, log_g, temperatures, n_sites, kB=1.0):
    """Canonical energy and specific heat per site from a density of states.

    .. math::

        \\langle E^n \\rangle_\\beta = \\frac{\\sum_E E^n g(E) e^{-\\beta E}}{\\sum_E g(E) e^{-\\beta E}},
        \\qquad
        C_v = \\frac{\\langle E^2 \\rangle - \\langle E \\rangle^2}{k_B T^2 N}.

    Parameters
    ----------
    energies : array_like
        Energy levels.
    log_g : array_like
        :math:`\\ln g(E)` on those levels; ``-inf`` marks an empty level.
    temperatures : array_like
        Temperatures at which to evaluate.
    n_sites : int
        Number of sites, to report per-site quantities.
    kB : float, default=1.0
        Boltzmann constant.

    Returns
    -------
    dict of str -> ndarray
        Keys ``"T"``, ``"E"`` (per site), ``"C_v"`` (per site) and
        ``"log_Z"``.

    Examples
    --------
    >>> E, g = ising_density_of_states_exact(2)
    >>> out = canonical_from_density_of_states(E, np.log(g), [1e-3], n_sites=4)
    >>> float(out["E"][0])
    -2.0
    """
    E = np.asarray(energies, dtype=np.float64)
    lg = np.asarray(log_g, dtype=np.float64)
    T = np.atleast_1d(np.asarray(temperatures, dtype=np.float64))
    out_E = np.empty(T.size)
    out_C = np.empty(T.size)
    log_Z = np.empty(T.size)
    with np.errstate(divide="ignore"):
        for n, t in enumerate(T):
            w = lg - E / (kB * t)
            log_Z[n] = logsumexp(w)
            p = np.exp(w - log_Z[n])
            mean = np.sum(p * E)
            out_E[n] = mean / n_sites
            out_C[n] = np.sum(p * (E - mean) ** 2) / (kB * t**2 * n_sites)
    return {"T": T, "E": out_E, "C_v": out_C, "log_Z": log_Z}


class WangLandauIsing:
    """Wang-Landau estimate of the density of states of the 2D Ising model.

    The walk proposes single-spin flips accepted with probability
    :math:`\\min(1, g(E_{\\text{old}})/g(E_{\\text{new}}))` and raises
    :math:`\\ln g` of the current level by :math:`\\ln f` after every
    proposal. Whenever the visit histogram is flat (every visited level
    within ``flatness`` of the mean), it is reset and :math:`\\ln f` is
    halved, until :math:`\\ln f` falls below ``log_f_final``.

    Parameters
    ----------
    L : int, default=8
        Linear lattice size.
    J : float, default=1.0
        Ferromagnetic coupling.
    seed : int, optional
        Seed for the initial configuration and the walk.

    Examples
    --------
    >>> wl = WangLandauIsing(L=4, seed=0)
    >>> E, log_g = wl.run(log_f_final=1e-3)
    >>> round(float(np.exp(log_g[0])), 6)  # normalized to the two ground states
    2.0
    """

    def __init__(self, L=8, J=1.0, seed=None):
        self.L = L
        self.J = J
        self.n_sites = L * L
        rng = np.random.default_rng(seed)
        if seed is not None:
            seed_numba_random(seed)
        self.spins = rng.choice(np.array([-1, 1], dtype=np.int64), size=(L, L))
        self.energies = -2 * J * self.n_sites + 4 * J * np.arange(self.n_sites + 1)
        # E = +/-(2JN - 4J) cannot be reached by any configuration
        self.accessible = np.ones(self.n_sites + 1, dtype=bool)
        self.accessible[[1, self.n_sites - 1]] = False
        self.log_f_history: list[float] = []

    def run(self, log_f_initial=1.0, log_f_final=1e-6, flatness=0.8, sweeps_per_check=1000):
        """Run the walk to convergence and return the normalized :math:`\\ln g(E)`.

        Parameters
        ----------
        log_f_initial : float, default=1.0
            Starting modification factor :math:`\\ln f_0`.
        log_f_final : float, default=1e-6
            Stop once :math:`\\ln f` falls below this.
        flatness : float, default=0.8
            A histogram is flat when its minimum over the accessible levels
            is at least this fraction of its mean.
        sweeps_per_check : int, default=1000
            Sweeps between flatness checks.

        Returns
        -------
        energies : ndarray
            Energy levels :math:`E_k`.
        log_g : ndarray
            :math:`\\ln g(E)`, normalized so that :math:`g(E_0) = 2` (the
            two ground states), with ``-inf`` on the inaccessible levels.
        """
        log_g = np.zeros(self.n_sites + 1)
        hist = np.zeros(self.n_sites + 1, dtype=np.int64)
        log_f = log_f_initial
        self.log_f_history = []
        while log_f > log_f_final:
            wang_landau_sweeps_ising(self.spins, log_g, hist, log_f, sweeps_per_check, self.J)
            h = hist[self.accessible]
            if h.min() >= flatness * h.mean():
                self.log_f_history.append(log_f)
                hist[:] = 0
                log_f /= 2.0
        log_g = log_g - log_g[0] + np.log(2.0)
        log_g[~self.accessible] = -np.inf
        return self.energies.copy(), log_g
