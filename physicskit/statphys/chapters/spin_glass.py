"""Spin glasses: quenched disorder, frustration, and replica symmetry breaking.

Unlike the pure Ising model, where every bond has the same sign, a spin
glass's bond couplings are themselves random variables, fixed ("quenched")
for the lifetime of a given sample rather than fluctuating with the spins.
A plaquette with an odd number of antiferromagnetic bonds is *frustrated*:
no spin configuration can satisfy every one of its bonds simultaneously.
This produces a rugged, many-valley energy landscape and, below a spin-glass
transition temperature, a frozen but disordered ground state -- detected not
by a conventional magnetization but by the Edwards-Anderson order
parameter, the overlap between two independently thermalized replicas of
the same disorder realization.

:class:`EdwardsAndersonSpinGlass2D` is the short-range, nearest-neighbor
version of this story on a 2D lattice. :class:`SherringtonKirkpatrick` is
its infinite-range completion: every spin couples to every other spin, the
model Sherrington and Kirkpatrick introduced in 1975 and that Giorgio
Parisi's 1979-1980 replica-symmetry-breaking solution ultimately resolved,
revealing an equilibrium phase built not from one ground state but from a
hierarchy of infinitely many, organized by a nontrivial overlap
distribution :math:`P(q)` rather than a single frozen value.
"""

from __future__ import annotations

import numpy as np

from physicskit.statphys.core.monte_carlo import (
    metropolis_sweep_sk,
    metropolis_sweep_spin_glass,
    seed_numba_random,
    sk_total_energy,
    spin_glass_total_energy,
)

__all__ = ["EdwardsAndersonSpinGlass2D", "SherringtonKirkpatrick"]


class EdwardsAndersonSpinGlass2D:
    """The 2D +/-J Edwards-Anderson Ising spin glass on a periodic L x L lattice.

    Parameters
    ----------
    L : int, default=32
        Linear lattice size.
    J : float, default=1.0
        Bond magnitude; each bond independently takes the value ``+J`` or
        ``-J`` with equal probability (the "bimodal" Edwards-Anderson
        glass).
    kB : float, default=1.0
        Boltzmann constant.
    seed : int, optional
        Seed for the quenched disorder realization, the initial spin
        configuration, and the Metropolis dynamics.

    Attributes
    ----------
    spins : ndarray of shape (L, L)
        Current spin configuration, values in ``{-1, +1}``.
    J_right, J_down : ndarray of shape (L, L)
        Quenched bond couplings; see
        :func:`physicskit.statphys.core.monte_carlo.metropolis_sweep_spin_glass` for
        their indexing convention. Fixed for the model's lifetime.
    """

    def __init__(self, L=32, J=1.0, kB=1.0, seed=None):
        self.L = L
        self.J = J
        self.kB = kB
        self.n_sites = L * L
        self._rng = np.random.default_rng(seed)
        if seed is not None:
            seed_numba_random(seed)
        self.spins = self._rng.choice(np.array([-1, 1], dtype=np.int64), size=(L, L))
        self.J_right = self._rng.choice(np.array([-J, J]), size=(L, L))
        self.J_down = self._rng.choice(np.array([-J, J]), size=(L, L))

    def sweep(self, beta, n_sweeps=1):
        """Advance the spins by ``n_sweeps`` Metropolis-Hastings sweeps (bonds are fixed).

        Parameters
        ----------
        beta : float
            Inverse temperature.
        n_sweeps : int, default=1
            Number of sweeps to perform.
        """
        for _ in range(n_sweeps):
            metropolis_sweep_spin_glass(self.spins, beta, self.J_right, self.J_down)

    def energy(self):
        """Total energy :math:`-\\sum_{\\langle i,j\\rangle} J_{ij} s_i s_j` of the current configuration."""
        return spin_glass_total_energy(self.spins, self.J_right, self.J_down)

    def frustration_density(self):
        """Fraction of elementary plaquettes that are frustrated.

        A plaquette is frustrated when the product of its four bond
        couplings is negative -- no spin configuration can satisfy all four
        bonds at once. This is a fixed geometric property of the quenched
        disorder, independent of the current spin configuration.

        Returns
        -------
        float
        """
        product = self.J_right * np.roll(self.J_down, -1, axis=1) * np.roll(self.J_right, -1, axis=0) * self.J_down
        return float(np.mean(product < 0))

    def edwards_anderson_order_parameter(self, beta, n_equil=200, n_measure=200, measure_every=1):
        """Estimate :math:`\\langle q^2 \\rangle`, the mean-squared Edwards-Anderson replica overlap.

        Two independently initialized and thermalized replicas, sharing the
        *same* quenched bonds, are compared via their spin overlap
        :math:`q = \\frac{1}{N}\\sum_i s_i^{(1)} s_i^{(2)}`. Even though
        each replica's spins keep fluctuating, below the spin-glass
        transition they fluctuate around the same disorder-selected frozen
        pattern, so :math:`\\langle q^2 \\rangle` stays nonzero; above it,
        the replicas decorrelate and :math:`\\langle q^2 \\rangle \\to 0`.

        Parameters
        ----------
        beta : float
            Inverse temperature.
        n_equil : int, default=200
            Equilibration sweeps for each replica.
        n_measure : int, default=200
            Number of overlap measurements.
        measure_every : int, default=1
            Sweeps between successive measurements.

        Returns
        -------
        float
            Estimated :math:`\\langle q^2 \\rangle \\in [0, 1]`.
        """
        replica_a = self._rng.choice(np.array([-1, 1], dtype=np.int64), size=(self.L, self.L))
        replica_b = self._rng.choice(np.array([-1, 1], dtype=np.int64), size=(self.L, self.L))
        for _ in range(n_equil):
            metropolis_sweep_spin_glass(replica_a, beta, self.J_right, self.J_down)
            metropolis_sweep_spin_glass(replica_b, beta, self.J_right, self.J_down)

        overlaps = np.empty(n_measure)
        for k in range(n_measure):
            for _ in range(measure_every):
                metropolis_sweep_spin_glass(replica_a, beta, self.J_right, self.J_down)
                metropolis_sweep_spin_glass(replica_b, beta, self.J_right, self.J_down)
            overlaps[k] = np.mean(replica_a * replica_b)
        return float(np.mean(overlaps**2))


class SherringtonKirkpatrick:
    """The infinite-range Sherrington-Kirkpatrick Ising spin glass on N fully-connected spins.

    Every pair of spins is coupled by an independent Gaussian random bond,

    .. math::

        H = -\\sum_{i<j} J_{ij}\\, s_i s_j, \\qquad
        J_{ij} \\sim \\mathcal{N}\\!\\left(0, \\frac{J^2}{N}\\right),

    where the :math:`1/N` variance scaling is what keeps the energy
    extensive as :math:`N \\to \\infty`. Rather than attempt Parisi's
    analytic replica-symmetry-breaking solution, :meth:`overlap_distribution`
    measures its numerically observable fingerprint directly: the
    distribution :math:`P(q)` of the replica overlap
    :math:`q = \\frac{1}{N}\\sum_i s_i^{(1)} s_i^{(2)}`, pooled over many
    independent disorder realizations. A single sharp peak at :math:`q=0`
    signals the trivial (replica-symmetric) paramagnetic phase; a broad,
    non-Gaussian, non-self-averaging :math:`P(q)` is the finite-size
    signature of the many-valley spin-glass phase that Parisi's solution
    describes exactly in the :math:`N \\to \\infty` limit.

    Parameters
    ----------
    N : int, default=128
        Number of spins. Note one sweep costs :math:`O(N^2)`, since every
        spin couples to every other one.
    J : float, default=1.0
        Coupling scale; bond strengths are drawn from :math:`\\mathcal{N}(0,
        J^2/N)`.
    kB : float, default=1.0
        Boltzmann constant.
    seed : int, optional
        Seed for the quenched disorder, initial spins, and dynamics.

    Attributes
    ----------
    spins : ndarray of shape (N,)
        Current spin configuration, values in ``{-1, +1}``.
    J : ndarray of shape (N, N)
        Quenched, symmetric coupling matrix with zero diagonal. Fixed for
        the model's lifetime.
    """

    def __init__(self, N=128, J=1.0, kB=1.0, seed=None):
        self.N = N
        self.J_scale = J
        self.kB = kB
        self._rng = np.random.default_rng(seed)
        if seed is not None:
            seed_numba_random(seed)
        self.spins = self._rng.choice(np.array([-1, 1], dtype=np.int64), size=N)
        bonds = self._rng.normal(loc=0.0, scale=J / np.sqrt(N), size=(N, N))
        upper = np.triu(bonds, k=1)
        self.J = upper + upper.T

    def sweep(self, beta, n_sweeps=1):
        """Advance the spins by ``n_sweeps`` Metropolis-Hastings sweeps (bonds are fixed).

        Parameters
        ----------
        beta : float
            Inverse temperature.
        n_sweeps : int, default=1
            Number of sweeps to perform.
        """
        for _ in range(n_sweeps):
            metropolis_sweep_sk(self.spins, beta, self.J)

    def energy(self):
        """Total energy :math:`-\\sum_{i<j} J_{ij} s_i s_j` of the current configuration."""
        return sk_total_energy(self.spins, self.J)

    def overlap_distribution(self, beta, n_disorder=30, n_equil=300, n_measure=100, measure_every=1):
        """Sample the replica-overlap distribution P(q), pooled over independent disorder realizations.

        For each of ``n_disorder`` independent bond realizations, two
        replicas are initialized randomly, thermalized under that shared
        disorder, and their overlap :math:`q = \\frac{1}{N}\\sum_i
        s_i^{(1)} s_i^{(2)}` recorded repeatedly. Pooling samples across
        many disorder realizations (rather than just many measurements of
        one realization) is essential here: unlike the Edwards-Anderson
        model's single well-defined :math:`\\langle q^2 \\rangle`, the SK
        spin glass's overlap distribution is sample-dependent (it does not
        self-average) below the transition, and it is exactly this
        sample-to-sample variability that Parisi's solution organizes into
        a hierarchy of pure states.

        Parameters
        ----------
        beta : float
            Inverse temperature.
        n_disorder : int, default=30
            Number of independent quenched-disorder realizations to sample.
        n_equil : int, default=300
            Equilibration sweeps per replica, for each disorder realization.
        n_measure : int, default=100
            Number of overlap measurements per disorder realization.
        measure_every : int, default=1
            Sweeps between successive measurements.

        Returns
        -------
        ndarray of shape (n_disorder * n_measure,)
            Pooled overlap samples :math:`q \\in [-1, 1]`.
        """
        samples = np.empty(n_disorder * n_measure)
        idx = 0
        for _ in range(n_disorder):
            bonds = self._rng.normal(loc=0.0, scale=self.J_scale / np.sqrt(self.N), size=(self.N, self.N))
            upper = np.triu(bonds, k=1)
            J = upper + upper.T
            replica_a = self._rng.choice(np.array([-1, 1], dtype=np.int64), size=self.N)
            replica_b = self._rng.choice(np.array([-1, 1], dtype=np.int64), size=self.N)
            for _ in range(n_equil):
                metropolis_sweep_sk(replica_a, beta, J)
                metropolis_sweep_sk(replica_b, beta, J)
            for _ in range(n_measure):
                for _ in range(measure_every):
                    metropolis_sweep_sk(replica_a, beta, J)
                    metropolis_sweep_sk(replica_b, beta, J)
                samples[idx] = np.mean(replica_a * replica_b)
                idx += 1
        return samples
