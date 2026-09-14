"""Real-space renormalization group: Kadanoff block-spin coarse-graining.

Kadanoff's block-spin construction groups an Ising lattice into
:math:`b \\times b` blocks and replaces each block by a single effective spin
(here, majority rule with ``b = 2``), producing a new, smaller lattice at the
same nominal temperature. Iterating this map is a real-space renormalization
group transformation: starting configurations flow toward one of the
theory's fixed points -- a fully ordered state at :math:`T \\ll T_c`, a fully
disordered state at :math:`T \\gg T_c`, and a self-similar, scale-invariant
critical configuration at :math:`T = T_c`, where the coarse-grained lattice
looks statistically identical to the original.
"""

from __future__ import annotations

import numpy as np

from physicskit.statphys.chapters.ising_lattice import Ising2D

__all__ = ["BlockSpinRG"]


class BlockSpinRG:
    """Iterated 2x2 majority-rule block-spin coarse-graining of an Ising configuration.

    Parameters
    ----------
    L : int, default=64
        Linear size of the initial lattice. Should be a power of 2 times the
        largest block factor you intend to reach, e.g. ``L=64`` supports up
        to 6 coarse-graining steps.
    T : float, default=None
        Temperature at which to equilibrate the initial configuration, in
        units of ``J/kB``. Defaults to the exact critical temperature
        (``Ising2D.T_C``) if not given, i.e. the critical fixed point.
    J : float, default=1.0
        Ising coupling constant used to equilibrate the initial
        configuration.
    n_equil_sweeps : int, default=500
        Wolff cluster updates used to equilibrate the initial fine-grained
        configuration at temperature ``T`` before coarse-graining begins.
    seed : int, optional
        Seed for the initial equilibration and for majority-rule tie
        breaking.

    Attributes
    ----------
    initial_spins : ndarray of shape (L, L)
        The equilibrated fine-grained configuration coarse-graining starts
        from.
    """

    def __init__(self, L=64, T=None, J=1.0, n_equil_sweeps=500, seed=None):
        self.L = L
        self.J = J
        self._rng = np.random.default_rng(seed)

        model = Ising2D(L=L, J=J, seed=seed)
        self.T = model.T_C if T is None else T
        beta = 1.0 / self.T if self.T > 0 else np.inf
        if np.isfinite(beta):
            # Wolff cluster updates equilibrate far more reliably than single-spin
            # Metropolis: at low T, Metropolis dynamics from a random start undergoes
            # slow domain-wall coarsening that a modest sweep budget cannot outrun.
            model.sweep(beta, algorithm="wolff", n_sweeps=n_equil_sweeps)
        self.initial_spins = model.spins.copy()

    def coarse_grain_step(self, spins):
        """Apply one 2x2 majority-rule block-spin transformation.

        Each non-overlapping :math:`2 \\times 2` block of the input is
        replaced by a single spin equal to the sign of the block's sum; an
        exact tie (sum of zero) is broken by an independent fair coin flip.

        Parameters
        ----------
        spins : ndarray of shape (L, L)
            Input configuration. ``L`` must be even.

        Returns
        -------
        ndarray of shape (L // 2, L // 2)
            Coarse-grained spin configuration.
        """
        L = spins.shape[0]
        if L % 2 != 0:
            raise ValueError("lattice size must be even to block by 2x2")
        blocks = spins.reshape(L // 2, 2, L // 2, 2).sum(axis=(1, 3))
        new_spins = np.sign(blocks).astype(np.int64)
        ties = new_spins == 0
        if np.any(ties):
            coin = self._rng.choice(np.array([-1, 1], dtype=np.int64), size=ties.sum())
            new_spins[ties] = coin
        return new_spins

    def iterate(self, n_steps=None):
        """Repeatedly coarse-grain the initial configuration.

        Parameters
        ----------
        n_steps : int, optional
            Number of coarse-graining steps. Defaults to
            ``floor(log2(L)) - 1``, i.e. as many as keep the lattice at least
            :math:`2 \\times 2`.

        Returns
        -------
        list of ndarray
            The sequence of configurations
            ``[initial_spins, step_1, step_2, ...]``, of shrinking size
            :math:`L, L/2, L/4, \\ldots`

        Examples
        --------
        >>> rg = BlockSpinRG(L=16, T=100.0, seed=0)
        >>> grids = rg.iterate(n_steps=3)
        >>> [g.shape[0] for g in grids]
        [16, 8, 4, 2]
        """
        if n_steps is None:
            n_steps = int(np.floor(np.log2(self.L))) - 1
        grids = [self.initial_spins]
        current = self.initial_spins
        for _ in range(n_steps):
            current = self.coarse_grain_step(current)
            grids.append(current)
        return grids

    @staticmethod
    def order_parameter(spins):
        """Magnitude of the mean magnetization per site, :math:`|\\langle s \\rangle|`.

        A convenient scalar summary of a grid in the RG flow: it drifts to
        ``1`` under iteration when the starting temperature is below
        :math:`T_c` (flow to the ordered fixed point), to ``0`` when above
        (flow to the disordered fixed point), and hovers at an intermediate,
        scale-invariant value at :math:`T_c`.

        Parameters
        ----------
        spins : ndarray
            A spin configuration.

        Returns
        -------
        float
        """
        return float(np.abs(spins.mean()))
