"""Random walks: diffusion, the Einstein relation, and the central limit theorem.

The random walk is arguably the most foundational model in statistical
physics: it underlies Einstein's 1905 explanation of Brownian motion (which
provided the first direct evidence for the atomic hypothesis), and its
Gaussian long-time limit is the simplest nontrivial instance of the central
limit theorem, whose universality is itself the deep reason so many
different microscopic models flow to the same few macroscopic diffusion
laws.
"""

from __future__ import annotations

import numpy as np

from physicskit.statphys.core.monte_carlo import seed_numba_random
from physicskit.statphys.core.random_walk_engine import simulate_gaussian_walks, simulate_lattice_walks

__all__ = ["RandomWalk"]


class RandomWalk:
    """An ensemble of independent random walkers used to study diffusion.

    Parameters
    ----------
    n_walkers : int, default=500
        Number of independent walkers in the ensemble.
    n_steps : int, default=1000
        Number of steps each walker takes.
    dim : int, default=2
        Spatial dimension.
    kind : {"lattice", "gaussian"}, default="lattice"
        ``"lattice"`` walkers take unit steps along a randomly chosen axis
        (a simple-cubic lattice walk); ``"gaussian"`` walkers take
        continuous, independent Gaussian steps in every dimension at every
        time step (a discrete-time model of Brownian motion).
    step_std : float, default=1.0
        Per-axis standard deviation of a single step. Only used for
        ``kind="gaussian"``.
    seed : int, optional
        Seed for reproducible trajectories.

    Attributes
    ----------
    trajectories : ndarray of shape (n_steps + 1, n_walkers, dim) or None
        Walker positions at every time, populated by :meth:`run`.

    Examples
    --------
    >>> walk = RandomWalk(n_walkers=200, n_steps=200, dim=2, seed=0)
    >>> t, msd = walk.mean_squared_displacement()
    >>> bool(msd[-1] > msd[10])
    True
    """

    def __init__(self, n_walkers=500, n_steps=1000, dim=2, kind="lattice", step_std=1.0, seed=None):
        self.n_walkers = n_walkers
        self.n_steps = n_steps
        self.dim = dim
        self.kind = kind
        self.step_std = step_std
        if seed is not None:
            seed_numba_random(seed)
        self.trajectories = None

    def run(self):
        """Simulate every walker's full trajectory.

        Returns
        -------
        ndarray of shape (n_steps + 1, n_walkers, dim)
            The same array stored in :attr:`trajectories`.
        """
        if self.kind == "lattice":
            self.trajectories = simulate_lattice_walks(self.n_walkers, self.n_steps, self.dim).astype(np.float64)
        elif self.kind == "gaussian":
            self.trajectories = simulate_gaussian_walks(self.n_walkers, self.n_steps, self.dim, self.step_std)
        else:
            raise ValueError("kind must be 'lattice' or 'gaussian'")
        return self.trajectories

    def mean_squared_displacement(self):
        """Ensemble-averaged mean squared displacement :math:`\\text{MSD}(t) = \\langle |r(t)|^2 \\rangle`.

        Returns
        -------
        t : ndarray of shape (n_steps + 1,)
            Time (in steps), starting at 0.
        msd : ndarray of shape (n_steps + 1,)
            Mean squared displacement from the (shared) origin at each time.
        """
        if self.trajectories is None:
            self.run()
        disp2 = np.sum(self.trajectories**2, axis=2)
        msd = disp2.mean(axis=1)
        t = np.arange(self.n_steps + 1)
        return t, msd

    def diffusion_coefficient(self):
        """Estimate the diffusion coefficient D via the Einstein relation.

        Fits :math:`\\text{MSD}(t) = 2 d D t` (a line through the origin,
        least-squares) to the ensemble MSD, where :math:`d` is
        :attr:`dim`.

        Returns
        -------
        float
            Estimated diffusion coefficient.
        """
        t, msd = self.mean_squared_displacement()
        slope = np.sum(t[1:] * msd[1:]) / np.sum(t[1:] ** 2)
        return float(slope / (2.0 * self.dim))

    def final_displacement_histogram(self, axis=0, bins=40):
        """Density histogram of the final-time displacement along one axis.

        A demonstration of the central limit theorem: regardless of the
        (highly non-Gaussian) single-step distribution, the sum of many
        independent steps approaches a Gaussian.

        Parameters
        ----------
        axis : int, default=0
            Which spatial axis to histogram.
        bins : int, default=40
            Number of histogram bins.

        Returns
        -------
        centers : ndarray of shape (bins,)
            Bin center positions.
        density : ndarray of shape (bins,)
            Probability density in each bin.
        """
        if self.trajectories is None:
            self.run()
        final = self.trajectories[-1, :, axis]
        density, edges = np.histogram(final, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, density
