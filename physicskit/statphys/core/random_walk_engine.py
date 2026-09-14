"""Numba-accelerated random walk trajectory generation."""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = ["simulate_gaussian_walks", "simulate_lattice_walks"]


@njit(cache=True)
def simulate_lattice_walks(n_walkers, n_steps, dim):
    """Simulate independent nearest-neighbor random walks on a hypercubic lattice.

    Each step picks one of the ``dim`` axes uniformly at random and moves
    :math:`\\pm 1` along it.

    Parameters
    ----------
    n_walkers : int
        Number of independent walkers.
    n_steps : int
        Number of steps per walker.
    dim : int
        Spatial dimension.

    Returns
    -------
    ndarray of shape (n_steps + 1, n_walkers, dim), dtype int64
        Walker positions at every time, starting from the origin at ``t=0``.
    """
    positions = np.zeros((n_steps + 1, n_walkers, dim), dtype=np.int64)
    for w in range(n_walkers):
        pos = np.zeros(dim, dtype=np.int64)
        for t in range(1, n_steps + 1):
            axis = np.random.randint(0, dim)
            step = 1 if np.random.random() < 0.5 else -1
            pos[axis] += step
            positions[t, w, :] = pos
    return positions


@njit(cache=True)
def simulate_gaussian_walks(n_walkers, n_steps, dim, step_std):
    """Simulate independent continuous Brownian walks with Gaussian steps.

    Parameters
    ----------
    n_walkers : int
        Number of independent walkers.
    n_steps : int
        Number of steps per walker.
    dim : int
        Spatial dimension.
    step_std : float
        Standard deviation of each per-axis Gaussian step.

    Returns
    -------
    ndarray of shape (n_steps + 1, n_walkers, dim), dtype float64
        Walker positions at every time, starting from the origin at ``t=0``.
    """
    positions = np.zeros((n_steps + 1, n_walkers, dim), dtype=np.float64)
    for w in range(n_walkers):
        for t in range(1, n_steps + 1):
            for d in range(dim):
                positions[t, w, d] = positions[t - 1, w, d] + np.random.normal(0.0, step_std)
    return positions
