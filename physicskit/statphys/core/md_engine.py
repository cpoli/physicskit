"""Numba-accelerated molecular dynamics engine: Lennard-Jones forces and Velocity Verlet.

Implements a minimal-image, periodic-boundary N-body simulator for a 2D
Lennard-Jones gas. The pairwise force evaluation is the O(N^2) inner loop and
is the part that benefits from JIT compilation; everything else is thin
Python bookkeeping around it.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = ["initialize_maxwell_boltzmann_velocities", "lj_forces", "velocity_verlet_step"]


@njit(cache=True)
def lj_forces(positions, box_size, epsilon=1.0, sigma=1.0, cutoff=2.5):
    """Pairwise Lennard-Jones forces and potential energy under minimum-image PBC.

    Parameters
    ----------
    positions : ndarray of shape (N, 2)
        Particle positions, expected in ``[0, box_size)``.
    box_size : float
        Side length of the square periodic simulation box.
    epsilon : float, default=1.0
        Lennard-Jones well depth.
    sigma : float, default=1.0
        Lennard-Jones length scale (zero-crossing distance).
    cutoff : float, default=2.5
        Interaction cutoff radius, in the same length units as ``sigma``.

    Returns
    -------
    forces : ndarray of shape (N, 2)
        Net force on each particle.
    potential_energy : float
        Total (uncut-shifted) Lennard-Jones potential energy of the
        configuration.

    Notes
    -----
    The Lennard-Jones potential is
    :math:`V(r) = 4\\epsilon\\left[(\\sigma/r)^{12} - (\\sigma/r)^{6}\\right]`,
    giving the force magnitude
    :math:`f(r) = 24\\epsilon\\left[2(\\sigma/r)^{12} - (\\sigma/r)^{6}\\right] / r`.
    """
    n = positions.shape[0]
    forces = np.zeros((n, 2), dtype=np.float64)
    potential_energy = 0.0
    cutoff2 = cutoff * cutoff
    sigma2 = sigma * sigma
    for a in range(n):
        xa = positions[a, 0]
        ya = positions[a, 1]
        for b in range(a + 1, n):
            dx = xa - positions[b, 0]
            dy = ya - positions[b, 1]
            dx -= box_size * np.round(dx / box_size)
            dy -= box_size * np.round(dy / box_size)
            r2 = dx * dx + dy * dy
            if 1e-12 < r2 < cutoff2:
                inv_r2 = sigma2 / r2
                inv_r6 = inv_r2 * inv_r2 * inv_r2
                inv_r12 = inv_r6 * inv_r6
                f_over_r2 = 24.0 * epsilon * (2.0 * inv_r12 - inv_r6) / r2
                fx = f_over_r2 * dx
                fy = f_over_r2 * dy
                forces[a, 0] += fx
                forces[a, 1] += fy
                forces[b, 0] -= fx
                forces[b, 1] -= fy
                potential_energy += 4.0 * epsilon * (inv_r12 - inv_r6)
    return forces, potential_energy


def velocity_verlet_step(positions, velocities, forces, dt, box_size, mass=1.0, **lj_kwargs):
    """Advance positions and velocities by one Velocity Verlet step.

    Parameters
    ----------
    positions : ndarray of shape (N, 2)
        Current positions. Not modified; a new array is returned.
    velocities : ndarray of shape (N, 2)
        Current velocities.
    forces : ndarray of shape (N, 2)
        Forces evaluated at the current positions (from the previous step's
        call, so the force is only ever computed once per step).
    dt : float
        Integration time step.
    box_size : float
        Side length of the periodic simulation box.
    mass : float, default=1.0
        Common particle mass.
    **lj_kwargs
        Extra keyword arguments (``epsilon``, ``sigma``, ``cutoff``) forwarded
        to :func:`lj_forces`.

    Returns
    -------
    new_positions : ndarray of shape (N, 2)
        Positions after one step, wrapped into ``[0, box_size)``.
    new_velocities : ndarray of shape (N, 2)
        Velocities after one step.
    new_forces : ndarray of shape (N, 2)
        Forces evaluated at ``new_positions``, ready for the next call.
    potential_energy : float
        Potential energy at ``new_positions``.
    """
    new_positions = positions + velocities * dt + 0.5 * (forces / mass) * dt * dt
    new_positions = np.mod(new_positions, box_size)
    new_forces, potential_energy = lj_forces(new_positions, box_size, **lj_kwargs)
    new_velocities = velocities + 0.5 * (forces + new_forces) / mass * dt
    return new_positions, new_velocities, new_forces, potential_energy


def initialize_maxwell_boltzmann_velocities(n_particles, temperature, mass=1.0, kB=1.0, rng=None):
    """Draw initial 2D velocities from the Maxwell-Boltzmann distribution and remove drift.

    Each velocity component is drawn i.i.d. from
    :math:`\\mathcal{N}(0, k_B T / m)`, then the center-of-mass velocity is
    subtracted so the total momentum is zero.

    Parameters
    ----------
    n_particles : int
        Number of particles.
    temperature : float
        Target temperature.
    mass : float, default=1.0
        Particle mass.
    kB : float, default=1.0
        Boltzmann constant, in whatever unit system is being used.
    rng : numpy.random.Generator, optional
        Random number generator. A fresh default generator is created if not
        given.

    Returns
    -------
    ndarray of shape (n_particles, 2)
        Initial velocities with zero net momentum.

    Examples
    --------
    >>> v = initialize_maxwell_boltzmann_velocities(100, temperature=1.5, rng=np.random.default_rng(0))
    >>> v.shape
    (100, 2)
    >>> np.allclose(v.mean(axis=0), 0.0)
    True
    """
    if rng is None:
        rng = np.random.default_rng()
    std = np.sqrt(kB * temperature / mass)
    velocities = rng.normal(loc=0.0, scale=std, size=(n_particles, 2))
    velocities -= velocities.mean(axis=0, keepdims=True)
    return velocities
