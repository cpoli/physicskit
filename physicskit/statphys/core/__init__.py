"""Low-level, Numba-accelerated simulation kernels.

:mod:`physicskit.statphys.core.monte_carlo` holds the Metropolis-Hastings and Wolff
lattice-update kernels used by :mod:`physicskit.statphys.chapters.ising_lattice`.
:mod:`physicskit.statphys.core.md_engine` holds the Lennard-Jones force evaluation and
Velocity Verlet integrator used by :mod:`physicskit.statphys.chapters.molecular_dynamics`.

These functions operate on plain NumPy arrays and carry no state; the
stateful, user-facing model classes live in :mod:`physicskit.statphys.chapters`.
"""
