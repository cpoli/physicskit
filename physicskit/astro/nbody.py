r"""Direct-summation N-body gravitational dynamics.

Uses **gravitational units with** :math:`G=1` by default -- every function
accepts ``G`` as a keyword argument, the same convention
:mod:`physicskit.relativity` uses for its own geometrized units.

- :func:`gravitational_acceleration` -- pairwise Plummer-softened gravity.
- :func:`leapfrog_step` -- one symplectic kick-drift-kick integration step.
- :class:`NBodySystem` -- a stateful N-body simulation with energy and
  angular-momentum diagnostics.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from physicskit.integrators.fixed_step import leapfrog_step as _leapfrog_step

__all__ = ["gravitational_acceleration", "leapfrog_step", "NBodySystem", "figure_eight_initial_conditions"]


@njit(cache=True)
def _pairwise_acceleration(positions, masses, G, softening):
    n = positions.shape[0]
    acc = np.zeros((n, 3))
    eps2 = softening**2
    for i in range(n):
        ax, ay, az = 0.0, 0.0, 0.0
        for j in range(n):
            if i == j:
                continue
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dz = positions[j, 2] - positions[i, 2]
            dist2 = dx * dx + dy * dy + dz * dz + eps2
            inv_dist3 = dist2 ** (-1.5)
            f = G * masses[j] * inv_dist3
            ax += f * dx
            ay += f * dy
            az += f * dz
        acc[i, 0] = ax
        acc[i, 1] = ay
        acc[i, 2] = az
    return acc


def gravitational_acceleration(positions, masses, G=1.0, softening=0.0):
    r"""Pairwise Plummer-softened gravitational acceleration.

    .. math::

        \vec a_i = G\sum_{j\neq i} m_j\,
        \frac{\vec r_j-\vec r_i}{\left(|\vec r_j-\vec r_i|^2+\epsilon^2\right)^{3/2}}

    Parameters
    ----------
    positions : ndarray of shape (N, 3)
    masses : ndarray of shape (N,)
    G : float, default=1.0
    softening : float, default=0.0
        Plummer softening length :math:`\epsilon`.

    Returns
    -------
    ndarray of shape (N, 3)

    Examples
    --------
    >>> import numpy as np
    >>> pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    >>> m = np.array([1.0, 1.0])
    >>> acc = gravitational_acceleration(pos, m)
    >>> round(float(acc[0, 0]), 6)
    1.0
    """
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    masses = np.ascontiguousarray(masses, dtype=np.float64)
    return _pairwise_acceleration(positions, masses, float(G), float(softening))


@njit(cache=True)
def _nbody_accel(positions, t, params):
    """Adapt :func:`_pairwise_acceleration` to the shared ``force(pos, t,
    params) -> acc`` convention, packing ``masses``, ``G`` and ``softening``
    into a single ``params`` vector (``masses`` followed by ``[G,
    softening]``) so it can be passed as a first-class function into
    :func:`physicskit.integrators.leapfrog_step`."""
    n = positions.shape[0]
    masses = params[:n]
    G = params[n]
    softening = params[n + 1]
    return _pairwise_acceleration(positions, masses, G, softening)


def leapfrog_step(positions, velocities, masses, dt, G=1.0, softening=0.0):
    """One kick-drift-kick leapfrog (symplectic) integration step.

    Delegates to the shared :func:`physicskit.integrators.leapfrog_step`.

    Parameters
    ----------
    positions : ndarray of shape (N, 3)
    velocities : ndarray of shape (N, 3)
    masses : ndarray of shape (N,)
    dt : float
        Timestep.
    G : float, default=1.0
    softening : float, default=0.0

    Returns
    -------
    new_positions, new_velocities : ndarray of shape (N, 3)
    """
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    velocities = np.ascontiguousarray(velocities, dtype=np.float64)
    masses = np.ascontiguousarray(masses, dtype=np.float64)
    params = np.concatenate((masses, np.array([G, softening], dtype=np.float64)))
    return _leapfrog_step(_nbody_accel, positions, velocities, 0.0, dt, params)


class NBodySystem:
    """A stateful direct-summation N-body gravitational simulation.

    Parameters
    ----------
    positions : ndarray of shape (N, 3)
    velocities : ndarray of shape (N, 3)
    masses : ndarray of shape (N,)
    G : float, default=1.0
    softening : float, default=0.0
    """

    def __init__(self, positions, velocities, masses, G=1.0, softening=0.0):
        self.positions = np.array(positions, dtype=np.float64, copy=True)
        self.velocities = np.array(velocities, dtype=np.float64, copy=True)
        self.masses = np.array(masses, dtype=np.float64, copy=True)
        self.G = G
        self.softening = softening

    def step(self, dt):
        """Advance the system's stored state by one leapfrog step, in place."""
        self.positions, self.velocities = leapfrog_step(self.positions, self.velocities, self.masses, dt, self.G, self.softening)

    def simulate(self, dt, n_steps):
        """Advance ``n_steps`` steps, returning the position history.

        Returns
        -------
        ndarray of shape (n_steps + 1, N, 3)
        """
        n = self.positions.shape[0]
        history = np.zeros((n_steps + 1, n, 3))
        history[0] = self.positions
        for k in range(n_steps):
            self.step(dt)
            history[k + 1] = self.positions
        return history

    def total_energy(self):
        r"""Kinetic plus (softened) gravitational potential energy, at the current state."""
        kinetic = 0.5 * np.sum(self.masses[:, None] * self.velocities**2)
        n = self.positions.shape[0]
        potential = 0.0
        eps2 = self.softening**2
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(self.positions[i] - self.positions[j])
                potential -= self.G * self.masses[i] * self.masses[j] / np.sqrt(d**2 + eps2)
        return float(kinetic + potential)

    def total_angular_momentum(self):
        r"""Total angular momentum, :math:`\vec L=\sum_im_i\,\vec r_i\times\vec v_i`."""
        L = np.sum(self.masses[:, None] * np.cross(self.positions, self.velocities), axis=0)
        return L


def figure_eight_initial_conditions():
    r"""Initial conditions for the equal-mass planar figure-eight three-body choreography.

    Three equal masses, in :math:`G=1` units, chase each other forever
    around a single figure-eight-shaped curve -- a periodic solution
    discovered numerically by Moore (1993) and proven to exist by
    Chenciner & Montgomery (2000, *Annals of Mathematics* 152). Unlike
    typical three-body configurations it is planar, collision-free, and
    exactly periodic, making it a clean, visually striking demo for
    :class:`NBodySystem`.

    Returns
    -------
    positions : ndarray of shape (3, 3)
    velocities : ndarray of shape (3, 3)
    masses : ndarray of shape (3,)

    Examples
    --------
    >>> positions, velocities, masses = figure_eight_initial_conditions()
    >>> system = NBodySystem(positions, velocities, masses)
    >>> history = system.simulate(dt=0.001, n_steps=100)
    >>> history.shape
    (101, 3, 3)
    """
    r1 = np.array([0.97000436, -0.24308753, 0.0])
    v3 = np.array([0.93240737, 0.86473146, 0.0])
    positions = np.array([r1, -r1, [0.0, 0.0, 0.0]])
    velocities = np.array([-0.5 * v3, -0.5 * v3, v3])
    masses = np.array([1.0, 1.0, 1.0])
    return positions, velocities, masses
