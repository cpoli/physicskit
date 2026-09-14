"""Point-vortex N-body dynamics: the Biot-Savart law and the von Karman vortex street.

Concentrating all of a 2D flow's vorticity into a finite set of
infinitesimal points turns the vorticity-transport PDE into a system of
ordinary differential equations: each point vortex simply advects with the
velocity field induced by every *other* point vortex (a vortex induces no
velocity on itself), via the 2D Biot-Savart law. This is an exact solution
of the incompressible Euler equations for a singular vorticity distribution,
not merely a discretization of the continuous vorticity-transport equation
solved by :mod:`physicskit.fluids.systems.navier_stokes` -- and it is the
setting in which Theodore von Karman analyzed the staggered, alternating
vortex street shed behind a bluff body.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from physicskit.fluids.exceptions import InvalidParameterError

__all__ = [
    "point_vortex_velocities",
    "PointVortexSystem",
    "von_karman_vortex_street",
    "VON_KARMAN_SPACING_RATIO",
]

#: Theodore von Karman's spacing ratio ``h/l`` (row separation over
#: along-row spacing) at which a doubly-infinite staggered vortex row is
#: linearly stable to vortex-row perturbations (von Karman & Rubach, 1912).
VON_KARMAN_SPACING_RATIO: float = 0.2805


@njit(cache=True)
def _point_vortex_velocities(positions: NDArray[np.float64], circulations: NDArray[np.float64], core: float) -> NDArray[np.float64]:
    """Biot-Savart velocity induced on each point vortex by every other one.

    Parameters
    ----------
    positions : ndarray of float, shape (n, 2)
        Vortex positions ``(x, y)``.
    circulations : ndarray of float, shape (n,)
        Vortex circulations :math:`\\Gamma_i`.
    core : float
        Regularization (Rankine core) radius; keeps the induced velocity
        finite if two vortices coincide.

    Returns
    -------
    ndarray of float, shape (n, 2)
        Velocity at each vortex's own position.
    """
    n = positions.shape[0]
    velocities = np.zeros((n, 2))
    core2 = core * core
    for i in range(n):
        vx, vy = 0.0, 0.0
        xi, yi = positions[i, 0], positions[i, 1]
        for j in range(n):
            if i == j:
                continue
            dx = xi - positions[j, 0]
            dy = yi - positions[j, 1]
            r2 = dx * dx + dy * dy + core2
            factor = circulations[j] / (2.0 * np.pi * r2)
            vx -= factor * dy
            vy += factor * dx
        velocities[i, 0] = vx
        velocities[i, 1] = vy
    return velocities


@njit(cache=True)
def _point_vortex_rk4_integrate(
    positions0: NDArray[np.float64], circulations: NDArray[np.float64], core: float, dt: float, n_steps: int
) -> NDArray[np.float64]:
    """Integrate the point-vortex system with RK4.

    Parameters
    ----------
    positions0 : ndarray of float, shape (n, 2)
        Initial vortex positions.
    circulations : ndarray of float, shape (n,)
        Vortex circulations.
    core : float
        Regularization core radius.
    dt : float
        Time step.
    n_steps : int
        Number of steps to advance.

    Returns
    -------
    ndarray of float, shape (n_steps + 1, n, 2)
        Vortex positions at each step, including the initial positions.
    """
    n = positions0.shape[0]
    trajectory = np.empty((n_steps + 1, n, 2))
    trajectory[0] = positions0
    positions = positions0.copy()
    for step in range(n_steps):
        k1 = _point_vortex_velocities(positions, circulations, core)
        k2 = _point_vortex_velocities(positions + 0.5 * dt * k1, circulations, core)
        k3 = _point_vortex_velocities(positions + 0.5 * dt * k2, circulations, core)
        k4 = _point_vortex_velocities(positions + dt * k3, circulations, core)
        positions = positions + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        trajectory[step + 1] = positions
    return trajectory


def point_vortex_velocities(positions: ArrayLike, circulations: ArrayLike, core: float = 1e-6) -> NDArray[np.float64]:
    """Velocity induced on each point vortex by every other vortex (Biot-Savart law).

    The 2D Biot-Savart law: an isolated point vortex of circulation
    :math:`\\Gamma_j` at :math:`\\mathbf{r}_j` induces a purely azimuthal
    velocity :math:`\\Gamma_j/(2\\pi r)` at distance `r`, and a system of `n`
    vortices simply superposes these fields (Laplace's equation is linear,
    just as in :mod:`physicskit.fluids.systems.potential_flow` -- indeed a
    point vortex here *is* :func:`~physicskit.fluids.systems.potential_flow.point_vortex_potential`,
    evaluated at every other vortex's location rather than on a field grid).

    Parameters
    ----------
    positions : array_like of float, shape (n, 2)
        Vortex positions ``(x, y)``.
    circulations : array_like of float, shape (n,)
        Vortex circulations :math:`\\Gamma_i` (positive counterclockwise).
    core : float, default 1e-6
        Regularization (Rankine core) radius, keeping the induced velocity
        finite if two vortex cores coincide; negligible for any pair
        separated by much more than `core`.

    Returns
    -------
    ndarray of float, shape (n, 2)
        Velocity at each vortex's own position, induced by all the others.

    Examples
    --------
    A pair of counter-rotating vortices separated by distance `d` induces
    equal and opposite velocities of magnitude :math:`\\Gamma/(2\\pi d)` on
    each other, translating the pair sideways rather than rotating it:

    >>> import numpy as np
    >>> Gamma, d = 1.0, 2.0
    >>> positions = np.array([[0.0, d / 2], [0.0, -d / 2]])
    >>> circulations = np.array([Gamma, -Gamma])
    >>> vel = point_vortex_velocities(positions, circulations)
    >>> bool(np.allclose(vel[0], vel[1]))  # both translate together
    True
    >>> round(float(vel[0, 0]), 6) == round(Gamma / (2 * np.pi * d), 6)
    True
    """
    positions = np.asarray(positions, dtype=np.float64)
    circulations = np.asarray(circulations, dtype=np.float64)
    if positions.shape[0] != circulations.shape[0]:
        raise InvalidParameterError("positions and circulations must have the same number of vortices")
    return _point_vortex_velocities(positions, circulations, core)


class PointVortexSystem:
    """A system of interacting 2D point vortices, advected by their mutual Biot-Savart field.

    Parameters
    ----------
    positions : array_like of float, shape (n, 2)
        Initial vortex positions ``(x, y)``.
    circulations : array_like of float, shape (n,)
        Vortex circulations :math:`\\Gamma_i`; fixed for the system's lifetime
        (point vortices carry their circulation with them unchanged).
    core : float, default 1e-6
        Regularization core radius (see :func:`point_vortex_velocities`).

    Attributes
    ----------
    positions : ndarray of float, shape (n, 2)
        Current vortex positions.
    circulations : ndarray of float, shape (n,)
        Vortex circulations.
    core : float
        Regularization core radius.
    n_vortices : int
        Number of vortices, `n`.

    Raises
    ------
    InvalidParameterError
        If `positions` and `circulations` have inconsistent lengths.

    Examples
    --------
    >>> import numpy as np
    >>> system = PointVortexSystem(positions=[[1.0, 0.0], [-1.0, 0.0]], circulations=[1.0, 1.0])
    >>> t, trajectory = system.trajectory(dt=0.01, n_steps=10)
    >>> trajectory.shape
    (11, 2, 2)
    """

    def __init__(self, positions: ArrayLike, circulations: ArrayLike, core: float = 1e-6):
        positions = np.asarray(positions, dtype=np.float64)
        circulations = np.asarray(circulations, dtype=np.float64)
        if positions.shape[0] != circulations.shape[0]:
            raise InvalidParameterError("positions and circulations must have the same number of vortices")
        self.positions = positions
        self.circulations = circulations
        self.core = float(core)
        self.n_vortices = positions.shape[0]

    def velocities(self) -> NDArray[np.float64]:
        """Velocity currently induced on each vortex.

        Returns
        -------
        ndarray of float, shape (n, 2)
        """
        return _point_vortex_velocities(self.positions, self.circulations, self.core)

    def trajectory(self, dt: float = 0.01, n_steps: int = 1000) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Integrate the vortex trajectories forward with RK4.

        Parameters
        ----------
        dt : float, default 0.01
            Time step.
        n_steps : int, default 1000
            Number of steps to advance.

        Returns
        -------
        times : ndarray of float, shape (n_steps + 1,)
        positions : ndarray of float, shape (n_steps + 1, n, 2)
            Vortex positions at each step, including the current positions.
        """
        times = np.arange(n_steps + 1) * dt
        trajectory = _point_vortex_rk4_integrate(self.positions, self.circulations, self.core, dt, n_steps)
        return times, trajectory

    def __repr__(self) -> str:
        return f"PointVortexSystem(n_vortices={self.n_vortices}, circulations={self.circulations!r})"


def von_karman_vortex_street(
    n_pairs: int, spacing_l: float, spacing_ratio: float = VON_KARMAN_SPACING_RATIO
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build a finite, doubly staggered von Karman vortex street.

    Two parallel rows of point vortices, offset from each other by half the
    along-row spacing `spacing_l` and by a row separation
    ``h = spacing_ratio * spacing_l``, with alternating-sign circulation
    within each row so that adjacent vortices across the two rows always
    have *opposite* sign -- the staggered, alternating pattern observed in
    the wake behind a bluff body. Von Karman showed that a doubly *infinite*
    row of this form is linearly stable to vortex-row perturbations at
    exactly one spacing ratio, :data:`VON_KARMAN_SPACING_RATIO`; any other
    ratio (or any symmetric, unstaggered arrangement) grows unstable, which
    is why the observed spacing behind real bluff bodies clusters so
    consistently near this value.

    Parameters
    ----------
    n_pairs : int
        Number of vortex pairs (one from each row) to construct; must be at
        least 2 to form a discernible street.
    spacing_l : float
        Spacing between same-row, same-sign vortices along the street.
    spacing_ratio : float, default :data:`VON_KARMAN_SPACING_RATIO`
        Row separation over along-row spacing, ``h / spacing_l``.

    Returns
    -------
    positions : ndarray of float, shape (2*n_pairs, 2)
        Vortex positions.
    circulations : ndarray of float, shape (2*n_pairs,)
        Vortex circulations, alternating in sign.

    Raises
    ------
    InvalidParameterError
        If `n_pairs` is smaller than 2 or `spacing_l` is not positive.

    Examples
    --------
    >>> positions, circulations = von_karman_vortex_street(n_pairs=6, spacing_l=1.0)
    >>> positions.shape
    (12, 2)
    >>> bool(np.isclose(circulations.sum(), 0.0))  # equal and opposite in each row
    True
    """
    if n_pairs < 2:
        raise InvalidParameterError(f"n_pairs must be at least 2, got {n_pairs}")
    if spacing_l <= 0:
        raise InvalidParameterError(f"spacing_l must be positive, got {spacing_l}")
    h = spacing_ratio * spacing_l
    positions = np.empty((2 * n_pairs, 2))
    circulations = np.empty(2 * n_pairs)
    for i in range(n_pairs):
        sign = 1.0 if i % 2 == 0 else -1.0
        positions[2 * i] = [i * spacing_l, h / 2.0]
        circulations[2 * i] = sign
        positions[2 * i + 1] = [(i + 0.5) * spacing_l, -h / 2.0]
        circulations[2 * i + 1] = -sign
    return positions, circulations
