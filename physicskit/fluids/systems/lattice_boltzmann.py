"""Lattice Boltzmann method: the D2Q9 model with the BGK collision operator.

Instead of discretizing the Navier-Stokes equations, the lattice Boltzmann
method evolves particle distributions :math:`f_i(\\mathbf x, t)` moving
with nine discrete velocities :math:`\\mathbf e_i` on a square lattice
(D2Q9), relaxing each step toward a local equilibrium with a single
relaxation time :math:`\\tau` (Bhatnagar, Gross and Krook 1954):

.. math::

    f_i(\\mathbf x + \\mathbf e_i, t + 1) = f_i(\\mathbf x, t)
    - \\frac{1}{\\tau}\\big[f_i - f_i^{\\text{eq}}\\big] + F_i,
    \\qquad
    f_i^{\\text{eq}} = w_i\\rho\\left[1 + 3\\,\\mathbf e_i\\cdot\\mathbf u
    + \\tfrac92(\\mathbf e_i\\cdot\\mathbf u)^2 - \\tfrac32 u^2\\right]

(Qian, d'Humières and Lallemand 1992). A Chapman-Enskog expansion shows
that the density :math:`\\rho = \\sum_i f_i` and momentum
:math:`\\rho\\mathbf u = \\sum_i f_i\\mathbf e_i` then obey the weakly
compressible Navier-Stokes equations with sound speed
:math:`c_s^2 = 1/3` and kinematic viscosity

.. math::

    \\nu = c_s^2\\left(\\tau - \\tfrac12\\right)

(Chen and Doolen, Annu. Rev. Fluid Mech. 30, 329 (1998)). Everything is
in lattice units: :math:`\\Delta x = \\Delta t = 1`. A body force enters
through Guo's forcing term (Guo, Zheng and Shi, Phys. Rev. E 65, 046308
(2002)), and solid walls through halfway bounce-back, which places the
no-slip wall midway between a fluid node and a solid node. The model is
accurate for Mach numbers :math:`u/c_s \\ll 1`.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from physicskit.fluids.exceptions import InvalidParameterError

__all__ = [
    "D2Q9_VELOCITIES",
    "D2Q9_WEIGHTS",
    "LBM_SOUND_SPEED_SQUARED",
    "lbm_viscosity",
    "lbm_relaxation_time",
    "lbm_equilibrium",
    "LatticeBoltzmannD2Q9",
]

#: The nine D2Q9 lattice velocities: rest, four axis neighbors, four diagonals.
D2Q9_VELOCITIES = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.int64)
#: D2Q9 quadrature weights (4/9, 1/9 x 4, 1/36 x 4).
D2Q9_WEIGHTS = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
#: Lattice sound speed squared, :math:`c_s^2 = 1/3`.
LBM_SOUND_SPEED_SQUARED = 1.0 / 3.0
_OPPOSITE = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)


def lbm_viscosity(tau: float) -> float:
    """Kinematic viscosity of the BGK lattice Boltzmann model, :math:`\\nu = c_s^2(\\tau - 1/2)`.

    Parameters
    ----------
    tau : float
        Relaxation time, :math:`\\tau > 1/2`.

    Returns
    -------
    float
        Viscosity in lattice units.

    Examples
    --------
    >>> lbm_viscosity(1.0)
    0.16666666666666666
    """
    if tau <= 0.5:
        raise InvalidParameterError(f"tau must exceed 1/2 for positive viscosity, got {tau}")
    return LBM_SOUND_SPEED_SQUARED * (tau - 0.5)


def lbm_relaxation_time(viscosity: float) -> float:
    """Relaxation time giving a target viscosity, :math:`\\tau = \\nu/c_s^2 + 1/2`.

    Parameters
    ----------
    viscosity : float
        Kinematic viscosity in lattice units, positive.

    Returns
    -------
    float

    Examples
    --------
    >>> lbm_relaxation_time(1 / 6)
    1.0
    """
    if viscosity <= 0:
        raise InvalidParameterError(f"viscosity must be positive, got {viscosity}")
    return viscosity / LBM_SOUND_SPEED_SQUARED + 0.5


def lbm_equilibrium(rho: ArrayLike, ux: ArrayLike, uy: ArrayLike) -> NDArray[np.float64]:
    """D2Q9 equilibrium distributions :math:`f_i^{\\text{eq}}(\\rho, \\mathbf u)`.

    .. math::

        f_i^{\\text{eq}} = w_i\\rho\\left[1 + 3\\,\\mathbf e_i\\cdot\\mathbf u
        + \\tfrac92(\\mathbf e_i\\cdot\\mathbf u)^2 - \\tfrac32 u^2\\right]

    Its zeroth and first moments are exactly :math:`\\rho` and
    :math:`\\rho\\mathbf u`.

    Parameters
    ----------
    rho, ux, uy : array_like
        Density and velocity fields (broadcastable to a common shape).

    Returns
    -------
    ndarray, shape (9, *shape)

    Examples
    --------
    >>> feq = lbm_equilibrium(1.2, 0.05, -0.02)
    >>> round(float(feq.sum()), 12), round(float(feq @ D2Q9_VELOCITIES[:, 0]), 12)
    (1.2, 0.06)
    """
    rho, ux, uy = np.broadcast_arrays(*(np.asarray(v, dtype=np.float64) for v in (rho, ux, uy)))
    eu = D2Q9_VELOCITIES[:, 0].reshape((9,) + (1,) * ux.ndim) * ux + D2Q9_VELOCITIES[:, 1].reshape((9,) + (1,) * ux.ndim) * uy
    u2 = ux**2 + uy**2
    return D2Q9_WEIGHTS.reshape((9,) + (1,) * ux.ndim) * rho * (1.0 + 3.0 * eu + 4.5 * eu**2 - 1.5 * u2)


@njit(cache=True)
def _collide_stream(f, f_new, solid, ex, ey, w, opp, tau, gx, gy, n_steps):
    nx, ny = solid.shape
    omega = 1.0 / tau
    pref = 1.0 - 0.5 * omega
    post = np.empty(9)
    for _ in range(n_steps):
        for x in range(nx):
            for y in range(ny):
                if solid[x, y]:
                    continue
                rho = 0.0
                mx = 0.0
                my = 0.0
                for i in range(9):
                    fi = f[i, x, y]
                    rho += fi
                    mx += fi * ex[i]
                    my += fi * ey[i]
                # Guo forcing: the physical velocity includes half the force impulse
                Fx = rho * gx
                Fy = rho * gy
                ux = (mx + 0.5 * Fx) / rho
                uy = (my + 0.5 * Fy) / rho
                u2 = ux * ux + uy * uy
                for i in range(9):
                    eu = ex[i] * ux + ey[i] * uy
                    feq = w[i] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2)
                    Fi = pref * w[i] * (3.0 * ((ex[i] - ux) * Fx + (ey[i] - uy) * Fy) + 9.0 * eu * (ex[i] * Fx + ey[i] * Fy))
                    post[i] = f[i, x, y] - omega * (f[i, x, y] - feq) + Fi
                for i in range(9):
                    xn = (x + ex[i]) % nx
                    yn = (y + ey[i]) % ny
                    if solid[xn, yn]:
                        # halfway bounce-back: reflect into the opposite direction at the source node
                        f_new[opp[i], x, y] = post[i]
                    else:
                        f_new[i, xn, yn] = post[i]
        for i in range(9):
            for x in range(nx):
                for y in range(ny):
                    f[i, x, y] = f_new[i, x, y]


class LatticeBoltzmannD2Q9:
    """A D2Q9 BGK lattice Boltzmann fluid on a periodic box with optional solid obstacles.

    Both axes wrap periodically; channels, cylinders, and cavities are
    built by marking ``solid`` nodes, whose faces toward the fluid become
    no-slip walls by halfway bounce-back (the wall sits half a lattice
    spacing outside the last fluid node). A uniform body force
    (acceleration) ``body_force`` drives the flow, e.g. as the pressure
    gradient of a periodic channel.

    Parameters
    ----------
    nx, ny : int
        Lattice size.
    tau : float
        BGK relaxation time, :math:`\\tau > 1/2`; the viscosity is
        :math:`\\nu = (\\tau - 1/2)/3` (:func:`lbm_viscosity`).
    body_force : tuple of float, default=(0.0, 0.0)
        Acceleration :math:`\\mathbf g` acting on the fluid (force per unit
        mass), in lattice units.
    solid : array_like of bool, shape (nx, ny), optional
        Solid (wall) nodes.
    rho0 : float, default=1.0
        Initial uniform density.
    velocity : tuple of array_like, optional
        Initial velocity field ``(ux, uy)``; zero by default. The
        populations start at the matching equilibrium.

    Attributes
    ----------
    f : ndarray, shape (9, nx, ny)
        Particle distributions.
    time : int
        Number of steps taken.

    Examples
    --------
    Flow between two plates driven by a body force (plane Poiseuille flow):

    >>> solid = np.zeros((1, 22), dtype=bool)
    >>> solid[:, [0, -1]] = True
    >>> lb = LatticeBoltzmannD2Q9(1, 22, tau=0.8, body_force=(1e-6, 0.0), solid=solid)
    >>> lb.step(6000)
    >>> H, nu = 20.0, lb.viscosity
    >>> round(float(lb.velocity[0][0, 11] / (1e-6 * H**2 / (8 * nu))), 2)
    1.0
    """

    def __init__(self, nx, ny, tau, body_force=(0.0, 0.0), solid=None, rho0=1.0, velocity=None):
        self.nx, self.ny = int(nx), int(ny)
        self.tau = float(tau)
        self.viscosity = lbm_viscosity(self.tau)
        self.body_force = (float(body_force[0]), float(body_force[1]))
        self.solid = np.zeros((self.nx, self.ny), dtype=np.bool_) if solid is None else np.array(solid, dtype=np.bool_)
        if self.solid.shape != (self.nx, self.ny):
            raise InvalidParameterError(f"solid must have shape {(self.nx, self.ny)}, got {self.solid.shape}")
        ux, uy = (0.0, 0.0) if velocity is None else velocity
        ux = np.broadcast_to(np.asarray(ux, dtype=np.float64), (self.nx, self.ny)) * ~self.solid
        uy = np.broadcast_to(np.asarray(uy, dtype=np.float64), (self.nx, self.ny)) * ~self.solid
        self.f = np.ascontiguousarray(lbm_equilibrium(rho0, ux, uy))
        self._f_new = self.f.copy()
        self.time = 0

    def step(self, n_steps: int = 1) -> None:
        """Advance ``n_steps`` collide-and-stream updates.

        Parameters
        ----------
        n_steps : int, default=1
        """
        _collide_stream(
            self.f,
            self._f_new,
            self.solid,
            D2Q9_VELOCITIES[:, 0].copy(),
            D2Q9_VELOCITIES[:, 1].copy(),
            D2Q9_WEIGHTS,
            _OPPOSITE,
            self.tau,
            self.body_force[0],
            self.body_force[1],
            int(n_steps),
        )
        self.time += int(n_steps)

    @property
    def density(self) -> NDArray[np.float64]:
        """Density :math:`\\rho = \\sum_i f_i` (zero-padded meaning on solid nodes)."""
        return self.f.sum(axis=0)

    @property
    def velocity(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Velocity :math:`\\mathbf u = (\\sum_i f_i\\mathbf e_i + \\tfrac12\\rho\\mathbf g)/\\rho`, zero on solid nodes.

        Returns
        -------
        ux, uy : ndarray, shape (nx, ny)
        """
        rho = self.density
        safe = np.where(self.solid, 1.0, rho)
        ux = (np.tensordot(D2Q9_VELOCITIES[:, 0], self.f, axes=1) + 0.5 * rho * self.body_force[0]) / safe
        uy = (np.tensordot(D2Q9_VELOCITIES[:, 1], self.f, axes=1) + 0.5 * rho * self.body_force[1]) / safe
        return np.where(self.solid, 0.0, ux), np.where(self.solid, 0.0, uy)

    def vorticity(self) -> NDArray[np.float64]:
        """Vorticity :math:`\\partial_x u_y - \\partial_y u_x` by periodic central differences.

        Returns
        -------
        ndarray, shape (nx, ny)
        """
        ux, uy = self.velocity
        return 0.5 * (np.roll(uy, -1, 0) - np.roll(uy, 1, 0)) - 0.5 * (np.roll(ux, -1, 1) - np.roll(ux, 1, 1))
