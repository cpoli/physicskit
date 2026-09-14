"""Incompressible 2D Navier-Stokes via the vorticity-streamfunction formulation.

Solves :math:`\\partial_t \\omega + (\\mathbf{u}\\cdot\\nabla)\\omega = \\nu \\nabla^2 \\omega`
on a doubly periodic domain, with the incompressible velocity field
recovered from the streamfunction :math:`\\psi` (:math:`\\nabla^2\\psi=-\\omega`,
:math:`\\mathbf{u}=(\\partial_y\\psi,\\,-\\partial_x\\psi)`) via an exact spectral
Poisson solve at every step -- the standard pseudo-spectral vorticity
formulation used throughout this package for 2D turbulence and shear-flow
instabilities. Working in vorticity rather than velocity-pressure form
eliminates the pressure entirely (it never appears once the curl of the
momentum equation is taken) at the cost of being specific to two dimensions,
where vorticity is a scalar.

This module is the general-purpose incompressible-flow workhorse of
:mod:`physicskit.fluids`: :mod:`physicskit.fluids.systems.instabilities` grows
its Kelvin-Helmholtz and Rayleigh-Taylor initial conditions with it (or with
its Boussinesq extension), and :mod:`physicskit.fluids.utils.spectral_analysis`
measures the turbulent energy spectrum of the velocity fields it produces.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from physicskit.fluids.core.grid import poisson_solve_streamfunction, spectral_grid, velocity_from_streamfunction
from physicskit.fluids.core.timestepping import integrate_vorticity_streamfunction, vorticity_rhs
from physicskit.fluids.exceptions import InvalidParameterError

__all__ = ["NavierStokes2D", "simulate_vorticity_streamfunction"]


class NavierStokes2D:
    """A doubly periodic 2D incompressible Navier-Stokes solver, in vorticity-streamfunction form.

    Builds the pseudo-spectral grid once at construction and exposes
    :meth:`rhs`, :meth:`step`, and :meth:`simulate` against it, so that
    repeated calls (e.g. one per animation frame) do not re-derive the
    wavenumber grid on every call the way the module-level
    :func:`simulate_vorticity_streamfunction` convenience function must.

    Parameters
    ----------
    n : int, default 64
        Number of grid points along each axis.
    length : float, default ``2*pi``
        Physical domain size (the domain is ``[0, length) x [0, length)``).
    nu : float, default 0.01
        Kinematic viscosity. Must be positive; a truly inviscid flow is
        ill-posed on this grid (nothing controls the cascade of enstrophy to
        the grid scale), so ``nu=0`` is rejected.

    Attributes
    ----------
    n : int
        Grid resolution.
    length : float
        Domain size.
    nu : float
        Kinematic viscosity.
    X, Y : ndarray of float, shape (n, n)
        Real-space coordinate grids.
    KX, KY, K2 : ndarray of float, shape (n, n)
        Wavenumber grids.

    Raises
    ------
    InvalidParameterError
        If `nu` is not positive (see :func:`physicskit.fluids.core.grid.spectral_grid`
        for the conditions on `n` and `length`).

    Examples
    --------
    >>> import numpy as np
    >>> solver = NavierStokes2D(n=48, length=2 * np.pi, nu=0.05)
    >>> omega0 = np.sin(solver.X) * np.sin(solver.Y)
    >>> result = solver.simulate(omega0, dt=0.01, steps=20)
    >>> result["omega"].shape
    (48, 48)
    >>> bool(np.max(np.abs(result["omega"])) < np.max(np.abs(omega0)))
    True
    """

    def __init__(self, n: int = 64, length: float = 2.0 * np.pi, nu: float = 0.01):
        if nu <= 0:
            raise InvalidParameterError(f"nu (kinematic viscosity) must be positive, got {nu}")
        self.n = int(n)
        self.length = float(length)
        self.nu = float(nu)
        self.X, self.Y, self.KX, self.KY, self.K2 = spectral_grid(self.n, self.length)

    def rhs(self, omega: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the vorticity-transport right-hand side at `omega`.

        Parameters
        ----------
        omega : ndarray of float, shape (n, n)
            Vorticity field.

        Returns
        -------
        ndarray of float, shape (n, n)
            :math:`-(\\mathbf{u}\\cdot\\nabla)\\omega + \\nu\\nabla^2\\omega`.
        """
        return vorticity_rhs(omega, self.KX, self.KY, self.K2, self.nu)

    def velocity(self, omega: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Recover the velocity field associated with a vorticity field.

        Parameters
        ----------
        omega : ndarray of float, shape (n, n)
            Vorticity field.

        Returns
        -------
        u, v : ndarray of float, shape (n, n)
            Velocity components.
        """
        psi = poisson_solve_streamfunction(omega, self.K2)
        return velocity_from_streamfunction(psi, self.KX, self.KY)

    def step(self, omega: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
        """Advance the vorticity field by one RK4 step of size `dt`.

        Parameters
        ----------
        omega : ndarray of float, shape (n, n)
            Current vorticity field.
        dt : float
            Step size.

        Returns
        -------
        ndarray of float, shape (n, n)
            Vorticity field advanced by one step.
        """
        k1 = self.rhs(omega)
        k2 = self.rhs(omega + dt / 2 * k1)
        k3 = self.rhs(omega + dt / 2 * k2)
        k4 = self.rhs(omega + dt * k3)
        return omega + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def simulate(self, omega0: NDArray[np.float64], dt: float, steps: int) -> dict[str, NDArray[np.float64]]:
        """Time-step an initial vorticity field forward by ``steps`` RK4 steps.

        Parameters
        ----------
        omega0 : ndarray of float, shape (n, n)
            Initial vorticity field.
        dt : float
            Time step.
        steps : int
            Number of RK4 steps to advance.

        Returns
        -------
        dict
            ``{"omega": final vorticity, "psi": final streamfunction, "u": ..., "v": ...}``.
        """
        omega = np.array(omega0, dtype=float, copy=True)
        for _ in range(steps):
            omega = self.step(omega, dt)
        psi = poisson_solve_streamfunction(omega, self.K2)
        u, v = velocity_from_streamfunction(psi, self.KX, self.KY)
        return {"omega": omega, "psi": psi, "u": u, "v": v}

    def __repr__(self) -> str:
        return f"NavierStokes2D(n={self.n}, length={self.length}, nu={self.nu})"


def simulate_vorticity_streamfunction(omega0: NDArray[np.float64], nu: float, dt: float, steps: int, length: float) -> dict[str, NDArray[np.float64]]:
    """Time-step 2D incompressible vorticity-streamfunction Navier-Stokes with RK4.

    A one-shot functional convenience wrapper around :class:`NavierStokes2D`,
    for scripts that only need a single run rather than repeated stepping
    against the same grid.

    Parameters
    ----------
    omega0 : ndarray of float, shape (n, n)
        Initial vorticity field on a doubly periodic ``[0, length)^2`` domain.
    nu : float
        Kinematic viscosity; must be positive.
    dt : float
        Time step.
    steps : int
        Number of RK4 steps to advance.
    length : float
        Physical domain size.

    Returns
    -------
    dict
        ``{"omega": final vorticity, "psi": final streamfunction, "u": ..., "v": ...}``.

    Raises
    ------
    InvalidParameterError
        If `nu` is not positive.

    See Also
    --------
    NavierStokes2D : Reusable object form, for repeated stepping on one grid.
    physicskit.fluids.systems.instabilities.kelvin_helmholtz_ic :
        Shear-layer initial condition that grows via this solver.

    Examples
    --------
    >>> import numpy as np
    >>> n = 48
    >>> from physicskit.fluids.core.grid import spectral_grid
    >>> X, Y, KX, KY, K2 = spectral_grid(n, 2 * np.pi)
    >>> omega0 = np.sin(X) * np.sin(Y)
    >>> result = simulate_vorticity_streamfunction(omega0, nu=0.05, dt=0.01, steps=20, length=2 * np.pi)
    >>> result["omega"].shape
    (48, 48)
    """
    if nu <= 0:
        raise InvalidParameterError(f"nu (kinematic viscosity) must be positive, got {nu}")
    return integrate_vorticity_streamfunction(omega0, nu, dt, steps, length)
