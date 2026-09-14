"""Pseudo-spectral RK4 time-stepping for advection-diffusion transport on the spectral grid.

The right-hand side of the vorticity-transport equation involves an FFT at
every evaluation (to invert the Laplacian for the streamfunction, and to
differentiate the advected field), so -- unlike
:mod:`physicskit.chaos.core.integrators` -- these steppers are plain NumPy
rather than Numba ``@njit`` functions: Numba's ``nopython`` mode does not
support ``numpy.fft``. What *is* shared and reused here is the algorithm:
one classical 4th-order Runge-Kutta step applied to the pseudo-spectrally
evaluated right-hand side, exactly as :mod:`physicskit.chaos.core.integrators`
does with its Numba-jitted right-hand sides.

:func:`vorticity_rhs` and :func:`integrate_vorticity_streamfunction` are the
single-field (pure vorticity-transport) engine reused, unmodified, by
:mod:`physicskit.fluids.systems.navier_stokes` as the package's
general-purpose incompressible-flow workhorse, and in turn by
:mod:`physicskit.fluids.systems.instabilities` to grow a Kelvin-Helmholtz
shear layer. :func:`buoyant_vorticity_rhs` and
:func:`integrate_boussinesq` extend the same scheme with a passively
advected buoyancy (density) field and its baroclinic torque on vorticity --
the minimal addition needed to grow a Rayleigh-Taylor instability, which
requires two fluid layers of different density and therefore cannot be
represented by vorticity alone.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from physicskit.fluids.core.grid import poisson_solve_streamfunction, spectral_grid, velocity_from_streamfunction

__all__ = [
    "vorticity_rhs",
    "integrate_vorticity_streamfunction",
    "buoyant_vorticity_rhs",
    "integrate_boussinesq",
]


def vorticity_rhs(
    omega: NDArray[np.float64],
    KX: NDArray[np.float64],
    KY: NDArray[np.float64],
    K2: NDArray[np.float64],
    nu: float,
) -> NDArray[np.float64]:
    """Right-hand side of the vorticity transport equation, evaluated pseudo-spectrally.

    Parameters
    ----------
    omega : ndarray of float, shape (n, n)
        Vorticity field.
    KX, KY, K2 : ndarray of float, shape (n, n)
        Wavenumber grids from :func:`physicskit.fluids.core.grid.spectral_grid`.
    nu : float
        Kinematic viscosity.

    Returns
    -------
    ndarray of float, shape (n, n)
        :math:`-(\\mathbf{u}\\cdot\\nabla)\\omega + \\nu\\nabla^2\\omega`.
    """
    psi = poisson_solve_streamfunction(omega, K2)
    u, v = velocity_from_streamfunction(psi, KX, KY)
    omega_hat = np.fft.fft2(omega)
    domega_dx = np.real(np.fft.ifft2(1j * KX * omega_hat))
    domega_dy = np.real(np.fft.ifft2(1j * KY * omega_hat))
    advection = -(u * domega_dx + v * domega_dy)
    diffusion = np.real(np.fft.ifft2(-nu * K2 * omega_hat))
    return advection + diffusion


def integrate_vorticity_streamfunction(omega0: NDArray[np.float64], nu: float, dt: float, steps: int, length: float) -> dict[str, NDArray[np.float64]]:
    """Time-step the 2D incompressible vorticity-streamfunction equations with RK4.

    Parameters
    ----------
    omega0 : ndarray of float, shape (n, n)
        Initial vorticity field on a doubly periodic ``[0, length)^2`` domain.
    nu : float
        Kinematic viscosity.
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

    See Also
    --------
    physicskit.fluids.systems.navier_stokes.NavierStokes2D : Validated, documented wrapper.
    """
    n = omega0.shape[0]
    _, _, KX, KY, K2 = spectral_grid(n, length)
    omega = np.array(omega0, dtype=float, copy=True)
    for _ in range(steps):
        k1 = vorticity_rhs(omega, KX, KY, K2, nu)
        k2 = vorticity_rhs(omega + dt / 2 * k1, KX, KY, K2, nu)
        k3 = vorticity_rhs(omega + dt / 2 * k2, KX, KY, K2, nu)
        k4 = vorticity_rhs(omega + dt * k3, KX, KY, K2, nu)
        omega = omega + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    psi = poisson_solve_streamfunction(omega, K2)
    u, v = velocity_from_streamfunction(psi, KX, KY)
    return {"omega": omega, "psi": psi, "u": u, "v": v}


def buoyant_vorticity_rhs(
    omega: NDArray[np.float64],
    buoyancy: NDArray[np.float64],
    KX: NDArray[np.float64],
    KY: NDArray[np.float64],
    K2: NDArray[np.float64],
    nu: float,
    kappa: float,
    g: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Right-hand side of the Boussinesq vorticity-buoyancy system.

    Extends :func:`vorticity_rhs` with a passively advected-and-diffused
    dimensionless buoyancy field :math:`b = -\\delta\\rho/\\rho_0` (positive
    `b` means locally lighter than the reference density) and its
    baroclinic torque on vorticity, :math:`g\\,\\partial_x b`, which is
    exactly the term through which a horizontal density gradient generates
    vorticity under gravity -- the mechanism behind the Rayleigh-Taylor
    instability.

    .. math::

        \\partial_t \\omega + (\\mathbf{u}\\cdot\\nabla)\\omega
            = \\nu \\nabla^2 \\omega + g\\,\\partial_x b \\\\
        \\partial_t b + (\\mathbf{u}\\cdot\\nabla) b = \\kappa \\nabla^2 b

    Parameters
    ----------
    omega : ndarray of float, shape (n, n)
        Vorticity field.
    buoyancy : ndarray of float, shape (n, n)
        Dimensionless buoyancy field :math:`b`.
    KX, KY, K2 : ndarray of float, shape (n, n)
        Wavenumber grids from :func:`physicskit.fluids.core.grid.spectral_grid`.
    nu : float
        Kinematic viscosity (vorticity diffusivity).
    kappa : float
        Molecular (or eddy) diffusivity of the buoyancy field.
    g : float
        Gravitational acceleration, pointing from ``+y`` toward ``-y``.

    Returns
    -------
    domega_dt, dbuoyancy_dt : ndarray of float, shape (n, n)
        Time derivatives of vorticity and buoyancy.
    """
    psi = poisson_solve_streamfunction(omega, K2)
    u, v = velocity_from_streamfunction(psi, KX, KY)

    omega_hat = np.fft.fft2(omega)
    domega_dx = np.real(np.fft.ifft2(1j * KX * omega_hat))
    domega_dy = np.real(np.fft.ifft2(1j * KY * omega_hat))
    diffusion_omega = np.real(np.fft.ifft2(-nu * K2 * omega_hat))

    b_hat = np.fft.fft2(buoyancy)
    db_dx = np.real(np.fft.ifft2(1j * KX * b_hat))
    db_dy = np.real(np.fft.ifft2(1j * KY * b_hat))
    diffusion_b = np.real(np.fft.ifft2(-kappa * K2 * b_hat))

    domega_dt = -(u * domega_dx + v * domega_dy) + diffusion_omega + g * db_dx
    dbuoyancy_dt = -(u * db_dx + v * db_dy) + diffusion_b
    return domega_dt, dbuoyancy_dt


def integrate_boussinesq(
    omega0: NDArray[np.float64],
    buoyancy0: NDArray[np.float64],
    nu: float,
    kappa: float,
    g: float,
    dt: float,
    steps: int,
    length: float,
) -> dict[str, NDArray[np.float64]]:
    """Time-step the coupled Boussinesq vorticity-buoyancy equations with RK4.

    Parameters
    ----------
    omega0 : ndarray of float, shape (n, n)
        Initial vorticity field on a doubly periodic ``[0, length)^2`` domain.
    buoyancy0 : ndarray of float, shape (n, n)
        Initial buoyancy field.
    nu : float
        Kinematic viscosity.
    kappa : float
        Buoyancy diffusivity.
    g : float
        Gravitational acceleration.
    dt : float
        Time step.
    steps : int
        Number of RK4 steps to advance.
    length : float
        Physical domain size.

    Returns
    -------
    dict
        ``{"omega": ..., "buoyancy": ..., "psi": ..., "u": ..., "v": ...}`` at
        the final time.

    See Also
    --------
    physicskit.fluids.systems.instabilities.simulate_rayleigh_taylor : Validated, documented wrapper.
    """
    n = omega0.shape[0]
    _, _, KX, KY, K2 = spectral_grid(n, length)
    omega = np.array(omega0, dtype=float, copy=True)
    buoyancy = np.array(buoyancy0, dtype=float, copy=True)
    for _ in range(steps):
        k1o, k1b = buoyant_vorticity_rhs(omega, buoyancy, KX, KY, K2, nu, kappa, g)
        k2o, k2b = buoyant_vorticity_rhs(omega + dt / 2 * k1o, buoyancy + dt / 2 * k1b, KX, KY, K2, nu, kappa, g)
        k3o, k3b = buoyant_vorticity_rhs(omega + dt / 2 * k2o, buoyancy + dt / 2 * k2b, KX, KY, K2, nu, kappa, g)
        k4o, k4b = buoyant_vorticity_rhs(omega + dt * k3o, buoyancy + dt * k3b, KX, KY, K2, nu, kappa, g)
        omega = omega + dt / 6 * (k1o + 2 * k2o + 2 * k3o + k4o)
        buoyancy = buoyancy + dt / 6 * (k1b + 2 * k2b + 2 * k3b + k4b)
    psi = poisson_solve_streamfunction(omega, K2)
    u, v = velocity_from_streamfunction(psi, KX, KY)
    return {"omega": omega, "buoyancy": buoyancy, "psi": psi, "u": u, "v": v}
