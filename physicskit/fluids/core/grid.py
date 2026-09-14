"""Doubly-periodic pseudo-spectral grid and Poisson solve.

The shared spatial machinery behind every grid-based solver in
:mod:`physicskit.fluids`: a real-space/wavenumber grid pair on the doubly
periodic domain ``[0, length) x [0, length)``, an exact spectral Poisson
solve (used to recover a streamfunction from a vorticity field), and the
spectral velocity recovery that follows from it. Because differentiation is
exact multiplication by ``i*k`` in Fourier space, every derivative used here
is exact up to machine precision and aliasing, with none of a finite
difference's truncation error -- the standard reason pseudo-spectral methods
are the tool of choice for smooth, periodic 2D flow.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from physicskit.fluids.exceptions import InvalidParameterError

__all__ = [
    "spectral_grid",
    "poisson_solve_streamfunction",
    "velocity_from_streamfunction",
    "vorticity_from_velocity",
]


def spectral_grid(n: int, length: float) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Build a doubly periodic real-space and wavenumber grid.

    Parameters
    ----------
    n : int
        Number of grid points along each axis. Must be at least 8, so that
        the FFT-based derivatives resolve at least a few wavelengths of the
        lowest nonzero mode.
    length : float
        Physical domain size (the domain is ``[0, length) x [0, length)``).

    Returns
    -------
    X, Y : ndarray of float, shape (n, n)
        Real-space coordinate grids.
    KX, KY : ndarray of float, shape (n, n)
        Wavenumber grids (angular, i.e. :math:`2\\pi/\\lambda` convention).
    K2 : ndarray of float, shape (n, n)
        :math:`K_X^2 + K_Y^2`, with the zero mode set to 1 to make Poisson
        division safe (the zero mode of a vorticity field is always 0 for a
        physical flow, so this entry is unused downstream).

    Raises
    ------
    InvalidParameterError
        If `n` is smaller than 8 or `length` is not positive.

    Examples
    --------
    >>> X, Y, KX, KY, K2 = spectral_grid(64, 2 * 3.141592653589793)
    >>> X.shape
    (64, 64)
    >>> bool(K2[0, 0] == 1.0)
    True
    """
    if n < 8:
        raise InvalidParameterError(f"n must be >= 8 for a usable doubly periodic spectral grid, got {n}")
    if length <= 0:
        raise InvalidParameterError(f"length must be positive, got {length}")
    x = np.linspace(0, length, n, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    k = 2 * np.pi * np.fft.fftfreq(n, d=length / n)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    K2 = KX**2 + KY**2
    K2 = K2.copy()
    K2[0, 0] = 1.0
    return X, Y, KX, KY, K2


def poisson_solve_streamfunction(omega: NDArray[np.float64], K2: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solve :math:`\\nabla^2 \\psi = -\\omega` for the streamfunction via FFT.

    Parameters
    ----------
    omega : ndarray of float, shape (n, n)
        Vorticity field.
    K2 : ndarray of float, shape (n, n)
        Squared wavenumber grid from :func:`spectral_grid`.

    Returns
    -------
    ndarray of float, shape (n, n)
        Streamfunction :math:`\\psi`.

    Examples
    --------
    >>> import numpy as np
    >>> X, Y, KX, KY, K2 = spectral_grid(64, 2 * np.pi)
    >>> omega = np.sin(X) * np.sin(Y)
    >>> psi = poisson_solve_streamfunction(omega, K2)
    >>> expected = 0.5 * np.sin(X) * np.sin(Y)
    >>> bool(np.max(np.abs(psi - expected)) < 1e-10)
    True
    """
    omega_hat = np.fft.fft2(omega)
    psi_hat = omega_hat / K2
    psi_hat[0, 0] = 0.0
    return np.real(np.fft.ifft2(psi_hat))


def velocity_from_streamfunction(psi: NDArray[np.float64], KX: NDArray[np.float64], KY: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Recover the incompressible velocity field from a streamfunction.

    Parameters
    ----------
    psi : ndarray of float, shape (n, n)
        Streamfunction.
    KX, KY : ndarray of float, shape (n, n)
        Wavenumber grids from :func:`spectral_grid`.

    Returns
    -------
    u, v : ndarray of float, shape (n, n)
        Velocity components :math:`u=\\partial_y\\psi`, :math:`v=-\\partial_x\\psi`,
        which satisfy :math:`\\nabla\\cdot\\mathbf{u}=0` identically.

    Examples
    --------
    >>> import numpy as np
    >>> X, Y, KX, KY, K2 = spectral_grid(64, 2 * np.pi)
    >>> psi = 0.5 * np.sin(X) * np.sin(Y)
    >>> u, v = velocity_from_streamfunction(psi, KX, KY)
    >>> curl = vorticity_from_velocity(u, v, KX, KY)
    >>> bool(np.max(np.abs(curl - np.sin(X) * np.sin(Y))) < 1e-10)
    True
    """
    psi_hat = np.fft.fft2(psi)
    u = np.real(np.fft.ifft2(1j * KY * psi_hat))
    v = np.real(np.fft.ifft2(-1j * KX * psi_hat))
    return u, v


def vorticity_from_velocity(u: NDArray[np.float64], v: NDArray[np.float64], KX: NDArray[np.float64], KY: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the vorticity :math:`\\omega=\\partial_x v-\\partial_y u` spectrally.

    The inverse of :func:`velocity_from_streamfunction` followed by a curl;
    mainly useful for validating that a velocity field is consistent with
    some streamfunction, or for measuring the vorticity of a field built
    directly (e.g. by superposing elementary potential-flow solutions)
    rather than evolved from one.

    Parameters
    ----------
    u, v : ndarray of float, shape (n, n)
        Velocity components.
    KX, KY : ndarray of float, shape (n, n)
        Wavenumber grids from :func:`spectral_grid`.

    Returns
    -------
    ndarray of float, shape (n, n)
        Vorticity field :math:`\\omega`.
    """
    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)
    return np.real(np.fft.ifft2(1j * KX * v_hat - 1j * KY * u_hat))
