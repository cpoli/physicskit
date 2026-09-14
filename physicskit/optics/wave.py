"""Scalar wave optics: FFT-based Fresnel, Fraunhofer, and angular-spectrum diffraction.

Treats light as a scalar complex field :math:`U(x,y)` obeying the
Helmholtz equation, and propagates it between parallel planes using the
three workhorse methods of Fourier optics (Goodman, *Introduction to
Fourier Optics*):

- :func:`angular_spectrum_propagate` -- the exact scalar-diffraction
  solution, decomposing the field into plane waves (its 2D Fourier
  transform), advancing each by its own propagation phase
  :math:`e^{ik_z z}`, and re-synthesizing. Valid at any distance,
  including deep into the near field, as long as the grid resolves the
  field's spatial frequencies.
- :func:`fresnel_diffraction` -- the paraxial (parabolic-wave) near-field
  approximation, computable as a single Fourier transform of the aperture
  times a quadratic phase.
- :func:`fraunhofer_diffraction` -- the further far-field approximation
  valid once :math:`z` is large enough that the quadratic phase across the
  aperture itself is negligible; the diffraction pattern becomes simply
  the (scaled, phase-prefactored) Fourier transform of the aperture, the
  basis for classic single- and double-slit interference patterns.

All propagation routines expect a field sampled on a uniform square-pixel
grid of spacing ``dx`` (same length units as ``wavelength``), and treat
array index ``(Ny//2, Nx//2)`` as the on-axis origin.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "circular_aperture",
    "single_slit_aperture",
    "double_slit_aperture",
    "fraunhofer_diffraction",
    "fresnel_diffraction",
    "angular_spectrum_propagate",
    "intensity",
    "field_grid",
]


def field_grid(shape, dx):
    """Centered physical-coordinate grid for an array of the given ``shape``.

    Parameters
    ----------
    shape : tuple of int
        ``(Ny, Nx)`` array shape.
    dx : float
        Grid spacing (same units as ``wavelength`` elsewhere in this
        module).

    Returns
    -------
    X, Y : ndarray of shape (Ny, Nx)
        Cartesian coordinates, with the origin at index ``(Ny//2, Nx//2)``.

    Examples
    --------
    >>> X, Y = field_grid((4, 4), dx=1.0)
    >>> X[0].tolist()
    [-2.0, -1.0, 0.0, 1.0]
    """
    Ny, Nx = shape
    x = (np.arange(Nx) - Nx // 2) * dx
    y = (np.arange(Ny) - Ny // 2) * dx
    X, Y = np.meshgrid(x, y)
    return X, Y


def circular_aperture(shape, dx, radius):
    """A circular (disk) aperture, transmittance 1 inside, 0 outside.

    Parameters
    ----------
    shape : tuple of int
        ``(Ny, Nx)`` array shape.
    dx : float
        Grid spacing.
    radius : float
        Aperture radius.

    Returns
    -------
    ndarray of shape (Ny, Nx)
        Real-valued array of 1.0 (open) and 0.0 (blocked).

    Examples
    --------
    >>> ap = circular_aperture((5, 5), dx=1.0, radius=1.5)
    >>> float(ap[2, 2])
    1.0
    >>> float(ap[0, 0])
    0.0
    """
    X, Y = field_grid(shape, dx)
    return np.where(X**2 + Y**2 <= radius**2, 1.0, 0.0)


def single_slit_aperture(shape, dx, width):
    """A single slit of the given ``width`` along x, open along the full height in y.

    Parameters
    ----------
    shape : tuple of int
        ``(Ny, Nx)`` array shape.
    dx : float
        Grid spacing.
    width : float
        Full width of the slit opening (in the x direction).

    Returns
    -------
    ndarray of shape (Ny, Nx)
        Real-valued array of 1.0 (open) and 0.0 (blocked).

    Examples
    --------
    >>> ap = single_slit_aperture((3, 7), dx=1.0, width=2.5)
    >>> ap[0].tolist()
    [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    """
    X, _ = field_grid(shape, dx)
    return np.where(np.abs(X) <= width / 2.0, 1.0, 0.0)


def double_slit_aperture(shape, dx, width, separation):
    """Two parallel slits of the given ``width``, centers separated by ``separation``.

    Parameters
    ----------
    shape : tuple of int
        ``(Ny, Nx)`` array shape.
    dx : float
        Grid spacing.
    width : float
        Full width of each slit opening.
    separation : float
        Center-to-center distance between the two slits.

    Returns
    -------
    ndarray of shape (Ny, Nx)
        Real-valued array of 1.0 (open) and 0.0 (blocked).

    Examples
    --------
    >>> ap = double_slit_aperture((3, 11), dx=1.0, width=1.5, separation=6.0)
    >>> ap[0].tolist()
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    """
    X, _ = field_grid(shape, dx)
    left = np.abs(X + separation / 2.0) <= width / 2.0
    right = np.abs(X - separation / 2.0) <= width / 2.0
    return np.where(left | right, 1.0, 0.0)


def fraunhofer_diffraction(aperture, wavelength, z, dx):
    """Far-field (Fraunhofer) diffraction pattern of ``aperture`` at distance ``z``.

    Uses the standard single-Fourier-transform Fraunhofer formula (Goodman
    convention),

    .. math::

        U(x',y') = \\frac{e^{ikz}}{i\\lambda z}
            e^{i\\frac{k}{2z}(x'^2+y'^2)}
            \\iint U_0(x,y)\\,
            e^{-i\\frac{2\\pi}{\\lambda z}(x x' + y y')}\\,dx\\,dy,

    with :math:`k = 2\\pi/\\lambda`. The double integral is exactly a 2D
    Fourier transform of ``aperture`` evaluated at spatial frequency
    :math:`(f_x, f_y) = (x'/(\\lambda z), y'/(\\lambda z))`, computed here
    with a single FFT (the discrete sum is converted to a continuum
    integral by the ``dx**2`` pixel-area factor). The output array lives
    on its own natural grid of spacing :math:`\\lambda z / (N\\,dx)`, not
    the input grid.

    Parameters
    ----------
    aperture : ndarray of shape (Ny, Nx)
        Input field (real or complex) immediately after the aperture.
    wavelength : float
        Wavelength :math:`\\lambda`, in the same length units as ``dx``.
    z : float
        Propagation distance to the observation plane.
    dx : float
        Input grid spacing.

    Returns
    -------
    ndarray of shape (Ny, Nx), complex
        The far-field complex amplitude.

    See Also
    --------
    fresnel_diffraction : The near-field approximation this is a further limit of.

    Examples
    --------
    >>> ap = circular_aperture((64, 64), dx=0.01, radius=0.05)
    >>> U = fraunhofer_diffraction(ap, wavelength=0.5e-3, z=2.0, dx=0.01)
    >>> U.shape
    (64, 64)
    >>> bool(intensity(U)[32, 32] == intensity(U).max())
    True
    """
    aperture = np.asarray(aperture)
    k = 2.0 * np.pi / wavelength
    Ny, Nx = aperture.shape
    FT = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(aperture))) * dx * dx
    fx = np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(Ny, d=dx))
    Xp, Yp = np.meshgrid(wavelength * z * fx, wavelength * z * fy)
    prefactor = (np.exp(1j * k * z) / (1j * wavelength * z)) * np.exp(1j * k / (2.0 * z) * (Xp**2 + Yp**2))
    return prefactor * FT


def fresnel_diffraction(aperture, wavelength, z, dx):
    """Near-field (Fresnel) diffraction pattern of ``aperture`` at distance ``z``.

    The single-Fourier-transform ("one-step") Fresnel propagation method:
    multiply the input field by the Fresnel quadratic phase, Fourier
    transform, then multiply by the corresponding output quadratic phase
    and prefactor,

    .. math::

        U(x',y') = \\frac{e^{ikz}}{i\\lambda z}
            e^{i\\frac{k}{2z}(x'^2+y'^2)}\\,
            \\mathcal{F}\\!\\left[U_0(x,y)\\,
            e^{i\\frac{k}{2z}(x^2+y^2)}\\right]_{f_x=x'/(\\lambda z),\\ f_y=y'/(\\lambda z)},

    the paraxial approximation to exact scalar diffraction. As with
    :func:`fraunhofer_diffraction`, the output lives on its own natural
    grid of spacing :math:`\\lambda z/(N\\,dx)`.

    Parameters
    ----------
    aperture : ndarray of shape (Ny, Nx)
        Input field (real or complex).
    wavelength : float
        Wavelength :math:`\\lambda`.
    z : float
        Propagation distance.
    dx : float
        Input grid spacing.

    Returns
    -------
    ndarray of shape (Ny, Nx), complex
        The propagated complex amplitude.

    See Also
    --------
    angular_spectrum_propagate : The exact (non-paraxial) alternative, on the same input grid.

    Examples
    --------
    >>> ap = circular_aperture((64, 64), dx=0.01, radius=0.05)
    >>> U = fresnel_diffraction(ap, wavelength=0.5e-3, z=5.0, dx=0.01)
    >>> U.shape
    (64, 64)
    """
    aperture = np.asarray(aperture)
    k = 2.0 * np.pi / wavelength
    Ny, Nx = aperture.shape
    X, Y = field_grid((Ny, Nx), dx)
    integrand = aperture * np.exp(1j * k / (2.0 * z) * (X**2 + Y**2))
    FT = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(integrand))) * dx * dx
    fx = np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(Ny, d=dx))
    Xp, Yp = np.meshgrid(wavelength * z * fx, wavelength * z * fy)
    prefactor = (np.exp(1j * k * z) / (1j * wavelength * z)) * np.exp(1j * k / (2.0 * z) * (Xp**2 + Yp**2))
    return prefactor * FT


def angular_spectrum_propagate(U0, wavelength, z, dx):
    """Exact scalar diffraction propagation via the angular spectrum method.

    Decomposes the input field into plane-wave components (its 2D Fourier
    transform), advances each by its own longitudinal propagation phase,
    and re-synthesizes:

    .. math::

        U(x,y;z) = \\mathcal{F}^{-1}\\!\\left[
            \\mathcal{F}[U_0](f_x,f_y)\\; e^{ik_z z}
        \\right], \\qquad
        k_z = \\begin{cases}
            \\sqrt{k^2 - k_x^2 - k_y^2} & k_x^2+k_y^2 \\le k^2 \\quad \\text{(propagating)}\\\\
            i\\sqrt{k_x^2+k_y^2 - k^2} & k_x^2+k_y^2 > k^2 \\quad \\text{(evanescent)}
        \\end{cases}

    with :math:`k=2\\pi/\\lambda`, :math:`k_x = 2\\pi f_x`,
    :math:`k_y = 2\\pi f_y`. Evanescent orders get a purely imaginary
    :math:`k_z`, so :math:`e^{ik_z z}` decays exponentially rather than
    producing ``nan``/``inf``. Unlike :func:`fresnel_diffraction` and
    :func:`fraunhofer_diffraction`, this is not a paraxial approximation
    and the output remains on the *same* grid (shape and spacing ``dx``)
    as the input.

    Parameters
    ----------
    U0 : ndarray of shape (Ny, Nx)
        Input complex (or real) field.
    wavelength : float
        Wavelength :math:`\\lambda`.
    z : float
        Propagation distance.
    dx : float
        Grid spacing (both input and output).

    Returns
    -------
    ndarray of shape (Ny, Nx), complex
        The propagated field, on the same grid as the input.

    Examples
    --------
    >>> ap = circular_aperture((64, 64), dx=0.01, radius=0.05)
    >>> U = angular_spectrum_propagate(ap, wavelength=0.5e-3, z=0.05, dx=0.01)
    >>> U.shape
    (64, 64)
    >>> bool(np.isclose(intensity(U).sum(), intensity(ap).sum(), rtol=0.02))
    True
    """
    U0 = np.asarray(U0)
    Ny, Nx = U0.shape
    k = 2.0 * np.pi / wavelength
    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dx)
    FX, FY = np.meshgrid(fx, fy)
    kx = 2.0 * np.pi * FX
    ky = 2.0 * np.pi * FY
    kz_sq = k**2 - kx**2 - ky**2
    propagating = kz_sq >= 0.0
    kz = np.empty_like(kz_sq, dtype=complex)
    kz[propagating] = np.sqrt(kz_sq[propagating])
    kz[~propagating] = 1j * np.sqrt(-kz_sq[~propagating])
    H = np.exp(1j * kz * z)
    return np.fft.ifft2(np.fft.fft2(U0) * H)


def intensity(U):
    """Optical intensity :math:`|U|^2` of a complex field.

    Parameters
    ----------
    U : ndarray
        Complex (or real) field amplitude.

    Returns
    -------
    ndarray
        Real-valued intensity, same shape as ``U``.

    Examples
    --------
    >>> intensity(np.array([3.0 + 4.0j]))
    array([25.])
    """
    return np.abs(U) ** 2
