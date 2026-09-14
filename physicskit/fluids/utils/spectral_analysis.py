"""Turbulent kinetic energy spectra and the Kolmogorov -5/3 law.

A turbulent velocity field is usefully summarized not by its detailed
pointwise structure but by how its kinetic energy is distributed across
length scales -- the energy spectrum :math:`E(k)`, defined so that
:math:`\\int E(k)\\,dk` is the total kinetic energy per unit mass.
Kolmogorov's 1941 theory predicts a universal power law,
:math:`E(k)\\propto k^{-5/3}`, in the "inertial range" of scales small
enough to have forgotten how the turbulence was forced but large enough not
to feel viscosity yet -- one of the most-tested predictions in all of
classical physics, and the standard first check that a simulated 2D or 3D
flow (e.g. the output of
:class:`physicskit.fluids.systems.navier_stokes.NavierStokes2D`) is
resolving a genuine turbulent cascade.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from physicskit.fluids.exceptions import InvalidParameterError

__all__ = ["energy_spectrum", "kolmogorov_reference_slope"]


def energy_spectrum(u: NDArray[np.float64], v: NDArray[np.float64], length: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Isotropic (azimuthally averaged) kinetic energy spectrum of a 2D velocity field.

    Computes :math:`\\hat{u}(\\mathbf{k})` and :math:`\\hat{v}(\\mathbf{k})`
    by FFT, forms the kinetic energy density
    :math:`\\tfrac{1}{2}(|\\hat{u}|^2+|\\hat{v}|^2)` at each wavevector, and
    sums it over the discrete annulus of wavevectors with
    :math:`k \\le |\\mathbf{k}| < k+1` (in grid-index units) to give
    :math:`E(k)`, normalized so that :math:`\\sum_k E(k) \\approx` the
    domain-averaged kinetic energy per unit mass (Parseval's theorem).

    Parameters
    ----------
    u, v : ndarray of float, shape (n, n)
        Velocity components on a doubly periodic grid, e.g. from
        :meth:`physicskit.fluids.systems.navier_stokes.NavierStokes2D.velocity`.
    length : float
        Physical domain size.

    Returns
    -------
    k : ndarray of float, shape (n // 2,)
        Wavenumber bins (angular, :math:`2\\pi/\\lambda` convention).
    E : ndarray of float, shape (n // 2,)
        Kinetic energy spectral density at each wavenumber bin.

    Raises
    ------
    InvalidParameterError
        If `u` and `v` are not the same shape, or are not square.

    See Also
    --------
    kolmogorov_reference_slope : The -5/3 reference line to compare `E` against.

    Examples
    --------
    >>> import numpy as np
    >>> n, length = 64, 2 * np.pi
    >>> x = np.linspace(0, length, n, endpoint=False)
    >>> X, Y = np.meshgrid(x, x, indexing="ij")
    >>> u, v = np.sin(Y), np.sin(X)  # a single-mode flow
    >>> k, E = energy_spectrum(u, v, length)
    >>> bool(np.argmax(E) == 1)  # all the energy sits at the single k=1 mode
    True
    """
    if u.shape != v.shape:
        raise InvalidParameterError("u and v must have the same shape")
    if u.shape[0] != u.shape[1]:
        raise InvalidParameterError("u and v must be square (n, n) arrays")
    n = u.shape[0]
    u_hat = np.fft.fft2(u) / n**2
    v_hat = np.fft.fft2(v) / n**2
    energy_density = 0.5 * (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2) * n**2

    kfreq = np.fft.fftfreq(n, d=length / n) * 2.0 * np.pi
    KX, KY = np.meshgrid(kfreq, kfreq, indexing="ij")
    kmag_index = np.round(np.sqrt(KX**2 + KY**2) / (2.0 * np.pi / length)).astype(int)

    n_bins = n // 2
    E = np.zeros(n_bins)
    for shell in range(n_bins):
        mask = kmag_index == shell
        E[shell] = energy_density[mask].sum()
    k = np.arange(n_bins) * (2.0 * np.pi / length)
    return k, E


def kolmogorov_reference_slope(k: NDArray[np.float64], k0: float, E0: float) -> NDArray[np.float64]:
    """A Kolmogorov :math:`k^{-5/3}` reference line, anchored at one point.

    Kolmogorov's K41 theory predicts that, in the inertial range, the
    energy spectrum depends only on the wavenumber `k` and the (scale-independent)
    energy dissipation rate, and dimensional analysis alone then fixes the
    power law:

    .. math::

        E(k) = E_0 \\left(\\frac{k}{k_0}\\right)^{-5/3}.

    Plotting this line through one point ``(k0, E0)`` of a measured
    spectrum (see :func:`energy_spectrum`) on log-log axes is the standard
    visual check for an inertial range: a genuine turbulent cascade runs
    parallel to this line over at least a decade of `k`.

    Parameters
    ----------
    k : ndarray of float
        Wavenumbers at which to evaluate the reference line.
    k0 : float
        Anchor wavenumber; must be positive.
    E0 : float
        Spectral energy density at `k0`.

    Returns
    -------
    ndarray of float
        Reference spectral density :math:`E_0(k/k_0)^{-5/3}` at each `k`.

    Raises
    ------
    InvalidParameterError
        If `k0` is not positive.

    Examples
    --------
    >>> import numpy as np
    >>> k = np.array([1.0, 2.0, 4.0])
    >>> ref = kolmogorov_reference_slope(k, k0=1.0, E0=1.0)
    >>> round(float(ref[1]), 4)
    0.315
    """
    if k0 <= 0:
        raise InvalidParameterError(f"k0 must be positive, got {k0}")
    k = np.asarray(k, dtype=np.float64)
    return E0 * (k / k0) ** (-5.0 / 3.0)
