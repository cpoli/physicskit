r"""Bogomolny's semiclassical transfer operator on a billiard boundary.

Bogomolny (1992) replaced the Gutzwiller sum over periodic orbits with a
finite operator on a Poincare surface of section. Its kernel is the
semiclassical propagator from one point of the section to the next,

.. math::

   T(q,q';E) = \frac{1}{\sqrt{2\pi i\hbar}}
   \sqrt{\left|\frac{\partial^2 S}{\partial q\,\partial q'}\right|}\,
   \exp\!\left[\frac{i}{\hbar}S(q,q';E) - i\frac{\pi\nu}{2}\right],

summed over the classical paths that go from :math:`q'` to :math:`q`
without crossing the section in between. Quantized energies are the
zeros of :math:`\det[1-T(E)]`. Expanding :math:`\log\det[1-T]` in
traces of powers of :math:`T` gives back the Gutzwiller trace formula,
but the determinant needs only short orbits (one crossing of the
section) and a matrix whose size is set by the number of wavelengths
across the section.

For a convex billiard the boundary itself is the natural section. With
arc length :math:`s` along it, the path from :math:`s'` to :math:`s` is
the straight chord of length :math:`\ell`, :math:`S=\hbar k\ell`, and
:math:`|\partial^2\ell/\partial s\,\partial s'|=\cos\theta\cos\theta'/\ell`,
where :math:`\theta,\theta'` are the angles between the chord and the
inward normals. A Dirichlet wall adds a phase :math:`\pi` per bounce
(:math:`\nu=2`), a Neumann wall none.
:func:`bogomolny_transfer_operator` builds the discretized kernel and
:func:`bogomolny_quantization_function` the quantity whose minima give
the levels.

References
----------
E. B. Bogomolny, "Semiclassical quantization of multidimensional
systems," Nonlinearity **5**, 805-866 (1992).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "bogomolny_transfer_operator",
    "bogomolny_quantization_function",
]


def bogomolny_transfer_operator(points: np.ndarray, normals: np.ndarray, ds: np.ndarray, k: float, boundary: str = "dirichlet") -> np.ndarray:
    r"""Discretized Bogomolny transfer operator :math:`T(k)` on the boundary of a convex billiard.

    .. math::

       T_{ij} = \sqrt{\frac{k\,\cos\theta_i\cos\theta_j}{2\pi i\,\ell_{ij}}}\;
       e^{ik\ell_{ij} - i\pi\nu/2}\,\sqrt{\Delta s_i\,\Delta s_j},

    with :math:`\ell_{ij}` the chord between boundary points :math:`i` and
    :math:`j`. The symmetric weight :math:`\sqrt{\Delta s_i\Delta s_j}`
    turns the integral kernel into a matrix whose eigenvalues approximate
    those of the operator. Use at least a few points per wavelength
    :math:`2\pi/k`; extra points only resolve the evanescent part of
    :math:`T` better.

    Parameters
    ----------
    points : ndarray, shape (N, 2)
        Boundary points, ordered along the boundary.
    normals : ndarray, shape (N, 2)
        Unit inward normals at ``points``.
    ds : ndarray, shape (N,)
        Arc-length weight of each point.
    k : float
        Wavenumber.
    boundary : {"dirichlet", "neumann"}, default="dirichlet"
        Wall boundary condition (Maslov phase :math:`\nu=2` or 0 per bounce).

    Returns
    -------
    ndarray of complex, shape (N, N)
        The matrix :math:`T(k)`.

    Raises
    ------
    ValueError
        If ``boundary`` is not recognized.

    Notes
    -----
    Every chord must lie inside the billiard, so the boundary must be
    convex. Grazing chords (:math:`\cos\theta\to0`) carry whispering-gallery
    states, which this leading-order kernel describes least accurately.

    Examples
    --------
    On a boundary of length :math:`L` there are about :math:`kL/\pi`
    propagating directions. :math:`T` is close to unitary on them and
    close to zero on the rest; for the unit circle at :math:`k=20` that is
    about 40 singular values near 1:

    >>> import numpy as np
    >>> N = 200
    >>> phi = 2 * np.pi * np.arange(N) / N
    >>> pts = np.column_stack([np.cos(phi), np.sin(phi)])
    >>> T = bogomolny_transfer_operator(pts, -pts, np.full(N, 2 * np.pi / N), k=20.0)
    >>> T.shape
    (200, 200)
    >>> sv = np.linalg.svd(T, compute_uv=False)
    >>> int(np.sum(sv > 0.5))
    41
    """
    if boundary not in ("dirichlet", "neumann"):
        raise ValueError(f'boundary must be "dirichlet" or "neumann", got {boundary!r}.')
    points = np.asarray(points, dtype=float)
    normals = np.asarray(normals, dtype=float)
    ds = np.asarray(ds, dtype=float)
    chord = points[None, :, :] - points[:, None, :]  # chord[i, j] = r_j - r_i
    length = np.linalg.norm(chord, axis=-1)
    np.fill_diagonal(length, 1.0)
    cos_i = np.einsum("ik,ijk->ij", normals, chord) / length
    cos_j = -np.einsum("jk,ijk->ij", normals, chord) / length
    maslov = np.pi if boundary == "dirichlet" else 0.0
    amplitude = np.sqrt(k * np.abs(cos_i * cos_j) / (2j * np.pi * length))
    T = amplitude * np.exp(1j * (k * length - maslov)) * np.sqrt(ds[:, None] * ds[None, :])
    np.fill_diagonal(T, 0.0)
    return T


def bogomolny_quantization_function(k_grid, points: np.ndarray, normals: np.ndarray, ds: np.ndarray, boundary: str = "dirichlet") -> np.ndarray:
    r"""Smallest singular value of :math:`1-T(k)` on a grid of wavenumbers.

    Bogomolny's quantization condition is :math:`\det[1-T(k)]=0`. The
    determinant of the discretized operator is never exactly zero, and its
    magnitude varies by orders of magnitude with :math:`k`; the smallest
    singular value of :math:`1-T(k)` is a better-scaled measure of how close
    :math:`T` is to having an eigenvalue 1. Its sharp minima are the
    semiclassical levels, and a near-degenerate pair shows up as a single
    minimum.

    Parameters
    ----------
    k_grid : ndarray
        Wavenumbers at which to evaluate.
    points, normals, ds : ndarray
        Boundary discretization, as in :func:`bogomolny_transfer_operator`.
    boundary : {"dirichlet", "neumann"}, default="dirichlet"
        Wall boundary condition.

    Returns
    -------
    ndarray, shape matching ``k_grid``
        :math:`\sigma_{\min}[1-T(k)]`.

    Examples
    --------
    The lowest Dirichlet level of the unit disk is the first zero of
    :math:`J_0`, :math:`k=2.4048`:

    >>> import numpy as np
    >>> N = 120
    >>> phi = 2 * np.pi * np.arange(N) / N
    >>> pts = np.column_stack([np.cos(phi), np.sin(phi)])
    >>> ks = np.linspace(2.2, 2.6, 81)
    >>> f = bogomolny_quantization_function(ks, pts, -pts, np.full(N, 2 * np.pi / N))
    >>> round(float(ks[np.argmin(f)]), 1)
    2.4
    """
    k_grid = np.asarray(k_grid, dtype=float)
    identity = np.eye(len(points))
    out = np.empty(k_grid.shape)
    for idx, k in np.ndenumerate(k_grid):
        T = bogomolny_transfer_operator(points, normals, ds, float(k), boundary)
        out[idx] = np.linalg.svd(identity - T, compute_uv=False)[-1]
    return out
