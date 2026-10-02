"""Fourier's heat (diffusion) equation on a grid: explicit, Crank-Nicolson, and spectral solvers.

Solves

.. math::

    \\frac{\\partial u}{\\partial t} = \\alpha\\,\\nabla^2 u

(Fourier, *Théorie analytique de la chaleur*, 1822) for a temperature or
concentration field :math:`u` on a uniform 1D, 2D, or 3D grid, with
diffusivity :math:`\\alpha`. Three methods, the classic trade-offs of
parabolic PDEs:

- ``"explicit"`` -- forward-time centered-space (FTCS). Cheap per step but
  only stable for :math:`\\alpha\\,\\Delta t\\sum_a 1/\\Delta x_a^2 \\le 1/2`
  (von Neumann analysis; Press et al., *Numerical Recipes*, 3rd ed.,
  §20.2).
- ``"crank_nicolson"`` -- the trapezoidal rule in time (Crank and Nicolson
  1947): unconditionally stable and second order in :math:`\\Delta t`, at
  the cost of one sparse solve per step (factored once).
- :func:`heat_equation_spectral` -- the exact solution of the
  periodic problem, :math:`\\hat u(\\mathbf k, t) = \\hat u(\\mathbf k, 0)
  e^{-\\alpha k^2 t}`: Fourier's own method of decomposing the field into
  modes that each decay independently.

Units are whatever the caller uses consistently (``alpha`` in length^2 per
time).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.linalg import splu

from physicskit.fluids.exceptions import InvalidParameterError

__all__ = [
    "laplacian_matrix",
    "heat_equation",
    "heat_equation_spectral",
    "gaussian_heat_solution",
]

_BCS = ("dirichlet", "neumann", "periodic")


def _spacing(spacing, ndim: int) -> tuple[float, ...]:
    h = np.broadcast_to(np.asarray(spacing, dtype=np.float64), (ndim,))
    if np.any(h <= 0):
        raise InvalidParameterError(f"grid spacing must be positive, got {spacing}")
    return tuple(float(v) for v in h)


def _laplacian_1d(n: int, h: float, bc: str) -> sp.csr_matrix:
    main = -2.0 * np.ones(n)
    off = np.ones(n - 1)
    L = sp.diags([off, main, off], [-1, 0, 1], format="lil")
    if bc == "periodic":
        L[0, n - 1] = 1.0
        L[n - 1, 0] = 1.0
    elif bc == "neumann":
        # ghost node u_{-1} = u_1 (zero gradient at the wall)
        L[0, 1] = 2.0
        L[n - 1, n - 2] = 2.0
    return (L / h**2).tocsr()


def laplacian_matrix(shape, spacing, bc: str = "dirichlet") -> sp.csr_matrix:
    """Second-order finite-difference Laplacian on a uniform grid, as a sparse matrix.

    Built as a Kronecker sum of 1D three-point stencils
    :math:`(u_{i+1} - 2u_i + u_{i-1})/\\Delta x^2`, acting on the grid
    flattened in C order.

    Parameters
    ----------
    shape : tuple of int
        Grid shape (1 to 3 axes).
    spacing : float or sequence of float
        Grid spacing, one value or one per axis.
    bc : {"dirichlet", "neumann", "periodic"}, default="dirichlet"
        ``"dirichlet"`` zeroes the rows of every boundary node, so those
        values stay fixed under :func:`heat_equation`; ``"neumann"`` imposes
        zero normal gradient (an insulated wall) with a mirrored ghost node;
        ``"periodic"`` wraps each axis.

    Returns
    -------
    scipy.sparse.csr_matrix, shape (N, N)
        With ``N = prod(shape)``.

    Examples
    --------
    >>> L = laplacian_matrix((5,), 1.0, bc="periodic")
    >>> L.toarray()[0].tolist()
    [-2.0, 1.0, 0.0, 0.0, 1.0]
    """
    if bc not in _BCS:
        raise InvalidParameterError(f"bc must be one of {_BCS}, got {bc!r}")
    shape = tuple(int(n) for n in shape)
    h = _spacing(spacing, len(shape))
    if any(n < 3 for n in shape):
        raise InvalidParameterError("every grid axis needs at least 3 points")
    total = sp.csr_matrix((int(np.prod(shape)), int(np.prod(shape))))
    for axis, (n, ha) in enumerate(zip(shape, h)):
        block = _laplacian_1d(n, ha, bc)
        before = int(np.prod(shape[:axis]))
        after = int(np.prod(shape[axis + 1 :]))
        total = total + sp.kron(sp.kron(sp.identity(before), block), sp.identity(after), format="csr")
    if bc == "dirichlet":
        interior = np.zeros(shape, dtype=bool)
        interior[tuple(slice(1, -1) for _ in shape)] = True
        total = sp.diags(interior.ravel().astype(np.float64)) @ total
    return total.tocsr()


def heat_equation(
    u0: ArrayLike,
    spacing,
    alpha: float,
    t_max: float,
    dt: float,
    method: str = "crank_nicolson",
    bc: str = "dirichlet",
    n_frames: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evolve a field under the heat equation with an explicit or Crank-Nicolson scheme.

    With :math:`L` the finite-difference Laplacian of
    :func:`laplacian_matrix`,

    - explicit (FTCS): :math:`u^{n+1} = u^n + \\alpha\\Delta t\\,L u^n`;
    - Crank-Nicolson: :math:`(I - \\tfrac12\\alpha\\Delta t L)u^{n+1} =
      (I + \\tfrac12\\alpha\\Delta t L)u^n`.

    Parameters
    ----------
    u0 : array_like
        Initial field on a 1D, 2D, or 3D grid. With ``bc="dirichlet"`` its
        boundary values are held fixed.
    spacing : float or sequence of float
        Grid spacing, one value or one per axis.
    alpha : float
        Diffusivity (thermal diffusivity :math:`k/\\rho c_p`, or a
        diffusion coefficient).
    t_max : float
        Final time, rounded to a whole number of steps per frame.
    dt : float
        Time step.
    method : {"crank_nicolson", "explicit"}, default="crank_nicolson"
        Time-stepping scheme.
    bc : {"dirichlet", "neumann", "periodic"}, default="dirichlet"
        Boundary condition on every face.
    n_frames : int, default=1
        Number of recorded frames after the initial one.

    Returns
    -------
    times : ndarray, shape (n_frames + 1,)
    frames : ndarray, shape (n_frames + 1, *u0.shape)

    Raises
    ------
    InvalidParameterError
        If ``alpha`` or ``dt`` is not positive, the method or boundary
        condition is unknown, or the explicit scheme's stability bound
        :math:`\\alpha\\Delta t\\sum_a \\Delta x_a^{-2} \\le 1/2` is violated.

    Examples
    --------
    A sine mode between cold walls decays as :math:`e^{-\\alpha k^2 t}`, up
    to the grid's :math:`O(k^2\\Delta x^2)` error in the decay rate:

    >>> x = np.linspace(0.0, 1.0, 101)
    >>> t, u = heat_equation(np.sin(np.pi * x), 0.01, alpha=0.1, t_max=1.0, dt=0.01)
    >>> round(float(u[-1, 50] / np.exp(-0.1 * np.pi**2)), 3)
    1.0
    """
    if alpha <= 0 or dt <= 0:
        raise InvalidParameterError("alpha and dt must be positive")
    u = np.array(u0, dtype=np.float64)
    h = _spacing(spacing, u.ndim)
    L = laplacian_matrix(u.shape, h, bc)
    stride = max(1, int(round(t_max / (n_frames * dt))))
    frames = np.empty((n_frames + 1,) + u.shape)
    frames[0] = u
    flat = u.ravel()
    if method == "explicit":
        r = alpha * dt * sum(1.0 / hk**2 for hk in h)
        if r > 0.5 + 1e-12:
            raise InvalidParameterError(f"explicit scheme unstable: alpha*dt*sum(1/dx^2) = {r:.3g} > 1/2; reduce dt or use crank_nicolson")
        A = (sp.identity(flat.size) + alpha * dt * L).tocsr()
        for f in range(n_frames):
            for _ in range(stride):
                flat = A @ flat
            frames[f + 1] = flat.reshape(u.shape)
    elif method == "crank_nicolson":
        eye = sp.identity(flat.size, format="csc")
        lhs = splu((eye - 0.5 * alpha * dt * L).tocsc())
        rhs = (eye + 0.5 * alpha * dt * L).tocsr()
        for f in range(n_frames):
            for _ in range(stride):
                flat = lhs.solve(rhs @ flat)
            frames[f + 1] = flat.reshape(u.shape)
    else:
        raise InvalidParameterError(f"method must be 'explicit' or 'crank_nicolson', got {method!r}")
    return np.arange(n_frames + 1, dtype=np.float64) * stride * dt, frames


def heat_equation_spectral(u0: ArrayLike, spacing, alpha: float, times: ArrayLike) -> NDArray[np.float64]:
    """Exact solution of the periodic heat equation by Fourier modes.

    .. math::

        \\hat u(\\mathbf k, t) = \\hat u(\\mathbf k, 0)\\,e^{-\\alpha |\\mathbf k|^2 t}

    Each Fourier mode decays independently at its own rate, with no time
    stepping and no time-step error: Fourier's 1807/1822 solution method.

    Parameters
    ----------
    u0 : array_like
        Initial field on a periodic grid of any dimension.
    spacing : float or sequence of float
        Grid spacing, one value or one per axis (the period along each
        axis is ``n * spacing``).
    alpha : float
        Diffusivity.
    times : array_like of float
        Times at which to return the field.

    Returns
    -------
    ndarray, shape (len(times), *u0.shape)

    Examples
    --------
    The mean is conserved while a mode of wavenumber :math:`k` decays:

    >>> x = np.arange(64) * (2 * np.pi / 64)
    >>> u = heat_equation_spectral(1.0 + np.cos(3 * x), 2 * np.pi / 64, 0.5, [0.0, 1.0])
    >>> round(float(u[1].mean()), 12), round(float(u[1, 0] - 1.0), 6), round(float(np.exp(-4.5)), 6)
    (1.0, 0.011109, 0.011109)
    """
    if alpha <= 0:
        raise InvalidParameterError(f"alpha must be positive, got {alpha}")
    u = np.asarray(u0, dtype=np.float64)
    h = _spacing(spacing, u.ndim)
    k2 = np.zeros(u.shape)
    for axis, (n, hk) in enumerate(zip(u.shape, h)):
        k = 2.0 * np.pi * np.fft.fftfreq(n, d=hk)
        shape = [1] * u.ndim
        shape[axis] = n
        k2 = k2 + k.reshape(shape) ** 2
    u_hat = np.fft.fftn(u)
    times = np.atleast_1d(np.asarray(times, dtype=np.float64))
    return np.stack([np.fft.ifftn(u_hat * np.exp(-alpha * k2 * t)).real for t in times])


def gaussian_heat_solution(coords, t: float, alpha: float, sigma0: float, amplitude: float = 1.0, center=None) -> NDArray[np.float64]:
    """Closed-form spreading of a Gaussian under the heat equation in free space.

    A Gaussian of width :math:`\\sigma_0` convolved with the heat kernel stays
    Gaussian, with :math:`\\sigma^2(t) = \\sigma_0^2 + 2\\alpha t` and its peak
    lowered so that the total heat is conserved:

    .. math::

        u(\\mathbf x, t) = A\\left(\\frac{\\sigma_0^2}{\\sigma^2(t)}\\right)^{d/2}
        \\exp\\!\\left(-\\frac{|\\mathbf x - \\mathbf x_0|^2}{2\\sigma^2(t)}\\right)

    in :math:`d` dimensions (e.g. Crank, *The Mathematics of Diffusion*, 2nd
    ed., 1975, eq. 3.5a).

    Parameters
    ----------
    coords : ndarray or sequence of ndarray
        One coordinate array (1D) or one per axis (e.g. from
        :func:`numpy.meshgrid` with ``indexing="ij"``).
    t : float
        Time.
    alpha : float
        Diffusivity.
    sigma0 : float
        Initial width.
    amplitude : float, default=1.0
        Initial peak value :math:`A`.
    center : sequence of float, optional
        Center :math:`\\mathbf x_0`; the origin by default.

    Returns
    -------
    ndarray
        :math:`u(\\mathbf x, t)`.

    Examples
    --------
    >>> float(gaussian_heat_solution(np.array([0.0]), t=1.5, alpha=1.0, sigma0=1.0)[0])
    0.5
    """
    if isinstance(coords, np.ndarray):
        coords = (coords,)
    coords = [np.asarray(c, dtype=np.float64) for c in coords]
    center = np.zeros(len(coords)) if center is None else np.asarray(center, dtype=np.float64)
    r2 = sum((c - c0) ** 2 for c, c0 in zip(coords, center))
    s2 = sigma0**2 + 2.0 * alpha * t
    return amplitude * (sigma0**2 / s2) ** (len(coords) / 2.0) * np.exp(-r2 / (2.0 * s2))
