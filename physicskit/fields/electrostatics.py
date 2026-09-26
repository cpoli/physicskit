"""Electrostatics: Coulomb's law, Poisson's equation, the method of images, and multipoles.

Everything here solves (or sidesteps solving) Poisson's equation for the
electrostatic potential,

.. math::

    \\nabla^2 \\phi = -\\rho / \\varepsilon_0, \\qquad \\mathbf E = -\\nabla\\phi,

in SI units, like the rest of :mod:`physicskit.fields` (:data:`EPS0` is the
vacuum permittivity):

- :func:`coulomb_potential` / :func:`coulomb_field` -- the free-space Green's
  function :math:`1/(4\\pi\\varepsilon_0 |\\mathbf r - \\mathbf r'|)` summed
  over point charges (Coulomb 1785, plus superposition).
- :func:`solve_poisson` -- Jacobi or successive-over-relaxation (SOR)
  iteration of the second-order finite-difference Laplacian on a 2D or 3D
  grid, with Dirichlet values on the box faces and on any interior
  "conductor" cells (Press et al., *Numerical Recipes*, 3rd ed., §20.5).
- :func:`solve_poisson_fft` -- a direct spectral solve, periodic (FFT) or
  grounded-box (type-I discrete sine transform).
- :func:`image_charges_plane` / :func:`image_charges_sphere` -- Green's
  functions for a grounded plane and a sphere by the method of images
  (Jackson, *Classical Electrodynamics*, 3rd ed., §2.1-2.4).
- :func:`multipole_moments` / :func:`multipole_potential` -- the Cartesian
  multipole expansion to quadrupole order (Jackson §4.1, eqs. 4.9-4.10).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.fft
from numba import njit

from physicskit.fields.electrodynamics import EPS0

__all__ = [
    "PoissonSolution",
    "MultipoleMoments",
    "coulomb_potential",
    "coulomb_field",
    "solve_poisson",
    "solve_poisson_fft",
    "electric_field_from_potential",
    "image_charges_plane",
    "image_charges_sphere",
    "induced_charge_density_plane",
    "induced_charge_density_sphere",
    "multipole_moments",
    "multipole_potential",
    "dipole_field",
]

# ---------------------------------------------------------------------------
# Coulomb's law
# ---------------------------------------------------------------------------


def _as_charge_arrays(charges, positions) -> tuple[np.ndarray, np.ndarray]:
    q = np.atleast_1d(np.asarray(charges, dtype=np.float64))
    pos = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    if pos.shape[0] != q.shape[0]:
        raise ValueError(f"got {q.shape[0]} charges but {pos.shape[0]} positions")
    return q, pos


def coulomb_potential(charges, positions, points, eps: float = EPS0) -> np.ndarray:
    """Electrostatic potential of a set of point charges (Coulomb's law plus superposition).

    .. math::

        \\phi(\\mathbf r) = \\frac{1}{4\\pi\\varepsilon}\\sum_i
        \\frac{q_i}{|\\mathbf r - \\mathbf r_i|}

    (Jackson, *Classical Electrodynamics*, 3rd ed., eq. 1.17).

    Parameters
    ----------
    charges : array_like, shape (n,)
        Point charges in coulombs.
    positions : array_like, shape (n, d)
        Charge positions in meters, ``d`` = 2 or 3 (2D positions are points
        in the plane ``z = 0`` of ordinary 3D space).
    points : array_like, shape (..., d)
        Observation points.
    eps : float, default=EPS0
        Permittivity of the surrounding medium.

    Returns
    -------
    ndarray, shape (...)
        Potential in volts; ``inf`` exactly at a charge.

    Examples
    --------
    >>> phi = coulomb_potential([1e-9], [[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]])
    >>> round(float(phi[0]), 3)  # 1 nC at 1 m: about 8.988 V
    8.988
    """
    q, pos = _as_charge_arrays(charges, positions)
    pts = np.asarray(points, dtype=np.float64)
    r = np.linalg.norm(pts[..., None, :] - pos, axis=-1)
    with np.errstate(divide="ignore"):
        return np.sum(q / r, axis=-1) / (4.0 * np.pi * eps)


def coulomb_field(charges, positions, points, eps: float = EPS0) -> np.ndarray:
    """Electric field of a set of point charges (Coulomb's inverse-square law).

    .. math::

        \\mathbf E(\\mathbf r) = \\frac{1}{4\\pi\\varepsilon}\\sum_i
        q_i \\frac{\\mathbf r - \\mathbf r_i}{|\\mathbf r - \\mathbf r_i|^3}

    (Jackson eq. 1.5).

    Parameters
    ----------
    charges : array_like, shape (n,)
        Point charges in coulombs.
    positions : array_like, shape (n, d)
        Charge positions in meters.
    points : array_like, shape (..., d)
        Observation points.
    eps : float, default=EPS0
        Permittivity of the surrounding medium.

    Returns
    -------
    ndarray, shape (..., d)
        Electric field in V/m; ``nan`` exactly at a charge.

    Examples
    --------
    >>> E = coulomb_field([1e-9], [[0.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]])
    >>> round(float(E[0, 0]), 4)  # a quarter of the field at 1 m
    2.2469
    """
    q, pos = _as_charge_arrays(charges, positions)
    pts = np.asarray(points, dtype=np.float64)
    sep = pts[..., None, :] - pos
    r = np.linalg.norm(sep, axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sum(q[:, None] * sep / r**3, axis=-2) / (4.0 * np.pi * eps)


# ---------------------------------------------------------------------------
# Poisson's equation on a grid
# ---------------------------------------------------------------------------


class PoissonSolution(NamedTuple):
    """Result of :func:`solve_poisson`.

    Attributes
    ----------
    phi : ndarray
        Converged (or last) potential on the grid, in volts.
    n_iter : int
        Number of sweeps performed.
    converged : bool
        Whether the relative update fell below ``tol`` within ``max_iter``.
    """

    phi: np.ndarray
    n_iter: int
    converged: bool


@njit(cache=True)
def _relax_2d(phi, src, fixed, cx, cy, omega, jacobi, tol, max_iter):
    nx, ny = phi.shape
    diag = 2.0 * (cx + cy)
    old = phi.copy()
    scale = max(np.max(np.abs(phi)), 1e-300)
    for it in range(1, max_iter + 1):
        if jacobi:
            old[:, :] = phi
        max_delta = 0.0
        max_phi = scale
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                if fixed[i, j]:
                    continue
                if jacobi:
                    new = (cx * (old[i + 1, j] + old[i - 1, j]) + cy * (old[i, j + 1] + old[i, j - 1]) + src[i, j]) / diag
                else:
                    gs = (cx * (phi[i + 1, j] + phi[i - 1, j]) + cy * (phi[i, j + 1] + phi[i, j - 1]) + src[i, j]) / diag
                    new = phi[i, j] + omega * (gs - phi[i, j])
                delta = abs(new - phi[i, j])
                if delta > max_delta:
                    max_delta = delta
                phi[i, j] = new
                if abs(new) > max_phi:
                    max_phi = abs(new)
        if max_delta <= tol * max_phi:
            return it, True
    return max_iter, False


@njit(cache=True)
def _relax_3d(phi, src, fixed, cx, cy, cz, omega, jacobi, tol, max_iter):
    nx, ny, nz = phi.shape
    diag = 2.0 * (cx + cy + cz)
    old = phi.copy()
    scale = max(np.max(np.abs(phi)), 1e-300)
    for it in range(1, max_iter + 1):
        if jacobi:
            old[:, :, :] = phi
        max_delta = 0.0
        max_phi = scale
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    if fixed[i, j, k]:
                        continue
                    if jacobi:
                        new = (
                            cx * (old[i + 1, j, k] + old[i - 1, j, k])
                            + cy * (old[i, j + 1, k] + old[i, j - 1, k])
                            + cz * (old[i, j, k + 1] + old[i, j, k - 1])
                            + src[i, j, k]
                        ) / diag
                    else:
                        gs = (
                            cx * (phi[i + 1, j, k] + phi[i - 1, j, k])
                            + cy * (phi[i, j + 1, k] + phi[i, j - 1, k])
                            + cz * (phi[i, j, k + 1] + phi[i, j, k - 1])
                            + src[i, j, k]
                        ) / diag
                        new = phi[i, j, k] + omega * (gs - phi[i, j, k])
                    delta = abs(new - phi[i, j, k])
                    if delta > max_delta:
                        max_delta = delta
                    phi[i, j, k] = new
                    if abs(new) > max_phi:
                        max_phi = abs(new)
        if max_delta <= tol * max_phi:
            return it, True
    return max_iter, False


def _spacings(spacing, ndim: int) -> tuple[float, ...]:
    h = np.broadcast_to(np.asarray(spacing, dtype=np.float64), (ndim,))
    return tuple(float(v) for v in h)


def solve_poisson(
    rho,
    spacing,
    *,
    phi0=None,
    fixed=None,
    method: str = "sor",
    omega: float | None = None,
    tol: float = 1e-8,
    max_iter: int = 100_000,
    eps: float = EPS0,
) -> PoissonSolution:
    """Solve Poisson's equation on a 2D or 3D grid by Jacobi or SOR relaxation.

    Iterates the standard second-order finite-difference form of
    :math:`\\nabla^2\\phi = -\\rho/\\varepsilon`,

    .. math::

        \\phi_{ij}^{\\text{new}} = \\frac{(\\phi_{i+1,j}+\\phi_{i-1,j})/h_x^2
        + (\\phi_{i,j+1}+\\phi_{i,j-1})/h_y^2 + \\rho_{ij}/\\varepsilon}
        {2/h_x^2 + 2/h_y^2},

    either all at once from the previous sweep (Jacobi) or in place with
    over-relaxation :math:`\\phi \\leftarrow \\phi + \\omega(\\phi^{\\text{GS}} -
    \\phi)` (SOR; Press et al., *Numerical Recipes*, 3rd ed., §20.5). For an
    ``N x N`` grid SOR needs :math:`O(N)` sweeps at the optimal
    :math:`\\omega = 2/(1+\\sin(\\pi/N))`, against Jacobi's :math:`O(N^2)`.

    The outermost faces of the grid, and every cell where ``fixed`` is
    true, are Dirichlet cells held at their ``phi0`` values -- a grounded
    box by default, or electrodes/conductors at set voltages. In 2D the
    geometry is invariant along ``z``, so ``rho`` is still a volume density
    and point sources are really line charges.

    Parameters
    ----------
    rho : array_like, shape (nx, ny) or (nx, ny, nz)
        Charge density in C/m^3.
    spacing : float or sequence of float
        Grid spacing in meters, one value or one per axis.
    phi0 : array_like, optional
        Initial guess and Dirichlet values (same shape as ``rho``). Defaults
        to zero everywhere.
    fixed : array_like of bool, optional
        Interior cells to hold at ``phi0`` (conductors). The box faces are
        always held.
    method : {"sor", "jacobi"}, default="sor"
        Relaxation scheme.
    omega : float, optional
        SOR over-relaxation factor in ``(0, 2)``; defaults to
        ``2 / (1 + sin(pi / max(shape)))``. Ignored for Jacobi.
    tol : float, default=1e-8
        Stop once the largest single-sweep update is below ``tol`` times the
        largest ``|phi|``.
    max_iter : int, default=100000
        Maximum number of sweeps.
    eps : float, default=EPS0
        Permittivity of the medium.

    Returns
    -------
    PoissonSolution
        ``(phi, n_iter, converged)``.

    Examples
    --------
    Two plates at +-1 V, grounded box: the potential is odd about the
    midplane.

    >>> phi0 = np.zeros((41, 41))
    >>> fixed = np.zeros((41, 41), dtype=bool)
    >>> phi0[10:31, 15], phi0[10:31, 25] = 1.0, -1.0
    >>> fixed[10:31, 15] = fixed[10:31, 25] = True
    >>> sol = solve_poisson(np.zeros((41, 41)), 0.01, phi0=phi0, fixed=fixed)
    >>> sol.converged, round(float(sol.phi[20, 20]), 6)
    (True, 0.0)
    """
    src = np.asarray(rho, dtype=np.float64) / eps
    ndim = src.ndim
    if ndim not in (2, 3):
        raise ValueError(f"rho must be 2D or 3D, got ndim={ndim}")
    h = _spacings(spacing, ndim)
    phi = np.zeros_like(src) if phi0 is None else np.array(phi0, dtype=np.float64)
    if phi.shape != src.shape:
        raise ValueError(f"phi0 shape {phi.shape} does not match rho shape {src.shape}")
    mask = np.zeros(src.shape, dtype=np.bool_) if fixed is None else np.asarray(fixed, dtype=np.bool_)
    if method not in ("sor", "jacobi"):
        raise ValueError("method must be 'sor' or 'jacobi'")
    if omega is None:
        omega = 2.0 / (1.0 + np.sin(np.pi / max(src.shape)))
    if not 0.0 < omega < 2.0:
        raise ValueError(f"omega must lie in (0, 2), got {omega}")
    jacobi = method == "jacobi"
    coeffs = tuple(1.0 / hk**2 for hk in h)
    if ndim == 2:
        n_iter, converged = _relax_2d(phi, src, mask, *coeffs, float(omega), jacobi, float(tol), int(max_iter))
    else:
        n_iter, converged = _relax_3d(phi, src, mask, *coeffs, float(omega), jacobi, float(tol), int(max_iter))
    return PoissonSolution(phi, int(n_iter), bool(converged))


def solve_poisson_fft(rho, spacing, *, bc: str = "periodic", eps: float = EPS0) -> np.ndarray:
    """Solve Poisson's equation directly with a fast Fourier or sine transform.

    ``bc="periodic"`` treats the grid as one period of an infinite lattice
    and solves :math:`\\hat\\phi(\\mathbf k) = \\hat\\rho(\\mathbf k) /
    (\\varepsilon k^2)` with the exact continuum :math:`k^2` (spectral
    accuracy for smooth sources). The :math:`\\mathbf k = 0` mode is set to
    zero, which is the same as adding a uniform neutralizing background:
    only a net-neutral periodic system has a periodic potential.

    ``bc="dirichlet"`` holds every face of the box at :math:`\\phi = 0` (the
    grid's first and last nodes along each axis are the grounded walls) and
    diagonalizes the second-order finite-difference Laplacian with a
    type-I discrete sine transform, whose eigenvalues are
    :math:`\\sum_a (2 - 2\\cos(\\pi m_a/(n_a-1)))/h_a^2`. It therefore returns
    the exact solution of the same linear system :func:`solve_poisson`
    iterates towards with a zero ``phi0`` and no ``fixed`` cells, in one
    :math:`O(N\\log N)` step (Press et al., *Numerical Recipes*, 3rd ed.,
    §20.4).

    Parameters
    ----------
    rho : array_like
        Charge density in C/m^3, any number of dimensions.
    spacing : float or sequence of float
        Grid spacing in meters, one value or one per axis.
    bc : {"periodic", "dirichlet"}, default="periodic"
        Boundary condition, as described above.
    eps : float, default=EPS0
        Permittivity of the medium.

    Returns
    -------
    ndarray
        Potential in volts, same shape as ``rho``.

    Examples
    --------
    A sinusoidal charge wave has potential :math:`\\rho/(\\varepsilon k^2)`:

    >>> x = np.arange(64) * (2 * np.pi / 64)
    >>> phi = solve_poisson_fft(np.sin(x)[:, None] * np.ones((64, 8)), 2 * np.pi / 64, eps=1.0)
    >>> bool(np.allclose(phi[:, 0], np.sin(x)))
    True
    """
    src = np.asarray(rho, dtype=np.float64) / eps
    h = _spacings(spacing, src.ndim)
    if bc == "periodic":
        k2 = np.zeros(src.shape)
        for axis, (n, hk) in enumerate(zip(src.shape, h)):
            k = 2.0 * np.pi * np.fft.fftfreq(n, d=hk)
            shape = [1] * src.ndim
            shape[axis] = n
            k2 = k2 + k.reshape(shape) ** 2
        k2.flat[0] = 1.0
        phi_hat = np.fft.fftn(src) / k2
        phi_hat.flat[0] = 0.0
        return np.fft.ifftn(phi_hat).real
    if bc == "dirichlet":
        interior = src[tuple(slice(1, -1) for _ in range(src.ndim))]
        lam = np.zeros(interior.shape)
        for axis, (n, hk) in enumerate(zip(src.shape, h)):
            m = np.arange(1, n - 1)
            shape = [1] * src.ndim
            shape[axis] = n - 2
            lam = lam + ((2.0 - 2.0 * np.cos(np.pi * m / (n - 1))) / hk**2).reshape(shape)
        phi = np.zeros_like(src)
        phi[tuple(slice(1, -1) for _ in range(src.ndim))] = scipy.fft.idstn(scipy.fft.dstn(interior, type=1) / lam, type=1)
        return phi
    raise ValueError("bc must be 'periodic' or 'dirichlet'")


def electric_field_from_potential(phi, spacing) -> np.ndarray:
    """Electric field :math:`\\mathbf E = -\\nabla\\phi` from a gridded potential.

    Uses second-order central differences in the interior (one-sided at
    the faces), via :func:`numpy.gradient`.

    Parameters
    ----------
    phi : array_like
        Potential on the grid, in volts.
    spacing : float or sequence of float
        Grid spacing in meters, one value or one per axis.

    Returns
    -------
    ndarray, shape (phi.ndim, *phi.shape)
        Field components ``E[0]`` (along axis 0), ``E[1]``, ... in V/m.

    Examples
    --------
    >>> x = np.linspace(0.0, 1.0, 11)
    >>> E = electric_field_from_potential(np.outer(-5.0 * x, np.ones(4)), 0.1)
    >>> bool(np.allclose(E[0], 5.0)), bool(np.allclose(E[1], 0.0))
    (True, True)
    """
    phi = np.asarray(phi, dtype=np.float64)
    h = _spacings(spacing, phi.ndim)
    grads = np.gradient(phi, *h)
    if phi.ndim == 1:
        grads = [grads]
    return -np.stack(grads)


# ---------------------------------------------------------------------------
# Method of images (Green's functions for simple conductors)
# ---------------------------------------------------------------------------


def image_charges_plane(q: float, position) -> tuple[np.ndarray, np.ndarray]:
    """Image charge making the grounded conducting plane ``z = 0`` an equipotential.

    A charge :math:`q` at :math:`(x, y, d)` above a grounded plane sees,
    in the region :math:`z > 0`, exactly the potential of itself plus an
    image :math:`-q` at :math:`(x, y, -d)` (Jackson §2.1). The pair is
    Green's function for the half-space with Dirichlet boundary.

    Parameters
    ----------
    q : float
        The real charge, in coulombs.
    position : array_like, shape (3,)
        Its position ``(x, y, d)`` with ``d != 0``.

    Returns
    -------
    charges : ndarray, shape (1,)
        The image charge ``[-q]``.
    positions : ndarray, shape (1, 3)
        The image position ``[[x, y, -d]]``.

    Examples
    --------
    >>> qi, ri = image_charges_plane(1e-9, [0.0, 0.0, 0.5])
    >>> phi = coulomb_potential(np.r_[1e-9, qi], np.vstack([[0, 0, 0.5], ri]), [[0.3, -0.2, 0.0]])
    >>> abs(float(phi[0])) < 1e-12
    True
    """
    r0 = np.asarray(position, dtype=np.float64)
    if r0.shape != (3,) or r0[2] == 0.0:
        raise ValueError("position must be (x, y, d) with d != 0")
    return np.array([-q]), np.array([[r0[0], r0[1], -r0[2]]])


def image_charges_sphere(
    q: float,
    position,
    radius: float,
    center=(0.0, 0.0, 0.0),
    sphere_charge: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Image charge(s) for a point charge and a conducting sphere.

    For a sphere of radius :math:`R` and a charge :math:`q` at distance
    :math:`a` from its center, the image

    .. math::

        q' = -q\\frac{R}{a} \\quad\\text{at distance}\\quad \\frac{R^2}{a}

    along the same ray makes the sphere's surface an equipotential at zero
    potential (Jackson §2.2, eq. 2.4). This holds with the charge outside
    (:math:`a > R`, image inside) or inside a spherical cavity
    (:math:`a < R`, image outside). For an *isolated* sphere carrying total
    charge :math:`Q` (charge outside only), a second image
    :math:`Q - q'` at the center restores the net charge while keeping the
    surface an equipotential (Jackson §2.3).

    Parameters
    ----------
    q : float
        The real charge, in coulombs.
    position : array_like, shape (3,)
        Its position, in meters.
    radius : float
        Sphere radius ``R``.
    center : array_like, shape (3,), default=(0, 0, 0)
        Sphere center.
    sphere_charge : float, optional
        Total charge of an isolated sphere. ``None`` (default) means the
        sphere is grounded.

    Returns
    -------
    charges : ndarray, shape (1,) or (2,)
        Image charge(s).
    positions : ndarray, shape (1, 3) or (2, 3)
        Image position(s).

    Examples
    --------
    >>> qi, ri = image_charges_sphere(1.0, [2.0, 0.0, 0.0], radius=1.0)
    >>> float(qi[0]), float(ri[0, 0])
    (-0.5, 0.5)
    """
    r0 = np.asarray(position, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    rel = r0 - c
    a = float(np.linalg.norm(rel))
    if a == 0.0 or np.isclose(a, radius):
        raise ValueError("the charge must not sit at the center or on the sphere")
    q_img = -q * radius / a
    charges = [q_img]
    positions = [c + rel * (radius**2 / a**2)]
    if sphere_charge is not None:
        if a < radius:
            raise ValueError("sphere_charge applies only to a charge outside the sphere")
        charges.append(sphere_charge - q_img)
        positions.append(c.copy())
    return np.array(charges), np.array(positions)


def induced_charge_density_plane(q: float, d: float, rho) -> np.ndarray:
    """Surface charge induced on the grounded plane ``z = 0`` by a charge at height ``d``.

    .. math::

        \\sigma(\\varrho) = -\\frac{q\\, d}{2\\pi(\\varrho^2 + d^2)^{3/2}}

    (Griffiths, *Introduction to Electrodynamics*, 4th ed., eq. 3.10),
    which integrates to exactly :math:`-q` over the plane.

    Parameters
    ----------
    q : float
        The real charge, in coulombs.
    d : float
        Its height above the plane, in meters.
    rho : array_like
        Distance along the plane from the foot of the perpendicular.

    Returns
    -------
    ndarray
        Surface charge density in C/m^2.

    Examples
    --------
    >>> round(float(induced_charge_density_plane(1.0, 1.0, 0.0) * 2 * np.pi), 12)
    -1.0
    """
    rho = np.asarray(rho, dtype=np.float64)
    return -q * d / (2.0 * np.pi * (rho**2 + d**2) ** 1.5)


def induced_charge_density_sphere(q: float, a: float, radius: float, theta) -> np.ndarray:
    """Surface charge induced on a grounded sphere by an outside point charge.

    .. math::

        \\sigma(\\gamma) = -\\frac{q\\,(a^2 - R^2)}{4\\pi R\\,(R^2 + a^2 - 2aR\\cos\\gamma)^{3/2}}

    (Jackson eq. 2.5, rewritten in the sphere radius :math:`R` and charge
    distance :math:`a`), where :math:`\\gamma` is the angle from the ray
    through the charge. It integrates to the image charge :math:`-qR/a`.

    Parameters
    ----------
    q : float
        The real charge, in coulombs.
    a : float
        Distance of the charge from the center, ``a > radius``.
    radius : float
        Sphere radius ``R``.
    theta : array_like
        Polar angle(s) ``gamma`` on the sphere, measured from the charge's
        direction.

    Returns
    -------
    ndarray
        Surface charge density in C/m^2.

    Examples
    --------
    >>> from scipy.integrate import trapezoid
    >>> th = np.linspace(0.0, np.pi, 20001)
    >>> sigma = induced_charge_density_sphere(1.0, 2.0, 1.0, th)
    >>> total = trapezoid(sigma * 2 * np.pi * np.sin(th), th)
    >>> round(float(total), 6)  # the image charge -q R / a
    -0.5
    """
    theta = np.asarray(theta, dtype=np.float64)
    return -q * (a**2 - radius**2) / (4.0 * np.pi * radius * (radius**2 + a**2 - 2.0 * a * radius * np.cos(theta)) ** 1.5)


# ---------------------------------------------------------------------------
# Multipole expansion
# ---------------------------------------------------------------------------


class MultipoleMoments(NamedTuple):
    """Cartesian multipole moments of a charge distribution about ``origin``.

    Attributes
    ----------
    monopole : float
        Total charge :math:`q = \\sum_i q_i`.
    dipole : ndarray, shape (3,)
        Dipole moment :math:`\\mathbf p = \\sum_i q_i \\mathbf x_i`.
    quadrupole : ndarray, shape (3, 3)
        Traceless quadrupole tensor
        :math:`Q_{jk} = \\sum_i q_i (3x_{ij}x_{ik} - r_i^2\\delta_{jk})`.
    origin : ndarray, shape (3,)
        Expansion center the moments are taken about.
    """

    monopole: float
    dipole: np.ndarray
    quadrupole: np.ndarray
    origin: np.ndarray


def _to_3d(arr: np.ndarray) -> np.ndarray:
    if arr.shape[-1] == 3:
        return arr
    if arr.shape[-1] == 2:
        return np.concatenate([arr, np.zeros(arr.shape[:-1] + (1,))], axis=-1)
    raise ValueError("positions must have 2 or 3 components")


def multipole_moments(charges, positions, origin=(0.0, 0.0, 0.0)) -> MultipoleMoments:
    """Monopole, dipole and (traceless) quadrupole moments of a charge distribution.

    Uses Jackson's conventions (§4.1, eq. 4.9):

    .. math::

        q = \\sum_i q_i, \\quad
        \\mathbf p = \\sum_i q_i \\mathbf x_i, \\quad
        Q_{jk} = \\sum_i q_i\\,(3 x_{ij} x_{ik} - r_i^2 \\delta_{jk}),

    with positions measured from ``origin``. A continuous density on a grid
    is handled the same way: pass ``charges = rho * dV`` and the cell
    centers as ``positions``.

    Parameters
    ----------
    charges : array_like, shape (n,)
        Point charges, in coulombs.
    positions : array_like, shape (n, 3) or (n, 2)
        Positions in meters.
    origin : array_like, shape (3,), default=(0, 0, 0)
        Expansion center.

    Returns
    -------
    MultipoleMoments

    Examples
    --------
    A physical dipole +-q separated by d along z:

    >>> m = multipole_moments([1.0, -1.0], [[0, 0, 0.5], [0, 0, -0.5]])
    >>> float(m.monopole), m.dipole.tolist()
    (0.0, [0.0, 0.0, 1.0])
    """
    q, pos = _as_charge_arrays(charges, positions)
    c = np.asarray(origin, dtype=np.float64)
    x = _to_3d(pos) - c
    r2 = np.sum(x**2, axis=1)
    monopole = float(np.sum(q))
    dipole = q @ x
    quad = 3.0 * np.einsum("i,ij,ik->jk", q, x, x) - np.sum(q * r2) * np.eye(3)
    return MultipoleMoments(monopole, dipole, quad, c)


def multipole_potential(moments: MultipoleMoments, points, order: int = 2, eps: float = EPS0) -> np.ndarray:
    """Far-field potential from a truncated multipole expansion.

    .. math::

        \\phi(\\mathbf x) = \\frac{1}{4\\pi\\varepsilon}\\left[\\frac{q}{r}
        + \\frac{\\mathbf p\\cdot\\mathbf x}{r^3}
        + \\frac{1}{2}\\sum_{jk} Q_{jk}\\frac{x_j x_k}{r^5} + \\cdots\\right]

    (Jackson eq. 4.10), with :math:`\\mathbf x` measured from
    ``moments.origin``. The error of the order-:math:`\\ell` truncation falls
    off as :math:`r^{-(\\ell+2)}`.

    Parameters
    ----------
    moments : MultipoleMoments
        From :func:`multipole_moments`.
    points : array_like, shape (..., 3) or (..., 2)
        Observation points.
    order : {0, 1, 2}, default=2
        Highest multipole kept: monopole, dipole, or quadrupole.
    eps : float, default=EPS0
        Permittivity of the medium.

    Returns
    -------
    ndarray, shape (...)
        Potential in volts.

    Examples
    --------
    >>> m = multipole_moments([1.0, -1.0], [[0, 0, 0.5], [0, 0, -0.5]])
    >>> far = [[0.0, 0.0, 100.0]]
    >>> approx = multipole_potential(m, far, order=1, eps=1.0)
    >>> exact = coulomb_potential([1.0, -1.0], [[0, 0, 0.5], [0, 0, -0.5]], far, eps=1.0)
    >>> bool(abs(approx[0] / exact[0] - 1) < 1e-4)
    True
    """
    if order not in (0, 1, 2):
        raise ValueError("order must be 0, 1 or 2")
    x = _to_3d(np.asarray(points, dtype=np.float64)) - moments.origin
    r = np.linalg.norm(x, axis=-1)
    phi = moments.monopole / r
    if order >= 1:
        phi = phi + (x @ moments.dipole) / r**3
    if order >= 2:
        phi = phi + 0.5 * np.einsum("...j,jk,...k->...", x, moments.quadrupole, x) / r**5
    return phi / (4.0 * np.pi * eps)


def dipole_field(p, points, origin=(0.0, 0.0, 0.0), eps: float = EPS0) -> np.ndarray:
    """Electric field of an ideal point dipole.

    .. math::

        \\mathbf E(\\mathbf x) = \\frac{1}{4\\pi\\varepsilon}
        \\frac{3\\hat{\\mathbf n}(\\mathbf p\\cdot\\hat{\\mathbf n}) - \\mathbf p}{r^3}

    (Jackson eq. 4.13, away from the origin). On the dipole axis this is
    :math:`2p/(4\\pi\\varepsilon r^3)`, twice the equatorial magnitude.

    Parameters
    ----------
    p : array_like, shape (3,)
        Dipole moment in C m.
    points : array_like, shape (..., 3)
        Observation points.
    origin : array_like, shape (3,), default=(0, 0, 0)
        Dipole location.
    eps : float, default=EPS0
        Permittivity of the medium.

    Returns
    -------
    ndarray, shape (..., 3)
        Field in V/m.

    Examples
    --------
    >>> E = dipole_field([0, 0, 1.0], [[0, 0, 1.0], [1.0, 0, 0]], eps=1 / (4 * np.pi))
    >>> E.round(12).tolist()
    [[0.0, 0.0, 2.0], [0.0, 0.0, -1.0]]
    """
    p = np.asarray(p, dtype=np.float64)
    x = np.asarray(points, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    r = np.linalg.norm(x, axis=-1, keepdims=True)
    n = x / r
    return (3.0 * n * np.sum(n * p, axis=-1, keepdims=True) - p) / (4.0 * np.pi * eps * r**3)
