"""Two-dimensional linear elasticity: plane stress and plane strain by finite elements.

A linear elastic solid in equilibrium satisfies
:math:`\\nabla\\cdot\\boldsymbol\\sigma + \\mathbf f = 0` with Hooke's law
:math:`\\boldsymbol\\sigma = \\lambda\\,\\mathrm{tr}(\\boldsymbol\\varepsilon)\\mathbf I
+ 2\\mu\\boldsymbol\\varepsilon` (Navier 1821, Cauchy 1822). In 2D two
idealizations are standard:

- *plane stress* -- a thin plate loaded in its plane,
  :math:`\\sigma_{zz} = 0`;
- *plane strain* -- a long body loaded uniformly along its length (a
  pipe, a dam), :math:`\\varepsilon_{zz} = 0`.

:func:`plane_elasticity_solve` discretizes either with linear
(constant-strain) triangles, the element of Turner, Clough, Martin and Topp
(1956) that started the finite element method, assembled into a sparse
stiffness matrix. :func:`lame_thick_cylinder` gives Lamé's (1852)
closed-form stresses in a pressurized thick-walled cylinder, the standard
benchmark. Units are whatever the caller uses consistently (e.g. Pa and m).
This module sits in :mod:`physicskit.fluids` with the package's other
continuum-mechanics solvers (heat conduction and acoustics), although the
material here is solid.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.linalg import spsolve

from physicskit.fluids.exceptions import InvalidParameterError

__all__ = [
    "PlaneElasticityResult",
    "lame_parameters",
    "elasticity_matrix",
    "rectangle_mesh",
    "annulus_mesh",
    "pressure_load",
    "plane_elasticity_solve",
    "lame_thick_cylinder",
    "polar_stress",
]


def lame_parameters(E: float, nu: float) -> tuple[float, float]:
    """Lamé's first parameter :math:`\\lambda` and shear modulus :math:`\\mu` from :math:`E` and :math:`\\nu`.

    .. math::

        \\lambda = \\frac{E\\nu}{(1+\\nu)(1-2\\nu)}, \\qquad \\mu = \\frac{E}{2(1+\\nu)}

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio, in ``(-1, 1/2)``.

    Returns
    -------
    lam, mu : float

    Examples
    --------
    >>> lame_parameters(2.5, 0.25)
    (1.0, 1.0)
    """
    if E <= 0 or not -1.0 < nu < 0.5:
        raise InvalidParameterError(f"need E > 0 and -1 < nu < 1/2, got E={E}, nu={nu}")
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


def elasticity_matrix(E: float, nu: float, plane: str = "stress") -> NDArray[np.float64]:
    """Constitutive matrix :math:`D` with :math:`(\\sigma_{xx}, \\sigma_{yy}, \\sigma_{xy}) = D(\\varepsilon_{xx}, \\varepsilon_{yy}, \\gamma_{xy})`.

    Plane stress:

    .. math::

        D = \\frac{E}{1-\\nu^2}\\begin{pmatrix}1&\\nu&0\\\\ \\nu&1&0\\\\ 0&0&\\tfrac{1-\\nu}{2}\\end{pmatrix}.

    Plane strain:

    .. math::

        D = \\frac{E}{(1+\\nu)(1-2\\nu)}\\begin{pmatrix}1-\\nu&\\nu&0\\\\ \\nu&1-\\nu&0\\\\ 0&0&\\tfrac{1-2\\nu}{2}\\end{pmatrix}.

    Engineering shear strain :math:`\\gamma_{xy} = 2\\varepsilon_{xy}`
    (Zienkiewicz and Taylor, *The Finite Element Method*, 5th ed., vol. 1,
    §4.2).

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    plane : {"stress", "strain"}, default="stress"

    Returns
    -------
    ndarray, shape (3, 3)

    Examples
    --------
    >>> elasticity_matrix(1.0, 0.0).tolist()
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]]
    """
    lame_parameters(E, nu)  # validates
    if plane == "stress":
        return E / (1 - nu**2) * np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1 - nu)]])
    if plane == "strain":
        return E / ((1 + nu) * (1 - 2 * nu)) * np.array([[1 - nu, nu, 0.0], [nu, 1 - nu, 0.0], [0.0, 0.0, 0.5 * (1 - 2 * nu)]])
    raise InvalidParameterError(f"plane must be 'stress' or 'strain', got {plane!r}")


def rectangle_mesh(Lx: float, Ly: float, nx: int, ny: int) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Structured triangulation of the rectangle :math:`[0, L_x]\\times[0, L_y]`.

    Each of the ``nx * ny`` cells is split into two counter-clockwise
    triangles.

    Parameters
    ----------
    Lx, Ly : float
        Rectangle size.
    nx, ny : int
        Number of cells along each axis.

    Returns
    -------
    nodes : ndarray, shape ((nx + 1) * (ny + 1), 2)
        Node coordinates; node ``i * (ny + 1) + j`` is at ``(i Lx/nx, j Ly/ny)``.
    triangles : ndarray of int, shape (2 * nx * ny, 3)

    Examples
    --------
    >>> nodes, tris = rectangle_mesh(2.0, 1.0, 2, 1)
    >>> nodes.shape, tris.shape
    ((6, 2), (4, 3))
    """
    x = np.linspace(0.0, Lx, nx + 1)
    y = np.linspace(0.0, Ly, ny + 1)
    X, Y = np.meshgrid(x, y, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel()])
    return nodes, _grid_triangles(nx, ny)


def _grid_triangles(n1: int, n2: int) -> NDArray[np.int64]:
    i, j = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")
    a = (i * (n2 + 1) + j).ravel()
    b = a + (n2 + 1)
    c = b + 1
    d = a + 1
    return np.concatenate([np.column_stack([a, b, c]), np.column_stack([a, c, d])]).astype(np.int64)


def annulus_mesh(inner_radius: float, outer_radius: float, n_r: int, n_theta: int, theta_max: float = 0.5 * np.pi):
    """Structured triangulation of an annular sector :math:`a \\le r \\le b`, :math:`0 \\le \\theta \\le \\theta_{\\max}`.

    A quarter annulus (the default) with symmetry conditions on its two
    straight edges models a full pressurized cylinder at a quarter of the
    cost.

    Parameters
    ----------
    inner_radius, outer_radius : float
        Radii :math:`a < b`.
    n_r, n_theta : int
        Number of cells radially and around.
    theta_max : float, default=pi/2
        Opening angle of the sector.

    Returns
    -------
    nodes : ndarray, shape ((n_r + 1) * (n_theta + 1), 2)
    triangles : ndarray of int, shape (2 * n_r * n_theta, 3)
    boundaries : dict
        ``"inner"`` and ``"outer"``: arrays of shape ``(n_theta, 2)`` of
        boundary edges, each directed with the solid on its right (the
        convention of :func:`pressure_load`); ``"theta0"`` and
        ``"theta_max"``: node indices on the two straight edges.

    Examples
    --------
    >>> nodes, tris, bnd = annulus_mesh(1.0, 2.0, 4, 8)
    >>> len(nodes), len(tris), bnd["inner"].shape
    (45, 64, (8, 2))
    """
    if not 0 < inner_radius < outer_radius:
        raise InvalidParameterError("need 0 < inner_radius < outer_radius")
    r = np.linspace(inner_radius, outer_radius, n_r + 1)
    th = np.linspace(0.0, theta_max, n_theta + 1)
    R, T = np.meshgrid(r, th, indexing="ij")
    nodes = np.column_stack([(R * np.cos(T)).ravel(), (R * np.sin(T)).ravel()])
    tris = _grid_triangles(n_r, n_theta)
    idx = np.arange(nodes.shape[0]).reshape(n_r + 1, n_theta + 1)
    inner = np.column_stack([idx[0, :-1], idx[0, 1:]])  # counter-clockwise: solid (larger r) on the right
    outer = np.column_stack([idx[-1, 1:], idx[-1, :-1]])  # clockwise: solid (smaller r) on the right
    return nodes, tris, {"inner": inner, "outer": outer, "theta0": idx[:, 0], "theta_max": idx[:, -1]}


def pressure_load(nodes: ArrayLike, edges: ArrayLike, pressure: float, thickness: float = 1.0) -> NDArray[np.float64]:
    """Consistent nodal forces from a uniform pressure on boundary edges.

    Each directed edge :math:`a \\to b` carries a normal traction of
    magnitude ``pressure`` pushing to its right, i.e. into the solid when
    edges are listed with the solid on their right (as
    :func:`annulus_mesh` returns them). For a linear element the traction
    splits equally between the two end nodes: each receives
    :math:`\\tfrac12 p\\,t\\,\\ell\\,\\hat{\\mathbf n}`.

    Parameters
    ----------
    nodes : array_like, shape (n_nodes, 2)
    edges : array_like of int, shape (n_edges, 2)
    pressure : float
    thickness : float, default=1.0

    Returns
    -------
    ndarray, shape (n_nodes, 2)

    Examples
    --------
    >>> F = pressure_load([[0, 0], [1, 0]], [[0, 1]], 2.0)
    >>> F.tolist()  # pushes toward -y, the right of the edge direction +x
    [[0.0, -1.0], [0.0, -1.0]]
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)
    d = nodes[edges[:, 1]] - nodes[edges[:, 0]]
    right_normal_times_length = np.column_stack([d[:, 1], -d[:, 0]])
    f = 0.5 * pressure * thickness * right_normal_times_length
    F = np.zeros_like(nodes)
    np.add.at(F, edges[:, 0], f)
    np.add.at(F, edges[:, 1], f)
    return F


class PlaneElasticityResult(NamedTuple):
    """Solution of :func:`plane_elasticity_solve`.

    Attributes
    ----------
    displacement : ndarray, shape (n_nodes, 2)
        Nodal displacements :math:`(u_x, u_y)`.
    strain : ndarray, shape (n_triangles, 3)
        Per-element :math:`(\\varepsilon_{xx}, \\varepsilon_{yy}, \\gamma_{xy})`.
    stress : ndarray, shape (n_triangles, 3)
        Per-element :math:`(\\sigma_{xx}, \\sigma_{yy}, \\sigma_{xy})`.
    centroids : ndarray, shape (n_triangles, 2)
        Element centroids, where the (constant) strain and stress apply best.
    """

    displacement: np.ndarray
    strain: np.ndarray
    stress: np.ndarray
    centroids: np.ndarray


def _strain_displacement(nodes: np.ndarray, tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy = nodes[tris]  # (n_el, 3, 2)
    x, y = xy[..., 0], xy[..., 1]
    b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], axis=1)
    c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], axis=1)
    area2 = x[:, 0] * b[:, 0] + x[:, 1] * b[:, 1] + x[:, 2] * b[:, 2]
    if np.any(area2 <= 0):
        raise InvalidParameterError("triangles must be non-degenerate and counter-clockwise")
    B = np.zeros((tris.shape[0], 3, 6))
    B[:, 0, 0::2] = b
    B[:, 1, 1::2] = c
    B[:, 2, 0::2] = c
    B[:, 2, 1::2] = b
    return B / area2[:, None, None], 0.5 * area2


def plane_elasticity_solve(
    nodes: ArrayLike,
    triangles: ArrayLike,
    E: float,
    nu: float,
    *,
    plane: str = "stress",
    fixed: ArrayLike | None = None,
    fixed_values: ArrayLike | None = None,
    forces: ArrayLike | None = None,
    thickness: float = 1.0,
) -> PlaneElasticityResult:
    """Solve 2D linear elasticity with constant-strain triangles.

    Each triangle contributes the stiffness
    :math:`K_e = t A_e B_e^{\\mathsf T} D B_e`, with :math:`B_e` the
    (constant) strain-displacement matrix of linear shape functions and
    :math:`D` from :func:`elasticity_matrix` (Zienkiewicz and Taylor §4.2).
    The global system :math:`K\\mathbf u = \\mathbf f` is solved for the free
    degrees of freedom after moving prescribed displacements to the
    right-hand side. The element passes the patch test: any uniform strain
    field is reproduced exactly.

    Parameters
    ----------
    nodes : array_like, shape (n_nodes, 2)
        Node coordinates.
    triangles : array_like of int, shape (n_triangles, 3)
        Counter-clockwise node indices.
    E, nu : float
        Young's modulus and Poisson's ratio.
    plane : {"stress", "strain"}, default="stress"
        Plane idealization.
    fixed : array_like of bool, shape (n_nodes, 2), optional
        Which displacement components are prescribed. At least enough must
        be fixed to remove rigid-body motion.
    fixed_values : array_like, shape (n_nodes, 2), optional
        Prescribed displacement values (zero by default).
    forces : array_like, shape (n_nodes, 2), optional
        External nodal forces, e.g. from :func:`pressure_load`.
    thickness : float, default=1.0
        Out-of-plane thickness (per unit length for plane strain).

    Returns
    -------
    PlaneElasticityResult

    Examples
    --------
    Uniaxial tension of a plate: :math:`\\sigma_{xx} = \\sigma` everywhere and
    :math:`u_x = \\sigma x / E`. The loaded right edge is listed top to
    bottom, so the plate lies on its right, and a negative pressure pulls:

    >>> nodes, tris = rectangle_mesh(2.0, 1.0, 4, 2)
    >>> fixed = np.zeros((len(nodes), 2), dtype=bool)
    >>> fixed[nodes[:, 0] == 0, 0] = True
    >>> fixed[0, 1] = True
    >>> right = np.flatnonzero(nodes[:, 0] == 2.0)
    >>> F = pressure_load(nodes, np.column_stack([right[1:], right[:-1]]), -3.0)
    >>> res = plane_elasticity_solve(nodes, tris, 100.0, 0.3, fixed=fixed, forces=F)
    >>> bool(np.allclose(res.stress[:, 0], 3.0)), round(float(res.displacement[right, 0].mean()), 6)
    (True, 0.06)
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    tris = np.asarray(triangles, dtype=np.int64)
    n_dof = 2 * nodes.shape[0]
    D = elasticity_matrix(E, nu, plane)
    B, area = _strain_displacement(nodes, tris)
    Ke = thickness * area[:, None, None] * np.einsum("eki,kl,elj->eij", B, D, B)
    dofs = np.empty((tris.shape[0], 6), dtype=np.int64)
    dofs[:, 0::2] = 2 * tris
    dofs[:, 1::2] = 2 * tris + 1
    rows = np.repeat(dofs, 6, axis=1).ravel()
    cols = np.tile(dofs, (1, 6)).ravel()
    K = sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=(n_dof, n_dof))

    f = np.zeros(n_dof) if forces is None else np.asarray(forces, dtype=np.float64).ravel().copy()
    is_fixed = np.zeros(n_dof, dtype=bool) if fixed is None else np.asarray(fixed, dtype=bool).ravel()
    if not is_fixed.any():
        raise InvalidParameterError("fix at least three displacement components to remove rigid-body motion")
    u = np.zeros(n_dof) if fixed_values is None else np.asarray(fixed_values, dtype=np.float64).ravel().copy()
    free = ~is_fixed
    rhs = f[free] - K[free][:, is_fixed] @ u[is_fixed]
    u[free] = spsolve(K[free][:, free].tocsc(), rhs)
    u_el = u[dofs]
    strain = np.einsum("eij,ej->ei", B, u_el)
    stress = strain @ D.T
    centroids = nodes[tris].mean(axis=1)
    return PlaneElasticityResult(u.reshape(-1, 2), strain, stress, centroids)


def polar_stress(stress: ArrayLike, points: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Rotate Cartesian stresses to polar components about the origin.

    .. math::

        \\sigma_{rr} = \\sigma_{xx}\\cos^2\\theta + \\sigma_{yy}\\sin^2\\theta + 2\\sigma_{xy}\\sin\\theta\\cos\\theta,\\\\
        \\sigma_{\\theta\\theta} = \\sigma_{xx}\\sin^2\\theta + \\sigma_{yy}\\cos^2\\theta - 2\\sigma_{xy}\\sin\\theta\\cos\\theta,\\\\
        \\sigma_{r\\theta} = (\\sigma_{yy} - \\sigma_{xx})\\sin\\theta\\cos\\theta + \\sigma_{xy}(\\cos^2\\theta - \\sin^2\\theta).

    Parameters
    ----------
    stress : array_like, shape (n, 3)
        :math:`(\\sigma_{xx}, \\sigma_{yy}, \\sigma_{xy})`.
    points : array_like, shape (n, 2)
        Where each stress applies.

    Returns
    -------
    sigma_rr, sigma_tt, sigma_rt : ndarray, shape (n,)

    Examples
    --------
    >>> [round(float(v[0]), 12) for v in polar_stress([[1.0, 2.0, 0.0]], [[0.0, 1.0]])]
    [2.0, 1.0, 0.0]
    """
    s = np.asarray(stress, dtype=np.float64)
    p = np.asarray(points, dtype=np.float64)
    th = np.arctan2(p[:, 1], p[:, 0])
    c, n = np.cos(th), np.sin(th)
    sxx, syy, sxy = s[:, 0], s[:, 1], s[:, 2]
    srr = sxx * c**2 + syy * n**2 + 2 * sxy * n * c
    stt = sxx * n**2 + syy * c**2 - 2 * sxy * n * c
    srt = (syy - sxx) * n * c + sxy * (c**2 - n**2)
    return srr, stt, srt


def lame_thick_cylinder(
    r: ArrayLike,
    inner_radius: float,
    outer_radius: float,
    p_in: float,
    p_out: float = 0.0,
    E: float | None = None,
    nu: float | None = None,
    plane: str = "strain",
):
    """Lamé's solution for a thick-walled cylinder under internal and external pressure.

    With :math:`a, b` the inner and outer radii,

    .. math::

        \\sigma_{rr} = A - \\frac{B}{r^2}, \\quad \\sigma_{\\theta\\theta} = A + \\frac{B}{r^2}, \\quad
        A = \\frac{p_i a^2 - p_o b^2}{b^2 - a^2}, \\quad B = \\frac{(p_i - p_o)a^2 b^2}{b^2 - a^2}

    (Lamé, *Leçons sur la théorie mathématique de l'élasticité des corps
    solides*, 1852; Timoshenko and Goodier, *Theory of Elasticity*, 3rd ed.,
    §28), and the radial displacement

    .. math::

        u_r = \\frac{1+\\nu}{E}\\left[(1-2\\nu)Ar + \\frac{B}{r}\\right] \\;\\text{(plane strain)}, \\qquad
        u_r = \\frac{1}{E}\\left[(1-\\nu)Ar + (1+\\nu)\\frac{B}{r}\\right] \\;\\text{(plane stress)}.

    Parameters
    ----------
    r : array_like
        Radius, :math:`a \\le r \\le b`.
    inner_radius, outer_radius : float
        :math:`a` and :math:`b`.
    p_in : float
        Internal pressure :math:`p_i`.
    p_out : float, default=0.0
        External pressure :math:`p_o`.
    E, nu : float, optional
        Elastic constants; needed only for the displacement.
    plane : {"strain", "stress"}, default="strain"
        Idealization for the displacement (the in-plane stresses are the
        same for both).

    Returns
    -------
    sigma_rr, sigma_tt : ndarray
    u_r : ndarray or None
        ``None`` when ``E`` or ``nu`` is not given.

    Examples
    --------
    The hoop stress at the bore of a cylinder with :math:`b = 2a`:

    >>> srr, stt, _ = lame_thick_cylinder([1.0, 2.0], 1.0, 2.0, p_in=3.0)
    >>> srr.tolist(), stt.tolist()
    ([-3.0, 0.0], [5.0, 2.0])
    """
    a, b = inner_radius, outer_radius
    if not 0 < a < b:
        raise InvalidParameterError("need 0 < inner_radius < outer_radius")
    r = np.asarray(r, dtype=np.float64)
    A = (p_in * a**2 - p_out * b**2) / (b**2 - a**2)
    Bc = (p_in - p_out) * a**2 * b**2 / (b**2 - a**2)
    srr = A - Bc / r**2
    stt = A + Bc / r**2
    if E is None or nu is None:
        return srr, stt, None
    lame_parameters(E, nu)
    if plane == "strain":
        ur = (1 + nu) / E * ((1 - 2 * nu) * A * r + Bc / r)
    elif plane == "stress":
        ur = ((1 - nu) * A * r + (1 + nu) * Bc / r) / E
    else:
        raise InvalidParameterError(f"plane must be 'stress' or 'strain', got {plane!r}")
    return srr, stt, ur
