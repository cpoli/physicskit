"""Plane elasticity FEM against Lamé's pressurized cylinder and the patch test."""

import numpy as np
import pytest

from physicskit.fluids.exceptions import InvalidParameterError
from physicskit.fluids.systems.elasticity import (
    annulus_mesh,
    elasticity_matrix,
    lame_parameters,
    lame_thick_cylinder,
    plane_elasticity_solve,
    polar_stress,
    pressure_load,
    rectangle_mesh,
)


def _pressurized_quarter_cylinder(plane, n_r=24, n_theta=48, a=1.0, b=2.0, p=10.0, E=1000.0, nu=0.3):
    nodes, tris, bnd = annulus_mesh(a, b, n_r, n_theta)
    fixed = np.zeros((len(nodes), 2), dtype=bool)
    fixed[bnd["theta0"], 1] = True  # symmetry: u_y = 0 on y = 0
    fixed[bnd["theta_max"], 0] = True  # symmetry: u_x = 0 on x = 0
    F = pressure_load(nodes, bnd["inner"], p)
    return nodes, plane_elasticity_solve(nodes, tris, E, nu, plane=plane, fixed=fixed, forces=F)


@pytest.mark.parametrize("plane", ["strain", "stress"])
def test_lame_pressurized_cylinder(plane):
    """Radial displacement and hoop/radial stresses match Lamé (Timoshenko and Goodier §28)."""
    a, b, p, E, nu = 1.0, 2.0, 10.0, 1000.0, 0.3
    nodes, res = _pressurized_quarter_cylinder(plane)
    r_nodes = np.hypot(nodes[:, 0], nodes[:, 1])
    u_r = np.sum(res.displacement * nodes, axis=1) / r_nodes
    _, _, exact_u = lame_thick_cylinder(r_nodes, a, b, p, E=E, nu=nu, plane=plane)
    assert np.max(np.abs(u_r - exact_u)) < 1e-2 * np.max(exact_u)
    # tangential displacement vanishes by axisymmetry
    u_t = (-res.displacement[:, 0] * nodes[:, 1] + res.displacement[:, 1] * nodes[:, 0]) / r_nodes
    assert np.max(np.abs(u_t)) < 1e-3 * np.max(exact_u)
    srr, stt, srt = polar_stress(res.stress, res.centroids)
    r_c = np.hypot(res.centroids[:, 0], res.centroids[:, 1])
    ex_rr, ex_tt, _ = lame_thick_cylinder(r_c, a, b, p)
    # constant-strain triangles give first-order stresses (the error halves with h, see
    # test_lame_error_converges_under_refinement), least accurate at the bore where the
    # Lame stresses vary fastest
    assert np.max(np.abs(stt - ex_tt)) < 0.1 * p
    assert np.max(np.abs(srr - ex_rr)) < 0.1 * p
    mid = (r_c > 1.2) & (r_c < 1.8)
    assert np.max(np.abs(stt[mid] - ex_tt[mid])) < 0.05 * p
    assert np.max(np.abs(srr[mid] - ex_rr[mid])) < 0.05 * p
    assert np.max(np.abs(srt)) < 0.05 * p


def test_lame_error_converges_under_refinement():
    """Displacement error falls as h^2 and the (element-constant) stress error as h."""
    u_errs, s_errs = [], []
    for n in (8, 16):
        nodes, res = _pressurized_quarter_cylinder("strain", n_r=n, n_theta=2 * n)
        r = np.hypot(nodes[:, 0], nodes[:, 1])
        u_r = np.sum(res.displacement * nodes, axis=1) / r
        u_errs.append(np.max(np.abs(u_r - lame_thick_cylinder(r, 1.0, 2.0, 10.0, E=1000.0, nu=0.3)[2])))
        _, stt, _ = polar_stress(res.stress, res.centroids)
        exact_tt = lame_thick_cylinder(np.hypot(res.centroids[:, 0], res.centroids[:, 1]), 1.0, 2.0, 10.0)[1]
        s_errs.append(np.sqrt(np.mean((stt - exact_tt) ** 2)))
    assert u_errs[0] / u_errs[1] == pytest.approx(4.0, rel=0.25)
    assert s_errs[0] / s_errs[1] == pytest.approx(2.0, rel=0.1)


def test_lame_closed_form_boundary_values():
    a, b, pi, po = 0.5, 1.5, 7.0, 2.0
    srr, stt, _ = lame_thick_cylinder([a, b], a, b, pi, po)
    assert srr == pytest.approx([-pi, -po])
    # the axial-strain-free sum sigma_rr + sigma_tt is uniform
    r = np.linspace(a, b, 5)
    srr, stt, _ = lame_thick_cylinder(r, a, b, pi, po)
    assert np.ptp(srr + stt) < 1e-12


@pytest.mark.parametrize("plane", ["stress", "strain"])
def test_patch_test_reproduces_uniform_strain_exactly(plane):
    """Prescribe the linear field u = (e_xx x + g y/2, e_yy y + g x/2) on the boundary: every
    element must carry exactly that strain and D times it."""
    nodes, tris = rectangle_mesh(3.0, 2.0, 6, 5)
    rng = np.random.default_rng(0)
    interior = (nodes[:, 0] > 0) & (nodes[:, 0] < 3) & (nodes[:, 1] > 0) & (nodes[:, 1] < 2)
    nodes[interior] += rng.uniform(-0.1, 0.1, (interior.sum(), 2))  # distort interior nodes
    exx, eyy, g = 1e-3, -4e-4, 6e-4
    exact = np.column_stack([exx * nodes[:, 0] + 0.5 * g * nodes[:, 1], eyy * nodes[:, 1] + 0.5 * g * nodes[:, 0]])
    fixed = np.zeros_like(nodes, dtype=bool)
    fixed[~interior] = True
    res = plane_elasticity_solve(nodes, tris, 200.0, 0.25, plane=plane, fixed=fixed, fixed_values=exact)
    assert res.displacement == pytest.approx(exact, abs=1e-12)
    assert res.strain == pytest.approx(np.tile([exx, eyy, g], (len(tris), 1)), abs=1e-12)
    assert res.stress == pytest.approx(np.tile(elasticity_matrix(200.0, 0.25, plane) @ [exx, eyy, g], (len(tris), 1)), abs=1e-10)


def test_plane_strain_matrix_matches_lame_form():
    E, nu = 3.0, 0.2
    lam, mu = lame_parameters(E, nu)
    D = elasticity_matrix(E, nu, "strain")
    assert D == pytest.approx(np.array([[lam + 2 * mu, lam, 0], [lam, lam + 2 * mu, 0], [0, 0, mu]]))
    # plane stress: sigma_zz = 0 condenses lambda to 2 lam mu / (lam + 2 mu)
    lam_star = 2 * lam * mu / (lam + 2 * mu)
    assert elasticity_matrix(E, nu, "stress") == pytest.approx(np.array([[lam_star + 2 * mu, lam_star, 0], [lam_star, lam_star + 2 * mu, 0], [0, 0, mu]]))


def test_pressure_load_totals_and_validation():
    nodes, tris, bnd = annulus_mesh(1.0, 2.0, 2, 16)
    F = pressure_load(nodes, bnd["inner"], 5.0)
    # on a quarter circle of radius a, the net force of pressure p is p a (1, 1)
    assert F.sum(axis=0) == pytest.approx([5.0, 5.0], rel=5e-3)
    with pytest.raises(InvalidParameterError):
        lame_parameters(1.0, 0.5)
    with pytest.raises(InvalidParameterError):
        elasticity_matrix(1.0, 0.3, "shell")
    with pytest.raises(InvalidParameterError):
        plane_elasticity_solve(nodes, tris, 1.0, 0.3)
    with pytest.raises(InvalidParameterError):
        plane_elasticity_solve(nodes, tris[:, ::-1], 1.0, 0.3, fixed=np.ones_like(nodes, dtype=bool))
