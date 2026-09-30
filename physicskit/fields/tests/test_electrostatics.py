"""Electrostatics against closed-form results: Coulomb, Poisson solvers, images, multipoles."""

import numpy as np
import pytest
from scipy.integrate import quad

from physicskit.fields.electrodynamics import EPS0
from physicskit.fields.electrostatics import (
    coulomb_field,
    coulomb_potential,
    dipole_field,
    electric_field_from_potential,
    image_charges_plane,
    image_charges_sphere,
    induced_charge_density_plane,
    induced_charge_density_sphere,
    multipole_moments,
    multipole_potential,
    solve_poisson,
    solve_poisson_fft,
)

K = 1.0 / (4.0 * np.pi * EPS0)


# --- Coulomb's law -----------------------------------------------------------


def test_coulomb_field_is_minus_gradient_of_potential():
    q = [1e-9, -2e-9, 0.5e-9]
    pos = [[0.0, 0.0, 0.0], [0.3, 0.1, -0.2], [-0.2, 0.4, 0.1]]
    x = np.array([0.7, -0.3, 0.5])
    h = 1e-6
    grad = np.array([(coulomb_potential(q, pos, x + h * e) - coulomb_potential(q, pos, x - h * e)) / (2 * h) for e in np.eye(3)])
    assert coulomb_field(q, pos, x) == pytest.approx(-grad, rel=1e-6)


def test_coulomb_inverse_square_law():
    r = np.array([[0.5, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    E = coulomb_field([1e-9], [[0, 0, 0]], r)[:, 0]
    assert E == pytest.approx(K * 1e-9 / r[:, 0] ** 2, rel=1e-12)


# --- Poisson solvers --------------------------------------------------------


def _charged_sphere_setup(n=49, L=1.0, R=0.3, rho0=1.0):
    x = np.linspace(-L, L, n)
    h = x[1] - x[0]
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    r = np.sqrt(X**2 + Y**2 + Z**2)
    rho = np.where(r <= R, rho0, 0.0)
    Q = rho.sum() * h**3  # the staircased sphere's actual charge
    phi0 = np.zeros_like(rho)
    faces = np.ones_like(rho, dtype=bool)
    faces[1:-1, 1:-1, 1:-1] = False
    phi0[faces] = Q / (4 * np.pi * r[faces])  # eps = 1
    return x, h, r, rho, Q, phi0


def test_charged_sphere_potential_and_field_by_sor():
    """A uniformly charged ball: phi = Q(3R^2 - r^2)/(8 pi R^3) inside and
    Q/(4 pi r) outside; |E| = Q r/(4 pi R^3) inside, Q/(4 pi r^2) outside
    (Griffiths Example 2.8), with eps = 1 and exact potential on the box."""
    R = 0.3
    x, h, r, rho, Q, phi0 = _charged_sphere_setup(R=R)
    sol = solve_poisson(rho, h, phi0=phi0, eps=1.0, tol=1e-10)
    assert sol.converged
    c = len(x) // 2
    assert sol.phi[c, c, c] == pytest.approx(3 * Q / (8 * np.pi * R), rel=0.02)
    for i in (c + 12, c + 18):  # outside the ball, along x
        assert sol.phi[i, c, c] == pytest.approx(Q / (4 * np.pi * x[i]), rel=0.01)
    E = electric_field_from_potential(sol.phi, h)
    i_out = c + 15
    assert E[0][i_out, c, c] == pytest.approx(Q / (4 * np.pi * x[i_out] ** 2), rel=0.02)
    i_in = c + 3
    assert E[0][i_in, c, c] == pytest.approx(Q * x[i_in] / (4 * np.pi * R**3), rel=0.05)


def test_dst_solver_matches_sor_on_grounded_box():
    """Both solve the identical discrete system, so agree to solver tolerance."""
    rng = np.random.default_rng(0)
    rho = rng.normal(size=(33, 29))
    sol = solve_poisson(rho, (0.1, 0.07), eps=1.0, tol=1e-13)
    direct = solve_poisson_fft(rho, (0.1, 0.07), bc="dirichlet", eps=1.0)
    assert np.max(np.abs(sol.phi - direct)) < 1e-9 * np.max(np.abs(direct))


def test_dst_solver_3d_matches_sor():
    x, h, r, rho, Q, _ = _charged_sphere_setup(n=25)
    sor = solve_poisson(rho, h, eps=1.0, tol=1e-12).phi
    direct = solve_poisson_fft(rho, h, bc="dirichlet", eps=1.0)
    assert np.max(np.abs(sor - direct)) < 1e-8 * np.max(np.abs(direct))


def test_periodic_fft_solver_on_plane_wave_charge():
    n = 64
    x = np.arange(n) * (2 * np.pi / n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = np.sin(2 * X) * np.cos(3 * Y)
    phi = solve_poisson_fft(rho, 2 * np.pi / n, eps=2.0)
    assert phi == pytest.approx(rho / (2.0 * 13.0), abs=1e-12)


def test_jacobi_and_sor_agree_and_sor_is_faster():
    phi0 = np.zeros((41, 41))
    phi0[:, -1] = 1.0
    jac = solve_poisson(np.zeros((41, 41)), 1.0, phi0=phi0, method="jacobi", tol=1e-10)
    sor = solve_poisson(np.zeros((41, 41)), 1.0, phi0=phi0, method="sor", tol=1e-10)
    assert jac.converged and sor.converged
    assert sor.n_iter * 10 < jac.n_iter
    assert np.max(np.abs(jac.phi - sor.phi)) < 1e-6


def test_parallel_plate_capacitor_field_is_v_over_d():
    """Wide plates, close together: E = V/d between them, away from the edges."""
    n, L = 241, 1.2
    h = L / (n - 1)
    V, gap = 1.0, 0.1
    phi0 = np.zeros((n, n))
    fixed = np.zeros((n, n), dtype=bool)
    c = n // 2
    half_gap = round(gap / (2 * h))
    half_width = round(0.4 / h)
    plates = slice(c - half_width, c + half_width + 1)
    phi0[plates, c - half_gap], phi0[plates, c + half_gap] = V / 2, -V / 2
    fixed[plates, c - half_gap] = fixed[plates, c + half_gap] = True
    sol = solve_poisson(np.zeros((n, n)), h, phi0=phi0, fixed=fixed, tol=1e-10)
    E = electric_field_from_potential(sol.phi, h)
    d = 2 * half_gap * h
    assert E[1][c, c] == pytest.approx(V / d, rel=0.01)
    assert abs(E[0][c, c]) < 1e-6 * V / d


def test_coaxial_capacitor_logarithmic_potential():
    """phi(r) = V ln(b/r)/ln(b/a) between coaxial conductors (Griffiths Problem 2.43)."""
    n = 161
    x = np.linspace(-0.5, 0.5, n)
    h = x[1] - x[0]
    X, Y = np.meshgrid(x, x, indexing="ij")
    r = np.hypot(X, Y)
    a, b, V = 0.08, 0.4, 1.0
    fixed = (r <= a) | (r >= b)
    phi0 = np.where(r <= a, V, 0.0)
    sol = solve_poisson(np.zeros((n, n)), h, phi0=phi0, fixed=fixed, tol=1e-10)
    # The staircased conductors shift the effective radii a and b by O(h), so
    # check the functional form -- phi exactly linear in ln r -- and the slope.
    between = (r > a + 2 * h) & (r < b - 2 * h)
    slope, intercept = np.polyfit(np.log(r[between]), sol.phi[between], 1)
    assert np.max(np.abs(sol.phi[between] - (slope * np.log(r[between]) + intercept))) < 0.01 * V
    assert slope == pytest.approx(-V / np.log(b / a), rel=0.03)


# --- Method of images -------------------------------------------------------


def test_grounded_plane_is_equipotential_and_force_is_image_force():
    q, d = 2e-9, 0.3
    qi, ri = image_charges_plane(q, [0.1, -0.2, d])
    charges, pos = np.r_[q, qi], np.vstack([[0.1, -0.2, d], ri])
    rng = np.random.default_rng(1)
    plane = np.column_stack([rng.uniform(-2, 2, (50, 2)), np.zeros(50)])
    assert np.max(np.abs(coulomb_potential(charges, pos, plane))) < 1e-9 * K * q / d
    F = q * coulomb_field(qi, ri, [0.1, -0.2, d])
    assert F == pytest.approx([0.0, 0.0, -K * q**2 / (2 * d) ** 2], rel=1e-12, abs=1e-30)


def test_induced_charge_on_plane_totals_minus_q():
    q, d = 1.0, 0.7
    total, _ = quad(lambda s: induced_charge_density_plane(q, d, s) * 2 * np.pi * s, 0, np.inf)
    assert total == pytest.approx(-q, rel=1e-8)
    # and it equals eps0 E_z just above the plane (E from the charge plus image)
    s = 0.4
    E = coulomb_field([q, -q], [[0, 0, d], [0, 0, -d]], [s, 0.0, 0.0])
    assert induced_charge_density_plane(q, d, s) == pytest.approx(EPS0 * E[2], rel=1e-10)


@pytest.mark.parametrize("a", [1.5, 4.0, 0.4])
def test_grounded_sphere_surface_is_equipotential(a):
    q, R = 1e-9, 1.0
    src = np.array([a, 0.0, 0.0])
    qi, ri = image_charges_sphere(q, src, R)
    rng = np.random.default_rng(2)
    u = rng.normal(size=(100, 3))
    surface = R * u / np.linalg.norm(u, axis=1, keepdims=True)
    phi = coulomb_potential(np.r_[q, qi], np.vstack([src, ri]), surface)
    assert np.max(np.abs(phi)) < 1e-9 * K * q / abs(a - R)


def test_grounded_sphere_attracts_with_closed_form_force():
    """F = q^2 R a / (4 pi eps0 (a^2 - R^2)^2), toward the sphere (Griffiths Problem 3.8)."""
    q, R, a = 1e-9, 0.5, 1.3
    qi, ri = image_charges_sphere(q, [0, 0, a], R, center=(0, 0, 0))
    F = q * coulomb_field(qi, ri, [0, 0, a])[2]
    assert F == pytest.approx(-K * q**2 * R * a / (a**2 - R**2) ** 2, rel=1e-12)


def test_isolated_sphere_is_equipotential_with_its_charge():
    q, R, a, Qs = 1e-9, 1.0, 2.5, 3e-9
    qi, ri = image_charges_sphere(q, [0, a, 0], R, sphere_charge=Qs)
    assert qi.sum() == pytest.approx(Qs)
    rng = np.random.default_rng(3)
    u = rng.normal(size=(60, 3))
    surface = R * u / np.linalg.norm(u, axis=1, keepdims=True)
    phi = coulomb_potential(np.r_[q, qi], np.vstack([[0, a, 0], ri]), surface)
    # constant on the surface, at the potential of its center image Qs + q R / a
    assert np.ptp(phi) < 1e-9 * np.mean(phi)
    assert np.mean(phi) == pytest.approx(K * (Qs + q * R / a) / R, rel=1e-10)


def test_induced_charge_on_sphere_totals_image_charge():
    q, a, R = 1.0, 3.0, 1.2
    total, _ = quad(lambda th: induced_charge_density_sphere(q, a, R, th) * 2 * np.pi * R**2 * np.sin(th), 0, np.pi)
    assert total == pytest.approx(-q * R / a, rel=1e-10)


# --- Multipole expansion ----------------------------------------------------


def test_dipole_far_field_on_axis_and_equator():
    """A physical dipole +-q at +-d/2 has p = q d and far field
    2p/(4 pi eps0 r^3) on its axis, -p/(4 pi eps0 r^3) on its equator."""
    q, d = 1e-9, 1e-3
    charges, pos = [q, -q], [[0, 0, d / 2], [0, 0, -d / 2]]
    m = multipole_moments(charges, pos)
    assert m.monopole == pytest.approx(0.0, abs=1e-24)
    assert m.dipole == pytest.approx([0, 0, q * d])
    r = 0.5
    axis = coulomb_field(charges, pos, [0, 0, r])[2]
    equator = coulomb_field(charges, pos, [r, 0, 0])[2]
    assert axis == pytest.approx(2 * K * q * d / r**3, rel=1e-5)
    assert equator == pytest.approx(-K * q * d / r**3, rel=1e-5)
    assert dipole_field(m.dipole, [[0, 0, r], [r, 0, 0]])[:, 2] == pytest.approx([axis, equator], rel=1e-5)


def test_dipole_potential_error_falls_as_r_minus_2():
    """For a symmetric dipole the first neglected term is the octupole, so the
    relative error of the dipole truncation falls as (d/r)^2."""
    charges, pos = [1.0, -1.0], [[0.1, 0.2, 0.5], [0.1, 0.2, -0.5]]
    m = multipole_moments(charges, pos, origin=(0.1, 0.2, 0.0))
    direction = np.array([1.0, 2.0, 2.0]) / 3.0

    def rel_err(r):
        x = np.array([0.1, 0.2, 0.0]) + r * direction
        exact = coulomb_potential(charges, pos, x, eps=1.0)
        return abs(multipole_potential(m, x, order=1, eps=1.0) / exact - 1)

    assert rel_err(20.0) / rel_err(40.0) == pytest.approx(4.0, rel=0.01)


def test_linear_quadrupole_moments_and_potential():
    """Charges q, -2q, q at z = d, 0, -d: Q_zz = 4 q d^2, Q_xx = Q_yy = -2 q d^2."""
    q, d = 1.0, 0.1
    charges, pos = [q, -2 * q, q], [[0, 0, d], [0, 0, 0], [0, 0, -d]]
    m = multipole_moments(charges, pos)
    assert m.monopole == 0.0
    assert m.dipole == pytest.approx([0, 0, 0], abs=1e-15)
    assert np.diag(m.quadrupole) == pytest.approx([-2 * q * d**2, -2 * q * d**2, 4 * q * d**2])
    assert np.trace(m.quadrupole) == pytest.approx(0.0, abs=1e-15)
    assert np.all(multipole_potential(m, [[0.3, 0.4, 1.2]], order=1) == 0.0)
    # the next nonzero term is the hexadecapole, so the relative error of the
    # quadrupole truncation falls as (d/r)^2
    direction = np.array([2.0, -1.0, 2.0]) / 3.0

    def rel_err(r):
        exact = coulomb_potential(charges, pos, r * direction)
        return abs(multipole_potential(m, r * direction, order=2) / exact - 1)

    assert rel_err(2.0) < 1e-2
    assert rel_err(2.0) / rel_err(4.0) == pytest.approx(4.0, rel=0.01)


def test_grid_density_moments_match_point_charge():
    """A Gaussian blob is a pure monopole: zero dipole about its own center."""
    x = np.linspace(-1, 1, 41)
    h = x[1] - x[0]
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    rho = np.exp(-((X - 0.1) ** 2 + Y**2 + Z**2) / 0.02)
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    m = multipole_moments(rho.ravel() * h**3, pts, origin=(0.1, 0.0, 0.0))
    assert np.linalg.norm(m.dipole) < 1e-10 * m.monopole
    assert np.max(np.abs(m.quadrupole)) < 1e-8 * m.monopole


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        solve_poisson(np.zeros(10), 1.0)
    with pytest.raises(ValueError):
        solve_poisson(np.zeros((5, 5)), 1.0, method="gauss")
    with pytest.raises(ValueError):
        solve_poisson_fft(np.zeros((5, 5)), 1.0, bc="neumann")
    with pytest.raises(ValueError):
        image_charges_plane(1.0, [0, 0, 0])
    with pytest.raises(ValueError):
        image_charges_sphere(1.0, [0.5, 0, 0], 1.0, sphere_charge=0.0)
    with pytest.raises(ValueError):
        multipole_potential(multipole_moments([1.0], [[0, 0, 0]]), [[1, 0, 0]], order=3)
