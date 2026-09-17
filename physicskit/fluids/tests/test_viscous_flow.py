import numpy as np
import pytest

from physicskit.fluids.exceptions import InvalidParameterError
from physicskit.fluids.systems.viscous_flow import (
    blasius_boundary_layer_thickness,
    blasius_skin_friction_coefficient,
    blasius_solve,
    couette_flow_velocity,
    poiseuille_flow_rate,
    poiseuille_flow_velocity,
    stokes_drag,
)


def test_couette_profile_is_linear_and_matches_wall_speeds():
    y = np.linspace(0, 2.0, 21)
    u = couette_flow_velocity(y, U_wall=3.0, h=2.0)
    assert u[0] == 0.0
    assert u[-1] == pytest.approx(3.0)
    np.testing.assert_allclose(np.diff(u), np.diff(u)[0])  # constant slope


def test_poiseuille_flow_rate_matches_integral_of_velocity_profile():
    dpdx, mu, h = -6.0, 0.8, 1.3
    y = np.linspace(0, h, 20001)
    u = poiseuille_flow_velocity(y, dpdx, mu, h)
    Q_numeric = np.trapz(u, y) if hasattr(np, "trapz") else np.sum(u) * (y[1] - y[0])
    Q_formula = poiseuille_flow_rate(dpdx, mu, h)
    assert Q_numeric == pytest.approx(Q_formula, rel=1e-3)


def test_stokes_drag_scales_linearly_with_velocity_and_radius():
    F1 = stokes_drag(mu=1e-3, radius=1e-4, velocity=0.01)
    F2 = stokes_drag(mu=1e-3, radius=1e-4, velocity=0.02)
    assert F2 == pytest.approx(2 * F1)
    F3 = stokes_drag(mu=1e-3, radius=2e-4, velocity=0.01)
    assert F3 == pytest.approx(2 * F1)


def test_blasius_wall_shear_matches_classic_constant():
    """f''(0), the Blasius shooting solution's wall-curvature, is the
    textbook constant ~0.332."""
    result = blasius_solve()
    assert result["fpp"][0] == pytest.approx(0.33206, abs=1e-4)


def test_blasius_far_field_velocity_recovers_free_stream():
    result = blasius_solve()
    assert result["fp"][-1] == pytest.approx(1.0, abs=1e-6)


def test_blasius_boundary_layer_thickness_grows_like_sqrt_x():
    x = np.array([1.0, 4.0, 9.0])
    delta = blasius_boundary_layer_thickness(x, U_inf=1.0, nu=1e-4)
    ratios = delta / np.sqrt(x)
    assert np.allclose(ratios, ratios[0], rtol=1e-8)


def test_blasius_skin_friction_decreases_with_reynolds_number():
    Re = np.array([1e3, 1e4, 1e5])
    cf = blasius_skin_friction_coefficient(Re)
    assert np.all(np.diff(cf) < 0)


def test_viscous_flow_rejects_nonpositive_parameters():
    with pytest.raises(InvalidParameterError):
        couette_flow_velocity(y=0.5, U_wall=1.0, h=0.0)
    with pytest.raises(InvalidParameterError):
        stokes_drag(mu=-1.0, radius=1.0, velocity=1.0)
    with pytest.raises(InvalidParameterError):
        blasius_boundary_layer_thickness(x=-1.0, U_inf=1.0, nu=1e-4)


def test_poiseuille_flow_velocity_rejects_nonpositive_mu_and_h():
    with pytest.raises(InvalidParameterError):
        poiseuille_flow_velocity(y=0.5, dpdx=-1.0, mu=0.0, h=1.0)
    with pytest.raises(InvalidParameterError):
        poiseuille_flow_velocity(y=0.5, dpdx=-1.0, mu=1.0, h=0.0)


def test_poiseuille_flow_rate_rejects_nonpositive_mu_and_h():
    with pytest.raises(InvalidParameterError):
        poiseuille_flow_rate(dpdx=-1.0, mu=0.0, h=1.0)
    with pytest.raises(InvalidParameterError):
        poiseuille_flow_rate(dpdx=-1.0, mu=1.0, h=0.0)


def test_stokes_drag_rejects_nonpositive_radius():
    with pytest.raises(InvalidParameterError):
        stokes_drag(mu=1.0, radius=0.0, velocity=1.0)


def test_blasius_boundary_layer_thickness_rejects_nonpositive_u_inf_and_nu():
    with pytest.raises(InvalidParameterError):
        blasius_boundary_layer_thickness(x=1.0, U_inf=0.0, nu=1e-4)
    with pytest.raises(InvalidParameterError):
        blasius_boundary_layer_thickness(x=1.0, U_inf=1.0, nu=0.0)


def test_blasius_skin_friction_coefficient_rejects_nonpositive_reynolds():
    with pytest.raises(InvalidParameterError):
        blasius_skin_friction_coefficient(reynolds_x=np.array([1e4, -1.0]))


def test_blasius_solve_expands_bracket_when_default_guess_undershoots():
    # eta_max=1.0 is too short for fpp0=1.0's shooting guess to reach f'=1,
    # forcing the initial bracket-expansion loop before bisection.
    result = blasius_solve(eta_max=1.0, n_points=50)
    assert result["fp"][-1] == pytest.approx(1.0, abs=1e-6)
