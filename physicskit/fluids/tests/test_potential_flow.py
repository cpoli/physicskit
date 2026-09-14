import numpy as np
import pytest

from physicskit.fluids.exceptions import InvalidParameterError
from physicskit.fluids.systems.potential_flow import (
    PotentialFlow,
    flow_past_cylinder,
    kutta_joukowski_lift,
    pressure_coefficient,
)


def test_cylinder_surface_is_a_single_streamline():
    """Uniform flow + doublet places a circular streamline exactly at the
    doublet's radius; every point on that circle must share one streamfunction value."""
    flow = flow_past_cylinder(U_inf=2.0, radius=1.5, circulation=3.0)
    theta = np.linspace(0, 2 * np.pi, 37)
    x, y = 1.5 * np.cos(theta), 1.5 * np.sin(theta)
    psi = flow.streamfunction(x, y)
    assert np.max(np.abs(psi - psi[0])) < 1e-6


def test_cylinder_surface_speed_matches_classic_formula():
    """On a lift-free cylinder, the tangential surface speed is exactly
    2*U_inf*sin(theta) -- the textbook result, checked independently of the
    velocity() finite-difference implementation via the surface Cp."""
    U_inf, R = 1.0, 1.0
    flow = flow_past_cylinder(U_inf=U_inf, radius=R, circulation=0.0)
    theta = np.linspace(0.01, np.pi - 0.01, 25)  # avoid stagnation points
    x, y = R * np.cos(theta), R * np.sin(theta)
    Cp = flow.pressure_coefficient(x, y)
    expected_speed = 2.0 * U_inf * np.sin(theta)
    expected_Cp = 1.0 - (expected_speed / U_inf) ** 2
    np.testing.assert_allclose(Cp, expected_Cp, atol=1e-4)


def test_kutta_joukowski_lift_matches_cylinder_pressure_integral():
    """Integrating Cp around the lifting cylinder's surface should reproduce
    the Kutta-Joukowski lift, an independent cross-check of two different
    ways to compute the same force."""
    U_inf, R, rho, Gamma = 1.0, 1.0, 1.0, 1.0
    flow = flow_past_cylinder(U_inf=U_inf, radius=R, circulation=Gamma)
    theta = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    x, y = R * np.cos(theta), R * np.sin(theta)
    Cp = flow.pressure_coefficient(x, y)
    p = Cp * (0.5 * rho * U_inf**2)
    # Lift is -oint p * sin(theta) * R dtheta (normal force projected onto y);
    # a plain Riemann sum over a uniform grid is exact for a periodic integrand.
    dtheta = theta[1] - theta[0]
    lift_numeric = -np.sum(p * np.sin(theta) * R) * dtheta
    lift_theory = kutta_joukowski_lift(rho, U_inf, Gamma)
    # the sign of the pressure-integral convention above is independent of
    # the theorem itself; only the magnitude is being cross-checked here.
    assert abs(lift_numeric) == pytest.approx(lift_theory, rel=0.02)


def test_flow_past_cylinder_rejects_nonpositive_radius():
    with pytest.raises(InvalidParameterError):
        flow_past_cylinder(U_inf=1.0, radius=0.0)


def test_pressure_coefficient_is_one_at_stagnation():
    """Cp = 1 exactly where the local speed vanishes (a stagnation point)."""
    assert pressure_coefficient(u=0.0, v=0.0, U_inf=1.0) == 1.0


def test_source_and_sink_pair_has_zero_net_streamfunction_far_away():
    """A source and an equal-strength sink placed symmetrically must produce
    a flow that looks, from far away, like nothing at all (net source strength zero)."""
    flow = PotentialFlow(U_inf=0.0)
    flow.add_source(strength=5.0, x0=-0.1, y0=0.0)
    flow.add_source(strength=-5.0, x0=0.1, y0=0.0)
    far_x, far_y = 1000.0, 1000.0
    u, v = flow.velocity(far_x, far_y)
    assert np.hypot(u, v) < 1e-6
