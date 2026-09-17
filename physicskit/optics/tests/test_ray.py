import numpy as np
import pytest

from physicskit.optics.ray import (
    OpticalElement,
    OpticalSystem,
    cavity_round_trip_matrix,
    cavity_stability,
    curved_interface,
    flat_interface,
    free_space,
    grin_medium,
    spherical_mirror,
    thick_lens,
    thin_lens,
)


def test_free_space_matrix():
    M = free_space(3.0)
    assert np.allclose(M, [[1.0, 3.0], [0.0, 1.0]])


def test_thin_lens_matrix():
    M = thin_lens(0.25)
    assert np.allclose(M, [[1.0, 0.0], [-4.0, 1.0]])


def test_flat_interface_matrix():
    M = flat_interface(1.0, 1.5)
    assert np.allclose(M, [[1.0, 0.0], [0.0, 1.0 / 1.5]])


def test_curved_interface_reduces_to_flat_interface_for_large_radius():
    M_curved = curved_interface(R=1.0e10, n1=1.0, n2=1.5)
    M_flat = flat_interface(1.0, 1.5)
    assert np.allclose(M_curved, M_flat, atol=1e-8)


def test_spherical_mirror_matrix():
    M = spherical_mirror(4.0)
    assert np.allclose(M, [[1.0, 0.0], [-0.5, 1.0]])


def test_thick_lens_composition_order_and_unit_determinant():
    R1, R2, t, n = 0.1, -0.1, 0.01, 1.5
    M = thick_lens(R1, R2, t, n)
    expected = curved_interface(R2, n, 1.0) @ free_space(t) @ curved_interface(R1, 1.0, n)
    assert np.allclose(M, expected)
    # a lossless system's ABCD matrix always has unit determinant
    assert np.linalg.det(M) == pytest.approx(1.0, abs=1e-10)


def test_thin_lens_focuses_parallel_ray_bundle_to_a_point():
    # A bundle of rays parallel to the axis (theta0=0), of any height y0,
    # converges to the axis after propagating one focal length past the lens.
    f = 0.2
    M = free_space(f) @ thin_lens(f)
    for y0 in (-0.05, -0.01, 0.0, 0.01, 0.05, 1.0):
        y_out, theta_out = M @ np.array([y0, 0.0])
        assert y_out == pytest.approx(0.0, abs=1e-12)
        assert theta_out == pytest.approx(-y0 / f)


def test_grin_medium_reduces_to_free_space_as_n2_coeff_vanishes():
    d = 2.5
    M_grin = grin_medium(n0=1.0, n2_coeff=1e-9, d=d)
    M_free = free_space(d)
    assert np.allclose(M_grin, M_free, atol=1e-6)


def test_grin_medium_zero_n2_coeff_is_exactly_homogeneous_propagation():
    n0, d = 1.3, 2.5
    assert np.array_equal(grin_medium(n0=n0, n2_coeff=0.0, d=d), np.array([[1.0, d / n0], [0.0, 1.0]]))


def test_grin_medium_matches_closed_form_for_finite_gradient():
    n0, n2_coeff, d = 1.5, 4.0, 0.3
    M = grin_medium(n0, n2_coeff, d)
    sqrt_n2 = np.sqrt(n2_coeff)
    arg = sqrt_n2 * d
    expected = np.array(
        [
            [np.cos(arg), np.sin(arg) / (n0 * sqrt_n2)],
            [-n0 * sqrt_n2 * np.sin(arg), np.cos(arg)],
        ]
    )
    assert np.allclose(M, expected)


def test_optical_element_stores_attributes():
    M = thin_lens(0.5)
    elem = OpticalElement(M, name="my lens", length=0.0)
    assert elem.name == "my lens"
    assert elem.length == 0.0
    assert np.allclose(elem.matrix, M)


def test_optical_system_matrix_multiplication_order():
    # elements[0] is hit first, so it must be the RIGHTMOST factor:
    # M = M2 @ M1, not M1 @ M2 (these differ since matrices don't commute).
    m1 = thin_lens(0.3)
    m2 = free_space(1.0)
    system = OpticalSystem([OpticalElement(m1, name="lens"), OpticalElement(m2, name="gap")])
    expected = m2 @ m1
    assert np.allclose(system.system_matrix(), expected)
    assert not np.allclose(system.system_matrix(), m1 @ m2)


def test_optical_system_trace_ray_is_cumulative():
    f = 0.5
    system = OpticalSystem(
        [
            OpticalElement(thin_lens(f), name="lens"),
            OpticalElement(free_space(f), name="gap"),
        ]
    )
    y0, theta0 = 0.02, 0.0
    trajectory = system.trace_ray(y0, theta0)
    assert trajectory.shape == (3, 2)
    assert np.allclose(trajectory[0], [y0, theta0])
    assert np.allclose(trajectory[1], thin_lens(f) @ [y0, theta0])
    assert np.allclose(trajectory[2], free_space(f) @ thin_lens(f) @ [y0, theta0])
    # the whole bundle refocuses to the axis after one focal length
    assert trajectory[-1, 0] == pytest.approx(0.0, abs=1e-12)


def test_stability_agrees_with_cavity_stability_for_stable_cavity():
    # Two R=2.0 mirrors separated by d=1.0: g = 1 - d/R = 0.5, g1*g2 = 0.25,
    # which lies in [0, 1] -- a stable resonator by the standard g-parameter
    # criterion (Kogelnik & Li 1966).
    R, d = 2.0, 1.0
    elements = [
        OpticalElement(spherical_mirror(R), name="M1"),
        OpticalElement(free_space(d), name="gap1"),
        OpticalElement(spherical_mirror(R), name="M2"),
        OpticalElement(free_space(d), name="gap2"),
    ]
    system = OpticalSystem(elements)
    M = system.system_matrix()
    assert M is not None
    assert system.stability_parameter == pytest.approx((M[0, 0] + M[1, 1]) / 2.0)
    assert system.is_stable() is True
    assert cavity_stability(cavity_round_trip_matrix(elements)) is True
    assert cavity_stability(system.system_matrix()) == system.is_stable()


def test_stability_agrees_with_cavity_stability_for_unstable_cavity():
    # Two R=0.5 mirrors separated by d=2.0: g = 1 - d/R = -3, g1*g2 = 9,
    # well outside [0, 1] -- an unstable resonator.
    R, d = 0.5, 2.0
    elements = [
        OpticalElement(spherical_mirror(R), name="M1"),
        OpticalElement(free_space(d), name="gap1"),
        OpticalElement(spherical_mirror(R), name="M2"),
        OpticalElement(free_space(d), name="gap2"),
    ]
    system = OpticalSystem(elements)
    assert system.is_stable() is False
    assert cavity_stability(cavity_round_trip_matrix(elements)) is False
    assert cavity_stability(system.system_matrix()) == system.is_stable()


def test_cavity_round_trip_matrix_matches_optical_system():
    elements = [
        OpticalElement(spherical_mirror(1.0), name="M1"),
        OpticalElement(free_space(0.4), name="gap1"),
        OpticalElement(spherical_mirror(1.5), name="M2"),
        OpticalElement(free_space(0.4), name="gap2"),
    ]
    assert np.allclose(cavity_round_trip_matrix(elements), OpticalSystem(elements).system_matrix())


def test_cavity_stability_boundary_cases():
    assert cavity_stability(np.array([[1.0, 0.0], [0.0, 1.0]])) is True  # trace=2
    assert cavity_stability(np.array([[-1.0, 0.0], [0.0, -1.0]])) is True  # trace=-2
    assert cavity_stability(np.array([[2.0, 0.0], [0.0, 1.01]])) is False  # trace=3.01
