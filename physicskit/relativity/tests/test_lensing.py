import numpy as np
import pytest

from physicskit.relativity.chapters.lensing import PointMassLens, exact_deflection_angle
from physicskit.relativity.chapters.schwarzschild import SchwarzschildBlackHole


def test_einstein_ring_at_perfect_alignment():
    lens = PointMassLens(M=1.0, D_L=1000.0, D_S=2000.0)
    theta_plus, theta_minus = lens.image_angles(beta=0.0)
    assert theta_plus == pytest.approx(lens.einstein_angle())
    assert theta_minus == pytest.approx(-lens.einstein_angle())


def test_einstein_angle_formula():
    lens = PointMassLens(M=1.0, D_L=1000.0, D_S=2000.0)
    expected = np.sqrt(4.0 * 1.0 * 1000.0 / (1000.0 * 2000.0))
    assert lens.einstein_angle() == pytest.approx(expected)


def test_images_straddle_the_lens_for_any_offset():
    lens = PointMassLens(M=1.0, D_L=1000.0, D_S=2000.0)
    theta_plus, theta_minus = lens.image_angles(beta=0.01)
    assert theta_plus > 0.0
    assert theta_minus < 0.0


def test_primary_image_approaches_source_position_far_from_lens():
    lens = PointMassLens(M=1.0, D_L=1000.0, D_S=2000.0)
    beta = 10.0 * lens.einstein_angle()
    theta_plus, _ = lens.image_angles(beta)
    assert theta_plus == pytest.approx(beta, rel=0.05)


def test_magnification_diverges_toward_perfect_alignment():
    lens = PointMassLens(M=1.0, D_L=1000.0, D_S=2000.0)
    theta_E = lens.einstein_angle()
    _, _, total_far = lens.magnification(1.0 * theta_E)
    _, _, total_near = lens.magnification(0.01 * theta_E)
    assert total_near > total_far > 1.0


def test_magnification_canonical_value_at_u_equals_one():
    lens = PointMassLens(M=1.0, D_L=1000.0, D_S=2000.0)
    theta_E = lens.einstein_angle()
    _, _, total = lens.magnification(theta_E)
    assert total == pytest.approx(1.34164, abs=1.0e-4)


def test_invalid_distances_raise():
    with pytest.raises(ValueError):
        PointMassLens(M=1.0, D_L=2000.0, D_S=1000.0)


def test_exact_deflection_matches_weak_field_for_large_impact_parameter():
    # A finite r_far leaves a small residual bias (the flat-space "straight
    # line" total angle swept between two finite radii isn't exactly pi;
    # see the light-bending example script), so allow a generous tolerance
    # rather than an unrealistically tight one.
    bh = SchwarzschildBlackHole(M=1.0)
    b = 50.0
    exact = exact_deflection_angle(bh, b)
    weak = bh.light_deflection_angle(b)
    assert exact == pytest.approx(weak, rel=0.1)


def test_exact_deflection_exceeds_weak_field_closer_to_the_photon_sphere():
    # n_steps is increased (at fixed default r_far) so the ray has enough
    # affine-parameter budget to complete the round trip out to r_far; see
    # exact_deflection_angle's default dtau = 3*r_far/n_steps.
    bh = SchwarzschildBlackHole(M=1.0)
    exact_far = exact_deflection_angle(bh, 200.0, n_steps=600000)
    exact_near = exact_deflection_angle(bh, 15.0, n_steps=600000)
    assert exact_near > bh.light_deflection_angle(15.0)
    assert exact_near > exact_far
