import numpy as np
import pytest

from physicskit.relativity.core.raytracer import OUTCOME_CAPTURED, OUTCOME_DISK, OUTCOME_ESCAPED, camera_basis
from physicskit.relativity.visualizers.shadow_render import render_black_hole_image


def test_camera_basis_is_orthonormal():
    view_dir, right, up = camera_basis(inclination=1.2)
    for v in (view_dir, right, up):
        assert np.linalg.norm(v) == pytest.approx(1.0)
    assert np.dot(view_dir, right) == pytest.approx(0.0, abs=1e-10)
    assert np.dot(view_dir, up) == pytest.approx(0.0, abs=1e-10)
    assert np.dot(right, up) == pytest.approx(0.0, abs=1e-10)


def test_shadow_fraction_matches_critical_impact_parameter_without_disk():
    M = 1.0
    screen_half_width = 15.0
    result = render_black_hole_image(
        M=M,
        ny=120,
        nx=120,
        screen_half_width=screen_half_width,
        r_disk_inner=1.0,
        r_disk_outer=0.0,
        n_steps=2000,
    )
    outcomes = result["outcomes"]
    captured_fraction = np.mean(outcomes == OUTCOME_CAPTURED)
    b_crit = 3.0 * np.sqrt(3.0) * M
    expected_fraction = np.pi * b_crit**2 / (2.0 * screen_half_width) ** 2
    assert captured_fraction == pytest.approx(expected_fraction, rel=0.15)


def test_all_pixels_produce_a_valid_outcome():
    result = render_black_hole_image(M=1.0, ny=30, nx=30, n_steps=1500)
    outcomes = result["outcomes"]
    assert np.all(outcomes >= 0)
    assert np.all(outcomes <= 3)


def test_disk_pixels_report_radius_within_disk_bounds():
    result = render_black_hole_image(M=1.0, ny=80, nx=80, r_disk_inner=6.0, r_disk_outer=20.0, n_steps=2000)
    outcomes = result["outcomes"]
    hit_radii = result["hit_radii"]
    disk_mask = outcomes == OUTCOME_DISK
    assert np.any(disk_mask)
    assert np.all(hit_radii[disk_mask] >= 6.0 - 1e-6)
    assert np.all(hit_radii[disk_mask] <= 20.0 + 1e-6)


def test_face_on_view_is_rotationally_symmetric_without_disk():
    # at inclination 0, the (diskless) shadow is a circle around the image center
    result = render_black_hole_image(M=1.0, ny=61, nx=61, inclination=1e-6, r_disk_inner=1.0, r_disk_outer=0.0, n_steps=1500)
    outcomes = result["outcomes"]
    captured = outcomes == OUTCOME_CAPTURED
    # symmetric under a vertical flip for a face-on, non-tilted view
    assert np.array_equal(captured, captured[::-1, :]) or np.mean(captured == captured[::-1, :]) > 0.9


def test_escaped_pixels_are_far_from_center_pixels_dominated_by_capture():
    result = render_black_hole_image(M=1.0, ny=41, nx=41, r_disk_inner=1.0, r_disk_outer=0.0, n_steps=1500)
    outcomes = result["outcomes"]
    center = outcomes.shape[0] // 2
    assert outcomes[center, center] == OUTCOME_CAPTURED
    assert outcomes[0, 0] == OUTCOME_ESCAPED
