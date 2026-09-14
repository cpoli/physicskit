import numpy as np
import pytest

from physicskit.chaos.systems.billiards import (
    BunimovichStadium,
    CircleBilliard,
    EllipseBilliard,
    RectangleBilliard,
    SinaiBilliard,
    TruncatedCircleBilliard,
)

ALL_BILLIARDS = [
    lambda: CircleBilliard(radius=1.0),
    lambda: RectangleBilliard(width=2.0, height=1.0),
    lambda: SinaiBilliard(cell_size=2.0, scatterer_radius=0.5),
    lambda: BunimovichStadium(radius=1.0, straight_length=2.0),
    lambda: TruncatedCircleBilliard(radius=1.0, cut=0.3),
    # Fewer segments than the default (2000) to keep the test suite fast; the
    # polygon-approximation physics is exercised the same way either way.
    lambda: EllipseBilliard(semi_major=1.5, semi_minor=1.0, n_segments=300),
]


@pytest.mark.parametrize("make_billiard", ALL_BILLIARDS)
def test_speed_conserved_across_bounces(make_billiard):
    """Specular reflection must preserve |v| = 1 at every bounce."""
    billiard = make_billiard()
    pos = billiard.sample_interior_point()
    rng = np.random.default_rng(0)
    for angle in rng.uniform(0, 2 * np.pi, size=10):
        vel = np.array([np.cos(angle), np.sin(angle)])
        result = billiard.simulate(pos, vel, n_bounces=200)
        speeds = np.hypot(result["vx"], result["vy"])
        np.testing.assert_allclose(speeds, 1.0, atol=1e-9)


@pytest.mark.parametrize("make_billiard", ALL_BILLIARDS)
def test_bounce_points_lie_within_perimeter_range(make_billiard):
    """Boundary arclength coordinate s must stay within [0, perimeter)."""
    billiard = make_billiard()
    pos = billiard.sample_interior_point()
    result = billiard.simulate(pos, np.array([1.0, 0.3]), n_bounces=200)
    assert np.all(result["s"] >= -1e-6)
    assert np.all(result["s"] <= billiard.perimeter() + 1e-6)


@pytest.mark.parametrize("make_billiard", ALL_BILLIARDS)
def test_sin_phi_within_unit_range(make_billiard):
    billiard = make_billiard()
    pos = billiard.sample_interior_point()
    result = billiard.simulate(pos, np.array([0.7, 0.5]), n_bounces=200)
    assert np.all(result["sin_phi"] >= -1.0 - 1e-9)
    assert np.all(result["sin_phi"] <= 1.0 + 1e-9)


@pytest.mark.parametrize("make_billiard", ALL_BILLIARDS)
def test_bounce_points_on_or_inside_boundary(make_billiard):
    """No bounce point should end up strictly outside the billiard's convex-ish
    bounding region (a coarse sanity check against ray-tracing bugs)."""
    billiard = make_billiard()
    pos = billiard.sample_interior_point()
    result = billiard.simulate(pos, np.array([0.4, 0.9]), n_bounces=100)
    boundary = billiard.boundary_polyline()
    margin = 1e-6
    # boundary_polyline() separates disjoint closed components (e.g. Sinai's
    # outer wall and inner scatterer) with a row of NaN; use nanmax/nanmin so
    # that doesn't turn the bounding box into NaN.
    assert result["x"].max() <= np.nanmax(boundary[:, 0]) + margin
    assert result["x"].min() >= np.nanmin(boundary[:, 0]) - margin
    assert result["y"].max() <= np.nanmax(boundary[:, 1]) + margin
    assert result["y"].min() >= np.nanmin(boundary[:, 1]) - margin


def test_circle_billiard_is_integrable_constant_sin_phi():
    """In a circular billiard, sin(phi) is an exact invariant of the motion."""
    billiard = CircleBilliard(radius=1.0)
    result = billiard.simulate(np.array([0.3, 0.0]), np.array([0.2, 1.0]), n_bounces=100)
    np.testing.assert_allclose(result["sin_phi"], result["sin_phi"][0], atol=1e-8)


def test_sinai_billiard_rejects_scatterer_larger_than_cell():
    with pytest.raises(ValueError):
        SinaiBilliard(cell_size=2.0, scatterer_radius=1.5)


def test_truncated_circle_rejects_invalid_cut():
    with pytest.raises(ValueError):
        TruncatedCircleBilliard(radius=1.0, cut=1.5)


def test_ellipse_billiard_rejects_invalid_axes():
    with pytest.raises(ValueError):
        EllipseBilliard(semi_major=1.0, semi_minor=1.5)  # major must exceed minor
    with pytest.raises(ValueError):
        EllipseBilliard(semi_major=-1.0, semi_minor=0.5)
    with pytest.raises(ValueError):
        EllipseBilliard(semi_major=1.0, semi_minor=0.0)


def test_ellipse_billiard_foci_at_known_positions():
    billiard = EllipseBilliard(semi_major=1.5, semi_minor=1.0)
    f1, f2 = billiard.foci()
    c = np.sqrt(1.5**2 - 1.0**2)
    np.testing.assert_allclose(f1, [c, 0.0])
    np.testing.assert_allclose(f2, [-c, 0.0])


def test_ellipse_billiard_sin_phi_stays_bounded_by_caustic():
    """A single ellipse-billiard trajectory stays tangent to one confocal
    caustic, so its sin(phi) values are confined to a bounded sub-range of
    [-1, 1] rather than filling it (unlike a chaotic billiard)."""
    billiard = EllipseBilliard(semi_major=1.5, semi_minor=1.0, n_segments=1000)
    result = billiard.simulate(billiard.sample_interior_point(), np.array([0.3, 1.0]), n_bounces=300)
    assert result["sin_phi"].max() - result["sin_phi"].min() < 1.0
