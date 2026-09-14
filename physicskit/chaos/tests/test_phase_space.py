import numpy as np
import pytest

from physicskit.chaos.systems.billiards import BunimovichStadium, CircleBilliard, SinaiBilliard
from physicskit.chaos.visualizers.phase_space import plot_billiard_trajectory, plot_poincare_section

ALL_BILLIARDS = [
    lambda: CircleBilliard(radius=1.0),
    lambda: SinaiBilliard(cell_size=2.0, scatterer_radius=0.5),
    lambda: BunimovichStadium(radius=1.0, straight_length=2.0),
]


@pytest.mark.parametrize("make_billiard", ALL_BILLIARDS)
def test_simulate_many_rays_matches_per_ray_simulate(make_billiard):
    """The parallel batch kernel must reproduce plain per-ray simulate() exactly."""
    billiard = make_billiard()
    pos = billiard.sample_interior_point()
    angles = np.array([0.3, 1.1, 2.7, 4.0, 5.9])

    batch = billiard.simulate_many_rays(pos, angles, n_bounces=30)

    expected_s, expected_sin_phi = [], []
    for angle in angles:
        vel = np.array([np.cos(angle), np.sin(angle)])
        result = billiard.simulate(pos, vel, n_bounces=30)
        expected_s.append(result["s"])
        expected_sin_phi.append(result["sin_phi"])

    np.testing.assert_allclose(batch["s"], np.concatenate(expected_s))
    np.testing.assert_allclose(batch["sin_phi"], np.concatenate(expected_sin_phi))


def test_plot_poincare_section_returns_pooled_scatter_within_bounds():
    billiard = SinaiBilliard(cell_size=2.0, scatterer_radius=0.5)
    _, ax = plot_poincare_section(billiard, n_rays=8, n_bounces=20, seed=0)
    collection = ax.collections[0]
    offsets = collection.get_offsets()
    assert offsets.shape == (8 * 20, 2)
    s, sin_phi = offsets[:, 0], offsets[:, 1]
    assert np.all(s >= 0.0) and np.all(s <= billiard.perimeter())
    assert np.all(sin_phi >= -1.0) and np.all(sin_phi <= 1.0)


def test_plot_billiard_trajectory_draws_boundary_and_path():
    billiard = CircleBilliard(radius=1.0)
    pos = billiard.sample_interior_point()
    _, ax = plot_billiard_trajectory(billiard, pos, vel=(1.0, 0.3), n_bounces=15)
    assert len(ax.lines) == 2


def test_plot_poincare_section_defaults_to_billiards_own_interior_point():
    """Rays launched from a circle's exact center (its default interior
    point) all hit the boundary head-on, so every ray shares sin(phi) = 0 --
    a documented degenerate case (see test below), not a bug in the default
    itself: the default must still be ``sample_interior_point()``."""
    billiard = CircleBilliard(radius=1.0)
    _, ax = plot_poincare_section(billiard, n_rays=10, n_bounces=20, seed=0)
    sin_phi = ax.collections[0].get_offsets()[:, 1]
    assert np.allclose(sin_phi, 0.0, atol=1e-9)


def test_plot_poincare_section_pos_override_avoids_circle_center_degeneracy():
    """An off-center pos, unlike the default center, must produce a genuine
    family of distinct invariant curves (one sin(phi) per ray, not all the
    same value)."""
    billiard = CircleBilliard(radius=1.0)
    _, ax = plot_poincare_section(billiard, n_rays=10, n_bounces=20, pos=(0.3, 0.0), seed=0)
    offsets = ax.collections[0].get_offsets()
    sin_phi_per_ray = np.asarray(offsets[:, 1]).reshape(10, 20)
    per_ray_values = sin_phi_per_ray[:, 0]
    assert not np.allclose(sin_phi_per_ray, 0.0, atol=1e-9)
    assert np.unique(np.round(per_ray_values, 6)).size == 10
