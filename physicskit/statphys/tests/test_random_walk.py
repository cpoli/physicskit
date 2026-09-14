import numpy as np
import pytest

from physicskit.statphys.chapters.random_walk import RandomWalk


def test_lattice_walk_shape():
    walk = RandomWalk(n_walkers=10, n_steps=50, dim=2, kind="lattice", seed=0)
    traj = walk.run()
    assert traj.shape == (51, 10, 2)


def test_walks_start_at_origin():
    walk = RandomWalk(n_walkers=20, n_steps=30, dim=3, kind="gaussian", seed=1)
    traj = walk.run()
    assert np.all(traj[0] == 0.0)


def test_msd_increases_with_time():
    walk = RandomWalk(n_walkers=200, n_steps=200, dim=2, kind="lattice", seed=1)
    _t, msd = walk.mean_squared_displacement()
    assert msd[-1] > msd[10]


def test_diffusion_coefficient_positive():
    walk = RandomWalk(n_walkers=300, n_steps=300, dim=1, kind="gaussian", step_std=1.0, seed=2)
    assert walk.diffusion_coefficient() > 0


def test_gaussian_walk_diffusion_coefficient_matches_theory():
    # For 1D Gaussian steps with std sigma, D = sigma^2 / 2 (Einstein relation, dt=1).
    walk = RandomWalk(n_walkers=3000, n_steps=500, dim=1, kind="gaussian", step_std=1.5, seed=3)
    D = walk.diffusion_coefficient()
    assert pytest.approx(1.5**2 / 2.0, rel=0.2) == D


def test_invalid_kind_raises():
    walk = RandomWalk(n_walkers=5, n_steps=10, kind="bogus")
    with pytest.raises(ValueError):
        walk.run()

    with pytest.raises(ValueError):
        walk.mean_squared_displacement()


def test_final_displacement_histogram_normalizes():
    walk = RandomWalk(n_walkers=500, n_steps=100, dim=2, kind="lattice", seed=4)
    centers, density = walk.final_displacement_histogram(bins=30)
    width = centers[1] - centers[0]
    assert np.sum(density) * width == pytest.approx(1.0, abs=0.05)
