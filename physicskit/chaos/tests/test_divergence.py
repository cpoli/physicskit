import numpy as np
import pytest

from physicskit.chaos.systems.billiards import CircleBilliard, TruncatedCircleBilliard
from physicskit.chaos.systems.continuous import Lorenz
from physicskit.chaos.visualizers.divergence import (
    billiard_trajectory_divergence,
    plot_billiard_divergence,
    plot_trajectory_ensemble,
    trajectory_ensemble,
)
from physicskit.chaos.visualizers.dynamic_plots import animate_billiard_divergence


def test_circle_billiard_separation_is_exactly_conserved():
    """The circle is integrable (sin(phi) is an exact invariant), so two
    nearby rays' separation should stay essentially unchanged forever --
    no exponential growth, because there is no chaos to be sensitive to."""
    billiard = CircleBilliard(radius=1.0)
    delta_0 = 1e-8
    _, separation = billiard_trajectory_divergence(
        billiard,
        pos=billiard.sample_interior_point(),
        vel=(0.2, 1.0),
        delta_0=delta_0,
        n_bounces=100,
    )
    assert np.allclose(separation, delta_0, rtol=1e-6)


@pytest.mark.parametrize("cut", [0.3, 0.05, 0.02])
def test_truncated_circle_billiard_diverges_for_any_cut(cut):
    """Even a tiny truncation destroys integrability: separation must grow
    by many orders of magnitude over enough bounces, for any cut depth."""
    billiard = TruncatedCircleBilliard(radius=1.0, cut=cut)
    delta_0 = 1e-8
    _, separation = billiard_trajectory_divergence(
        billiard,
        pos=billiard.sample_interior_point(),
        vel=(0.2, 1.0),
        delta_0=delta_0,
        n_bounces=300,
    )
    assert separation.max() > 1e-3


def test_billiard_trajectory_divergence_shapes():
    billiard = TruncatedCircleBilliard(radius=1.0, cut=0.05)
    bounce, separation = billiard_trajectory_divergence(billiard, pos=billiard.sample_interior_point(), vel=(0.2, 1.0), n_bounces=50)
    assert bounce.shape == (50,)
    assert separation.shape == (50,)
    assert np.array_equal(bounce, np.arange(1, 51))


def test_plot_billiard_divergence_estimates_positive_exponent_for_chaotic_billiard():
    billiard = TruncatedCircleBilliard(radius=1.0, cut=0.05)
    _, ax, lam = plot_billiard_divergence(billiard, pos=billiard.sample_interior_point(), vel=(0.2, 1.0), n_bounces=100)
    assert lam > 0.0
    assert ax.get_xlabel() == "bounce number k"


def test_animate_billiard_divergence_runs():
    billiard = TruncatedCircleBilliard(radius=1.0, cut=0.05)
    anim = animate_billiard_divergence(billiard, pos=billiard.sample_interior_point(), vel=(0.2, 1.0), n_bounces=30, trail=10)
    anim._draw_frame(20)


def test_trajectory_ensemble_shape_and_zero_spread_members_share_the_reference():
    """With spread=0, every ensemble member is launched from exactly the
    same state as the reference, so all members' trajectories must be
    identical to each other."""
    system = Lorenz()
    all_states = trajectory_ensemble(system, n_members=5, spread=0.0, dt=0.01, n_steps=50, seed=0)
    assert all_states.shape == (5, 51, system.dim)
    for member in all_states[1:]:
        np.testing.assert_allclose(member, all_states[0])


def test_trajectory_ensemble_members_start_within_the_requested_spread():
    system = Lorenz()
    state0 = system.initial_state()
    all_states = trajectory_ensemble(system, state0=state0, n_members=10, spread=1e-3, dt=0.01, n_steps=10, seed=0)
    offsets = np.linalg.norm(all_states[:, 0, :] - state0, axis=1)
    assert np.all(offsets < 1e-3 + 1e-12)


def test_plot_trajectory_ensemble_draws_one_line_per_member():
    system = Lorenz()
    _, ax = plot_trajectory_ensemble(system, n_members=6, spread=1e-4, dt=0.01, n_steps=20, seed=0)
    assert len(ax.get_lines()) == 6
