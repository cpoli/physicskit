import numpy as np
import pytest

from physicskit.chaos.systems.maps import BakersMap, LogisticMap


def test_bakers_map_state_stays_in_unit_square():
    """Every iterate of the (area-preserving) baker's map must stay in [0, 1) x [0, 1)."""
    system = BakersMap(alpha=0.5)
    traj = system.trajectory(np.array([0.123, 0.456]), n_iter=1000)
    assert np.all(traj >= 0.0)
    assert np.all(traj < 1.0)


@pytest.mark.parametrize("alpha", [0.2, 0.5, 0.7])
def test_bakers_map_state_stays_in_unit_square_asymmetric(alpha):
    system = BakersMap(alpha=alpha)
    traj = system.trajectory(np.array([0.37, 0.81]), n_iter=1000)
    assert np.all(traj >= 0.0)
    assert np.all(traj < 1.0)


def test_bakers_map_symmetric_lyapunov_exponents_are_log2():
    """For alpha=0.5, the analytic Lyapunov exponents are exactly +-ln(2)."""
    system = BakersMap(alpha=0.5)
    lam_plus, lam_minus = system.lyapunov_exponents()
    assert lam_plus == pytest.approx(np.log(2.0))
    assert lam_minus == pytest.approx(-np.log(2.0))


def test_bakers_map_lyapunov_exponents_are_symmetric_and_positive():
    system = BakersMap(alpha=0.3)
    lam_plus, lam_minus = system.lyapunov_exponents()
    assert lam_plus > 0.0
    assert lam_minus == pytest.approx(-lam_plus)


def test_bakers_map_symmetric_case_doubles_x_separation_each_step():
    """For alpha=0.5, two points starting in the same branch have their
    x-separation exactly double at every step, until a branch-crossing wraps
    one of them -- the literal "stretch" half of "stretch, cut, and stack"."""
    system = BakersMap(alpha=0.5)
    state_a = np.array([0.1, 0.2])
    state_b = np.array([0.1 + 1e-6, 0.2])
    for _ in range(10):
        dx_before = abs(state_b[0] - state_a[0])
        same_branch = (state_a[0] < 0.5) == (state_b[0] < 0.5)
        state_a = system.step(state_a)
        state_b = system.step(state_b)
        if same_branch:
            dx_after = abs(state_b[0] - state_a[0])
            assert dx_after == pytest.approx(2.0 * dx_before, rel=1e-6)


def test_bakers_map_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        BakersMap(alpha=0.0)
    with pytest.raises(ValueError):
        BakersMap(alpha=1.0)
    with pytest.raises(ValueError):
        BakersMap(alpha=1.5)


def test_logistic_map_step_matches_trajectory_first_iteration():
    """LogisticMap.trajectory() uses a fast, standalone numba loop rather
    than calling .step() itself; check .step() directly gives the same
    first iteration as trajectory() does, so the two paths can't diverge."""
    system = LogisticMap(r=3.9)
    state0 = np.array([0.37])
    stepped = system.step(state0)
    traj = system.trajectory(state0, n_iter=1)
    assert stepped.shape == (1,)
    np.testing.assert_allclose(stepped, traj[1])


def test_logistic_map_state_stays_in_unit_interval():
    system = LogisticMap(r=3.9)
    traj = system.trajectory(np.array([0.37]), n_iter=2000)
    assert traj.shape == (2001, 1)
    assert np.all(traj >= 0.0)
    assert np.all(traj <= 1.0)


def test_logistic_map_converges_to_known_fixed_point_below_r3():
    """For 1 < r < 3, every trajectory converges to the stable fixed point x* = 1 - 1/r."""
    r = 2.5
    system = LogisticMap(r=r)
    traj = system.trajectory(np.array([0.1]), n_iter=500)
    x_star = 1.0 - 1.0 / r
    assert traj[-1, 0] == pytest.approx(x_star, abs=1e-6)


def test_logistic_map_period_two_cycle_just_above_r3():
    """Just above the first period-doubling bifurcation at r=3, orbits settle
    into a period-2 cycle: x(n+2) == x(n) but x(n+1) != x(n)."""
    system = LogisticMap(r=3.2)
    traj = system.trajectory(np.array([0.1]), n_iter=2000)
    tail = traj[-4:, 0]
    assert tail[0] == pytest.approx(tail[2], abs=1e-6)
    assert tail[1] == pytest.approx(tail[3], abs=1e-6)
    assert abs(tail[0] - tail[1]) > 1e-3


def test_logistic_map_sensitive_dependence_at_chaotic_r():
    """At r=4 (fully chaotic), two nearby trajectories must diverge noticeably
    within a modest number of iterations."""
    system = LogisticMap(r=4.0)
    traj_a = system.trajectory(np.array([0.4]), n_iter=40)
    traj_b = system.trajectory(np.array([0.4 + 1e-10]), n_iter=40)
    assert abs(traj_a[-1, 0] - traj_b[-1, 0]) > 1e-3
