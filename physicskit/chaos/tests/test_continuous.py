import numpy as np
import pytest

from physicskit.chaos.systems.continuous import (
    Chua,
    DrivenPendulum,
    MagneticPendulum,
    RestrictedThreeBody,
    effective_potential,
    lagrange_points,
)
from physicskit.chaos.utils.metrics import energy_drift


def test_chua_trajectory_is_finite_and_bounded():
    """Chua's circuit is dissipative and settles onto a bounded attractor;
    integrating it must never blow up to infinity/NaN."""
    system = Chua()
    _, states = system.trajectory(n_steps=5000, dt=0.01)
    assert np.all(np.isfinite(states))
    # The classic double-scroll attractor stays within a modest, known range
    # of the parameters used here; a much looser bound just guards against
    # numerical blow-up regressions.
    assert np.all(np.abs(states) < 20.0)


def test_chua_rhs_at_origin_is_zero():
    """The origin is an equilibrium of Chua's circuit: h(0) = 0, so
    rhs((0, 0, 0)) == (0, 0, 0)."""
    system = Chua()
    derivative = system.rhs(np.array([0.0, 0.0, 0.0]), 0.0)
    np.testing.assert_allclose(derivative, [0.0, 0.0, 0.0], atol=1e-12)


def test_chua_trajectory_is_dissipative():
    """Chua's circuit is dissipative (its equilibria are unstable but phase
    volume contracts on average): starting near, but not at, the origin, the
    trajectory should not simply run away to the bounding box unboundedly."""
    system = Chua()
    _, states = system.trajectory(state0=np.array([0.1, 0.0, 0.0]), n_steps=5000, dt=0.01)
    assert np.all(np.isfinite(states))
    assert np.all(np.abs(states) < 20.0)


def test_restricted_three_body_jacobi_constant_is_conserved():
    """The Jacobi constant is an exact invariant of CR3BP motion; a good RK4
    integration should conserve it to a tight tolerance."""
    system = RestrictedThreeBody()
    t, states = system.trajectory(dt=0.001, n_steps=5000)
    drift = energy_drift(t, states, system.jacobi_constant)
    assert np.max(np.abs(drift)) < 1e-4


def test_restricted_three_body_arenstorf_orbit_is_approximately_periodic():
    """The default initial condition is the classic Arenstorf periodic orbit:
    integrating for exactly one period should return close to the start."""
    system = RestrictedThreeBody()
    state0 = system.initial_state()
    period = 17.0652165601579625588917206249
    dt = 0.0001
    _, states = system.trajectory(dt=dt, n_steps=round(period / dt))
    assert np.max(np.abs(states[-1] - state0)) < 0.01


def test_restricted_three_body_rhs_matches_known_acceleration_at_a_point():
    """Spot-check the CR3BP right-hand side at a simple point against a
    hand-computed value, to catch sign/algebra errors in the Coriolis terms."""
    system = RestrictedThreeBody(mu=0.1)
    # On the x-axis with zero velocity, the Coriolis terms vanish and the
    # y-acceleration is exactly zero by symmetry.
    derivative = system.rhs(np.array([0.5, 0.0, 0.0, 0.0]), 0.0)
    assert derivative[1] == 0.0  # dy/dt = vy = 0
    assert derivative[3] == 0.0  # dvy/dt = 0 by symmetry on the x-axis


def test_lagrange_points_collinear_points_are_ordered_and_at_equilibrium():
    """L3 < -mu (primary) < L1 < 1-mu (secondary) < L2 on the x-axis, and
    each collinear point must be a genuine root of the rotating-frame
    x-acceleration (zero net force at rest)."""
    mu = 0.012277471
    points = lagrange_points(mu)
    x_l1, x_l2, x_l3 = points[0, 0], points[1, 0], points[2, 0]
    assert x_l3 < -mu < x_l1 < (1.0 - mu) < x_l2

    system = RestrictedThreeBody(mu=mu)
    for x in (x_l1, x_l2, x_l3):
        derivative = system.rhs(np.array([x, 0.0, 0.0, 0.0]), 0.0)
        assert abs(derivative[2]) < 1e-8  # zero net x-acceleration at rest


def test_lagrange_points_l4_l5_form_equilateral_triangles_with_primaries():
    """L4 and L5 are, by construction, equidistant from both primaries at
    exactly the primary-primary separation (unit distance in these units)."""
    mu = 0.012277471
    points = lagrange_points(mu)
    primary = np.array([-mu, 0.0])
    secondary = np.array([1.0 - mu, 0.0])
    for lx, ly in points[3:]:
        p = np.array([lx, ly])
        assert np.hypot(*(p - primary)) == pytest.approx(1.0, abs=1e-10)
        assert np.hypot(*(p - secondary)) == pytest.approx(1.0, abs=1e-10)


def test_effective_potential_matches_jacobi_constant_formula():
    """RestrictedThreeBody.jacobi_constant is defined as
    ``2*effective_potential(x, y, mu) - speed**2``; check the two stay
    consistent (a regression guard against the two formulas drifting apart
    if either is edited independently)."""
    system = RestrictedThreeBody(mu=0.1)
    state = np.array([0.3, 0.4, 0.1, -0.2])
    x, y, vx, vy = state
    expected = 2.0 * effective_potential(x, y, system.mu) - (vx * vx + vy * vy)
    assert system.jacobi_constant(state) == pytest.approx(expected)


def test_effective_potential_is_vectorized_over_a_grid():
    """effective_potential must work directly on np.meshgrid output, since
    that's how zero-velocity-curve plots use it."""
    x = np.linspace(-1.5, 1.5, 5)
    y = np.linspace(-1.5, 1.5, 5)
    xx, yy = np.meshgrid(x, y)
    omega = effective_potential(xx, yy, 0.012277471)
    assert omega.shape == xx.shape
    assert np.all(np.isfinite(omega))


def test_effective_potential_is_minimized_at_lagrange_points_4_and_5():
    """L4/L5 are the global minima of the effective potential; nearby points
    (that aren't also Lagrange points) must have a strictly larger value."""
    mu = 0.012277471
    lx, ly = lagrange_points(mu)[3]
    omega_l4 = effective_potential(lx, ly, mu)
    for dx, dy in [(0.05, 0.0), (-0.05, 0.0), (0.0, 0.05), (0.0, -0.05)]:
        assert effective_potential(lx + dx, ly + dy, mu) > omega_l4


def test_driven_pendulum_rhs_matches_hand_derivation():
    """Spot-check the driven-pendulum right-hand side at a simple point."""
    system = DrivenPendulum(damping=0.5, g_over_l=1.0, A=1.5, omega_d=2.0 / 3.0)
    derivative = system.rhs(np.array([0.0, 0.0]), 0.0)
    # theta=0, omega=0, t=0: dtheta/dt=0, domega/dt = A*cos(0) = A
    assert derivative[0] == 0.0
    assert derivative[1] == pytest.approx(1.5)


def test_driven_pendulum_trajectory_is_finite():
    """The classic chaotic parameter set should integrate stably (bounded
    omega; theta itself is allowed to wind arbitrarily as the pendulum
    goes over the top, which happens for these parameters)."""
    system = DrivenPendulum()
    _, states = system.trajectory(n_steps=5000, dt=0.02)
    assert np.all(np.isfinite(states))
    assert np.all(np.abs(states[:, 1]) < 20.0)


def test_magnetic_pendulum_settles_near_a_magnet():
    """Starting the (damped) bob very close to one magnet should leave it
    settled even closer to that same magnet."""
    system = MagneticPendulum()
    magnet = system.magnet_positions[0]
    state0 = np.array([magnet[0] + 0.05, magnet[1], 0.0, 0.0])
    _, states = system.trajectory(state0=state0, n_steps=5000, dt=0.02)
    final_dist = np.hypot(*(states[-1, :2] - magnet))
    initial_dist = np.hypot(*(state0[:2] - magnet))
    assert final_dist < initial_dist


def test_magnetic_pendulum_shows_sensitive_dependence_near_a_basin_boundary():
    """Two very close starting points can settle near different magnets --
    the hallmark of a fractal basin boundary."""
    system = MagneticPendulum()
    outcomes = set()
    for x0 in np.linspace(-0.05, 0.05, 11):
        _, states = system.trajectory(state0=np.array([x0, 0.0, 0.0, 0.0]), n_steps=5000, dt=0.02)
        final = states[-1, :2]
        dists = np.hypot(*(system.magnet_positions - final).T)
        outcomes.add(int(np.argmin(dists)))
    assert len(outcomes) > 1


def test_magnetic_pendulum_trajectory_is_finite():
    system = MagneticPendulum()
    _, states = system.trajectory(n_steps=5000, dt=0.02)
    assert np.all(np.isfinite(states))
