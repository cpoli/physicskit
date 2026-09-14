import numpy as np
import pytest

from physicskit.astro.nbody import NBodySystem, figure_eight_initial_conditions, gravitational_acceleration, leapfrog_step


def test_gravitational_acceleration_two_body():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    masses = np.array([1.0, 1.0])
    acc = gravitational_acceleration(positions, masses, G=1.0)
    assert acc[0] == pytest.approx([1.0, 0.0, 0.0], abs=1e-10)
    assert acc[1] == pytest.approx([-1.0, 0.0, 0.0], abs=1e-10)


def test_leapfrog_step_shapes():
    positions = np.random.default_rng(0).normal(size=(4, 3))
    velocities = np.zeros((4, 3))
    masses = np.ones(4)
    new_pos, new_vel = leapfrog_step(positions, velocities, masses, dt=0.01)
    assert new_pos.shape == (4, 3)
    assert new_vel.shape == (4, 3)


def _circular_two_body_system(G=1.0, m=1.0, d=2.0):
    v = np.sqrt(G * m / (2 * d))
    positions = np.array([[d / 2, 0.0, 0.0], [-d / 2, 0.0, 0.0]])
    velocities = np.array([[0.0, v, 0.0], [0.0, -v, 0.0]])
    masses = np.array([m, m])
    return NBodySystem(positions, velocities, masses, G=G), v, d


def test_two_body_circular_orbit_conserves_energy_and_angular_momentum():
    system, v, d = _circular_two_body_system()
    E0 = system.total_energy()
    L0 = system.total_angular_momentum()

    period = 2 * np.pi * (d / 2) / v
    n_steps = 2000
    dt = period * 5 / n_steps  # five orbits
    system.simulate(dt, n_steps)

    E1 = system.total_energy()
    L1 = system.total_angular_momentum()
    assert E1 == pytest.approx(E0, rel=1e-3)
    assert L1 == pytest.approx(L0, rel=1e-3)


def test_two_body_orbit_returns_near_starting_position_after_one_period():
    system, v, d = _circular_two_body_system()
    period = 2 * np.pi * (d / 2) / v
    n_steps = 500
    dt = period / n_steps
    history = system.simulate(dt, n_steps)
    assert history[-1] == pytest.approx(history[0], abs=1e-2)


def test_figure_eight_initial_conditions_has_zero_net_momentum():
    """The choreography's whole premise is that the three equal masses
    share a common center of mass at rest; a nonzero net momentum would
    mean the figure drifts rather than retracing itself in place."""
    positions, velocities, masses = figure_eight_initial_conditions()
    momentum = np.sum(masses[:, None] * velocities, axis=0)
    assert momentum == pytest.approx([0.0, 0.0, 0.0], abs=1e-8)


def test_figure_eight_initial_conditions_is_periodic_and_conserves_energy():
    """Integrating for one full period (T ~ 6.32591398, the well-known
    value for this choreography) should return each body close to its
    starting position, with energy conserved by the symplectic leapfrog
    integrator."""
    positions, velocities, masses = figure_eight_initial_conditions()
    system = NBodySystem(positions, velocities, masses)
    E0 = system.total_energy()

    period = 6.32591398
    n_steps = 20000
    dt = period / n_steps
    system.simulate(dt, n_steps)

    assert system.total_energy() == pytest.approx(E0, rel=1e-6)
    assert system.positions == pytest.approx(positions, abs=1e-2)
