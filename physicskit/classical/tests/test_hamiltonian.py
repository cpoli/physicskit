"""Tests for physicskit.classical.systems.hamiltonian's action-angle helper and
PendulumSwarm.per_particle_energy(), neither of which any other test in
the suite calls (test_conservation.py only checks PendulumSwarm's total
energy/Liouville area, and pendulum_action_angle isn't used anywhere)."""

from __future__ import annotations

import numpy as np
import pytest

from physicskit.classical.systems.hamiltonian import PendulumSwarm, pendulum_action_angle


def test_pendulum_action_angle_small_oscillation_limit_matches_harmonic_period():
    """As E -> -g/l (the bottom of the well), a pendulum's period should
    approach the small-oscillation harmonic result T = 2*pi/sqrt(g/l)."""
    g_over_l = 2.0
    _, period = pendulum_action_angle(-g_over_l + 1e-6, g_over_l)
    assert period == pytest.approx(2.0 * np.pi / np.sqrt(g_over_l), rel=1e-3)


def test_pendulum_action_angle_rejects_energy_outside_librating_range():
    g_over_l = 1.0
    with pytest.raises(ValueError):
        pendulum_action_angle(g_over_l, g_over_l)
    with pytest.raises(ValueError):
        pendulum_action_angle(-g_over_l, g_over_l)
    with pytest.raises(ValueError):
        pendulum_action_angle(2.0 * g_over_l, g_over_l)


def test_pendulum_swarm_per_particle_energy_sums_to_the_total_energy():
    swarm = PendulumSwarm([0.1, 0.2, 0.3], [0.05, -0.05, 0.0], g_over_l=1.5)
    per_particle = swarm.per_particle_energy()
    assert per_particle.shape == (3,)
    assert np.sum(per_particle) == pytest.approx(swarm.energy())


def test_pendulum_swarm_per_particle_energy_accepts_explicit_state_override():
    swarm = PendulumSwarm([0.1, 0.2, 0.3], [0.05, -0.05, 0.0], g_over_l=1.5)
    default = swarm.per_particle_energy()
    overridden = swarm.per_particle_energy(q=np.zeros(3), p=np.zeros(3))
    np.testing.assert_allclose(overridden, -1.5)
    assert not np.allclose(default, overridden)
