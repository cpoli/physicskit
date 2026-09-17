"""Tests for physicskit.statphys.core.md_engine.initialize_maxwell_boltzmann_velocities,
which is otherwise only exercised via its doctest (not part of the
regular pytest/coverage run). lj_forces and velocity_verlet_step are
exercised indirectly through LennardJonesGas in test_molecular_dynamics.py.
"""

from __future__ import annotations

import numpy as np

from physicskit.statphys.core.md_engine import initialize_maxwell_boltzmann_velocities


def test_initialize_maxwell_boltzmann_velocities_has_zero_net_momentum():
    rng = np.random.default_rng(0)
    v = initialize_maxwell_boltzmann_velocities(200, temperature=1.5, mass=1.0, kB=1.0, rng=rng)
    assert v.shape == (200, 2)
    np.testing.assert_allclose(v.mean(axis=0), 0.0, atol=1e-9)


def test_initialize_maxwell_boltzmann_velocities_uses_default_rng_when_none_given():
    v = initialize_maxwell_boltzmann_velocities(50, temperature=1.0)
    assert v.shape == (50, 2)
    assert np.all(np.isfinite(v))


def test_initialize_maxwell_boltzmann_velocities_variance_scales_with_temperature():
    rng = np.random.default_rng(1)
    v_cold = initialize_maxwell_boltzmann_velocities(5000, temperature=0.5, mass=1.0, kB=1.0, rng=rng)
    v_hot = initialize_maxwell_boltzmann_velocities(5000, temperature=4.0, mass=1.0, kB=1.0, rng=rng)
    assert v_hot.var() > v_cold.var()
