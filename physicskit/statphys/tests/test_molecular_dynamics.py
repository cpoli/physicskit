import numpy as np
import pytest

from physicskit.statphys.chapters.molecular_dynamics import LennardJonesGas


def test_particles_start_inside_box():
    gas = LennardJonesGas(n_particles=36, box_size=12.0, seed=0)
    assert np.all(gas.positions >= 0.0)
    assert np.all(gas.positions < 12.0)


def test_zero_net_momentum_initially():
    gas = LennardJonesGas(n_particles=50, box_size=15.0, seed=1)
    assert np.allclose(gas.velocities.mean(axis=0), 0.0, atol=1e-9)


def test_maxwell_boltzmann_initial_velocity_distribution_has_zero_net_momentum():
    gas = LennardJonesGas(n_particles=50, box_size=15.0, initial_velocity_distribution="maxwell_boltzmann", seed=7)
    assert np.allclose(gas.velocities.mean(axis=0), 0.0, atol=1e-9)


def test_invalid_initial_velocity_distribution_raises():
    with pytest.raises(ValueError):
        LennardJonesGas(n_particles=10, box_size=10.0, initial_velocity_distribution="bogus", seed=0)


def test_step_advances_time_and_keeps_particles_in_box():
    gas = LennardJonesGas(n_particles=36, box_size=12.0, dt=0.005, seed=2)
    gas.step(n_steps=20)
    assert gas.time == pytest.approx(20 * 0.005)
    assert np.all(gas.positions >= 0.0)
    assert np.all(gas.positions < 12.0)


def test_energy_is_approximately_conserved():
    gas = LennardJonesGas(n_particles=36, box_size=14.0, dt=0.002, seed=3)
    E0 = gas.total_energy()
    gas.step(n_steps=300)
    E1 = gas.total_energy()
    assert pytest.approx(E0, rel=0.05) == E1


def test_h_function_is_finite():
    gas = LennardJonesGas(n_particles=64, box_size=16.0, seed=4)
    H = gas.h_function()
    assert np.isfinite(H)


def test_run_returns_expected_history_shape():
    gas = LennardJonesGas(n_particles=49, box_size=14.0, seed=5)
    history = gas.run(n_steps=100, steps_per_record=25)
    assert history["H"].shape == (5,)
    assert history["time"][0] == 0.0
    assert history["time"][-1] == pytest.approx(100 * gas.dt)


def test_h_theorem_relaxation_direction():
    """A far-from-equilibrium delta-function velocity distribution should see H
    decrease (or stay flat) on average as the gas thermalizes toward Maxwell-Boltzmann."""
    gas = LennardJonesGas(
        n_particles=100,
        box_size=18.0,
        temperature_init=1.0,
        initial_velocity_distribution="delta",
        dt=0.003,
        seed=6,
    )
    history = gas.run(n_steps=1500, steps_per_record=100)
    early_H = history["H"][:3].mean()
    late_H = history["H"][-3:].mean()
    assert late_H <= early_H + 0.05
