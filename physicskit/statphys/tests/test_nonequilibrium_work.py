import numpy as np
import pytest

from physicskit.statphys.chapters.nonequilibrium_work import JarzynskiHarmonicTrap


def test_run_protocol_returns_one_work_sample_per_trajectory():
    model = JarzynskiHarmonicTrap(seed=0)
    work = model.run_protocol(lambda_0=0.0, lambda_1=1.0, tau=2.0, n_steps=100, n_trajectories=50)
    assert work.shape == (50,)
    assert np.all(np.isfinite(work))


def test_jarzynski_estimate_of_constant_work_is_exact():
    model = JarzynskiHarmonicTrap(kB=1.0, T=1.0)
    work = np.full(1000, 2.5)
    assert model.jarzynski_free_energy_estimate(work) == pytest.approx(2.5)


def test_jarzynski_equality_recovers_zero_free_energy_for_harmonic_trap():
    # Only the trap center moves, never the stiffness, so the true
    # equilibrium free energy difference is exactly zero regardless of the
    # (finite-speed, dissipative) protocol used to drag it there.
    model = JarzynskiHarmonicTrap(k=1.0, gamma=1.0, kB=1.0, T=1.0, seed=7)
    work = model.run_protocol(lambda_0=0.0, lambda_1=2.0, tau=3.0, n_steps=300, n_trajectories=3000)
    dF_estimate = model.jarzynski_free_energy_estimate(work)
    assert dF_estimate == pytest.approx(0.0, abs=0.3)


def test_second_law_mean_work_exceeds_free_energy_difference():
    model = JarzynskiHarmonicTrap(k=1.0, gamma=1.0, kB=1.0, T=1.0, seed=8)
    work = model.run_protocol(lambda_0=0.0, lambda_1=2.0, tau=1.0, n_steps=200, n_trajectories=2000)
    assert work.mean() > 0.0
