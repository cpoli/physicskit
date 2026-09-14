import numpy as np

from physicskit.statphys.chapters.ising_lattice import XYModel2D


def test_theta_in_valid_range():
    model = XYModel2D(L=12, seed=0)
    assert np.all(model.theta >= 0.0)
    assert np.all(model.theta < 2.0 * np.pi)


def test_sweep_keeps_theta_in_range():
    model = XYModel2D(L=12, seed=1)
    model.sweep(beta=1.0, n_sweeps=10)
    assert np.all(model.theta >= 0.0)
    assert np.all(model.theta < 2.0 * np.pi)


def test_kt_temperature_is_positive_and_reasonable():
    model = XYModel2D(J=1.0, kB=1.0)
    assert 0.5 < model.T_KT < 1.5


def test_uniform_configuration_has_zero_vorticity():
    model = XYModel2D(L=10, seed=2)
    model.theta[:] = 0.0
    q = model.vorticity()
    assert np.allclose(q, 0.0, atol=1e-9)
    n_v, n_av = model.vortex_count()
    assert n_v == 0 and n_av == 0


def test_low_temperature_has_small_ordered_magnetization():
    model = XYModel2D(L=14, seed=3)
    model.theta[:] = model._rng.uniform(-0.05, 0.05, size=model.theta.shape) % (2 * np.pi)
    model.sweep(beta=10.0, n_sweeps=30)
    mag = model.magnetization_vector()
    norm = np.linalg.norm(mag) / model.n_sites
    assert norm > 0.5  # strongly aligned spins retain a large net moment


def test_energy_is_finite():
    model = XYModel2D(L=10, seed=4)
    model.sweep(beta=1.0, n_sweeps=5)
    assert np.isfinite(model.energy())
