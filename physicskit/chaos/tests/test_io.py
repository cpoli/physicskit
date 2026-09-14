import numpy as np

from physicskit.chaos.systems.billiards import SinaiBilliard
from physicskit.chaos.systems.continuous import Lorenz
from physicskit.chaos.utils.io import load_arrays, load_system_config, save_arrays, save_system_config


def test_save_and_load_system_config_roundtrips_scalar_params(tmp_path):
    system = Lorenz(sigma=11.0, rho=27.5, beta=8.0 / 3.0)
    path = tmp_path / "lorenz.json"
    save_system_config(system, path)
    restored = load_system_config(path)

    assert type(restored) is Lorenz
    assert restored.sigma == system.sigma
    assert restored.rho == system.rho
    assert restored.beta == system.beta


def test_save_and_load_system_config_roundtrips_billiard_params(tmp_path):
    billiard = SinaiBilliard(cell_size=2.5, scatterer_radius=0.7)
    path = tmp_path / "sinai.json"
    save_system_config(billiard, path)
    restored = load_system_config(path)

    assert type(restored) is SinaiBilliard
    assert restored.cell_size == billiard.cell_size
    assert restored.scatterer_radius == billiard.scatterer_radius


def test_save_and_load_arrays_roundtrips_exactly(tmp_path):
    t = np.linspace(0.0, 10.0, 50)
    states = np.random.default_rng(0).normal(size=(50, 3))
    path = tmp_path / "run.npz"

    save_arrays(path, t=t, states=states)
    loaded = load_arrays(path)

    assert set(loaded.keys()) == {"t", "states"}
    np.testing.assert_array_equal(loaded["t"], t)
    np.testing.assert_array_equal(loaded["states"], states)
