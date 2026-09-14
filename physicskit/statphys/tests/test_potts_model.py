import numpy as np
import pytest

from physicskit.statphys.chapters.ising_lattice import PottsModel2D


def test_critical_temperature_formula():
    model = PottsModel2D(q=3, J=1.0, kB=1.0)
    assert pytest.approx(1.0 / np.log(1.0 + np.sqrt(3.0)), rel=1e-10) == model.T_C


def test_states_in_valid_range():
    model = PottsModel2D(L=12, q=5, seed=0)
    assert model.spins.min() >= 0
    assert model.spins.max() < 5


def test_order_parameter_bounds():
    model = PottsModel2D(L=12, q=3, seed=1)
    m = model.order_parameter()
    assert 0.0 <= m <= 1.0 + 1e-9


def test_ordered_state_has_order_parameter_one():
    model = PottsModel2D(L=10, q=4, seed=2)
    model.spins[:] = 0
    assert model.order_parameter() == pytest.approx(1.0)


def test_sweep_reduces_energy_at_low_temperature():
    model = PottsModel2D(L=12, q=3, seed=3)
    E0 = model.energy()
    model.sweep(beta=20.0, n_sweeps=50)
    E1 = model.energy()
    assert E1 <= E0
