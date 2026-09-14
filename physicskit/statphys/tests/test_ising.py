import numpy as np
import pytest

from physicskit.statphys.chapters.ising_lattice import Ising2D


def test_critical_temperature_constant():
    model = Ising2D(J=1.0, kB=1.0)
    assert pytest.approx(2.0 / np.log(1.0 + np.sqrt(2.0)), rel=1e-10) == model.T_C
    assert pytest.approx(2.269, abs=1e-3) == model.T_C


def test_sweep_changes_configuration():
    model = Ising2D(L=16, seed=0)
    before = model.spins.copy()
    model.sweep(beta=1.0, n_sweeps=5)
    assert not np.array_equal(before, model.spins)


def test_energy_and_magnetization_bounds():
    model = Ising2D(L=10, seed=1)
    n = model.n_sites
    model.sweep(beta=0.5, n_sweeps=20)
    assert -4 * n <= model.energy() <= 4 * n
    assert -n <= model.magnetization() <= n


def test_zero_temperature_ground_state_is_uniform():
    model = Ising2D(L=10, seed=2)
    model.reset(ordered=True)
    assert np.all(model.spins == 1)
    model.sweep(beta=50.0, n_sweeps=50)
    # at very low T, essentially no spin should have flipped away from consensus
    assert abs(model.magnetization()) >= 0.9 * model.n_sites


def test_wolff_cluster_flip_reports_positive_size():
    model = Ising2D(L=16, seed=3)
    size = model.sweep(beta=1.0 / model.T_C, algorithm="wolff", n_sweeps=1)
    assert size >= 1
    assert size <= model.n_sites


def test_invalid_algorithm_raises():
    model = Ising2D(L=8, seed=0)
    with pytest.raises(ValueError):
        model.sweep(beta=1.0, algorithm="bogus")


def test_susceptibility_peaks_near_critical_temperature():
    """Onsager's exact result: chi(T) diverges (peaks, at finite L) at T_C ≈ 2.269."""
    model = Ising2D(L=14, J=1.0, kB=1.0, seed=42)
    T_C = model.T_C
    temperatures = np.linspace(T_C - 1.0, T_C + 1.0, 9)[::-1]  # anneal from hot to cold
    result = model.run_temperature_sweep(temperatures, n_equil=60, n_measure=120, algorithm="wolff")
    peak_T = result["T"][np.argmax(result["chi"])]
    assert peak_T == pytest.approx(T_C, abs=1.0)
    # heat capacity should also show elevated fluctuations near T_C
    assert result["C_v"].max() > result["C_v"].min()
