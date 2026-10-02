import numpy as np
import pytest

from physicskit.statphys.chapters.ising_lattice import Ising2D
from physicskit.statphys.chapters.wang_landau import (
    WangLandauIsing,
    canonical_from_density_of_states,
    ising_density_of_states_exact,
)
from physicskit.statphys.core.monte_carlo import seed_numba_random, swendsen_wang_step_ising


def _log(g):
    return np.log(g, out=np.full(g.shape, -np.inf), where=g > 0)


def test_swendsen_wang_infinite_temperature_flips_every_site_independently():
    # at beta = 0 no bond is activated, so every site is its own cluster
    seed_numba_random(0)
    spins = np.ones((8, 8), dtype=np.int64)
    assert swendsen_wang_step_ising(spins, 0.0, 1.0) == 64


def test_swendsen_wang_zero_temperature_ordered_lattice_is_one_cluster():
    seed_numba_random(1)
    spins = np.ones((8, 8), dtype=np.int64)
    assert swendsen_wang_step_ising(spins, 50.0, 1.0) == 1
    assert abs(spins.sum()) == 64


def test_swendsen_wang_energy_matches_exact_small_lattice():
    # canonical <E> from the exact 4x4 density of states
    T = 2.5
    E, g = ising_density_of_states_exact(4)
    exact = canonical_from_density_of_states(E, _log(g), [T], n_sites=16)["E"][0]
    model = Ising2D(L=4, seed=3)
    out = model.run_temperature_sweep([T], n_equil=500, n_measure=20000, algorithm="swendsen-wang")
    assert out["E"][0] == pytest.approx(exact, abs=0.02)


def test_sweep_rejects_unknown_algorithm():
    with pytest.raises(ValueError):
        Ising2D(L=4, seed=0).sweep(1.0, algorithm="heat-bath")


def test_exact_density_of_states_counts_every_configuration():
    E, g = ising_density_of_states_exact(4)
    assert g.sum() == 2**16
    assert g[0] == g[-1] == 2
    assert g[1] == g[-2] == 0
    np.testing.assert_array_equal(g, g[::-1])


def test_wang_landau_recovers_exact_density_of_states():
    E, g = ising_density_of_states_exact(4)
    _, log_g = WangLandauIsing(L=4, seed=0).run(log_f_final=1e-5)
    ok = g > 0
    np.testing.assert_allclose(log_g[ok], np.log(g[ok]), atol=0.3)
    assert np.all(np.isneginf(log_g[~ok]))


def test_canonical_low_temperature_limit_is_ground_state():
    E, g = ising_density_of_states_exact(4)
    out = canonical_from_density_of_states(E, _log(g), [0.05], n_sites=16)
    assert out["E"][0] == pytest.approx(-2.0)
    assert out["C_v"][0] == pytest.approx(0.0, abs=1e-10)
    assert out["log_Z"][0] == pytest.approx(np.log(2) + 32 / 0.05)
