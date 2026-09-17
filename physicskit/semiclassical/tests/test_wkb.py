"""Physics-correctness tests for physicskit.semiclassical.core.wkb."""

from __future__ import annotations

import numpy as np

from physicskit.semiclassical.core.wkb import bohr_sommerfeld_energies, turning_points, wkb_wavefunction


def test_bohr_sommerfeld_matches_harmonic_oscillator_exact_spectrum():
    V = lambda x: 0.5 * x**2
    energies = bohr_sommerfeld_energies(V, m=1.0, x_min=-20, x_max=20, n_max=6)
    assert np.allclose(energies, np.arange(6) + 0.5, atol=1e-8)


def test_turning_points_finds_root_landing_exactly_on_a_search_grid_point():
    # n_search=3 puts a grid point exactly at x=0, where E - V(x) = 0 - 0 = 0.
    roots = turning_points(E=0.0, V=lambda x: x, x_min=-1.0, x_max=1.0, n_search=3)
    assert roots == [0.0]


def test_wkb_wavefunction_has_n_nodes():
    V = lambda x: 0.5 * x**2
    for n in range(4):
        E_n = bohr_sommerfeld_energies(V, m=1.0, x_min=-20, x_max=20, n_max=n + 1)[n]
        x = np.linspace(-10, 10, 4000)
        psi = wkb_wavefunction(x, E_n, V, m=1.0)
        nonzero = psi[psi != 0]
        n_nodes = np.sum(np.diff(np.sign(nonzero)) != 0)
        assert n_nodes == n
