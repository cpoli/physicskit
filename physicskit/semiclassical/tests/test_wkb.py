"""Physics-correctness tests for physicskit.semiclassical.core.wkb."""

from __future__ import annotations

import numpy as np
import pytest

from physicskit.semiclassical.core.wkb import bohr_sommerfeld_energies, langer_corrected_wkb, turning_points, wkb_wavefunction


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


@pytest.mark.parametrize("l", [0, 1, 3])
def test_langer_corrected_wkb_gives_exact_hydrogen_levels(l):
    energies = langer_corrected_wkb(lambda r: -1.0 / r, l=l, m=1.0, r_max=400.0, n_max=4)
    assert np.allclose(energies, -0.5 / (np.arange(4) + l + 1) ** 2, rtol=1e-8)


def test_langer_corrected_wkb_gives_exact_3d_oscillator_levels():
    energies = langer_corrected_wkb(lambda r: 0.5 * r**2, l=2, m=1.0, r_max=20.0, n_max=4)
    assert np.allclose(energies, 2 * np.arange(4) + 2 + 1.5, rtol=1e-8)


def test_uncorrected_radial_wkb_misses_hydrogen_levels():
    exact = -0.5 / (np.arange(3) + 2) ** 2
    energies = langer_corrected_wkb(lambda r: -1.0 / r, l=1, m=1.0, r_max=400.0, n_max=3, langer=False)
    assert np.all(energies < exact)  # l(l+1) < (l+1/2)^2 makes the centrifugal barrier too weak
    assert np.max(np.abs(energies / exact - 1)) > 0.03


def test_uncorrected_s_wave_integrates_from_the_origin():
    # With no centrifugal barrier the inner limit is r_min, giving E = -1/(2(n+1/2)^2).
    energies = langer_corrected_wkb(lambda r: -1.0 / r, l=0, m=1.0, r_max=400.0, n_max=3, langer=False)
    assert np.allclose(energies, -0.5 / (np.arange(3) + 0.5) ** 2, rtol=1e-6)


def test_langer_corrected_wkb_raises_when_domain_too_small():
    with pytest.raises(ValueError, match="increase r_max"):
        langer_corrected_wkb(lambda r: -1.0 / r, l=0, m=1.0, r_max=10.0, n_max=5)
