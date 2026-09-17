"""Physics-correctness tests for physicskit.semiclassical.core.gutzwiller."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import find_peaks

from physicskit.semiclassical.core.gutzwiller import (
    classical_period,
    gutzwiller_amplitude_from_monodromy,
    gutzwiller_density_of_states,
)
from physicskit.semiclassical.core.wkb import bohr_sommerfeld_energies


@pytest.mark.slow
def test_gutzwiller_trace_formula_peaks_match_bohr_sommerfeld_spectrum():
    V = lambda x: 0.5 * x**2
    E_grid = np.linspace(0.2, 4.5, 200)
    dos = gutzwiller_density_of_states(E_grid, V, m=1.0, x_min=-20, x_max=20)
    peak_idx, _ = find_peaks(dos, height=0.3 * dos.max())
    exact = bohr_sommerfeld_energies(V, m=1.0, x_min=-20, x_max=20, n_max=4)
    assert np.allclose(E_grid[peak_idx], exact, atol=0.05)


def test_classical_period_matches_harmonic_oscillator():
    V = lambda x: 0.5 * x**2
    T = classical_period(E=3.0, V=V, m=1.0, x_min=-20, x_max=20)
    assert abs(T - 2 * np.pi) < 1e-4


def test_gutzwiller_amplitude_from_monodromy():
    M = np.array([[2.0, 0.0], [0.0, 0.5]])
    assert abs(gutzwiller_amplitude_from_monodromy(M) - 1.0 / np.sqrt(0.5)) < 1e-10
