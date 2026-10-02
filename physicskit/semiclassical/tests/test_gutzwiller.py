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


def _disk_levels(k_max):
    from scipy.special import jn_zeros

    levels = []
    for m in range(int(k_max) + 2):
        zeros = jn_zeros(m, int(k_max) + 2)
        zeros = zeros[zeros < k_max]
        levels.extend(zeros if m == 0 else np.repeat(zeros, 2))
    return np.sort(levels)


def test_balian_bloch_counting_function_has_zero_mean_residual_for_disk():
    from physicskit.semiclassical.core.gutzwiller import balian_bloch_counting_function

    levels = _disk_levels(60.0)
    ks = np.linspace(20.0, 60.0, 4000)
    residual = np.searchsorted(levels, ks) - balian_bloch_counting_function(ks, area=np.pi, perimeter=2 * np.pi, total_curvature=2 * np.pi)
    assert abs(residual.mean()) < 0.1
    weyl_only = np.searchsorted(levels, ks) - np.pi * ks**2 / (4 * np.pi)
    assert weyl_only.mean() < -5  # the area term alone overcounts by L k / 4 pi


def test_balian_bloch_constant_for_rectangle_and_neumann_sign():
    from physicskit.semiclassical.core.gutzwiller import balian_bloch_counting_function, balian_bloch_level_density

    corners = [np.pi / 2] * 4
    assert balian_bloch_counting_function(0.0, 2.0, 6.0, corner_angles=corners) == pytest.approx(0.25)
    dirichlet = balian_bloch_counting_function(5.0, 2.0, 6.0, corner_angles=corners)
    neumann = balian_bloch_counting_function(5.0, 2.0, 6.0, corner_angles=corners, boundary="neumann")
    assert neumann - dirichlet == pytest.approx(2 * 6.0 * 5.0 / (4 * np.pi))
    k = np.array([3.0, 7.0])
    h = 1e-6
    derivative = (balian_bloch_counting_function(k + h, 2.0, 6.0) - balian_bloch_counting_function(k - h, 2.0, 6.0)) / (2 * h)
    assert np.allclose(balian_bloch_level_density(k, 2.0, 6.0), derivative)


def test_balian_bloch_rejects_unknown_boundary():
    from physicskit.semiclassical.core.gutzwiller import balian_bloch_counting_function, balian_bloch_level_density

    with pytest.raises(ValueError):
        balian_bloch_counting_function(1.0, 1.0, 1.0, boundary="robin")
    with pytest.raises(ValueError):
        balian_bloch_level_density(1.0, 1.0, 1.0, boundary="robin")
