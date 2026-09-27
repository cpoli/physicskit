"""Tests for physicskit.particle.lattice_gauge: Monte Carlo 2D U(1) Wilson
loops against the exact torus solution, the area law and Creutz ratio,
gauge invariance, and strong coupling in 3D."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import iv

from physicskit.particle.lattice_gauge import (
    U1LatticeGauge,
    creutz_ratio,
    u1_2d_string_tension,
    u1_2d_wilson_loop_exact,
)


def test_exact_2d_formulas():
    beta = 1.3
    ratio = iv(1, beta) / iv(0, beta)
    np.testing.assert_allclose(u1_2d_wilson_loop_exact(beta, np.arange(1, 6)), ratio ** np.arange(1, 6))
    assert u1_2d_string_tension(beta) == pytest.approx(-np.log(ratio))
    # torus correction vanishes for a large lattice and is 1 for the full area's complement symmetry A -> V - A
    assert u1_2d_wilson_loop_exact(beta, 4, volume=10**4) == pytest.approx(ratio**4, rel=1e-12)
    V = 36
    assert u1_2d_wilson_loop_exact(beta, 0, volume=V) == pytest.approx(1.0)
    assert u1_2d_wilson_loop_exact(beta, 10, volume=V) == pytest.approx(u1_2d_wilson_loop_exact(beta, V - 10, volume=V))


def test_cold_start_and_gauge_invariance():
    lat = U1LatticeGauge(L=6, beta=1.0, seed=3)
    lat.sweep(5)
    before = (lat.plaquette(), lat.wilson_loop(2, 3), lat.wilson_loop(3, 1))
    lat.gauge_transform(np.random.default_rng(0).uniform(-np.pi, np.pi, 36))
    after = (lat.plaquette(), lat.wilson_loop(2, 3), lat.wilson_loop(3, 1))
    np.testing.assert_allclose(after, before, atol=1e-12)
    assert U1LatticeGauge(L=4, beta=1.0, dim=3, hot_start=False).wilson_loop(2, 2) == 1.0


@pytest.fixture(scope="module")
def measured_2d():
    beta, L = 1.5, 16
    lat = U1LatticeGauge(L=L, beta=beta, seed=11)
    lat.thermalize(300)
    loops = [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3)]
    return beta, L, lat.measure(loops, n_measurements=400, sweeps_between=5)


def test_2d_wilson_loops_match_exact_solution(measured_2d):
    beta, L, m = measured_2d
    for R, T in [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3)]:
        mean, err = m[(R, T)]
        exact = u1_2d_wilson_loop_exact(beta, R * T, volume=L * L)
        assert abs(mean - exact) < 4 * err + 2e-3
    assert m["plaquette"][0] == pytest.approx(m[(1, 1)][0])


def test_2d_area_law_and_creutz_ratio(measured_2d):
    beta, _, m = measured_2d
    sigma = u1_2d_string_tension(beta)
    assert creutz_ratio(m, 2, 2) == pytest.approx(sigma, abs=0.05)
    areas = np.array([1, 2, 4])
    logs = np.log([m[(1, 1)][0], m[(1, 2)][0], m[(2, 2)][0]])
    slope = np.polyfit(areas, logs, 1)[0]
    assert -slope == pytest.approx(sigma, abs=0.03)


def test_3d_strong_coupling_plaquette():
    # <P> = I1(beta)/I0(beta) + O(beta^5) at strong coupling in any dimension
    beta = 0.3
    lat = U1LatticeGauge(L=6, beta=beta, dim=3, seed=5)
    lat.thermalize(200)
    mean, err = lat.measure([], n_measurements=200, sweeps_between=2)["plaquette"]
    assert abs(mean - iv(1, beta) / iv(0, beta)) < 4 * err + 2e-3


def test_rejects_one_dimension():
    with pytest.raises(ValueError):
        U1LatticeGauge(L=4, beta=1.0, dim=1)
