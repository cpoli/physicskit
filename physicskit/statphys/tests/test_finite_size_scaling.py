import numpy as np
import pytest

from physicskit.statphys.utils.finite_size_scaling import (
    binder_cumulant_crossing,
    estimate_beta_over_nu,
    estimate_gamma_over_nu,
    estimate_nu_from_binder_slope,
    power_law_exponent,
)


def test_power_law_exponent_recovers_known_power():
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    y = 3.0 * x**1.5
    p, A = power_law_exponent(x, y)
    assert p == pytest.approx(1.5, abs=1e-6)
    assert pytest.approx(3.0, abs=1e-6) == A


def test_estimate_gamma_over_nu():
    L = np.array([8.0, 16.0, 32.0, 64.0])
    chi_max = 2.0 * L**1.75  # exact 2D Ising gamma/nu = 7/4
    assert estimate_gamma_over_nu(L, chi_max) == pytest.approx(1.75, abs=1e-6)


def test_estimate_beta_over_nu():
    L = np.array([8.0, 16.0, 32.0, 64.0])
    m_tc = 1.5 * L ** (-0.125)  # exact 2D Ising beta/nu = 1/8
    assert estimate_beta_over_nu(L, m_tc) == pytest.approx(0.125, abs=1e-6)


def test_estimate_nu_from_binder_slope_with_synthetic_data():
    # Construct U4(T) curves whose slope at T_c scales as L^(1/nu) with nu=1.
    T_c = 2.269
    L_values = [8, 16, 32]
    T_grids, U4_grids = [], []
    for L in L_values:
        T = np.linspace(T_c - 1.0, T_c + 1.0, 21)
        slope_at_tc = -(L ** (1.0 / 1.0))  # nu = 1
        U4 = 0.6 + slope_at_tc * (T - T_c)
        T_grids.append(T)
        U4_grids.append(U4)
    nu = estimate_nu_from_binder_slope(L_values, T_grids, U4_grids, T_c)
    assert nu == pytest.approx(1.0, rel=0.05)


def test_binder_cumulant_crossing_finds_known_intersection():
    T = np.linspace(2.0, 2.5, 100)
    U4_by_L = {
        16: 0.6 - 0.5 * (T - 2.25),
        32: 0.6 - 1.5 * (T - 2.25),
    }
    T_cross = binder_cumulant_crossing(T, U4_by_L)
    assert T_cross == pytest.approx(2.25, abs=0.01)


def test_binder_cumulant_crossing_raises_without_intersection():
    T = np.linspace(2.0, 2.5, 50)
    U4_by_L = {16: np.full(50, 0.5), 32: np.full(50, 0.7)}
    with pytest.raises(ValueError):
        binder_cumulant_crossing(T, U4_by_L)
