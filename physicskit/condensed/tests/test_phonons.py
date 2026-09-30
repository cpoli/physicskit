"""Tests for physicskit.condensed.phonons: acoustic-branch sound speeds,
zone-boundary frequencies, the Debye T^3 and Dulong-Petit limits, and
lattice heat capacities from sampled dispersions."""

from __future__ import annotations

import numpy as np
import pytest

from physicskit.condensed.phonons import (
    debye_heat_capacity,
    debye_temperature,
    diatomic_chain_dispersion,
    lattice_heat_capacity,
    monatomic_chain_dispersion,
    square_lattice_dynamical_matrix,
    square_lattice_phonon_dispersion,
)

K_SMALL = 1e-5


def test_monatomic_sound_speed_and_zone_edge():
    K, m, a = 2.0, 0.5, 1.5
    assert monatomic_chain_dispersion(K_SMALL, K, m, a) / K_SMALL == pytest.approx(a * np.sqrt(K / m), rel=1e-9)
    assert monatomic_chain_dispersion(np.pi / a, K, m, a) == pytest.approx(2 * np.sqrt(K / m))
    k = np.linspace(-np.pi / a, np.pi / a, 11)
    np.testing.assert_allclose(monatomic_chain_dispersion(k, K, m, a), monatomic_chain_dispersion(k + 2 * np.pi / a, K, m, a), atol=1e-12)


def test_monatomic_chain_matches_finite_ring_normal_modes():
    # a ring of N masses has normal modes at k = 2 pi j / (N a)
    N, K, m = 12, 1.3, 0.7
    D = (K / m) * (2 * np.eye(N) - np.roll(np.eye(N), 1, axis=0) - np.roll(np.eye(N), -1, axis=0))
    omega_modes = np.sqrt(np.clip(np.linalg.eigvalsh(D), 0, None))
    k = 2 * np.pi * np.arange(N) / N
    np.testing.assert_allclose(np.sort(monatomic_chain_dispersion(k, K, m)), omega_modes, atol=1e-7)


def test_diatomic_branches():
    K, m1, m2, a = 1.0, 1.0, 3.0, 2.0
    ac, op = diatomic_chain_dispersion(K_SMALL, K, m1, m2, a)
    assert ac / K_SMALL == pytest.approx(a * np.sqrt(K / (2 * (m1 + m2))), rel=1e-8)
    assert op == pytest.approx(np.sqrt(2 * K * (1 / m1 + 1 / m2)), rel=1e-8)
    ac_edge, op_edge = diatomic_chain_dispersion(np.pi / a, K, m1, m2, a)
    assert ac_edge == pytest.approx(np.sqrt(2 * K / m2))
    assert op_edge == pytest.approx(np.sqrt(2 * K / m1))


def test_diatomic_equal_masses_folds_monatomic_chain():
    # equal masses: unit cell a = 2 a0 folds the monatomic band
    K, m, a0 = 1.0, 1.0, 1.0
    k = np.linspace(0, np.pi / (2 * a0), 20)
    ac, op = diatomic_chain_dispersion(k, K, m, m, 2 * a0)
    np.testing.assert_allclose(ac, monatomic_chain_dispersion(k, K, m, a0), atol=1e-7)
    np.testing.assert_allclose(op, monatomic_chain_dispersion(k - np.pi / a0, K, m, a0), atol=1e-7)


def test_square_lattice_sound_speeds_along_x():
    K1, K2, m, a = 1.0, 0.4, 2.0, 1.2
    w = square_lattice_phonon_dispersion(K_SMALL, 0.0, K1, K2, m, a) / K_SMALL
    assert w[0] == pytest.approx(a * np.sqrt(K2 / m), rel=1e-7)  # transverse
    assert w[1] == pytest.approx(a * np.sqrt((K1 + K2) / m), rel=1e-7)  # longitudinal


def test_square_lattice_dynamical_matrix_is_symmetric_and_periodic():
    rng = np.random.default_rng(0)
    kx, ky = rng.uniform(-np.pi, np.pi, (2, 50))
    D = square_lattice_dynamical_matrix(kx, ky, 1.0, 0.3)
    np.testing.assert_allclose(D, np.swapaxes(D, -1, -2))
    np.testing.assert_allclose(D, square_lattice_dynamical_matrix(kx + 2 * np.pi, ky - 2 * np.pi, 1.0, 0.3), atol=1e-12)
    assert np.all(np.linalg.eigvalsh(D) >= -1e-12)


def test_debye_limits():
    theta = 300.0
    T_low = np.array([1.0, 3.0, 6.0])
    np.testing.assert_allclose(debye_heat_capacity(T_low, theta), 12 * np.pi**4 / 5 * (T_low / theta) ** 3, rtol=1e-6)
    assert debye_heat_capacity(1e5, theta) == pytest.approx(3.0, rel=1e-5)
    # tabulated Debye function: C/(3 N k_B) = 0.952 at T = Theta_D
    assert debye_heat_capacity(theta, theta) / 3 == pytest.approx(0.9517, abs=1e-4)
    assert debye_heat_capacity(0.0, theta) == 0.0


def test_debye_temperature():
    assert debye_temperature(2.0, 1.0) == pytest.approx(2.0 * (6 * np.pi**2) ** (1 / 3))


def test_lattice_heat_capacity_high_T_equipartition():
    k = np.linspace(-np.pi, np.pi, 200, endpoint=False)
    omega = monatomic_chain_dispersion(k)
    assert lattice_heat_capacity(omega, 1e4) == pytest.approx(1.0, rel=1e-8)


def test_lattice_heat_capacity_1d_linear_low_T():
    # 1D Debye limit: C/N = pi T a / (3 c) with c = a sqrt(K/m)
    N = 200000
    k = 2 * np.pi * np.arange(N) / N - np.pi
    omega = monatomic_chain_dispersion(k)
    T = np.array([0.005, 0.01, 0.02])
    np.testing.assert_allclose(lattice_heat_capacity(omega, T), np.pi * T / 3, rtol=2e-3)


def test_square_lattice_heat_capacity_t_squared_law():
    # 2D Debye: C/N_modes ∝ T^2 at low T
    n = 400
    k = 2 * np.pi * np.arange(n) / n - np.pi
    KX, KY = np.meshgrid(k, k)
    omega = square_lattice_phonon_dispersion(KX, KY, 1.0, 0.5)
    T = np.array([0.01, 0.02])
    C = lattice_heat_capacity(omega, T)
    assert C[1] / C[0] == pytest.approx(4.0, rel=0.02)
