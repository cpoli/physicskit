"""Tests for physicskit.quantum.chapters.relativistic: the Clifford algebra,
free Dirac/Klein-Gordon solutions, the Dirac hydrogen energy formula, and
Klein-paradox step scattering."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import fine_structure

from physicskit.quantum.chapters.relativistic import (
    dirac_hamiltonian,
    dirac_hydrogen_energy,
    dirac_plane_wave_spinor,
    dirac_step_scattering,
    fine_structure_expansion,
    gamma_matrices,
    klein_gordon_dispersion,
    klein_gordon_plane_wave,
    klein_gordon_step_scattering,
)

METRIC = np.diag([1.0, -1.0, -1.0, -1.0])


def test_clifford_algebra():
    g = gamma_matrices()
    for mu in range(4):
        for nu in range(4):
            np.testing.assert_allclose(g[mu] @ g[nu] + g[nu] @ g[mu], 2 * METRIC[mu, nu] * np.eye(4), atol=1e-15)


def test_klein_gordon_plane_wave_solves_the_equation():
    p, m, dx, dt = 0.8, 1.3, 1e-4, 1e-4
    x = np.linspace(-2, 2, 7)
    for sign in (+1, -1):
        phi = lambda x, t: klein_gordon_plane_wave(x, t, p, m, sign)  # noqa: E731
        d2t = (phi(x, 0.3 + dt) - 2 * phi(x, 0.3) + phi(x, 0.3 - dt)) / dt**2
        d2x = (phi(x + dx, 0.3) - 2 * phi(x, 0.3) + phi(x - dx, 0.3)) / dx**2
        np.testing.assert_allclose(d2t - d2x + m**2 * phi(x, 0.3), 0, atol=1e-5)
    assert klein_gordon_dispersion(0.0, 2.0) == 2.0


@pytest.mark.parametrize("p", [[0.0, 0.0, 0.0], [0.3, -0.2, 0.5], [2.0, 1.0, -3.0]])
@pytest.mark.parametrize("spin", [1, -1])
def test_dirac_spinors_are_eigenvectors_with_standard_normalization(p, spin):
    p = np.asarray(p)
    m = 0.7
    E = np.sqrt(p @ p + m**2)
    H = dirac_hamiltonian(p, m)
    u = dirac_plane_wave_spinor(p, m, spin, +1)
    w = dirac_plane_wave_spinor(p, m, spin, -1)
    np.testing.assert_allclose(H @ u, E * u, atol=1e-12)
    np.testing.assert_allclose(H @ w, -E * w, atol=1e-12)
    assert np.vdot(u, u).real == pytest.approx(2 * E)
    assert np.vdot(w, w).real == pytest.approx(2 * E)
    ubar_u = (u.conj() @ gamma_matrices()[0] @ u).real
    assert ubar_u == pytest.approx(2 * m)
    assert abs(np.vdot(u, w)) < 1e-12


def test_dirac_hamiltonian_spectrum():
    E = np.linalg.eigvalsh(dirac_hamiltonian([1.0, 2.0, 2.0], m=4.0))
    np.testing.assert_allclose(E, [-np.sqrt(25), -5, 5, 5], atol=1e-12)


@pytest.mark.parametrize("Z", [1, 20, 80])
def test_dirac_ground_state_closed_form(Z):
    assert dirac_hydrogen_energy(1, 0.5, Z) == pytest.approx(np.sqrt(1 - (Z * fine_structure) ** 2), rel=1e-14)


@pytest.mark.parametrize("n, j", [(1, 0.5), (2, 0.5), (2, 1.5), (3, 1.5), (3, 2.5), (5, 3.5)])
def test_dirac_energy_matches_fine_structure_expansion(n, j):
    Z = 1
    exact = dirac_hydrogen_energy(n, j, Z) - 1
    approx = fine_structure_expansion(n, j, Z)
    assert abs(exact - approx) < 2 * (Z * fine_structure) ** 6


def test_dirac_levels_depend_only_on_n_and_j():
    # 2S_1/2 and 2P_1/2 share j = 1/2, and the 2P fine-structure splitting is alpha^4/32
    a = fine_structure
    split = dirac_hydrogen_energy(2, 1.5) - dirac_hydrogen_energy(2, 0.5)
    assert split == pytest.approx(a**4 / 32, rel=1e-4)
    # hydrogen 2P fine structure ~ 10.9 GHz ~ 4.53e-5 eV
    assert split * 510998.95 == pytest.approx(4.53e-5, rel=5e-3)


def test_dirac_energy_rejects_bad_quantum_numbers():
    with pytest.raises(ValueError):
        dirac_hydrogen_energy(1, 1.5)
    with pytest.raises(ValueError):
        dirac_hydrogen_energy(1, 0.5, Z=140)


E_GRID = 2.0
# includes the band edges V0 = E -/+ m exactly; avoids the KG singularity V0 = 2E
V_GRID = np.r_[np.linspace(0.0, 3.9, 79), np.linspace(4.1, 12.0, 80)]


@pytest.mark.parametrize("func", [dirac_step_scattering, klein_gordon_step_scattering])
@pytest.mark.parametrize("group_velocity", [True, False])
def test_step_current_conservation(func, group_velocity):
    R, T = func(E_GRID, V_GRID, group_velocity=group_velocity)
    np.testing.assert_allclose(R + T, 1.0, atol=1e-12)


@pytest.mark.parametrize("func", [dirac_step_scattering, klein_gordon_step_scattering])
def test_step_regimes(func):
    m, E = 1.0, 2.0
    R0, _ = func(E, 0.0)
    assert R0 == pytest.approx(0.0, abs=1e-15)
    R_gap, T_gap = func(E, np.array([E - m + 0.1, E, E + m - 0.1]))
    np.testing.assert_allclose(R_gap, 1.0, atol=1e-12)
    np.testing.assert_allclose(T_gap, 0.0, atol=1e-12)


def test_klein_paradox_dirac_vs_klein_gordon():
    E, m = 2.0, 1.0
    V_klein = np.linspace(E + m + 0.1, 50, 50)
    R_d, T_d = dirac_step_scattering(E, V_klein)
    assert np.all((T_d > 0) & (R_d < 1))
    R_kg, T_kg = klein_gordon_step_scattering(E, V_klein)
    assert np.all((T_kg < 0) & (R_kg > 1))
    # naive phase-velocity choice flips the Dirac result to R > 1
    R_naive, _ = dirac_step_scattering(E, V_klein, group_velocity=False)
    assert np.all(R_naive > 1)
    # V0 -> infinity: T -> 4 k/(1+k)^2, k = sqrt((E+m)/(E-m))
    kinf = np.sqrt((E + m) / (E - m))
    assert dirac_step_scattering(E, 1e9)[1] == pytest.approx(4 * kinf / (1 + kinf) ** 2, rel=1e-6)


def test_dirac_step_reduces_to_schrodinger_step():
    m = 1.0
    T_kin, V0 = 1e-4, 4e-5  # non-relativistic: T_kin, V0 << m
    E = m + T_kin
    R, _ = dirac_step_scattering(E, V0, m)
    k, kp = np.sqrt(2 * m * T_kin), np.sqrt(2 * m * (T_kin - V0))
    assert R == pytest.approx(((k - kp) / (k + kp)) ** 2, rel=1e-3)
