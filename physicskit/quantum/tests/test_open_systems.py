"""Tests for physicskit.quantum.chapters.open_systems: T1/T2 decay against
closed forms, Lindblad evolution vs. the Kraus channels, trace/positivity
preservation, and the quantum-trajectory average vs. the Lindblad result."""

from __future__ import annotations

import numpy as np
import pytest

from physicskit.quantum.chapters.open_systems import (
    QuantumTrajectories,
    amplitude_damping_kraus,
    apply_kraus,
    dephasing_kraus,
    is_trace_preserving,
    lindblad_rhs,
    lindblad_steady_state,
    lindblad_superoperator,
    sigma_minus,
    solve_lindblad,
    t1_t2_collapse_operators,
)
from physicskit.quantum.core.operators import sigma_x, sigma_z

PLUS = np.array([1, 1]) / np.sqrt(2)


def test_t1_decay_is_exponential():
    gamma = 0.7
    t = np.linspace(0, 8, 41)
    rho = solve_lindblad(np.array([0, 1]), np.zeros((2, 2)), [np.sqrt(gamma) * sigma_minus()], t)
    np.testing.assert_allclose(rho[:, 1, 1].real, np.exp(-gamma * t), atol=1e-12)
    np.testing.assert_allclose(rho[:, 0, 0].real, 1 - np.exp(-gamma * t), atol=1e-12)


@pytest.mark.parametrize("T1, T2", [(1.0, 2.0), (1.0, 0.8), (3.0, 0.5)])
def test_t2_dephasing_of_off_diagonal(T1, T2):
    t = np.linspace(0, 5, 26)
    rho = solve_lindblad(PLUS, np.zeros((2, 2)), t1_t2_collapse_operators(T1, T2), t)
    np.testing.assert_allclose(np.abs(rho[:, 0, 1]), 0.5 * np.exp(-t / T2), atol=1e-12)
    np.testing.assert_allclose(rho[:, 1, 1].real, 0.5 * np.exp(-t / T1), atol=1e-12)


def test_t1_t2_rejects_unphysical_rates():
    with pytest.raises(ValueError):
        t1_t2_collapse_operators(1.0, 2.5)


def test_coherences_rotate_at_qubit_frequency_while_dephasing():
    omega, T2 = 2.0, 1.5
    H = 0.5 * omega * sigma_z
    t = np.linspace(0, 4, 17)
    rho = solve_lindblad(PLUS, H, t1_t2_collapse_operators(1e9, T2), t)
    np.testing.assert_allclose(rho[:, 0, 1], 0.5 * np.exp(-1j * omega * t - t / T2), atol=1e-8)


def test_superoperator_matches_rhs_on_random_three_level_system():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    H = A + A.conj().T
    ops = [rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)) for _ in range(2)]
    B = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    rho = B @ B.conj().T
    rho /= np.trace(rho)
    S = lindblad_superoperator(H, ops)
    np.testing.assert_allclose(S @ rho.reshape(-1), lindblad_rhs(rho, H, ops).reshape(-1), atol=1e-12)


def test_trace_hermiticity_and_positivity_preserved():
    rng = np.random.default_rng(7)
    A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    H = A + A.conj().T
    ops = [0.5 * (rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))) for _ in range(3)]
    psi0 = rng.normal(size=4) + 1j * rng.normal(size=4)
    psi0 /= np.linalg.norm(psi0)
    rho = solve_lindblad(psi0, H, ops, np.linspace(0, 5, 51))
    np.testing.assert_allclose(np.trace(rho, axis1=1, axis2=2), 1.0, atol=1e-10)
    np.testing.assert_allclose(rho, np.conj(np.transpose(rho, (0, 2, 1))), atol=1e-10)
    assert np.linalg.eigvalsh(rho).min() > -1e-10


@pytest.mark.parametrize("Gamma, t", [(0.4, 1.0), (1.3, 2.5)])
def test_lindblad_decay_equals_amplitude_damping_channel(Gamma, t):
    rho0 = np.array([[0.3, 0.2 - 0.1j], [0.2 + 0.1j, 0.7]])
    rho_t = solve_lindblad(rho0, np.zeros((2, 2)), [np.sqrt(Gamma) * sigma_minus()], np.array([0.0, t]))[-1]
    np.testing.assert_allclose(rho_t, apply_kraus(rho0, amplitude_damping_kraus(1 - np.exp(-Gamma * t))), atol=1e-12)


def test_lindblad_pure_dephasing_equals_dephasing_channel():
    gamma_phi, t = 0.8, 1.7
    rho0 = np.array([[0.6, 0.3j], [-0.3j, 0.4]])
    L = np.sqrt(gamma_phi / 2) * sigma_z
    rho_t = solve_lindblad(rho0, np.zeros((2, 2)), [L], np.array([0.0, t]))[-1]
    p = 0.5 * (1 - np.exp(-gamma_phi * t))
    np.testing.assert_allclose(rho_t, apply_kraus(rho0, dephasing_kraus(p)), atol=1e-12)


@pytest.mark.parametrize("kraus", [amplitude_damping_kraus(0.37), dephasing_kraus(0.21)])
def test_channels_are_trace_preserving_and_positive(kraus):
    assert is_trace_preserving(kraus)
    rho = apply_kraus(np.outer(PLUS, PLUS), kraus)
    assert np.trace(rho).real == pytest.approx(1.0)
    assert np.linalg.eigvalsh(rho).min() > -1e-12


def test_resonance_fluorescence_steady_state():
    Omega, gamma = 1.3, 0.6
    H = 0.5 * Omega * sigma_x
    rho = lindblad_steady_state(H, [np.sqrt(gamma) * sigma_minus()])
    assert rho[1, 1].real == pytest.approx(Omega**2 / (gamma**2 + 2 * Omega**2), abs=1e-12)
    np.testing.assert_allclose(lindblad_rhs(rho, H, [np.sqrt(gamma) * sigma_minus()]), 0, atol=1e-12)


def test_steady_state_rejects_degenerate_generator():
    with pytest.raises(ValueError):
        lindblad_steady_state(np.zeros((2, 2)), [])


def test_trajectory_average_matches_lindblad():
    Omega, gamma = 2.0, 0.5
    H = 0.5 * Omega * sigma_x
    ops = [np.sqrt(gamma) * sigma_minus(), np.sqrt(0.1) * sigma_z]
    t = np.linspace(0, 6, 31)
    n_traj = 4000
    res = QuantumTrajectories(H, ops).run(np.array([1, 0]), t, n_traj=n_traj, dt=0.002, rng=np.random.default_rng(11))
    rho_mc = res.density_matrices()
    rho_ex = solve_lindblad(np.array([1, 0]), H, ops, t)
    # binomial sampling error of a population estimate is <= 0.5/sqrt(N)
    assert np.max(np.abs(rho_mc[:, 1, 1] - rho_ex[:, 1, 1])) < 5 * 0.5 / np.sqrt(n_traj)
    assert np.max(np.abs(rho_mc[:, 0, 1] - rho_ex[:, 0, 1])) < 5 * 0.5 / np.sqrt(n_traj)
    np.testing.assert_allclose(res.expectation(np.diag([0.0, 1.0])), rho_mc[:, 1, 1].real, atol=1e-12)


def test_trajectories_stay_normalized_and_single_jump_decay_statistics():
    gamma, n_traj = 1.0, 3000
    t = np.linspace(0, 30, 7)
    res = QuantumTrajectories(np.zeros((2, 2)), [np.sqrt(gamma) * sigma_minus()]).run(
        np.array([0, 1]), t, n_traj=n_traj, dt=0.005, rng=np.random.default_rng(5)
    )
    np.testing.assert_allclose(np.linalg.norm(res.states, axis=2), 1.0, atol=1e-12)
    assert all(len(j) == 1 for j in res.jump_times)
    # waiting time to the single jump is exponential with mean 1/gamma
    waits = np.array([j[0] for j in res.jump_times])
    assert waits.mean() == pytest.approx(1 / gamma, abs=4 / np.sqrt(n_traj) + 0.005)


def test_trajectories_without_collapse_operators_are_unitary():
    H = 0.5 * sigma_x
    t = np.linspace(0, np.pi, 5)
    res = QuantumTrajectories(H, []).run(np.array([1, 0]), t, n_traj=3, rng=np.random.default_rng(0))
    np.testing.assert_allclose(res.expectation(np.diag([0.0, 1.0])), np.sin(t / 2) ** 2, atol=1e-12)
    assert all(len(j) == 0 for j in res.jump_times)
