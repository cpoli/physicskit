"""Tests for physicskit.quantum.core.operators, none of which any other
test in the suite exercises directly (test_physics_checks.py etc. work
at the wavefunction/solver level, not the operator-algebra level)."""

from __future__ import annotations

import numpy as np
import pytest

from physicskit.quantum.core.operators import (
    annihilation_operator,
    anticommutator,
    commutator,
    creation_operator,
    expectation,
    is_hermitian,
    momentum_operator,
    number_operator,
    position_operator,
    sigma_x,
    sigma_y,
    sigma_z,
    spin_operator,
)


def test_spin_operator_half_reduces_to_scaled_pauli_matrices():
    np.testing.assert_allclose(spin_operator("x", 0.5), 0.5 * sigma_x)
    np.testing.assert_allclose(spin_operator("y", 0.5), 0.5 * sigma_y)
    np.testing.assert_allclose(spin_operator("z", 0.5), 0.5 * sigma_z)


def test_spin_operator_ladder_operators_reconstruct_x_and_y():
    s_plus = spin_operator("+", 0.5)
    s_minus = spin_operator("-", 0.5)
    np.testing.assert_allclose(0.5 * (s_plus + s_minus), spin_operator("x", 0.5))
    np.testing.assert_allclose((s_plus - s_minus) / (2j), spin_operator("y", 0.5))
    np.testing.assert_allclose(s_minus, s_plus.conj().T)


@pytest.mark.parametrize("s", [0.5, 1.0, 1.5, 2.0])
def test_spin_operator_satisfies_angular_momentum_commutation_relations(s):
    """[Sx, Sy] = i*hbar*Sz (and cyclic permutations), for both integer
    and half-integer spin, not just the spin-1/2 Pauli case."""
    sx, sy, sz = spin_operator("x", s), spin_operator("y", s), spin_operator("z", s)
    np.testing.assert_allclose(commutator(sx, sy), 1j * sz, atol=1e-10)
    np.testing.assert_allclose(commutator(sy, sz), 1j * sx, atol=1e-10)
    np.testing.assert_allclose(commutator(sz, sx), 1j * sy, atol=1e-10)


def test_spin_operator_rejects_unknown_axis():
    with pytest.raises(ValueError):
        spin_operator("q")


def test_creation_operator_is_hermitian_conjugate_of_annihilation():
    a = annihilation_operator(6)
    a_dagger = creation_operator(6)
    np.testing.assert_allclose(a_dagger, a.conj().T)


def test_number_operator_is_diagonal_with_fock_state_eigenvalues():
    n_max = 6
    N = number_operator(n_max)
    np.testing.assert_allclose(np.diag(N).real, np.arange(n_max))
    off_diag = N - np.diag(np.diag(N))
    np.testing.assert_allclose(off_diag, 0.0)


def test_position_and_momentum_operators_satisfy_canonical_commutation():
    """[x, p] = i*hbar*I, except at the last row/column where truncating
    the Fock basis necessarily breaks the identity (a itself already
    truncates there)."""
    n_max = 8
    x = position_operator(n_max, m=2.0, omega=3.0, hbar=1.0)
    p = momentum_operator(n_max, m=2.0, omega=3.0, hbar=1.0)
    comm = commutator(x, p)
    np.testing.assert_allclose(np.diag(comm)[:-1], 1j * np.ones(n_max - 1), atol=1e-10)


def test_commutator_and_anticommutator_of_pauli_matrices():
    # Standard su(2) identities: [sigma_x, sigma_y] = 2i*sigma_z,
    # {sigma_x, sigma_y} = 0 (distinct Pauli matrices anticommute).
    np.testing.assert_allclose(commutator(sigma_x, sigma_y), 2j * sigma_z)
    np.testing.assert_allclose(anticommutator(sigma_x, sigma_y), np.zeros((2, 2)), atol=1e-12)
    np.testing.assert_allclose(anticommutator(sigma_x, sigma_x), 2 * np.eye(2))


def test_expectation_of_sigma_z_on_basis_states():
    up = np.array([1.0, 0.0])
    down = np.array([0.0, 1.0])
    assert expectation(sigma_z, up) == pytest.approx(1.0)
    assert expectation(sigma_z, down) == pytest.approx(-1.0)


def test_expectation_flattens_non_1d_state_input():
    up_column = np.array([[1.0], [0.0]])
    assert expectation(sigma_z, up_column) == pytest.approx(1.0)


def test_is_hermitian_true_for_pauli_matrices_false_for_non_hermitian():
    assert is_hermitian(sigma_x)
    assert is_hermitian(sigma_y)
    assert is_hermitian(sigma_z)
    assert not is_hermitian(np.array([[0.0, 1.0], [0.0, 0.0]]))
