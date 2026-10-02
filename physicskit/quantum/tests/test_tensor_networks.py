import numpy as np
import pytest
from scipy.linalg import expm

from physicskit.condensed.spin_chains import (
    entanglement_entropy,
    lowest_eigenstates,
    tfim_hamiltonian,
    xxz_hamiltonian,
)
from physicskit.quantum.chapters.tensor_networks import MPS, tebd, tfim_bond_hamiltonians, xxz_bond_hamiltonians


def _dense_from_bonds(bonds, N):
    H = np.zeros((2**N, 2**N), dtype=complex)
    for j, h in enumerate(bonds):
        H += np.kron(np.kron(np.eye(2**j), h), np.eye(2 ** (N - j - 2)))
    # kron order uses physical index 0 = up; flip to the condensed convention
    return H[::-1, ::-1]


@pytest.mark.parametrize(
    "make",
    [
        lambda N: (xxz_bond_hamiltonians(N, 1.0, 0.7, 0.3), xxz_hamiltonian(N, 1.0, 0.7, 0.3, periodic=False)[0]),
        lambda N: (tfim_bond_hamiltonians(N, 1.0, 0.8), tfim_hamiltonian(N, 1.0, 0.8, periodic=False)[0]),
    ],
)
def test_bond_terms_sum_to_condensed_hamiltonian(make):
    N = 6
    bonds, H = make(N)
    np.testing.assert_allclose(_dense_from_bonds(bonds, N), H.toarray(), atol=1e-12)


def test_product_state_conventions():
    psi = MPS.product_state([0, 1, 1])  # up, down, down = binary 100
    assert np.argmax(np.abs(psi.to_dense())) == 4
    np.testing.assert_allclose(psi.expectation_local(np.diag([1.0, -1.0])), [1, -1, -1])
    assert psi.entanglement_entropy().tolist() == [0.0, 0.0]


def test_real_time_quench_matches_exact_evolution():
    N = 10
    H = xxz_hamiltonian(N, periodic=False)[0].toarray()
    psi = MPS.product_state([0, 1] * (N // 2))
    v0 = psi.to_dense()
    out = tebd(psi, xxz_bond_hamiltonians(N), dt=0.02, n_steps=150, chi_max=64)
    v_exact = expm(-1j * H * out["t"][-1]) @ v0
    assert abs(np.vdot(v_exact, psi.to_dense())) == pytest.approx(1.0, abs=1e-6)
    assert out["entropy"][-1] == pytest.approx(entanglement_entropy(v_exact, N, N // 2), abs=1e-4)
    # a unitary evolution conserves the energy up to the Trotter error
    np.testing.assert_allclose(out["energy"], out["energy"][0], atol=1e-3)


def test_imaginary_time_ground_states_match_exact_diagonalization():
    N = 10
    E_xxz = lowest_eigenstates(xxz_hamiltonian(N, periodic=False, n_up=N // 2)[0], k=1)[0][0]
    out = tebd(MPS.product_state([0, 1] * (N // 2)), xxz_bond_hamiltonians(N), dt=0.05, n_steps=600, chi_max=32, imaginary=True, measure_every=600)
    assert out["energy"][-1] == pytest.approx(E_xxz, abs=1e-5)
    E_tfim = lowest_eigenstates(tfim_hamiltonian(N, 1.0, 1.5, periodic=False)[0], k=1)[0][0]
    out = tebd(MPS.product_state([0] * N), tfim_bond_hamiltonians(N, 1.0, 1.5), dt=0.02, n_steps=400, chi_max=32, imaginary=True, measure_every=400)
    assert out["energy"][-1] == pytest.approx(E_tfim, abs=1e-5)


def test_truncation_reports_discarded_weight():
    N = 12
    psi = MPS.product_state([0, 1] * (N // 2))
    out = tebd(psi, xxz_bond_hamiltonians(N), dt=0.05, n_steps=60, chi_max=4)
    assert max(psi.bond_dimensions) <= 4
    assert out["discarded_weight"][-1] > 0


def test_canonicalize_preserves_state_and_gives_right_canonical_tensors():
    psi = MPS.product_state([0, 1] * 4)
    tebd(psi, xxz_bond_hamiltonians(8), dt=0.1, n_steps=5)
    v = psi.to_dense()
    psi.canonicalize()
    np.testing.assert_allclose(psi.to_dense(), v, atol=1e-12)
    for b in psi.B:
        np.testing.assert_allclose(np.tensordot(b, b.conj(), axes=([1, 2], [1, 2])), np.eye(b.shape[0]), atol=1e-12)
