import numpy as np
import pytest

from physicskit.quantum.chapters.quantum_circuits import (
    QuantumCircuit,
    deutsch_jozsa_circuit,
    grover_circuit,
    grover_optimal_iterations,
    grover_success_probability,
    modular_exponentiation_circuit,
    qft_circuit,
    shor_period_from_measurement,
)


def test_gate_ordering_qubit_zero_is_most_significant():
    assert np.argmax(np.abs(QuantumCircuit(3).x(0).run())) == 4
    assert np.argmax(np.abs(QuantumCircuit(3).x(2).run())) == 1


def test_cnot_truth_table():
    U = QuantumCircuit(2).cx(0, 1).unitary()
    np.testing.assert_allclose(U, np.eye(4)[[0, 1, 3, 2]])
    U_rev = QuantumCircuit(2).cx(1, 0).unitary()
    np.testing.assert_allclose(U_rev, np.eye(4)[[0, 3, 2, 1]])


def test_circuit_is_unitary_and_inverse_undoes_it():
    qc = QuantumCircuit(3).h(0).cx(0, 2).phase(0.3, 1).cphase(1.1, 2, 0).swap(1, 2).diagonal([0, 1, 2, 3, 4, 5, 6, 7])
    U = qc.unitary()
    np.testing.assert_allclose(U.conj().T @ U, np.eye(8), atol=1e-12)
    full = QuantumCircuit(3).append(qc).append(qc.inverse())
    np.testing.assert_allclose(full.unitary(), np.eye(8), atol=1e-12)


def test_marginal_probabilities_respect_requested_order():
    psi = QuantumCircuit(3).x(0).run()  # |100>
    assert QuantumCircuit.probabilities(psi, 3, (0, 2)).tolist() == [0, 0, 1, 0]
    assert QuantumCircuit.probabilities(psi, 3, (2, 0)).tolist() == [0, 1, 0, 0]


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_qft_equals_dft_matrix(n):
    N = 2**n
    F = np.exp(2j * np.pi * np.outer(np.arange(N), np.arange(N)) / N) / np.sqrt(N)
    np.testing.assert_allclose(qft_circuit(n).unitary(), F, atol=1e-12)
    counts = qft_circuit(n).gate_count()
    assert counts["H"] == n and counts.get("CP", 0) == n * (n - 1) // 2


def test_deutsch_jozsa_distinguishes_constant_and_balanced_in_one_query():
    rng = np.random.default_rng(0)
    n = 4
    for f in (np.zeros(16, int), np.ones(16, int)):
        assert abs(deutsch_jozsa_circuit(f).run()[0]) ** 2 == pytest.approx(1.0)
    for _ in range(5):
        f = rng.permutation(np.repeat([0, 1], 2 ** (n - 1)))
        assert abs(deutsch_jozsa_circuit(f).run()[0]) ** 2 == pytest.approx(0.0, abs=1e-24)
    assert deutsch_jozsa_circuit(np.zeros(16, int)).gate_count()["D"] == 1


@pytest.mark.parametrize("n, marked", [(4, [3]), (6, [5, 40]), (8, [200])])
def test_grover_success_probability_matches_rotation_formula(n, marked):
    N = 2**n
    for k in range(0, grover_optimal_iterations(N, len(marked)) + 2):
        psi = grover_circuit(n, marked, iterations=k).run()
        p = np.sum(np.abs(psi[marked]) ** 2)
        assert p == pytest.approx(grover_success_probability(k, N, len(marked)), abs=1e-10)


def test_grover_optimal_iterations_scale_as_sqrt_N():
    assert grover_optimal_iterations(2**10) == 25
    assert grover_success_probability(25, 2**10) > 0.999


def test_shor_period_finding_for_fifteen():
    n_count = 8
    qc, m = modular_exponentiation_circuit(7, 15, n_count)
    qc.append(qft_circuit(n_count).inverse())
    p = QuantumCircuit.probabilities(qc.run(), n_count + m, tuple(range(n_count)))
    peaks = np.nonzero(p > 1e-9)[0]
    np.testing.assert_array_equal(peaks, [0, 64, 128, 192])
    np.testing.assert_allclose(p[peaks], 0.25)
    r = shor_period_from_measurement(64, n_count, 15)
    assert r == 4 and pow(7, r, 15) == 1
    assert sorted({np.gcd(7 ** (r // 2) - 1, 15), np.gcd(7 ** (r // 2) + 1, 15)}) == [3, 5]


def test_bad_inputs_rejected():
    with pytest.raises(ValueError):
        QuantumCircuit(2).gate(np.eye(2), 0, 1)
    with pytest.raises(ValueError):
        modular_exponentiation_circuit(5, 15, 4)
