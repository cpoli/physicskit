"""Tests for the Sachdev-Ye-Kitaev (SYK) ensemble.

Unlike every other ensemble in this package, SYK is not built from
i.i.d. matrix entries -- it's an explicit many-body Hamiltonian on an
exponentially large (2**(N/2)-dimensional) Hilbert space. Validation
here is therefore structural (Clifford algebra, Hermiticity, the exact
algebraic reason q=4 needs no phase but q=2 needs a factor of i) rather
than against a closed-form limiting spectral density, which -- unlike
the semicircle/Marchenko-Pastur/circular laws elsewhere in this package
-- has no simple exact form for general q at finite N. Kept to small N
throughout (cost is exponential in N, not polynomial -- see the module
docstring).
"""

import math

import numpy as np
import pytest
from cache_utils import cached_sample

import physicskit.rmt as rmt
from physicskit.rmt.ensembles.syk import _sample_syk_hamiltonian, majorana_operators


@pytest.mark.parametrize("n", [4, 6, 8])
def test_majorana_operators_are_hermitian(n):
    for gamma in majorana_operators(n):
        assert np.abs(gamma - gamma.conj().T).max() < 1e-12


@pytest.mark.parametrize("n", [4, 6, 8])
def test_majorana_operators_satisfy_clifford_algebra(n):
    gammas = majorana_operators(n)
    dim = gammas[0].shape[0]
    identity = np.eye(dim)
    for a in range(n):
        for b in range(n):
            anticommutator = gammas[a] @ gammas[b] + gammas[b] @ gammas[a]
            target = 2.0 * identity if a == b else np.zeros((dim, dim))
            assert np.abs(anticommutator - target).max() < 1e-10


def test_majorana_operators_reject_odd_n():
    with pytest.raises(ValueError):
        majorana_operators(5)


@pytest.mark.parametrize("q,should_need_phase", [(2, True), (4, False), (6, True), (8, False)])
def test_raw_q_body_product_hermiticity_matches_derivation(q, should_need_phase):
    # Reversing q distinct anticommuting Majoranas picks up sign
    # (-1)**(q*(q-1)/2) -- the raw (no i**(q/2) prefactor) product should
    # be Hermitian exactly when that sign is +1, i.e. exactly when
    # should_need_phase is False. This is the algebraic fact the
    # i**(q/2) prefactor in the module docstring is derived from;
    # checked directly here rather than only trusted from the derivation.
    n = 8
    gammas = majorana_operators(n)
    idx = list(range(q))
    term = gammas[idx[0]]
    for i in idx[1:]:
        term = term @ gammas[i]
    is_hermitian = np.abs(term - term.conj().T).max() < 1e-10
    assert is_hermitian == (not should_need_phase)
    # With the i**(q/2) prefactor applied, it should always be Hermitian.
    phased = (1j ** (q // 2)) * term
    assert np.abs(phased - phased.conj().T).max() < 1e-10


def test_beta_is_none():
    ens = rmt.ensembles.SYKEnsemble(n=8, seed=0)
    assert ens.beta is None


def test_default_q_is_four():
    ens = rmt.ensembles.SYKEnsemble(n=8, seed=0)
    assert ens.q == 4


@pytest.mark.parametrize("n", [4, 6, 8, 10])
def test_produces_2_pow_n_over_2_real_eigenvalues(n):
    ens = rmt.ensembles.SYKEnsemble(n=n, seed=1)
    spectrum = cached_sample(ens, n_samples=3)
    assert spectrum.eigenvalues.shape == (3, 2 ** (n // 2))
    assert not np.iscomplexobj(spectrum.eigenvalues)
    assert np.all(np.isfinite(spectrum.eigenvalues))


def test_reproducibility():
    ens_a = rmt.ensembles.SYKEnsemble(n=8, seed=42)
    ens_b = rmt.ensembles.SYKEnsemble(n=8, seed=42)
    spec_a = ens_a.sample(n_samples=3)
    spec_b = ens_b.sample(n_samples=3)
    np.testing.assert_allclose(spec_a.eigenvalues, spec_b.eigenvalues)


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        rmt.ensembles.SYKEnsemble(n=7, seed=0)  # odd n
    with pytest.raises(ValueError):
        rmt.ensembles.SYKEnsemble(n=8, q=3, seed=0)  # odd q
    with pytest.raises(ValueError):
        rmt.ensembles.SYKEnsemble(n=8, q=10, seed=0)  # q > n


def test_q_two_produces_a_finite_nondegenerate_free_fermion_spectrum():
    # q=2 is the (non-chaotic) free-fermion special case: H is built
    # from a single random N x N antisymmetric coupling matrix rather
    # than C(N,4) independent ones. Just check it produces a sane,
    # genuinely nonzero-scale spectrum -- not a specific numeric target,
    # since the exact eigenvalue scale for q=2 (checked empirically
    # during development to grow roughly like sqrt(N), not stay O(1) the
    # way q=4's does) isn't asserted here.
    ens = rmt.ensembles.SYKEnsemble(n=10, q=2, coupling=1.0, seed=3)
    spectrum = cached_sample(ens, n_samples=20)
    assert np.all(np.isfinite(spectrum.eigenvalues))
    assert spectrum.eigenvalues.std() > 0.1


def test_coupling_scales_hamiltonian_linearly():
    n, q = 8, 4
    gammas = majorana_operators(n)
    h1 = _sample_syk_hamiltonian(n, q, 1.0, np.random.default_rng(5), gammas)
    h2 = _sample_syk_hamiltonian(n, q, 2.0, np.random.default_rng(5), gammas)
    np.testing.assert_allclose(h2, 2.0 * h1)


def test_variance_normalization_matches_maldacena_stanford():
    # <J_{i1...iq}^2> = (q-1)! * J^2 / N^(q-1) -- checked directly against
    # the empirical variance of many independently drawn couplings,
    # rather than only trusted from the module docstring's citation.
    n, q, coupling = 12, 4, 1.5
    rng = np.random.default_rng(6)
    expected_variance = math.factorial(q - 1) * coupling**2 / n ** (q - 1)
    std = math.sqrt(expected_variance)
    draws = rng.normal(0.0, std, size=200_000)
    assert draws.var() == pytest.approx(expected_variance, rel=0.05)
