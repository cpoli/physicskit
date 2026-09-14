"""Tests for the embedded Gaussian ensembles (EmbeddedGaussianEnsemble,
TwoBodyRandomEnsemble).

Unlike every classical ensemble in this package, the many-body
Hamiltonian here is built by embedding a k-body random interaction into
a Fock space, via hand-rolled Jordan-Wigner fermionic operators -- a
genuinely new piece of machinery with real correctness risk (an initial
implementation had the raising/lowering matrices swapped, caught only
by the k=m exact-match check below, not by the Hermiticity or particle-
number-conservation checks, which both silently passed anyway). These
tests reproduce that verification chain directly rather than only
trusting the module docstring's account of it.
"""

import numpy as np
import pytest
from cache_utils import cached_sample

import physicskit.rmt as rmt
from physicskit.rmt.ensembles.embedded import _fermion_ladder_operators


def test_ladder_operators_satisfy_fermionic_algebra():
    n = 5
    creators = _fermion_ladder_operators(n)
    annihilators = [c.conj().T for c in creators]
    dim = 2**n
    identity = np.eye(dim, dtype=complex)
    for i in range(n):
        for j in range(n):
            anticomm_ac = annihilators[i] @ creators[j] + creators[j] @ annihilators[i]
            target = identity if i == j else np.zeros((dim, dim))
            assert np.abs(anticomm_ac - target).max() < 1e-10

            anticomm_aa = annihilators[i] @ annihilators[j] + annihilators[j] @ annihilators[i]
            assert np.abs(anticomm_aa).max() < 1e-10


@pytest.mark.parametrize("k,m,n_levels", [(2, 2, 4), (3, 3, 6)])
def test_k_equals_m_reduces_exactly_to_the_interaction_matrix(k, m, n_levels):
    # The construction's most important structural check: with only m=k
    # particles present, the embedding is trivial and H, restricted to
    # the m-particle sector, must equal the k-body interaction matrix V
    # EXACTLY (not merely statistically) -- this is precisely the check
    # that caught an operator-ordering sign bug during development.
    from itertools import combinations

    from physicskit.rmt.ensembles.embedded import _fermion_ladder_operators

    creators = _fermion_ladder_operators(n_levels)
    tuples = list(combinations(range(n_levels), k))
    n_tuples = len(tuples)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n_tuples, n_tuples))
    v = (x + x.T) / np.sqrt(2.0)

    def multi_create(idxs):
        op = creators[idxs[0]]
        for idx in idxs[1:]:
            op = op @ creators[idx]
        return op

    p_ops = [multi_create(t) for t in tuples]
    dim = p_ops[0].shape[0]
    h = np.zeros((dim, dim), dtype=complex)
    for a_idx in range(n_tuples):
        q = sum(v[a_idx, b_idx] * p_ops[b_idx].conj().T for b_idx in range(n_tuples))
        h += p_ops[a_idx] @ q

    def occ_to_index(occ, n):
        idx = 0
        for site in occ:
            idx |= 1 << (n - 1 - site)
        return idx

    basis = [occ_to_index(t, n_levels) for t in tuples]
    h_sub = h[np.ix_(basis, basis)].real
    np.testing.assert_allclose(h_sub, v, atol=1e-10)


@pytest.mark.parametrize("beta", [1, 2])
def test_hermiticity_and_particle_number_conservation(beta):
    n_levels = 6
    rng = np.random.default_rng(1)
    # Directly rebuild the dense (pre-restriction) Hamiltonian to check
    # Hermiticity and particle-number conservation on the FULL Fock
    # space, not just the already-restricted eigenvalue output.
    from itertools import combinations

    from physicskit.rmt.ensembles.embedded import _fermion_ladder_operators

    creators = _fermion_ladder_operators(n_levels)
    tuples = list(combinations(range(n_levels), 2))
    n_tuples = len(tuples)
    if beta == 1:
        x = rng.standard_normal((n_tuples, n_tuples))
        v = (x + x.T) / np.sqrt(2.0)
    else:
        real_part = rng.standard_normal((n_tuples, n_tuples))
        imag_part = rng.standard_normal((n_tuples, n_tuples))
        x = (real_part + 1j * imag_part) / np.sqrt(2.0)
        v = (x + x.conj().T) / 2.0

    def multi_create(idxs):
        op = creators[idxs[0]]
        for idx in idxs[1:]:
            op = op @ creators[idx]
        return op

    p_ops = [multi_create(t) for t in tuples]
    dim = p_ops[0].shape[0]
    h = np.zeros((dim, dim), dtype=complex)
    for a_idx in range(n_tuples):
        q = sum(v[a_idx, b_idx] * p_ops[b_idx].conj().T for b_idx in range(n_tuples))
        h += p_ops[a_idx] @ q

    assert np.abs(h - h.conj().T).max() < 1e-10

    annihilators = [c.conj().T for c in creators]
    n_total = sum(creators[i] @ annihilators[i] for i in range(n_levels))
    assert np.abs(h @ n_total - n_total @ h).max() < 1e-8


def test_beta_and_k_are_stored():
    ens = rmt.ensembles.TwoBodyRandomEnsemble(n_particles=3, n_levels=6, beta=2, seed=0)
    assert ens.k == 2
    assert ens.beta == 2
    assert ens.n_levels == 6


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        rmt.ensembles.EmbeddedGaussianEnsemble(n_particles=3, n_levels=6, beta=3, seed=0)
    with pytest.raises(ValueError):
        rmt.ensembles.EmbeddedGaussianEnsemble(n_particles=3, n_levels=6, k=1, seed=0)
    with pytest.raises(ValueError):
        rmt.ensembles.EmbeddedGaussianEnsemble(n_particles=8, n_levels=6, k=2, seed=0)


def test_produces_binomial_n_levels_choose_m_eigenvalues():
    from math import comb

    ens = rmt.ensembles.TwoBodyRandomEnsemble(n_particles=3, n_levels=6, seed=2)
    spectrum = cached_sample(ens, n_samples=3)
    assert spectrum.eigenvalues.shape == (3, comb(6, 3))


def test_eigenvalues_are_real():
    ens = rmt.ensembles.TwoBodyRandomEnsemble(n_particles=3, n_levels=6, beta=2, seed=3)
    spectrum = cached_sample(ens, n_samples=3)
    assert not np.iscomplexobj(spectrum.eigenvalues)
    assert np.all(np.isfinite(spectrum.eigenvalues))


def test_reproducibility():
    ens_a = rmt.ensembles.TwoBodyRandomEnsemble(n_particles=3, n_levels=6, seed=42)
    ens_b = rmt.ensembles.TwoBodyRandomEnsemble(n_particles=3, n_levels=6, seed=42)
    spec_a = ens_a.sample(n_samples=2)
    spec_b = ens_b.sample(n_samples=2)
    np.testing.assert_allclose(spec_a.eigenvalues, spec_b.eigenvalues)
