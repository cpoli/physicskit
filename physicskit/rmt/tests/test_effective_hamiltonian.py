"""Tests for the non-Hermitian effective Hamiltonian ensembles (EffGOE,
EffGUE, EffGSE) -- Feshbach-projection/chaotic-scattering resonance
ensembles built on GOE/GUE/GSE.

Like BdG (see ``tests/test_bdg.py``), there is no simple closed-form
bulk density to benchmark here, so validation is structural: causality
(widths never negative -- a property proved directly from Hermiticity
and positive semi-definiteness in the module docstring, not merely
observed), the m=0 Hermitian limit, and beta=4's Kramers degeneracy.
"""

import numpy as np
import pytest
from cache_utils import cached_sample

import physicskit.rmt as rmt

EFFECTIVE = [
    ("EffGOE", rmt.ensembles.EffGOE, 1),
    ("EffGUE", rmt.ensembles.EffGUE, 2),
    ("EffGSE", rmt.ensembles.EffGSE, 4),
]


@pytest.mark.parametrize("name,cls,beta", EFFECTIVE)
def test_beta_is_correct(name, cls, beta):
    ens = cls(n=30, m=5, seed=0)
    assert ens.beta == beta


@pytest.mark.parametrize("name,cls,beta", EFFECTIVE)
def test_eigenvalues_are_complex(name, cls, beta):
    ens = cls(n=40, m=5, seed=1)
    spectrum = cached_sample(ens, n_samples=5)
    assert np.iscomplexobj(spectrum.eigenvalues)


@pytest.mark.parametrize("name,cls,beta", EFFECTIVE)
def test_produces_n_eigenvalues(name, cls, beta):
    # beta=4 internally builds a 2n x 2n embedding but de-duplicates the
    # exact Kramers pairs, so all three classes return exactly n
    # eigenvalues per sample -- see module docstring.
    ens = cls(n=25, m=4, seed=2)
    spectrum = cached_sample(ens, n_samples=6)
    assert spectrum.eigenvalues.shape == (6, 25)


@pytest.mark.parametrize("name,cls,beta", EFFECTIVE)
def test_reproducibility(name, cls, beta):
    ens_a = cls(n=30, m=5, seed=42)
    ens_b = cls(n=30, m=5, seed=42)
    spec_a = ens_a.sample(n_samples=3)
    spec_b = ens_b.sample(n_samples=3)
    np.testing.assert_allclose(spec_a.eigenvalues, spec_b.eigenvalues)


@pytest.mark.parametrize("name,cls,beta", EFFECTIVE)
def test_widths_are_never_positive_causality(name, cls, beta):
    # Gamma_k = -2*Im(lambda_k) >= 0 for every eigenvalue of every
    # realization -- proved directly (not just observed) in the module
    # docstring from H Hermitian + Gamma Hermitian PSD.
    ens = cls(n=35, m=6, coupling=1.3, seed=3)
    spectrum = cached_sample(ens, n_samples=20)
    assert np.all(spectrum.eigenvalues.imag <= 1e-9)


@pytest.mark.parametrize("name,cls,beta", EFFECTIVE)
def test_zero_channels_recovers_hermitian_limit(name, cls, beta):
    # m=0 -> Gamma=0 -> H_eff=H exactly -> real eigenvalues to machine
    # precision, regardless of beta.
    ens = cls(n=30, m=0, seed=4)
    spectrum = cached_sample(ens, n_samples=5)
    assert np.abs(spectrum.eigenvalues.imag).max() < 1e-9


@pytest.mark.parametrize("name,cls,beta", EFFECTIVE)
def test_more_channels_increase_total_width(name, cls, beta):
    # More open channels -> more total decay -> larger mean |Im(lambda)|
    # (monotonic on average, not realization-by-realization).
    total_widths = []
    for m in (1, 4, 10):
        ens = cls(n=40, m=m, seed=5)
        spectrum = cached_sample(ens, n_samples=15)
        total_widths.append(np.mean(-spectrum.eigenvalues.imag))
    assert total_widths[0] < total_widths[1] < total_widths[2]


def test_effgse_eigenvalues_form_exact_kramers_pairs():
    # Direct, un-cached check against the raw (non-deduplicated) 2n
    # eigenvalues: EffGSE's Kramers degeneracy should be exact (machine
    # precision), unlike GinSE's merely-conjugate (distinct) pairs.
    from physicskit.rmt.ensembles.effective_hamiltonian import (
        _sample_coupling,
        _sample_hamiltonian,
    )

    rng = np.random.default_rng(6)
    n, m = 15, 3
    h = _sample_hamiltonian(n, rng, beta=4)
    v = _sample_coupling(n, m, rng, beta=4)
    gamma = v @ v.conj().T
    h_eff = h - 0.5j * gamma

    eigs = np.linalg.eigvals(h_eff)
    order = np.lexsort((eigs.imag, eigs.real))
    sorted_eigs = eigs[order]
    pair_diffs = np.abs(sorted_eigs[0::2] - sorted_eigs[1::2])
    assert pair_diffs.max() < 1e-8


def test_zero_coupling_is_equivalent_to_zero_channels():
    # coupling=0 with m>0 should give the same Hermitian-limit behavior
    # as m=0 (Gamma=0 either way).
    ens = rmt.ensembles.EffGUE(n=25, m=6, coupling=0.0, seed=7)
    spectrum = cached_sample(ens, n_samples=5)
    assert np.abs(spectrum.eigenvalues.imag).max() < 1e-9
