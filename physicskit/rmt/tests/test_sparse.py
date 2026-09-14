"""Tests for the sparse random matrix ensembles (ErdosRenyiEnsemble,
BernoulliWignerEnsemble).
"""

import numpy as np
import pytest
from cache_utils import cached_sample

import physicskit.rmt as rmt


def test_invalid_p_rejected():
    with pytest.raises(ValueError):
        rmt.ensembles.ErdosRenyiEnsemble(n=10, p=0.0, seed=0)
    with pytest.raises(ValueError):
        rmt.ensembles.BernoulliWignerEnsemble(n=10, p=1.5, seed=0)


def test_erdos_renyi_reproducibility():
    ens_a = rmt.ensembles.ErdosRenyiEnsemble(n=40, p=0.2, seed=42)
    ens_b = rmt.ensembles.ErdosRenyiEnsemble(n=40, p=0.2, seed=42)
    spec_a = ens_a.sample(n_samples=3)
    spec_b = ens_b.sample(n_samples=3)
    np.testing.assert_allclose(spec_a.eigenvalues, spec_b.eigenvalues)


def test_erdos_renyi_has_furedi_komlos_outlier():
    # The largest eigenvalue should sit near the mean degree n*p,
    # well separated from the sqrt(n*p*(1-p))-scale bulk.
    n, p = 1500, 0.05
    ens = rmt.ensembles.ErdosRenyiEnsemble(n=n, p=p, seed=1)
    spectrum = cached_sample(ens, n_samples=3)
    mean_degree = n * p
    largest = spectrum.eigenvalues.max(axis=1)
    assert np.all(np.abs(largest - mean_degree) < 0.15 * mean_degree)


def test_erdos_renyi_bulk_approaches_semicircle_scale_when_dense():
    # In the crossover regime (n*p large), the BULK (all but the top
    # outlier) should stay within the semicircle radius
    # 2*sqrt(n*p*(1-p)) up to a modest finite-size margin.
    n, p = 2000, 0.1
    ens = rmt.ensembles.ErdosRenyiEnsemble(n=n, p=p, seed=2)
    spectrum = cached_sample(ens, n_samples=3)
    scale = ens.natural_scale()
    for row in spectrum.eigenvalues:
        bulk = np.sort(row)[:-1]  # drop the single largest (outlier)
        assert bulk.max() < 2.2 * scale
        assert bulk.min() > -2.2 * scale


def test_bernoulli_wigner_reproducibility():
    ens_a = rmt.ensembles.BernoulliWignerEnsemble(n=40, p=0.3, seed=42)
    ens_b = rmt.ensembles.BernoulliWignerEnsemble(n=40, p=0.3, seed=42)
    spec_a = ens_a.sample(n_samples=3)
    spec_b = ens_b.sample(n_samples=3)
    np.testing.assert_allclose(spec_a.eigenvalues, spec_b.eigenvalues)


@pytest.mark.parametrize("p", [0.3, 0.05, 0.01])
def test_bernoulli_wigner_matches_semicircle_even_when_sparse(p):
    # No Furedi-Komlos outlier here (mean-zero entries) -- the bulk
    # should match the semicircle law directly, validated against the
    # already-implemented WignerSemicircle benchmark, even at
    # substantial sparsity (as long as n*p stays reasonably large).
    n = 2000
    ens = rmt.ensembles.BernoulliWignerEnsemble(n=n, p=p, seed=3)
    spectrum = cached_sample(ens, n_samples=10)
    benchmark = rmt.validation.WignerSemicircle()
    result = benchmark.validate(spectrum, seed=3)
    assert result.ks_statistic < 0.03
