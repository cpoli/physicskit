"""Tests for GOEGUECrossoverEnsemble.

The natural_scale formula sqrt(n*(1+lambda^2)) and the crossover
behavior in level statistics (ratio statistic KS distance to the GOE
and GUE surmises) were both verified numerically during development
before being used here -- see module docstring in
``physicskit.rmt/ensembles/crossover.py``.
"""

import numpy as np
import pytest
from cache_utils import cached_sample
from scipy.stats import kstest

import physicskit.rmt as rmt


def test_lambda_0_recovers_goe_natural_scale():
    ens = rmt.ensembles.GOEGUECrossoverEnsemble(n=200, lam=0.0, seed=0)
    assert ens.natural_scale() == pytest.approx(np.sqrt(200))


def test_natural_scale_matches_verified_formula():
    n, lam = 300, 0.5
    ens = rmt.ensembles.GOEGUECrossoverEnsemble(n=n, lam=lam, seed=0)
    assert ens.natural_scale() == pytest.approx(np.sqrt(n * (1 + lam**2)))


def test_negative_lambda_rejected():
    with pytest.raises(ValueError):
        rmt.ensembles.GOEGUECrossoverEnsemble(n=10, lam=-1.0, seed=0)


def test_reproducibility():
    ens_a = rmt.ensembles.GOEGUECrossoverEnsemble(n=40, lam=0.3, seed=42)
    ens_b = rmt.ensembles.GOEGUECrossoverEnsemble(n=40, lam=0.3, seed=42)
    spec_a = ens_a.sample(n_samples=3)
    spec_b = ens_b.sample(n_samples=3)
    np.testing.assert_allclose(spec_a.eigenvalues, spec_b.eigenvalues)


def test_lambda_0_matches_goe_ratio_statistics():
    ens = rmt.ensembles.GOEGUECrossoverEnsemble(n=400, lam=0.0, seed=1)
    spectrum = cached_sample(ens, n_samples=20)
    ratios = rmt.stats.ratio_statistics(spectrum)
    ks = kstest(ratios, rmt.stats.RatioSurmise(beta=1).cdf).statistic
    assert ks < 0.03


def test_large_lambda_matches_gue_ratio_statistics():
    ens = rmt.ensembles.GOEGUECrossoverEnsemble(n=400, lam=1.0, seed=2)
    spectrum = cached_sample(ens, n_samples=20)
    ratios = rmt.stats.ratio_statistics(spectrum)
    ks = kstest(ratios, rmt.stats.RatioSurmise(beta=2).cdf).statistic
    assert ks < 0.03


def test_crossover_is_monotonic_between_goe_and_gue():
    # As lambda grows from 0, KS distance to GOE should grow while KS
    # distance to GUE should shrink -- the defining crossover behavior.
    goe_model = rmt.stats.RatioSurmise(beta=1)
    gue_model = rmt.stats.RatioSurmise(beta=2)
    ks_goe = []
    ks_gue = []
    for lam in [0.0, 0.05, 1.0]:
        ens = rmt.ensembles.GOEGUECrossoverEnsemble(n=400, lam=lam, seed=3)
        spectrum = cached_sample(ens, n_samples=20)
        ratios = rmt.stats.ratio_statistics(spectrum)
        ks_goe.append(kstest(ratios, goe_model.cdf).statistic)
        ks_gue.append(kstest(ratios, gue_model.cdf).statistic)
    assert ks_goe[0] < ks_goe[1] < ks_goe[2]
    assert ks_gue[0] > ks_gue[1] > ks_gue[2]
