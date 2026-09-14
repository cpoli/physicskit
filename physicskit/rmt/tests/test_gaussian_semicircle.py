"""Tests for the Gaussian ensembles (GOE/GUE/GSE) and the Wigner semicircle
benchmark.

Per the design plan (see ``physicskit.rmt.validation.base``), we do NOT gate on a
single p > 0.05 KS test at one fixed N -- KS power grows with N, so that
criterion becomes *less* likely to pass as N grows, which is backwards for
demonstrating asymptotic convergence. Instead we assert that the distance
to the theoretical semicircle law shrinks as N increases.
"""

import numpy as np
import pytest
from cache_utils import cached_sample
from scipy.stats import linregress

import physicskit.rmt as rmt

CLASSICAL_ENSEMBLES = [
    ("GOE", rmt.ensembles.GOE, 1.0),
    ("GUE", rmt.ensembles.GUE, 2.0),
    ("GSE", rmt.ensembles.GSE, 4.0),
]


@pytest.mark.parametrize("name,cls,expected_beta", CLASSICAL_ENSEMBLES)
def test_beta_is_correct(name, cls, expected_beta):
    ens = cls(n=50, seed=0)
    assert ens.beta == expected_beta


@pytest.mark.parametrize("name,cls,expected_beta", CLASSICAL_ENSEMBLES)
def test_sample_shape_and_reproducibility(name, cls, expected_beta):
    ens_a = cls(n=200, seed=123)
    ens_b = cls(n=200, seed=123)
    spec_a = ens_a.sample(n_samples=5)
    spec_b = ens_b.sample(n_samples=5)

    assert spec_a.eigenvalues.shape == (5, 200)
    # same seed -> identical draws
    np.testing.assert_allclose(spec_a.eigenvalues, spec_b.eigenvalues)


@pytest.mark.parametrize("name,cls,expected_beta", CLASSICAL_ENSEMBLES)
def test_eigenvalues_are_sorted_and_real(name, cls, expected_beta):
    ens = cls(n=100, seed=1)
    spectrum = cached_sample(ens, n_samples=3)
    for row in spectrum.eigenvalues:
        assert np.all(np.isfinite(row))
        assert np.all(np.diff(row) >= 0)  # eigh_tridiagonal returns ascending order


@pytest.mark.parametrize("name,cls,expected_beta", CLASSICAL_ENSEMBLES)
def test_rescaled_support_approaches_semicircle_radius(name, cls, expected_beta):
    # At large n, essentially all rescaled eigenvalues should fall within
    # [-2, 2] (+ small finite-size overshoot), and the extremes should be
    # close to +/-2 (Tracy-Widom edge fluctuations are O(n^{-2/3}), much
    # smaller than the bulk scale checked here).
    ens = cls(n=3000, seed=2)
    spectrum = cached_sample(ens, n_samples=1)
    rescaled = spectrum.rescaled.ravel()
    assert rescaled.max() < 2.5
    assert rescaled.min() > -2.5
    assert rescaled.max() > 1.8
    assert rescaled.min() < -1.8


@pytest.mark.parametrize("name,cls,expected_beta", CLASSICAL_ENSEMBLES)
def test_semicircle_convergence_rate_decreases_with_n(name, cls, expected_beta):
    # NOTE on methodology: eigenvalues within a single sampled matrix are
    # NOT independent (level repulsion correlates them), so the
    # distance-to-theory at any *single* N is a noisy statistic -- a
    # strict pairwise decrease across only 2-3 N values is not a
    # statistically robust thing to assert (it can fail by chance even
    # when the ensemble is implemented correctly). Instead, fit the
    # log-log trend across five N values and require a clearly negative
    # slope; this is the same "convergence curve, not a single p > 0.05
    # gate" philosophy from the design plan, made robust to per-point noise.
    #
    # Looped manually here (rather than via benchmark.convergence_curve)
    # so each N's sample can be disk-cached individually.
    benchmark = rmt.validation.WignerSemicircle()
    n_values = [100, 300, 900, 2700, 8100]
    results = []
    for n in n_values:
        ens = cls(n=n, seed=42)
        spectrum = cached_sample(ens, n_samples=30)
        results.append(benchmark.validate(spectrum, seed=42))

    ks_stats = np.array([r.ks_statistic for r in results])
    wasserstein = np.array([r.wasserstein_distance for r in results])

    ks_slope = linregress(np.log(n_values), np.log(ks_stats)).slope
    # KS convergence is the clean, low-noise signal here: theory predicts
    # roughly -1/2 to -1; require a clearly negative trend well short of
    # that range to keep the test robust to run-to-run noise.
    assert ks_slope < -0.3, f"{name}: expected shrinking KS distance, slope={ks_slope}"

    # Wasserstein distance (vs a large fixed reference sample) is a noisier
    # signal at this sample budget -- use it as a coarser sanity check
    # (final smaller than initial) rather than a strict per-step trend.
    assert wasserstein[-1] < wasserstein[0]

    # By the largest N tested, agreement should already be tight in
    # absolute terms. The Wasserstein floor here is set by residual
    # Monte Carlo noise (correlated eigenvalues within a sample reduce
    # the effective sample size below n * n_samples), not by any
    # remaining bias -- verified empirically to sit around 0.002-0.003
    # at n=8100 with n_samples=30 for GOE/GUE/GSE.
    assert ks_stats[-1] < 0.001
    assert wasserstein[-1] < 0.005


def test_continuum_beta_also_matches_semicircle():
    # beta need not be 1, 2, or 4 -- the semicircle law (and this
    # construction) holds for any beta > 0.
    ens = rmt.ensembles.HermiteBetaEnsemble(n=2000, beta=1.7, seed=3)
    spectrum = cached_sample(ens, n_samples=25)
    benchmark = rmt.validation.WignerSemicircle()
    result = benchmark.validate(spectrum, seed=3)
    assert result.ks_statistic < 0.01


def test_invalid_beta_rejected():
    with pytest.raises(ValueError):
        rmt.ensembles.HermiteBetaEnsemble(n=100, beta=0, seed=0)
    with pytest.raises(ValueError):
        rmt.ensembles.HermiteBetaEnsemble(n=100, beta=-1, seed=0)
