"""Tests for LOE/LUE/LSE and the Marchenko-Pastur law benchmark.

The bidiagonal beta-Laguerre construction (see
``physicskit.rmt.utils.tridiagonal.sample_laguerre_beta_eigenvalues``) was
verified during development against dense X^T X / m constructions at
beta=1, 2 (two-sample KS test between pooled eigenvalues, matching to
within Monte Carlo noise) and against the exact Marchenko-Pastur law
directly (including support edges) at beta=1, 2, 4. Like the semicircle
law, Marchenko-Pastur is an *exact* N -> infinity limit (unlike the
Wigner/Atas surmises), so convergence-rate assertions here follow the
same "shrinks as N grows" philosophy as the semicircle tests.
"""

import numpy as np
import pytest
from cache_utils import cached_sample

import physicskit.rmt as rmt

CLASSICAL_WISHART = [
    ("LOE", rmt.ensembles.LOE, 1.0),
    ("LUE", rmt.ensembles.LUE, 2.0),
    ("LSE", rmt.ensembles.LSE, 4.0),
]

GAMMA = 0.3  # aspect ratio n/m used across most tests


@pytest.mark.parametrize("name,cls,expected_beta", CLASSICAL_WISHART)
def test_beta_is_correct(name, cls, expected_beta):
    ens = cls(n=50, m=200, seed=0)
    assert ens.beta == expected_beta


@pytest.mark.parametrize("name,cls,expected_beta", CLASSICAL_WISHART)
def test_gamma_property(name, cls, expected_beta):
    ens = cls(n=100, m=400, seed=0)
    assert ens.gamma == pytest.approx(0.25)


@pytest.mark.parametrize("name,cls,expected_beta", CLASSICAL_WISHART)
def test_sample_shape_reproducibility_and_nonnegativity(name, cls, expected_beta):
    ens_a = cls(n=200, m=int(200 / GAMMA), seed=123)
    ens_b = cls(n=200, m=int(200 / GAMMA), seed=123)
    spec_a = ens_a.sample(n_samples=5)
    spec_b = ens_b.sample(n_samples=5)

    assert spec_a.eigenvalues.shape == (5, 200)
    np.testing.assert_allclose(spec_a.eigenvalues, spec_b.eigenvalues)
    # Wishart matrices are positive semi-definite -- eigenvalues >= 0.
    assert np.all(spec_a.eigenvalues >= -1e-10)


@pytest.mark.parametrize("name,cls,expected_beta", CLASSICAL_WISHART)
def test_support_matches_marchenko_pastur_edges(name, cls, expected_beta):
    n = 3000
    m = int(n / GAMMA)
    ens = cls(n=n, m=m, seed=2)
    spectrum = cached_sample(ens, n_samples=1)
    lo, hi = rmt.stats.mp_support(ens.gamma)
    eigs = spectrum.rescaled.ravel()
    # small finite-size tolerance around the exact edges
    assert eigs.min() > lo - 0.15
    assert eigs.max() < hi + 0.15
    assert eigs.min() < lo + 0.2
    assert eigs.max() > hi - 0.2


@pytest.mark.parametrize("name,cls,expected_beta", CLASSICAL_WISHART)
def test_marchenko_pastur_convergence_shrinks_with_n(name, cls, expected_beta):
    # MP is an exact limit (like the semicircle law), so -- unlike the
    # Wigner/Atas surmises -- convergence-to-zero across N is the right
    # thing to check, via the same robust log-log trend as
    # test_gaussian_semicircle.test_semicircle_convergence_rate_decreases_with_n.
    # Looped manually (rather than via benchmark.convergence_curve) so
    # each N's sample can be disk-cached individually.
    from scipy.stats import linregress

    benchmark = rmt.validation.MarchenkoPastur(gamma=GAMMA)
    n_values = [50, 150, 450, 1350]

    results = []
    for n in n_values:
        ens = cls(n=n, m=int(n / GAMMA), seed=17)
        spectrum = cached_sample(ens, n_samples=25)
        results.append(benchmark.validate(spectrum, seed=17))

    ks_stats = np.array([r.ks_statistic for r in results])
    slope = linregress(np.log(n_values), np.log(ks_stats)).slope
    assert slope < -0.3, f"{name}: expected shrinking KS distance, slope={slope}"
    assert ks_stats[-1] < 0.005


def test_continuum_beta_also_matches_marchenko_pastur():
    ens = rmt.ensembles.LaguerreBetaEnsemble(n=800, m=int(800 / GAMMA), beta=1.7, seed=3)
    spectrum = cached_sample(ens, n_samples=25)
    benchmark = rmt.validation.MarchenkoPastur(gamma=GAMMA)
    result = benchmark.validate(spectrum, seed=3)
    assert result.ks_statistic < 0.01


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        rmt.ensembles.LOE(n=100, m=50, seed=0)  # m < n
    with pytest.raises(ValueError):
        rmt.ensembles.LaguerreBetaEnsemble(n=100, m=200, beta=0, seed=0)
    with pytest.raises(ValueError):
        rmt.validation.MarchenkoPastur(gamma=1.5)  # only (0, 1] supported
    with pytest.raises(ValueError):
        rmt.validation.MarchenkoPastur(gamma=0.0)


def test_mp_cdf_is_zero_below_and_one_above_support_and_monotonic():
    gamma = 0.5
    lo, hi = rmt.stats.mp_support(gamma)
    x = np.linspace(lo - 0.5, hi + 0.5, 20)
    cdf = rmt.stats.mp_cdf(x, gamma)
    assert cdf[0] == 0.0
    assert cdf[-1] == 1.0
    assert np.all(np.diff(cdf) >= -1e-12)
