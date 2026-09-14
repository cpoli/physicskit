"""Tests for the nearest-neighbor spacing distribution, the ratio
statistic, and their respective surmises.

Important methodological distinction from ``test_gaussian_semicircle.py``:
the Wigner semicircle law is the *exact* N -> infinity limit, so its KS
distance to a sampled ensemble should shrink toward zero as N grows. The
Wigner surmise (spacing) and the Atas et al. (2013) surmise (ratio) are
NOT exact limits -- they are known, very good, but imperfect
approximations (a fixed ~0.1-1% level discrepancy from the true limiting
distribution, which involves Fredholm-determinant sine-kernel statistics
with no elementary closed form). That gap does not vanish as N grows, so
these tests check "already small at moderate N" rather than "shrinks to
zero" -- asserting the latter would eventually fail once N is large
enough for the surmise's own approximation error to dominate over
finite-size noise, which is expected behavior, not a defect.
"""

import numpy as np
import pytest
from cache_utils import cached_sample

import physicskit.rmt as rmt

CLASSICAL = [
    ("GOE", rmt.ensembles.GOE, 1),
    ("GUE", rmt.ensembles.GUE, 2),
    ("GSE", rmt.ensembles.GSE, 4),
]

# Exact surmise-mean reference values (Atas et al. 2013), NOT the
# numerically-exact full-ensemble means (which differ at the ~1% level --
# see docstring in physicskit.rmt/stats/ratios.py).
RATIO_SURMISE_EXACT_MEANS = {
    1: 4 - 2 * np.sqrt(3),  # 0.535898...
}


@pytest.mark.parametrize("name,cls,beta", CLASSICAL)
def test_unfolded_mean_spacing_is_unity(name, cls, beta):
    ens = cls(n=1000, seed=10)
    spectrum = cached_sample(ens, n_samples=30)
    spacings = rmt.stats.nearest_neighbor_spacings(spectrum, rmt.stats.semicircle_cdf)
    assert spacings.mean() == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("name,cls,beta", CLASSICAL)
def test_spacing_distribution_matches_surmise_at_moderate_n(name, cls, beta):
    # Moderate N and sample count deliberately: pooling an enormous number
    # of spacings would give the KS test enough power to reject on the
    # surmise's own known small approximation error rather than anything
    # wrong with the implementation (see module docstring).
    ens = cls(n=1000, seed=20)
    spectrum = cached_sample(ens, n_samples=20)
    spacings = rmt.stats.nearest_neighbor_spacings(spectrum, rmt.stats.semicircle_cdf)

    benchmark = rmt.validation.WignerSurmise(beta=beta)
    result = benchmark.validate(spacings, seed=20)
    assert result.ks_statistic < 0.015
    assert result.wasserstein_distance < 0.01


@pytest.mark.parametrize("name,cls,beta", CLASSICAL)
def test_ratio_statistics_in_unit_interval(name, cls, beta):
    ens = cls(n=500, seed=30)
    spectrum = cached_sample(ens, n_samples=10)
    ratios = rmt.stats.ratio_statistics(spectrum)
    assert np.all(ratios >= 0.0)
    assert np.all(ratios <= 1.0)


@pytest.mark.parametrize("name,cls,beta", CLASSICAL)
def test_ratio_distribution_matches_surmise_at_moderate_n(name, cls, beta):
    ens = cls(n=1000, seed=40)
    spectrum = cached_sample(ens, n_samples=20)
    ratios = rmt.stats.ratio_statistics(spectrum)

    benchmark = rmt.validation.RatioDistribution(beta=beta)
    result = benchmark.validate(ratios, seed=40)
    assert result.ks_statistic < 0.02
    assert result.wasserstein_distance < 0.01


def test_ratio_surmise_exact_mean_matches_atas_et_al():
    # The surmise's OWN mean (not the full-ensemble numerical mean) has a
    # known closed form at beta=1: 4 - 2*sqrt(3). This checks the surmise
    # model itself, independent of any sampling.
    model = rmt.stats.RatioSurmise(beta=1)
    grid = np.linspace(0, 1, 200_000)
    mean_r = np.trapezoid(grid * model.pdf(grid), grid)
    assert mean_r == pytest.approx(RATIO_SURMISE_EXACT_MEANS[1], abs=1e-3)


@pytest.mark.parametrize("name,cls,beta", CLASSICAL)
def test_spacing_and_ratio_agree_independently(name, cls, beta):
    # The spacing distribution requires unfolding; the ratio statistic
    # does not. If an unfolding bug were inflating apparent agreement (or
    # disagreement) in the spacing test, the ratio test -- computed on
    # raw eigenvalues -- would not be affected the same way. Checking
    # both pass together (loosely) guards against that class of bug.
    ens = cls(n=1200, seed=50)
    spectrum = cached_sample(ens, n_samples=20)

    spacings = rmt.stats.nearest_neighbor_spacings(spectrum, rmt.stats.semicircle_cdf)
    spacing_result = rmt.validation.WignerSurmise(beta=beta).validate(spacings, seed=50)

    ratios = rmt.stats.ratio_statistics(spectrum)
    ratio_result = rmt.validation.RatioDistribution(beta=beta).validate(ratios, seed=50)

    assert spacing_result.ks_statistic < 0.02
    assert ratio_result.ks_statistic < 0.02
