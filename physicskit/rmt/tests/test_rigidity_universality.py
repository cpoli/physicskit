"""Tests for spectral rigidity (number variance) and universality
testing.

The GUE number variance formula is *exact* (derived directly from the
already-validated sine kernel, not an approximation), so it's tested
against a tight tolerance. The GOE formula is confirmed by two
independent secondary sources; the GSE formula follows the same
commonly-cited pattern but was verified only empirically here (against
Monte Carlo GSE) before being trusted -- both are large-L asymptotic,
so tested with a looser tolerance and only at moderately large L.
"""

import numpy as np
import pytest
from cache_utils import cached_sample

import physicskit.rmt as rmt


def test_poisson_number_variance_is_exactly_l():
    l_values = [1, 5, 10, 50]
    np.testing.assert_allclose(rmt.stats.number_variance_poisson(l_values), l_values)


def test_gue_number_variance_much_smaller_than_poisson():
    # The defining qualitative signature of spectral rigidity: GUE's
    # number variance grows logarithmically, Poisson's linearly, so at
    # any reasonably large L, GUE's variance is far smaller.
    l_values = [10, 50, 100]
    gue = rmt.stats.number_variance_gue_exact(l_values)
    poisson = rmt.stats.number_variance_poisson(l_values)
    assert np.all(gue < poisson / 3)


def test_gue_exact_matches_empirical_window_counting():
    ens = rmt.ensembles.GUE(n=4000, seed=1)
    spectrum = cached_sample(ens, n_samples=15)
    l_values = [2, 5, 10, 20]
    theory = rmt.stats.number_variance_gue_exact(l_values)
    empirical = rmt.stats.number_variance_empirical(spectrum, rmt.stats.semicircle_cdf, l_values, n_windows=200, seed=0)
    np.testing.assert_allclose(theory, empirical, atol=0.05)


def test_goe_asymptotic_matches_empirical_at_large_l():
    ens = rmt.ensembles.GOE(n=4000, seed=2)
    spectrum = cached_sample(ens, n_samples=15)
    l_values = [5, 10, 20]
    theory = rmt.stats.number_variance_goe_asymptotic(l_values)
    empirical = rmt.stats.number_variance_empirical(spectrum, rmt.stats.semicircle_cdf, l_values, n_windows=300, seed=0)
    np.testing.assert_allclose(theory, empirical, atol=0.06)


def test_gse_asymptotic_matches_empirical_at_large_l():
    ens = rmt.ensembles.GSE(n=4000, seed=3)
    spectrum = cached_sample(ens, n_samples=15)
    l_values = [5, 10, 20]
    theory = rmt.stats.number_variance_gse_asymptotic(l_values)
    empirical = rmt.stats.number_variance_empirical(spectrum, rmt.stats.semicircle_cdf, l_values, n_windows=300, seed=0)
    np.testing.assert_allclose(theory, empirical, atol=0.03)


def test_number_variance_theory_dispatch():
    l_values = [10]
    np.testing.assert_allclose(
        rmt.stats.number_variance_theory(l_values, beta=1),
        rmt.stats.number_variance_goe_asymptotic(l_values),
    )
    np.testing.assert_allclose(
        rmt.stats.number_variance_theory(l_values, beta=2),
        rmt.stats.number_variance_gue_exact(l_values),
    )
    np.testing.assert_allclose(
        rmt.stats.number_variance_theory(l_values, beta=4),
        rmt.stats.number_variance_gse_asymptotic(l_values),
    )
    with pytest.raises(ValueError):
        rmt.stats.number_variance_theory(l_values, beta=3)


# --- spectral rigidity (Delta_3) ---


def test_poisson_delta3_is_exactly_l_over_15():
    l_values = [3, 15, 30, 60]
    np.testing.assert_allclose(rmt.stats.spectral_rigidity_poisson(l_values), np.array(l_values) / 15.0)


def test_poisson_delta3_empirical_matches_exact():
    ens = rmt.ensembles.PoissonEnsemble(n=4000, seed=21)
    spectrum = cached_sample(ens, n_samples=8)
    l_values = [5, 10, 20, 40]

    def identity_cdf(x):
        return x / spectrum.n

    empirical = rmt.stats.spectral_rigidity_empirical(spectrum, identity_cdf, l_values, n_windows=300, seed=0)
    theory = rmt.stats.spectral_rigidity_poisson(l_values)
    np.testing.assert_allclose(theory, empirical, rtol=0.08)


@pytest.mark.slow
def test_gue_delta3_theory_matches_empirical():
    ens = rmt.ensembles.GUE(n=150, seed=22)
    spectrum = cached_sample(ens, n_samples=20)
    l_values = [3, 5, 8, 12]
    theory = rmt.stats.spectral_rigidity_theory(l_values, beta=2)
    empirical = rmt.stats.spectral_rigidity_empirical(spectrum, rmt.stats.semicircle_cdf, l_values, n_windows=200, seed=0)
    np.testing.assert_allclose(theory, empirical, rtol=0.05)


@pytest.mark.slow
def test_goe_delta3_theory_matches_empirical_at_large_l():
    ens = rmt.ensembles.GOE(n=200, seed=23)
    spectrum = cached_sample(ens, n_samples=20)
    l_values = [8, 12, 16, 20]
    theory = rmt.stats.spectral_rigidity_theory(l_values, beta=1)
    empirical = rmt.stats.spectral_rigidity_empirical(spectrum, rmt.stats.semicircle_cdf, l_values, n_windows=250, seed=0)
    np.testing.assert_allclose(theory, empirical, rtol=0.1)


@pytest.mark.slow
def test_gue_delta3_much_smaller_than_poisson():
    l_values = [10, 50, 100]
    gue = rmt.stats.spectral_rigidity_theory(l_values, beta=2)
    poisson = rmt.stats.spectral_rigidity_poisson(l_values)
    assert np.all(gue < poisson / 3)


# --- universality ---


@pytest.mark.slow
@pytest.mark.parametrize("beta", [1, 2])
def test_universality_holds_across_entry_distributions(beta):
    result = rmt.validation.check_universality(n=300, beta=beta, n_samples=10, seed=5)
    assert set(result.per_distribution.keys()) == set(rmt.validation.DEFAULT_ENTRY_DISTRIBUTIONS.keys())
    assert result.max_ks_statistic < 0.035


@pytest.mark.parametrize("beta", [1, 2])
def test_general_wigner_ensemble_matches_semicircle_directly(beta):
    ens = rmt.ensembles.GeneralWignerEnsemble(n=1000, entry_sampler=rmt.ensembles.rademacher, beta=beta, seed=7)
    spectrum = cached_sample(ens, n_samples=20)
    benchmark = rmt.validation.WignerSemicircle()
    result = benchmark.validate(spectrum, seed=7)
    assert result.ks_statistic < 0.01


def test_general_wigner_ensemble_rejects_beta_4():
    with pytest.raises(ValueError):
        rmt.ensembles.GeneralWignerEnsemble(n=100, entry_sampler=rmt.ensembles.rademacher, beta=4, seed=0)


def test_asymmetric_exponential_entries_still_converge():
    # The strongest universality stress test in the default set: a
    # skewed, unbounded entry distribution should still give the
    # symmetric semicircle law.
    ens = rmt.ensembles.GeneralWignerEnsemble(
        n=1000,
        entry_sampler=rmt.ensembles.exponential_centered_unit_variance,
        beta=2,
        seed=11,
    )
    spectrum = cached_sample(ens, n_samples=20)
    benchmark = rmt.validation.WignerSemicircle()
    result = benchmark.validate(spectrum, seed=11)
    assert result.ks_statistic < 0.01
