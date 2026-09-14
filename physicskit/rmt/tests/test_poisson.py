"""Tests for the Poisson ensemble -- the integrable-system (Berry-Tabor)
null model that RMT level-repulsion statistics are always contrasted
against.

Unlike GOE/GUE/GSE, whose spacing and ratio distributions are only
approximated by a surmise (exact for a reduced few-level model), the
Poisson process's spacing and ratio laws are exactly known in closed
form -- so tests here check against exact theory throughout, not a
surmise. The ratio-statistic law (2/(1+r)**2 for the min/max-normalized
r used by ``physicskit.rmt.stats.ratios.ratio_statistics``) is derived directly
in the module docstring from the ratio of two i.i.d. Exponential(1)
variables; ``test_ratio_statistic_matches_derived_exact_formula`` checks
that derivation numerically before it is used anywhere else.
"""

import numpy as np
import pytest
from cache_utils import cached_sample
from scipy.stats import expon, kstest

import physicskit.rmt as rmt


def test_beta_is_none():
    ens = rmt.ensembles.PoissonEnsemble(n=30, seed=0)
    assert ens.beta is None


def test_shape_and_reproducibility():
    ens_a = rmt.ensembles.PoissonEnsemble(n=50, seed=42)
    ens_b = rmt.ensembles.PoissonEnsemble(n=50, seed=42)
    spec_a = ens_a.sample(n_samples=5)
    spec_b = ens_b.sample(n_samples=5)
    assert spec_a.eigenvalues.shape == (5, 50)
    np.testing.assert_allclose(spec_a.eigenvalues, spec_b.eigenvalues)


def test_levels_are_strictly_increasing():
    # Partial sums of strictly positive Exponential draws.
    ens = rmt.ensembles.PoissonEnsemble(n=100, seed=1)
    spectrum = cached_sample(ens, n_samples=5)
    for row in spectrum.eigenvalues:
        assert np.all(np.diff(row) > 0)


def test_mean_spacing_is_one():
    ens = rmt.ensembles.PoissonEnsemble(n=2000, seed=2)
    spectrum = cached_sample(ens, n_samples=10)
    spacings = np.concatenate([np.diff(row) for row in spectrum.eigenvalues])
    assert spacings.mean() == pytest.approx(1.0, abs=0.03)


def test_spacing_distribution_matches_exact_exponential_law():
    # No unfolding needed (see module docstring) -- raw spacings are
    # already Exponential(1) by construction, checked directly against
    # theory rather than a surmise.
    ens = rmt.ensembles.PoissonEnsemble(n=2000, seed=3)
    spectrum = cached_sample(ens, n_samples=20)
    spacings = np.concatenate([np.diff(row) for row in spectrum.eigenvalues])
    ks = kstest(spacings, expon(scale=1.0).cdf)
    assert ks.statistic < 0.01


def test_ratio_statistic_matches_derived_exact_formula():
    # Verify 2/(1+r)**2 (module docstring derivation) directly against
    # Monte Carlo ratios of independent Exponential(1) pairs, before
    # trusting it against the ensemble itself.
    rng = np.random.default_rng(4)
    x = rng.exponential(1.0, size=2_000_000)
    y = rng.exponential(1.0, size=2_000_000)
    r = np.minimum(x, y) / np.maximum(x, y)

    def poisson_ratio_cdf(r):
        return 2.0 * r / (1.0 + r)

    ks = kstest(r, poisson_ratio_cdf)
    assert ks.statistic < 0.002


def test_ensemble_ratio_statistic_matches_exact_poisson_law():
    ens = rmt.ensembles.PoissonEnsemble(n=2000, seed=5)
    spectrum = cached_sample(ens, n_samples=20)
    ratios = rmt.stats.ratio_statistics(spectrum)

    def poisson_ratio_cdf(r):
        return 2.0 * r / (1.0 + r)

    ks = kstest(ratios, poisson_ratio_cdf)
    assert ks.statistic < 0.01


def test_number_variance_grows_linearly_like_poisson_theory():
    # No unfolding needed -- raw levels already have unit mean density,
    # so window-counting is done directly rather than via
    # ``number_variance_empirical`` (which expects a fixed-shape
    # theoretical CDF to unfold against, the wrong abstraction for an
    # unbounded, already-uniform-density process -- see module
    # docstring).
    ens = rmt.ensembles.PoissonEnsemble(n=4000, seed=6)
    spectrum = cached_sample(ens, n_samples=15)
    l_values = [2, 5, 10, 20]
    rng = np.random.default_rng(7)

    empirical = []
    for length in l_values:
        counts = []
        for row in spectrum.eigenvalues:
            lo, hi = row[0], row[-1]
            starts = rng.uniform(lo, hi - length, size=200)
            counts.extend(np.sum((row >= s) & (row < s + length)) for s in starts)
        empirical.append(np.var(counts))
    empirical = np.array(empirical)

    theory = rmt.stats.number_variance_poisson(l_values)
    np.testing.assert_allclose(empirical, theory, atol=0.6)


def test_poisson_number_variance_is_much_larger_than_sampled_gue():
    # The defining qualitative contrast this whole ensemble exists to
    # provide: GUE's number variance grows logarithmically (level
    # repulsion/rigidity), Poisson's grows linearly -- checked here
    # against an actually-sampled GUE spectrum, not just GUE's theory
    # curve (already covered by test_rigidity_universality.py).
    l_values = [10, 20]
    poisson_theory = rmt.stats.number_variance_poisson(l_values)

    gue_ens = rmt.ensembles.GUE(n=4000, seed=8)
    gue_spectrum = cached_sample(gue_ens, n_samples=15)
    gue_empirical = rmt.stats.number_variance_empirical(
        gue_spectrum,
        rmt.stats.semicircle_cdf,
        l_values,
        n_windows=200,
        seed=9,
    )
    assert np.all(gue_empirical < poisson_theory / 3)


def test_generalized_wigner_surmise_at_beta_zero_is_not_poisson():
    # Documents the important distinction from the module docstring: the
    # Gaussian-tailed generalized Wigner surmise does NOT reduce to the
    # exponential Poisson spacing law at beta=0 -- it gives a
    # half-Gaussian instead. Concretely, both densities are normalized
    # (integrate to 1) but disagree sharply at s=0: Poisson's exact
    # density there is exp(0)=1, the beta=0 surmise's is 2/pi.
    poisson_density_at_zero = 1.0
    surmise_density_at_zero = rmt.stats.wigner_surmise_pdf(0.0, beta=0)
    assert surmise_density_at_zero == pytest.approx(2.0 / np.pi)
    assert abs(surmise_density_at_zero - poisson_density_at_zero) > 0.3
