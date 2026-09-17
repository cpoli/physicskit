"""Tests for the single ring theorem and NonHermitianWishartEnsemble.

The exact Marchenko-Pastur moment identities E[X] = 1 and
E[1/X] = 1/(1-gamma) (gamma < 1), which give the theoretical ring radii
r_out=1, r_in=sqrt(1-gamma), were verified numerically against direct
Monte Carlo before being used here (see
``physicskit.rmt.stats.single_ring`` and ``physicskit.rmt.ensembles.single_ring`` module
docstrings). Like every other limiting-law benchmark in this package,
the single ring theorem is an N -> infinity result, so tests check that
KS distance to the theoretical annulus SHRINKS with increasing N, not
that it passes a fixed small-N threshold (confirmed during development:
KS went from 0.046 at n=150 to 0.015 at n=1200 for a fixed gamma).
"""

import numpy as np
import pytest
from cache_utils import cached_sample
from scipy.stats import linregress

import physicskit.rmt as rmt


def test_wishart_moment_identities():
    # E[X] = 1 and E[1/X] = 1/(1-gamma) for the Marchenko-Pastur
    # distribution -- the exact facts the theoretical ring radii rest
    # on. Checked directly here against a fresh Monte Carlo sample,
    # independent of the ensemble/benchmark machinery.
    from physicskit.rmt.utils.tridiagonal import sample_laguerre_beta_eigenvalues

    rng = np.random.default_rng(0)
    n, m = 500, 1000
    gamma = n / m
    eigs = sample_laguerre_beta_eigenvalues(m, n, 2.0, rng)
    assert eigs.mean() == pytest.approx(1.0, rel=0.02)
    assert np.mean(1.0 / eigs) == pytest.approx(1.0 / (1.0 - gamma), rel=0.02)


def test_single_ring_radii_from_uniform_singular_values():
    # Constant singular values s=1 for everything -> a genuine circle
    # (r_in = r_out = 1), the trivial sanity check of the raw formula.
    r_in, r_out = rmt.stats.single_ring_radii(np.ones(50))
    assert r_in == pytest.approx(1.0)
    assert r_out == pytest.approx(1.0)


def test_single_ring_radii_matches_wishart_theory():
    gamma = 0.4
    r_in, r_out = rmt.stats.single_ring_radii_wishart_theory(gamma)
    assert r_out == 1.0
    assert r_in == pytest.approx(np.sqrt(0.6))


def test_single_ring_radii_rejects_invalid_gamma():
    with pytest.raises(ValueError):
        rmt.stats.single_ring_radii_wishart_theory(1.0)
    with pytest.raises(ValueError):
        rmt.stats.single_ring_radii_wishart_theory(0.0)


def test_nonhermitian_wishart_rejects_gamma_geq_1():
    with pytest.raises(ValueError):
        rmt.ensembles.NonHermitianWishartEnsemble(n=100, m=100, seed=0)


def test_nonhermitian_wishart_rejects_nonpositive_beta():
    with pytest.raises(ValueError):
        rmt.ensembles.NonHermitianWishartEnsemble(n=10, m=20, beta=0.0, seed=0)


def test_single_ring_ensemble_and_nonhermitian_wishart_fresh_samples_are_finite():
    # Fresh (uncached) direct .sample() calls, to actually exercise
    # _sample_eigenvalues/natural_scale rather than risk a cache hit.
    ens1 = rmt.ensembles.SingleRingEnsemble(n=6, singular_value_sampler=lambda rng, n: np.ones(n), seed=321)
    spectrum1 = ens1.sample(n_samples=2)
    assert np.all(np.isfinite(spectrum1.eigenvalues))
    assert ens1.natural_scale() == 1.0

    ens2 = rmt.ensembles.NonHermitianWishartEnsemble(n=6, m=12, seed=321)
    spectrum2 = ens2.sample(n_samples=2)
    assert np.all(np.isfinite(spectrum2.eigenvalues))
    assert ens2.natural_scale() == 1.0


def test_nonhermitian_wishart_eigenvalue_radii_stay_within_ring_edges():
    n, m = 400, 1000
    ens = rmt.ensembles.NonHermitianWishartEnsemble(n=n, m=m, seed=1)
    spectrum = cached_sample(ens, n_samples=5)
    r_in, r_out = rmt.stats.single_ring_radii_wishart_theory(ens.gamma)
    radii = np.abs(spectrum.eigenvalues.ravel())
    # A modest margin for finite-n boundary fluctuation (the theorem is
    # an n -> infinity statement about the support, not a hard bound at
    # finite n).
    assert radii.min() > r_in - 0.15
    assert radii.max() < r_out + 0.15


def test_single_ring_theorem_convergence_shrinks_with_n():
    gamma_fixed = 0.5
    n_values = [100, 300, 900]
    r_in, r_out = rmt.stats.single_ring_radii_wishart_theory(gamma_fixed)
    benchmark = rmt.validation.SingleRingTheorem(r_in=r_in, r_out=r_out)

    results = []
    for n in n_values:
        m = int(n / gamma_fixed)
        ens = rmt.ensembles.NonHermitianWishartEnsemble(n=n, m=m, seed=5)
        spectrum = cached_sample(ens, n_samples=max(2, 3000 // n))
        results.append(benchmark.validate(spectrum, seed=5))

    ks_stats = np.array([r.ks_statistic for r in results])
    slope = linregress(np.log(n_values), np.log(ks_stats)).slope
    assert slope < -0.2, f"expected shrinking KS distance, slope={slope}"


def test_single_ring_theorem_rejects_invalid_radii():
    with pytest.raises(ValueError):
        rmt.validation.SingleRingTheorem(r_in=1.0, r_out=0.5)


def test_balanced_gamma_one_collapses_to_filled_disk():
    # gamma -> 1 (e.g. via a plain SingleRingEnsemble with Wishart-type
    # singular values at m=n) should give r_in ~ 0, consistent with the
    # ordinary (Ginibre) circular law rather than a genuine ring.
    from physicskit.rmt.utils.tridiagonal import sample_laguerre_beta_eigenvalues

    def wishart_singular_values(rng, n):
        return np.sqrt(sample_laguerre_beta_eigenvalues(n, n, 2.0, rng))

    ens = rmt.ensembles.SingleRingEnsemble(n=300, singular_value_sampler=wishart_singular_values, seed=2)
    spectrum = cached_sample(ens, n_samples=5)
    radii = np.abs(spectrum.eigenvalues.ravel())
    assert radii.min() < 0.1  # close to 0, not bounded away like a genuine ring
