"""Tests for the real-Ginibre real-eigenvalue-count statistic and
benchmark.

The asymptotic formula sqrt(2n/pi) + 1/2 was fit against, and confirmed
by, direct Monte Carlo real-eigenvalue counts at n = 2..128 during
development (see ``physicskit.rmt.stats.real_ginibre`` module docstring) -- it
is NOT the exact Edelman-Kostlan-Shub finite-n closed form, so tests use
a loose tolerance and larger n rather than expecting near-exact
agreement.
"""

import numpy as np
import pytest
from cache_utils import cached_sample

import physicskit.rmt as rmt


def test_real_eigenvalue_count_empirical_on_known_matrix():
    # A diagonal (all-real-eigenvalue) "spectrum": every eigenvalue
    # should be counted as real.
    eigenvalues = np.array([[1.0 + 0j, 2.0 + 0j, -3.0 + 0j]])
    counts = rmt.stats.real_eigenvalue_count_empirical(eigenvalues)
    np.testing.assert_array_equal(counts, [3])


def test_real_eigenvalue_count_empirical_mixed():
    eigenvalues = np.array([[1.0 + 0j, 2.0 + 3.0j, 2.0 - 3.0j, -1.0 + 0j]])
    counts = rmt.stats.real_eigenvalue_count_empirical(eigenvalues)
    np.testing.assert_array_equal(counts, [2])


def test_asymptotic_matches_n2_exact_value():
    # n=2 has a well-known exact closed form: E[N_real] = sqrt(2)
    # (P(both eigenvalues real) = 1/sqrt(2)). The asymptotic formula is
    # not expected to be exact this low, but should be in the right
    # ballpark.
    theory = rmt.stats.real_eigenvalue_count_asymptotic(2)
    assert theory == pytest.approx(np.sqrt(2), abs=0.3)


def test_asymptotic_matches_monte_carlo_at_moderate_n():
    n = 64
    ens = rmt.ensembles.GinOE(n=n, seed=0)
    spectrum = cached_sample(ens, n_samples=2000)
    counts = rmt.stats.real_eigenvalue_count_empirical(spectrum.eigenvalues)
    empirical_mean = counts.mean()
    theory = rmt.stats.real_eigenvalue_count_asymptotic(n)
    assert empirical_mean == pytest.approx(theory, rel=0.1)


def test_real_ginibre_eigenvalue_count_benchmark():
    n = 100
    ens = rmt.ensembles.GinOE(n=n, seed=1)
    spectrum = cached_sample(ens, n_samples=2000)
    benchmark = rmt.validation.RealGinibreEigenvalueCount()
    result = benchmark.validate(spectrum)
    assert result.n == n
    assert result.relative_error < 0.1


def test_mean_real_count_grows_like_sqrt_n():
    # The qualitative signature: E[N_real] should grow much slower than
    # n itself (like sqrt(n)), unlike a generic extensive quantity.
    small = rmt.stats.real_eigenvalue_count_asymptotic(25)
    large = rmt.stats.real_eigenvalue_count_asymptotic(100)
    # n quadruples (25->100); sqrt(n) doubles, so the count should
    # roughly double too, not quadruple.
    assert 1.5 < large / small < 2.5


# --- real-eigenvalue density (not just count) ---


def test_density_asymptotic_is_zero_outside_unit_disk():
    n = 100
    x = np.array([-1.5, -1.01, 1.01, 1.5])
    density = rmt.stats.real_eigenvalue_density_asymptotic(x, n)
    np.testing.assert_array_equal(density, 0.0)


def test_density_asymptotic_integrates_to_the_count():
    # The uniform-box density's total integral (height * width 2) should
    # equal the already-verified total count formula.
    n = 200
    x = np.linspace(-1.0, 1.0, 20001)
    density = rmt.stats.real_eigenvalue_density_asymptotic(x, n)
    integral = np.trapezoid(density, x)
    assert integral == pytest.approx(rmt.stats.real_eigenvalue_count_asymptotic(n), rel=1e-3)


def test_density_empirical_matches_asymptotic_in_the_bulk():
    n = 300
    ens = rmt.ensembles.GinOE(n=n, seed=0)
    spectrum = cached_sample(ens, n_samples=2000)
    centers, density = rmt.stats.real_eigenvalue_density_empirical(spectrum.eigenvalues, n)
    theory = rmt.stats.real_eigenvalue_density_asymptotic(centers, n)

    bulk = np.abs(centers) < 0.8
    assert density[bulk].mean() == pytest.approx(theory[bulk].mean(), rel=0.15)


def test_density_empirical_drops_near_the_edge():
    # Qualitative edge-cutoff check: density well outside the unit disk
    # (after rescaling by sqrt(n)) should be much smaller than in the bulk.
    n = 300
    ens = rmt.ensembles.GinOE(n=n, seed=0)
    spectrum = cached_sample(ens, n_samples=2000)
    centers, density = rmt.stats.real_eigenvalue_density_empirical(spectrum.eigenvalues, n, x_max=1.3)
    bulk_density = density[np.abs(centers) < 0.5].mean()
    edge_density = density[np.abs(centers) > 1.15].mean()
    assert edge_density < 0.1 * bulk_density
