"""Tests for the Girko ensembles (IIDEnsemble, GirkoElliptic) --
universality generalizations of the Ginibre ensembles: the circular and
elliptic laws hold for any i.i.d. entry distribution with the right
first two moments, not just Gaussian.

IIDEnsemble is validated against the SAME ``CircularLaw`` benchmark
already used for Ginibre in ``tests/test_ginibre_circular_law.py`` --
that reuse is itself the point: Girko's theorem says non-Gaussian iid
matrices obey the identical limiting law, so no new theoretical machinery
is needed. GirkoElliptic has no existing benchmark (the general 2-D
elliptic density isn't wired into the KS/Wasserstein ``Benchmark`` base
class), so it is checked via its known exact structural limits
(rho=+-1) and the known ellipse semi-axes (1+rho, 1-rho) at general rho.
"""

import numpy as np
import pytest
from cache_utils import cached_sample
from scipy.stats import linregress

import physicskit.rmt as rmt


def test_iid_ensemble_beta_is_none():
    ens = rmt.ensembles.IIDEnsemble(n=30, entry_sampler=rmt.ensembles.rademacher, seed=0)
    assert ens.beta is None


@pytest.mark.parametrize(
    "name,entry_sampler,complex_entries",
    [
        ("uniform-complex", rmt.ensembles.uniform_unit_variance, True),
        ("rademacher-complex", rmt.ensembles.rademacher, True),
        ("exponential-real", rmt.ensembles.exponential_centered_unit_variance, False),
    ],
)
def test_iid_ensemble_eigenvalues_are_complex(name, entry_sampler, complex_entries):
    ens = rmt.ensembles.IIDEnsemble(
        n=50,
        entry_sampler=entry_sampler,
        complex_entries=complex_entries,
        seed=1,
    )
    spectrum = cached_sample(ens, n_samples=5)
    assert np.iscomplexobj(spectrum.eigenvalues)
    assert spectrum.eigenvalues.shape == (5, 50)


def test_iid_ensemble_reproducibility():
    ens_a = rmt.ensembles.IIDEnsemble(n=40, entry_sampler=rmt.ensembles.rademacher, seed=42)
    ens_b = rmt.ensembles.IIDEnsemble(n=40, entry_sampler=rmt.ensembles.rademacher, seed=42)
    spec_a = ens_a.sample(n_samples=3)
    spec_b = ens_b.sample(n_samples=3)
    np.testing.assert_allclose(spec_a.eigenvalues, spec_b.eigenvalues)


@pytest.mark.parametrize(
    "name,entry_sampler,complex_entries,seed",
    [
        ("uniform", rmt.ensembles.uniform_unit_variance, True, 10),
        ("rademacher", rmt.ensembles.rademacher, True, 11),
        ("exponential", rmt.ensembles.exponential_centered_unit_variance, True, 12),
    ],
)
def test_circular_law_convergence_shrinks_with_n_non_gaussian(name, entry_sampler, complex_entries, seed):
    # Same benchmark, same convergence-rate philosophy, and (per the
    # module docstring) the same expected shallower log-log slope as
    # Ginibre's own non-Hermitian convergence -- as this is meant to be
    # the identical limiting law reached from a non-Gaussian starting
    # point, not a distinct, looser standard.
    benchmark = rmt.validation.CircularLaw()
    n_values = [40, 90, 200, 400]
    results = []
    for n in n_values:
        ens = rmt.ensembles.IIDEnsemble(
            n=n,
            entry_sampler=entry_sampler,
            complex_entries=complex_entries,
            seed=seed,
        )
        spectrum = cached_sample(ens, n_samples=12)
        results.append(benchmark.validate(spectrum, seed=seed))

    ks_stats = np.array([r.ks_statistic for r in results])
    slope = linregress(np.log(n_values), np.log(ks_stats)).slope
    assert slope < -0.1, f"{name}: expected shrinking KS distance, slope={slope}"
    assert ks_stats[-1] < 0.04


def test_girko_elliptic_rho_out_of_range_raises():
    with pytest.raises(ValueError):
        rmt.ensembles.GirkoElliptic(n=10, rho=1.5, seed=0)


def test_girko_elliptic_beta_is_none():
    ens = rmt.ensembles.GirkoElliptic(n=20, rho=0.5, seed=0)
    assert ens.beta is None


def test_girko_elliptic_reproducibility():
    ens_a = rmt.ensembles.GirkoElliptic(n=40, rho=0.3, seed=42)
    ens_b = rmt.ensembles.GirkoElliptic(n=40, rho=0.3, seed=42)
    spec_a = ens_a.sample(n_samples=3)
    spec_b = ens_b.sample(n_samples=3)
    np.testing.assert_allclose(spec_a.eigenvalues, spec_b.eigenvalues)


def test_girko_elliptic_rho_zero_matches_circular_law():
    # rho=0 -> plain i.i.d. real entries -> same limiting law as
    # IIDEnsemble(complex_entries=False) / GinOE (circular law).
    benchmark = rmt.validation.CircularLaw()
    ens = rmt.ensembles.GirkoElliptic(n=300, rho=0.0, seed=2)
    spectrum = cached_sample(ens, n_samples=15)
    result = benchmark.validate(spectrum, seed=2)
    assert result.ks_statistic < 0.05


def test_girko_elliptic_rho_plus_one_gives_exactly_real_eigenvalues():
    # rho=1 -> X exactly symmetric -> eigenvalues exactly real (Wigner).
    ens = rmt.ensembles.GirkoElliptic(n=60, rho=1.0, seed=3)
    spectrum = cached_sample(ens, n_samples=5)
    assert np.abs(spectrum.eigenvalues.imag).max() < 1e-8


def test_girko_elliptic_rho_minus_one_gives_exactly_imaginary_eigenvalues():
    # rho=-1 -> X exactly antisymmetric -> eigenvalues exactly purely
    # imaginary (real antisymmetric matrices have eigenvalues 0 or
    # imaginary conjugate pairs).
    ens = rmt.ensembles.GirkoElliptic(n=60, rho=-1.0, seed=4)
    spectrum = cached_sample(ens, n_samples=5)
    assert np.abs(spectrum.eigenvalues.real).max() < 1e-8


def test_correlated_pair_matrix_has_correct_variance_and_correlation():
    # Direct, un-cached check of the raw construction's second moments
    # (not just the downstream eigenvalue law) against the values
    # asserted in the module docstring, pooled over many independent
    # matrices for statistical accuracy.
    from physicskit.rmt.ensembles.girko import _correlated_pair_matrix, standard_normal

    rng = np.random.default_rng(5)
    n = 8
    rho = 0.6
    upper_vals, lower_vals = [], []
    for _ in range(4000):
        x = _correlated_pair_matrix(n, rho, standard_normal, rng)
        upper_vals.append(x[0, 1])
        lower_vals.append(x[1, 0])
    upper_vals = np.array(upper_vals)
    lower_vals = np.array(lower_vals)
    assert upper_vals.var() == pytest.approx(1.0, abs=0.1)
    assert lower_vals.var() == pytest.approx(1.0, abs=0.1)
    corr = np.corrcoef(upper_vals, lower_vals)[0, 1]
    assert corr == pytest.approx(rho, abs=0.05)


@pytest.mark.parametrize("rho", [-0.5, 0.0, 0.5, 0.8])
def test_girko_elliptic_semi_axes_bound_the_spectrum_at_large_n(rho):
    # Eigenvalues should (asymptotically) stay within the ellipse of
    # semi-axes (1+rho, 1-rho); allow a modest finite-n margin rather
    # than a tight asymptotic bound.
    n = 800
    ens = rmt.ensembles.GirkoElliptic(n=n, rho=rho, seed=6)
    spectrum = cached_sample(ens, n_samples=3)
    z = spectrum.rescaled.ravel()
    a, b = 1.0 + rho, max(1.0 - rho, 1e-6)
    elliptic_radius = np.sqrt((z.real / a) ** 2 + (z.imag / b) ** 2)
    frac_outside = np.mean(elliptic_radius > 1.15)
    assert frac_outside < 0.03, f"rho={rho}: {frac_outside:.3f} of mass outside the ellipse"
