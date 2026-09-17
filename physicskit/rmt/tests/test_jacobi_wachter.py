"""Tests for JOE, JUE and the Wachter distribution benchmark.

The Wachter formula and its parametrization (a=m1/n, b=m2/n, inverse
aspect ratios) were NOT taken from memory -- an initial guess at the
formula (with a,b as the more familiar n/m aspect ratios) gave support
edges outside [0, 1], which is impossible for this ensemble, immediately
flagging the error. The correct formula and parametrization were found
via a targeted search and then verified numerically against both real
(beta=1) and complex (beta=2) simulations before being used here -- see
``physicskit.rmt/stats/wachter.py`` and the design notes.
"""

import numpy as np
import pytest
from cache_utils import cached_sample
from scipy.stats import linregress

import physicskit.rmt as rmt

JACOBI = [
    ("JOE", rmt.ensembles.JOE, 1),
    ("JUE", rmt.ensembles.JUE, 2),
    ("JSE", rmt.ensembles.JSE, 4),
]


@pytest.mark.parametrize("name,cls,beta", JACOBI)
def test_beta_is_correct(name, cls, beta):
    ens = cls(n=50, m1=150, m2=200, seed=0)
    assert ens.beta == beta


@pytest.mark.parametrize("name,cls,beta", JACOBI)
def test_a_b_properties(name, cls, beta):
    ens = cls(n=100, m1=200, m2=300, seed=0)
    assert ens.a == pytest.approx(2.0)
    assert ens.b == pytest.approx(3.0)


@pytest.mark.parametrize("name,cls,beta", JACOBI)
def test_eigenvalues_in_unit_interval(name, cls, beta):
    ens = cls(n=200, m1=400, m2=600, seed=1)
    spectrum = cached_sample(ens, n_samples=10)
    assert np.all(spectrum.eigenvalues >= -1e-9)
    assert np.all(spectrum.eigenvalues <= 1.0 + 1e-9)


@pytest.mark.parametrize("name,cls,beta", JACOBI)
def test_reproducibility(name, cls, beta):
    ens_a = cls(n=80, m1=160, m2=240, seed=42)
    ens_b = cls(n=80, m1=160, m2=240, seed=42)
    spec_a = ens_a.sample(n_samples=3)
    spec_b = ens_b.sample(n_samples=3)
    np.testing.assert_allclose(spec_a.eigenvalues, spec_b.eigenvalues)


@pytest.mark.parametrize("name,cls,beta", JACOBI)
def test_support_matches_wachter_edges(name, cls, beta):
    n, m1, m2 = 2000, 4000, 6000
    ens = cls(n=n, m1=m1, m2=m2, seed=2)
    spectrum = cached_sample(ens, n_samples=1)
    lo, hi = rmt.stats.wachter_support(m1 / n, m2 / n)
    eigs = spectrum.eigenvalues.ravel()
    assert eigs.min() > lo - 0.05
    assert eigs.max() < hi + 0.05
    assert eigs.min() < lo + 0.08
    assert eigs.max() > hi - 0.08


@pytest.mark.parametrize("name,cls,beta", JACOBI)
def test_wachter_convergence_shrinks_with_n(name, cls, beta):
    # Looped manually (rather than via benchmark.convergence_curve) so
    # each N's sample can be disk-cached individually.
    m1_ratio, m2_ratio = 2.0, 3.0  # fixed inverse aspect ratios as n scales
    n_values = [50, 150, 450]

    benchmark = rmt.validation.Wachter(a=m1_ratio, b=m2_ratio)
    results = []
    for n in n_values:
        ens = cls(n=n, m1=int(n * m1_ratio), m2=int(n * m2_ratio), seed=9)
        spectrum = cached_sample(ens, n_samples=20)
        results.append(benchmark.validate(spectrum, seed=9))

    ks_stats = np.array([r.ks_statistic for r in results])
    slope = linregress(np.log(n_values), np.log(ks_stats)).slope
    assert slope < -0.3, f"{name}: expected shrinking KS distance, slope={slope}"
    assert ks_stats[-1] < 0.01


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        rmt.ensembles.JOE(n=100, m1=50, m2=200, seed=0)  # m1 < n
    with pytest.raises(ValueError):
        rmt.ensembles.JacobiBetaEnsemble(n=100, m1=200, m2=200, beta=3, seed=0)
    with pytest.raises(ValueError):
        rmt.validation.Wachter(a=0.5, b=2.0)  # a < 1 invalid


def test_jse_eigenvalue_count_is_n_not_2n():
    # JSE's quaternion embedding internally builds 2n x 2n matrices; only
    # the n distinct (post-Kramers-deduplication) eigenvalues should be
    # returned, matching GSE/CSE/EffGSE's convention -- see module
    # docstring in physicskit.rmt/ensembles/jacobi.py.
    n = 40
    ens = rmt.ensembles.JSE(n=n, m1=80, m2=120, seed=0)
    spectrum = ens.sample(n_samples=2)
    assert spectrum.eigenvalues.shape[1] == n


def test_wachter_pdf_zero_outside_support():
    a, b = 2.0, 3.0
    lo, hi = rmt.stats.wachter_support(a, b)
    assert rmt.stats.wachter_pdf(np.array([lo - 0.1, hi + 0.1]), a, b)[0] == 0.0
    assert rmt.stats.wachter_pdf(np.array([lo - 0.1, hi + 0.1]), a, b)[1] == 0.0


def test_wachter_pdf_scalar_integrand_is_zero_outside_support():
    """quad() integrates _wachter_pdf_scalar directly (not the vectorized
    wachter_pdf above) to build wachter_cdf; it needs the same
    zero-outside-support guard."""
    from physicskit.rmt.stats.wachter import _wachter_pdf_scalar

    a, b = 2.0, 3.0
    lo, hi = rmt.stats.wachter_support(a, b)
    assert _wachter_pdf_scalar(lo - 0.1, a, b, lo, hi) == 0.0
    assert _wachter_pdf_scalar(hi + 0.1, a, b, lo, hi) == 0.0


def test_wachter_cdf_is_zero_below_and_one_above_support_and_monotonic():
    a, b = 2.0, 3.0
    lo, hi = rmt.stats.wachter_support(a, b)
    x = np.linspace(lo - 0.1, hi + 0.1, 20)
    cdf = rmt.stats.wachter_cdf(x, a, b)
    assert cdf[0] == 0.0
    assert cdf[-1] == 1.0
    assert np.all(np.diff(cdf) >= -1e-12)
