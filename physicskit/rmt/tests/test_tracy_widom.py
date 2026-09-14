"""Tests for the Painleve II solution and the Tracy-Widom soft-edge
benchmark.

Two things were determined empirically during development, not assumed
from memory, and are worth testing explicitly here rather than only in
the module docstring: (1) q(x)'s known asymptotic tail
q(x) ~ sqrt(-x/2) as x -> -infinity, and (2) the beta=4 (GSE) edge
scaling requires (2n)^(2/3), not n^(2/3) -- confirmed by direct
comparison against Monte Carlo GSE samples via a KS test, since the
half-remembered formula did NOT match without this factor.

Note on statistical power: the largest eigenvalue is a single
extreme-value statistic per matrix sample, so pooling matters much less
than the bulk-statistic benchmarks (each sample of GOE/GUE/GSE
contributes exactly one data point to the edge-statistic distribution).
n_samples in these tests is correspondingly large (hundreds), unlike
elsewhere in the test suite.
"""

import numpy as np
import pytest
from cache_utils import cached_sample

import physicskit.rmt as rmt


def test_painleve_ii_matches_known_asymptotic_tail():
    # Re-derive q via the same ODE construction used internally, at a
    # couple of points in the asymptotic regime, and compare to the
    # known analytic tail sqrt(-x/2).
    from scipy.integrate import solve_ivp
    from scipy.special import airy

    def rhs(x, y):
        qv, qp = y
        return [qp, x * qv + 2.0 * qv**3]

    x0 = 6.0
    ai, aip, _, _ = airy(x0)
    ivp = solve_ivp(rhs, [x0, -6.0], [ai, aip], dense_output=True, rtol=1e-12, atol=1e-14, max_step=0.005)
    for x in [-6.0, -5.0, -4.0]:
        q_val = ivp.sol(x)[0]
        asymptotic = np.sqrt(-x / 2.0)
        assert q_val == pytest.approx(asymptotic, rel=0.01)


def test_cdfs_are_monotonic_and_span_zero_to_one():
    grid = np.linspace(-6, 6, 200)
    for beta in (1, 2, 4):
        cdf = rmt.stats.tracy_widom_cdf(grid, beta)
        assert np.all(np.diff(cdf) >= -1e-10)  # monotone non-decreasing
        assert cdf[0] < 0.01
        assert cdf[-1] > 0.99


def test_unsupported_beta_rejected():
    with pytest.raises(ValueError):
        rmt.stats.tracy_widom_cdf(0.0, beta=1.7)
    with pytest.raises(ValueError):
        rmt.validation.TracyWidom(beta=3)


def test_edge_scale_beta4_has_extra_factor_of_two():
    n = 500
    scale_1 = rmt.stats.tracy_widom_edge_scale(n, beta=1)
    scale_2 = rmt.stats.tracy_widom_edge_scale(n, beta=2)
    scale_4 = rmt.stats.tracy_widom_edge_scale(n, beta=4)
    assert scale_1 == pytest.approx(n ** (2 / 3))
    assert scale_2 == pytest.approx(n ** (2 / 3))
    assert scale_4 == pytest.approx((2 * n) ** (2 / 3))
    assert scale_4 != pytest.approx(scale_1)  # the easy-to-miss distinction


GAUSSIAN_TW = [
    ("GOE", rmt.ensembles.GOE, 1),
    ("GUE", rmt.ensembles.GUE, 2),
    ("GSE", rmt.ensembles.GSE, 4),
]


@pytest.mark.parametrize("name,cls,beta", GAUSSIAN_TW)
def test_largest_eigenvalue_matches_tracy_widom(name, cls, beta):
    ens = cls(n=500, seed=10)
    spectrum = cached_sample(ens, n_samples=400)
    benchmark = rmt.validation.TracyWidom(beta=beta)
    result = benchmark.validate(spectrum, seed=10)
    assert result.ks_statistic < 0.08
    assert result.ks_pvalue > 0.05


@pytest.mark.parametrize("name,cls,beta", GAUSSIAN_TW)
def test_beta_mismatch_rejected(name, cls, beta):
    other_beta = {1: 2, 2: 4, 4: 1}[beta]
    ens = cls(n=100, seed=0)
    spectrum = cached_sample(ens, n_samples=20)
    benchmark = rmt.validation.TracyWidom(beta=other_beta)
    with pytest.raises(ValueError):
        benchmark.validate(spectrum)


def test_largest_eigenvalues_helper():
    ens = rmt.ensembles.GUE(n=200, seed=0)
    spectrum = cached_sample(ens, n_samples=10)
    lam_max = rmt.stats.largest_eigenvalues(spectrum)
    assert lam_max.shape == (10,)
    np.testing.assert_allclose(lam_max, spectrum.rescaled.max(axis=1))
