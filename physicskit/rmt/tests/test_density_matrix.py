"""Tests for the random density matrix ensembles (InducedMeasureEnsemble,
HilbertSchmidtEnsemble, BuresHallEnsemble) and the Page-curve entropy
helpers.

Every formula used for validation here (mean purity for both measures,
Page's average-entropy formula, and -- most importantly -- the
Bures-Hall construction itself) was independently checked against exact
theory or independent numerical integration during development, not
assumed from a remembered citation; see the module docstrings in
``physicskit.rmt/ensembles/density_matrix.py`` and
``physicskit.rmt/stats/entanglement.py``.
"""

import numpy as np
import pytest
from cache_utils import cached_sample
from scipy import integrate

import physicskit.rmt as rmt


def _mean_purity(spectrum):
    return np.mean(np.sum(spectrum.eigenvalues**2, axis=1))


# --- structural checks common to all three ensembles ---

ALL_DM_ENSEMBLES = [
    (
        "InducedMeasure",
        lambda n, seed: rmt.ensembles.InducedMeasureEnsemble(n=n, k=n + 2, seed=seed),
    ),
    ("HilbertSchmidt", lambda n, seed: rmt.ensembles.HilbertSchmidtEnsemble(n=n, seed=seed)),
    ("BuresHall", lambda n, seed: rmt.ensembles.BuresHallEnsemble(n=n, seed=seed)),
]


@pytest.mark.parametrize("name,factory", ALL_DM_ENSEMBLES)
def test_eigenvalues_are_valid_probability_distribution(name, factory):
    ens = factory(8, 0)
    spectrum = cached_sample(ens, n_samples=20)
    assert np.all(spectrum.eigenvalues >= -1e-10)
    assert np.all(spectrum.eigenvalues <= 1.0 + 1e-10)
    np.testing.assert_allclose(spectrum.eigenvalues.sum(axis=1), 1.0, atol=1e-8)


@pytest.mark.parametrize("name,factory", ALL_DM_ENSEMBLES)
def test_reproducibility(name, factory):
    ens_a = factory(6, 42)
    ens_b = factory(6, 42)
    spec_a = ens_a.sample(n_samples=3)
    spec_b = ens_b.sample(n_samples=3)
    np.testing.assert_allclose(spec_a.eigenvalues, spec_b.eigenvalues)


def test_induced_measure_rejects_invalid_k():
    with pytest.raises(ValueError):
        rmt.ensembles.InducedMeasureEnsemble(n=5, k=0, seed=0)


def test_hilbert_schmidt_beta_is_two():
    ens = rmt.ensembles.HilbertSchmidtEnsemble(n=5, seed=0)
    assert ens.beta == 2


def test_bures_hall_beta_is_none():
    # Not one of Dyson's classical beta-ensembles -- extra 1/(li+lj)
    # factor in the joint density beyond a pure Vandermonde power.
    ens = rmt.ensembles.BuresHallEnsemble(n=5, seed=0)
    assert ens.beta is None


def test_k_equals_one_gives_pure_states():
    # k=1 forces rank-1 rho: exactly one eigenvalue = 1, rest = 0.
    ens = rmt.ensembles.InducedMeasureEnsemble(n=6, k=1, seed=1)
    spectrum = cached_sample(ens, n_samples=10)
    for row in spectrum.eigenvalues:
        assert np.sum(row > 1e-8) == 1
        assert np.max(row) == pytest.approx(1.0, abs=1e-8)


# --- exact mean-purity formulas (Zyczkowski-Sommers 2001) ---


@pytest.mark.parametrize("n", [2, 3, 5, 10])
def test_hilbert_schmidt_mean_purity_matches_exact_formula(n):
    ens = rmt.ensembles.HilbertSchmidtEnsemble(n=n, seed=2)
    spectrum = cached_sample(ens, n_samples=20000)
    theory = 2.0 * n / (n**2 + 1)
    assert _mean_purity(spectrum) == pytest.approx(theory, rel=0.03)


@pytest.mark.parametrize("n,k", [(3, 3), (3, 6), (3, 2), (4, 10)])
def test_induced_measure_mean_purity_matches_exact_formula(n, k):
    ens = rmt.ensembles.InducedMeasureEnsemble(n=n, k=k, seed=3)
    spectrum = cached_sample(ens, n_samples=20000)
    theory = (n + k) / (n * k + 1)
    assert _mean_purity(spectrum) == pytest.approx(theory, rel=0.03)


# --- Bures-Hall: verified directly against its own joint eigenvalue
# density (not merely against a remembered matrix-model recipe) ---


def test_bures_hall_matches_exact_n2_marginal_density():
    # For n=2, the exact Bures-Hall joint density on the simplex
    # lambda1+lambda2=1 gives a closed-form 1-D density for the smaller
    # eigenvalue: f(x) ~ (2x-1)^2 / sqrt(x(1-x)) on (0, 1), restricted
    # to (0, 0.5] and doubled (by exchange symmetry) for the SMALLER
    # eigenvalue's marginal CDF.
    def unnorm_pdf(x):
        return (2 * x - 1) ** 2 / np.sqrt(x * (1 - x))

    z, _ = integrate.quad(unnorm_pdf, 0, 1)

    def cdf_full(x):
        val, _ = integrate.quad(unnorm_pdf, 0, x)
        return val / z

    def smaller_eig_cdf(x):
        return np.clip(2.0 * np.vectorize(cdf_full)(np.clip(x, 0, 0.5)), 0.0, 1.0)

    ens = rmt.ensembles.BuresHallEnsemble(n=2, seed=4)
    spectrum = cached_sample(ens, n_samples=50000)
    smaller_eigs = spectrum.eigenvalues[:, 0]  # eigvalsh ascending -> smaller first

    from scipy.stats import kstest

    ks = kstest(smaller_eigs, smaller_eig_cdf)
    assert ks.statistic < 0.02


def test_bures_hall_matches_exact_n3_mean_purity_via_joint_density_integration():
    # For n=3, integrate the exact joint density over the 2-D simplex
    # directly (independent of the matrix construction) to get the
    # theoretical mean purity, and compare to Monte Carlo.
    def unnorm_density(l1, l2):
        l3 = 1.0 - l1 - l2
        if l3 <= 0 or l1 <= 0 or l2 <= 0:
            return 0.0
        vand = (l1 - l2) ** 2 * (l1 - l3) ** 2 * (l2 - l3) ** 2
        denom = (l1 + l2) * (l1 + l3) * (l2 + l3)
        weight = (l1 * l2 * l3) ** (-0.5)
        return vand / denom * weight

    def integrand_z(l2, l1):
        return unnorm_density(l1, l2)

    def integrand_purity(l2, l1):
        l3 = 1.0 - l1 - l2
        return unnorm_density(l1, l2) * (l1**2 + l2**2 + l3**2)

    z, _ = integrate.dblquad(integrand_z, 0, 1, lambda l1: 0, lambda l1: 1 - l1)
    p, _ = integrate.dblquad(integrand_purity, 0, 1, lambda l1: 0, lambda l1: 1 - l1)
    theory_mean_purity = p / z

    ens = rmt.ensembles.BuresHallEnsemble(n=3, seed=5)
    spectrum = cached_sample(ens, n_samples=40000)
    assert _mean_purity(spectrum) == pytest.approx(theory_mean_purity, rel=0.02)


# --- Page curve (bipartite / entanglement) ---


def test_von_neumann_entropy_handles_zero_eigenvalues():
    assert rmt.stats.von_neumann_entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert rmt.stats.von_neumann_entropy([0.5, 0.5]) == pytest.approx(np.log(2.0))


@pytest.mark.parametrize("n,k", [(2, 2), (2, 8), (3, 3), (3, 10)])
def test_page_curve_matches_induced_measure_ensemble(n, k):
    ens = rmt.ensembles.InducedMeasureEnsemble(n=n, k=k, seed=6)
    spectrum = cached_sample(ens, n_samples=15000)
    empirical = np.mean([rmt.stats.von_neumann_entropy(row) for row in spectrum.eigenvalues])
    theory = rmt.stats.page_curve_average_entropy(n, k)
    assert empirical == pytest.approx(theory, rel=0.02)
