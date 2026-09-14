"""Tests for the polynomial/biorthogonal ensemble (products of Ginibre
matrices).

L=1 is checked directly against the already-validated
``MarchenkoPastur`` benchmark (reused, not reimplemented -- it's the
exact same theoretical object). L >= 2 has no simple closed-form
density implemented here, so it is validated against the one thing that
*is* known in exact closed form for general L: the Fuss-Catalan moments
(Penson-Zyczkowski 2011), themselves checked against Monte Carlo before
being trusted (see ``test_fuss_catalan_moment_matches_monte_carlo``).
"""

import numpy as np
import pytest
from cache_utils import cached_sample

import physicskit.rmt as rmt
from physicskit.rmt.ensembles.polynomial import fuss_catalan_moment


def test_default_num_factors_is_two():
    ens = rmt.ensembles.PolynomialEnsemble(n=30, seed=0)
    assert ens.num_factors == 2


def test_eigenvalues_are_nonnegative():
    ens = rmt.ensembles.PolynomialEnsemble(n=50, num_factors=3, seed=1)
    spectrum = cached_sample(ens, n_samples=5)
    assert np.all(spectrum.eigenvalues >= -1e-8)


def test_reproducibility():
    ens_a = rmt.ensembles.PolynomialEnsemble(n=40, num_factors=2, seed=42)
    ens_b = rmt.ensembles.PolynomialEnsemble(n=40, num_factors=2, seed=42)
    spec_a = ens_a.sample(n_samples=3)
    spec_b = ens_b.sample(n_samples=3)
    np.testing.assert_allclose(spec_a.eigenvalues, spec_b.eigenvalues)


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        rmt.ensembles.PolynomialEnsemble(n=10, num_factors=0, seed=0)
    with pytest.raises(ValueError):
        rmt.ensembles.PolynomialEnsemble(n=10, beta=4, seed=0)


@pytest.mark.parametrize("beta", [1, 2])
def test_num_factors_one_matches_marchenko_pastur_exactly(beta):
    # L=1 is, EXACTLY (not approximately), the classical Wishart/LUE-LOE
    # ensemble -- reusing the already-validated MarchenkoPastur
    # benchmark directly, rather than a separate implementation, is
    # itself part of the point (see module docstring).
    n = 2000
    ens = rmt.ensembles.PolynomialEnsemble(n=n, num_factors=1, beta=beta, seed=2)
    spectrum = cached_sample(ens, n_samples=15)
    benchmark = rmt.validation.MarchenkoPastur(gamma=1.0)
    result = benchmark.validate(spectrum, seed=2)
    assert result.ks_statistic < 0.02


def test_fuss_catalan_moment_matches_monte_carlo():
    # Verify the exact moment formula itself (not assumed from the
    # literature citation) via a large, independent Monte Carlo
    # generalized-Ginibre-product simulation, for L=1..4.
    rng = np.random.default_rng(3)
    for num_factors in [1, 2, 3, 4]:
        n = 400
        x = np.eye(n, dtype=complex)
        for _ in range(num_factors):
            factor = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2.0)
            x = x @ factor
        eigs = np.linalg.eigvalsh(x.conj().T @ x) / n**num_factors
        for k in [1, 2]:
            empirical = np.mean(eigs**k)
            theory = fuss_catalan_moment(k, num_factors)
            assert empirical == pytest.approx(theory, rel=0.1)


@pytest.mark.parametrize("num_factors", [1, 2, 3, 4])
def test_ensemble_moments_match_fuss_catalan(num_factors):
    n = 300
    ens = rmt.ensembles.PolynomialEnsemble(n=n, num_factors=num_factors, beta=2, seed=4)
    spectrum = cached_sample(ens, n_samples=10)
    eigs = spectrum.rescaled.ravel()
    for k in [1, 2]:
        empirical = np.mean(eigs**k)
        theory = fuss_catalan_moment(k, num_factors)
        assert empirical == pytest.approx(theory, rel=0.1)


@pytest.mark.parametrize("num_factors", [1, 2, 3])
def test_spectrum_stays_near_fuss_catalan_support_edge(num_factors):
    # Largest rescaled eigenvalue should approach, but not meaningfully
    # exceed, the exact Fuss-Catalan support edge (L+1)**(L+1) / L**L.
    n = 400
    ens = rmt.ensembles.PolynomialEnsemble(n=n, num_factors=num_factors, beta=2, seed=5)
    spectrum = cached_sample(ens, n_samples=5)
    edge = (num_factors + 1) ** (num_factors + 1) / num_factors**num_factors
    max_eig = spectrum.rescaled.max()
    assert max_eig < edge * 1.1
    assert max_eig > edge * 0.7
