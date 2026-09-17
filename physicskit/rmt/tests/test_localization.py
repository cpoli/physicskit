"""Tests for eigenvector-based localization statistics (inverse
participation ratio) and the ``sample(return_eigenvectors=True)`` API.

``ipr_theory``'s Dirichlet-moment formula was verified during
development against direct Haar-random-vector Monte Carlo at beta=1, 2,
4 before being trusted (see ``physicskit.rmt.stats.localization``). The
tridiagonal-model ensembles (GOE/GUE/GSE via the fast
Dumitriu-Edelman sampler) do NOT get eigenvector support from that
formula for free: their raw eigenvectors were checked during development
and found to be more localized than a true Haar vector (IPR ~2-3x too
large, growing with n) -- only a genuinely dense diagonalization gives
Haar-distributed eigenvectors, which is why
``HermiteBetaEnsemble._sample_eigenvalues_and_vectors`` uses a separate
dense construction for beta in (1, 2, 4). GSE (beta=4) eigenvectors are
naturally 2n-dimensional complex vectors (n quaternionic "sites"), so
they need the dedicated ``inverse_participation_ratio_quaternionic``
(folding pairs of components first) rather than the plain
``inverse_participation_ratio`` -- verified during development that the
folded IPR matches ``ipr_theory(n, beta=4)`` to four significant figures.
"""

import numpy as np
import pytest

import physicskit.rmt as rmt


def test_ensemble_without_eigenvector_support_raises():
    ens = rmt.ensembles.LOE(n=20, m=25, seed=0)
    with pytest.raises(NotImplementedError):
        ens.sample(return_eigenvectors=True)


def test_gse_eigenvectors_are_normalized_and_shaped_2n_by_n():
    n = 20
    ens = rmt.ensembles.GSE(n=n, seed=0)
    spectrum = ens.sample(n_samples=2, return_eigenvectors=True)
    assert spectrum.eigenvectors is not None
    assert spectrum.eigenvalues.shape == (2, n)
    assert spectrum.eigenvectors.shape == (2, 2 * n, n)
    norms = np.sum(np.abs(spectrum.eigenvectors) ** 2, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-8)


def test_gse_quaternionic_ipr_matches_haar_theory():
    n = 200
    ens = rmt.ensembles.GSE(n=n, seed=1)
    spectrum = ens.sample(n_samples=20, return_eigenvectors=True)
    ipr = rmt.stats.inverse_participation_ratio_quaternionic(spectrum.eigenvectors)
    theory = rmt.stats.ipr_theory(n, beta=4)
    assert ipr.mean() == pytest.approx(theory, rel=0.15)


def test_naive_ipr_on_gse_eigenvectors_does_not_match_theory():
    # Documents the pitfall: calling the plain (non-quaternionic) IPR
    # function directly on GSE eigenvectors treats all 2n complex
    # components as independent sites, which does NOT match
    # ipr_theory(n, beta=4) -- inverse_participation_ratio_quaternionic
    # must be used instead (see module docstring).
    n = 200
    ens = rmt.ensembles.GSE(n=n, seed=1)
    spectrum = ens.sample(n_samples=20, return_eigenvectors=True)
    naive_ipr = rmt.stats.inverse_participation_ratio(spectrum.eigenvectors)
    theory = rmt.stats.ipr_theory(n, beta=4)
    assert naive_ipr.mean() != pytest.approx(theory, rel=0.15)


@pytest.mark.parametrize("cls", [rmt.ensembles.GOE, rmt.ensembles.GUE])
def test_dense_gaussian_ensemble_eigenvectors_are_normalized(cls):
    ens = cls(n=30, seed=0)
    spectrum = ens.sample(n_samples=2, return_eigenvectors=True)
    assert spectrum.eigenvectors is not None
    assert spectrum.eigenvectors.shape == (2, 30, 30)
    norms = np.sum(np.abs(spectrum.eigenvectors) ** 2, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-8)


@pytest.mark.parametrize("cls,beta", [(rmt.ensembles.GOE, 1), (rmt.ensembles.GUE, 2)])
def test_dense_gaussian_ensemble_ipr_matches_haar_theory(cls, beta):
    n = 250
    ens = cls(n=n, seed=1)
    spectrum = ens.sample(n_samples=30, return_eigenvectors=True)
    ipr = rmt.stats.inverse_participation_ratio(spectrum.eigenvectors)
    theory = rmt.stats.ipr_theory(n, beta)
    assert ipr.mean() == pytest.approx(theory, rel=0.15)


def test_ipr_theory_matches_direct_haar_vector_simulation():
    # Independent check of the closed-form formula itself, bypassing any
    # ensemble machinery: a Haar-random (Gaussian, then normalized) real
    # vector's mean IPR should match ipr_theory(n, beta=1) directly.
    rng = np.random.default_rng(7)
    n = 100
    x = rng.standard_normal((20_000, n))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    empirical = np.mean(np.sum(x**4, axis=1))
    assert empirical == pytest.approx(rmt.stats.ipr_theory(n, beta=1), rel=0.05)


def test_participation_ratio_is_reciprocal_of_ipr():
    rng = np.random.default_rng(3)
    vecs = rng.standard_normal((5, 10, 4))
    ipr = rmt.stats.inverse_participation_ratio(vecs)
    pr = rmt.stats.participation_ratio(vecs)
    np.testing.assert_allclose(pr, 1.0 / ipr)


def test_pbrm_localization_transition_via_ipr():
    # The defining physics of PowerLawBandedEnsemble (see its module
    # docstring): small b (narrow band) -> localized eigenvectors (IPR
    # much larger than the delocalized/GOE-like baseline); large b
    # (effectively unbanded) -> delocalized eigenvectors, IPR approaching
    # the Haar-vector theory value.
    n = 300
    localized = rmt.ensembles.PowerLawBandedEnsemble(n=n, b=0.5, alpha=2.0, seed=10)
    delocalized = rmt.ensembles.PowerLawBandedEnsemble(n=n, b=50.0, alpha=2.0, seed=11)

    # cache_utils.cached_sample never requests eigenvectors, so sample directly.
    spec_localized = localized.sample(n_samples=10, return_eigenvectors=True)
    spec_delocalized = delocalized.sample(n_samples=10, return_eigenvectors=True)

    ipr_localized = rmt.stats.inverse_participation_ratio(spec_localized.eigenvectors).mean()
    ipr_delocalized = rmt.stats.inverse_participation_ratio(spec_delocalized.eigenvectors).mean()
    theory_delocalized = rmt.stats.ipr_theory(n, beta=1)

    assert ipr_localized > 10 * ipr_delocalized
    assert ipr_delocalized == pytest.approx(theory_delocalized, rel=0.3)


def test_eigenvectors_absent_by_default():
    ens = rmt.ensembles.GOE(n=20, seed=0)
    spectrum = ens.sample(n_samples=2)
    assert spectrum.eigenvectors is None


# --- multifractal dimension spectrum ---


def test_generalized_ipr_matches_plain_ipr_at_q_2():
    rng = np.random.default_rng(4)
    vecs = rng.standard_normal((5, 10, 4))
    np.testing.assert_allclose(rmt.stats.generalized_ipr(vecs, q=2.0), rmt.stats.inverse_participation_ratio(vecs))


def test_generalized_ipr_rejects_q_equals_1():
    rng = np.random.default_rng(4)
    vecs = rng.standard_normal((5, 4))
    with pytest.raises(ValueError):
        rmt.stats.generalized_ipr(vecs, q=1.0)


def test_multifractal_dimension_is_one_for_delocalized_pbrm():
    # Large, fixed band-width b -> effectively GOE-like/delocalized, so
    # D_2 should be close to 1 (E[I_2] ~ 1/n exactly).
    def factory(n, seed):
        return rmt.ensembles.PowerLawBandedEnsemble(n=n, b=1000.0, alpha=2.0, seed=seed)

    d_2, r_squared = rmt.stats.multifractal_dimension(factory, n_values=[100, 200, 400, 800], q=2.0, n_samples=10, seed=0)
    assert d_2 == pytest.approx(1.0, abs=0.1)
    assert r_squared > 0.99


def test_multifractal_dimension_is_zero_for_localized_pbrm():
    # Small, fixed band-width b -> strongly localized, so E[I_2] stays
    # roughly n-independent, giving D_2 close to 0.
    def factory(n, seed):
        return rmt.ensembles.PowerLawBandedEnsemble(n=n, b=0.3, alpha=2.0, seed=seed)

    d_2, _ = rmt.stats.multifractal_dimension(factory, n_values=[100, 200, 400, 800], q=2.0, n_samples=10, seed=1)
    assert d_2 == pytest.approx(0.0, abs=0.2)


# --- f(alpha) singularity spectrum ---


def test_mass_exponent_is_exactly_zero_at_q_1():
    def factory(n, seed):
        return rmt.ensembles.PowerLawBandedEnsemble(n=n, b=1.0, alpha=1.0, seed=seed)

    tau_1, r_squared = rmt.stats.mass_exponent(factory, n_values=[100, 200, 400], q=1.0, n_samples=5, seed=0)
    assert tau_1 == 0.0
    assert r_squared == 1.0


def test_multifractal_dimension_rejects_q_1():
    def factory(n, seed):
        return rmt.ensembles.PowerLawBandedEnsemble(n=n, b=1.0, alpha=1.0, seed=seed)

    with pytest.raises(ValueError):
        rmt.stats.multifractal_dimension(factory, n_values=[100, 200], q=1.0, n_samples=5, seed=0)


@pytest.mark.slow
def test_mass_exponent_matches_multifractal_dimension_relation():
    def factory(n, seed):
        return rmt.ensembles.PowerLawBandedEnsemble(n=n, b=1000.0, alpha=2.0, seed=seed)

    n_values = [50, 100, 200, 400]
    tau_2, _ = rmt.stats.mass_exponent(factory, n_values, q=2.0, n_samples=10, seed=0)
    d_2, _ = rmt.stats.multifractal_dimension(factory, n_values, q=2.0, n_samples=10, seed=0)
    assert tau_2 / (2.0 - 1.0) == pytest.approx(d_2)


@pytest.mark.slow
def test_singularity_spectrum_is_a_single_point_for_delocalized_system():
    # Fully delocalized (D_q=1 for all q) means tau(q)=q-1 exactly, so
    # alpha(q)=1 and f(alpha)=1 for every q -- the degenerate
    # "monofractal" limit of the singularity spectrum.
    def factory(n, seed):
        return rmt.ensembles.PowerLawBandedEnsemble(n=n, b=1000.0, alpha=2.0, seed=seed)

    alpha, f_alpha = rmt.stats.singularity_spectrum(
        factory,
        n_values=[100, 200, 400, 800],
        q_values=np.array([0.5, 1.0, 1.5, 2.0]),
        n_samples=10,
        seed=0,
    )
    np.testing.assert_allclose(alpha, 1.0, atol=0.15)
    np.testing.assert_allclose(f_alpha, 1.0, atol=0.15)


@pytest.mark.slow
def test_singularity_spectrum_is_positive_and_concave_at_small_q_for_critical_pbrm():
    # At PBRM's multifractal critical point, the moderate-|q| portion of
    # f(alpha) should be positive (a genuine fractal dimension) and
    # concave (f(alpha) decreasing in |q-1|) -- see module/function
    # docstring for why larger |q| is not asserted here (harder
    # convergence, possible legitimate negative-dimension branch).
    def factory(n, seed):
        return rmt.ensembles.PowerLawBandedEnsemble(n=n, b=1.0, alpha=1.0, seed=seed)

    q_values = np.array([0.5, 1.0, 1.5])
    _, f_alpha = rmt.stats.singularity_spectrum(factory, n_values=[75, 150, 300, 600], q_values=q_values, n_samples=12, seed=2)
    assert np.all(f_alpha > 0.0)
    # Standard multifractal shape: f(alpha(q)) decreases monotonically as
    # q increases (alpha(q) itself decreases with q, with the global max
    # of f at q=0 -- not asserted here since q=0 needs even larger n for
    # reliable convergence); over this q range it should be decreasing.
    assert f_alpha[0] > f_alpha[1] > f_alpha[2]
