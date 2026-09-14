"""Tests for PTSymmetricEnsemble and the real-eigenvalue-fraction
statistic.

The pseudo-Hermiticity relation P H P = H^dagger (P the fixed +-1
signature matrix) was verified numerically to machine precision during
development for both beta=1 and beta=2 before this construction was
trusted -- see module docstring in
``physicskit.rmt/ensembles/pt_symmetric.py``.
"""

import numpy as np
import pytest
from cache_utils import cached_sample

import physicskit.rmt as rmt


@pytest.mark.parametrize("beta", [1, 2])
def test_pseudo_hermiticity_holds_exactly(beta):
    p, q = 8, 6
    ens = rmt.ensembles.PTSymmetricEnsemble(p=p, q=q, g=0.7, beta=beta, seed=0)
    h = ens._sample_matrix(np.random.default_rng(0))
    signature = np.diag(np.concatenate([np.ones(p), -np.ones(q)]))
    residual = signature @ h @ signature - h.conj().T
    assert np.abs(residual).max() < 1e-10


@pytest.mark.parametrize("beta", [1, 2])
def test_zero_coupling_gives_fully_real_spectrum(beta):
    ens = rmt.ensembles.PTSymmetricEnsemble(p=10, q=10, g=0.0, beta=beta, seed=1)
    spectrum = cached_sample(ens, n_samples=10)
    frac = rmt.stats.real_eigenvalue_fraction(spectrum.eigenvalues)
    np.testing.assert_allclose(frac, 1.0, atol=1e-6)


@pytest.mark.parametrize("beta", [1, 2])
def test_real_fraction_decreases_as_coupling_grows(beta):
    # The PT-symmetry-breaking transition: increasing coupling g should
    # drive the spectrum from fully real toward mostly complex.
    fracs = []
    for g in [0.0, 0.3, 1.0, 3.0]:
        ens = rmt.ensembles.PTSymmetricEnsemble(p=15, q=15, g=g, beta=beta, seed=2)
        spectrum = cached_sample(ens, n_samples=20)
        fracs.append(rmt.stats.real_eigenvalue_fraction(spectrum.eigenvalues).mean())
    assert fracs == sorted(fracs, reverse=True)
    assert fracs[0] == pytest.approx(1.0, abs=1e-6)
    assert fracs[-1] < 0.3


def test_eigenvalues_are_real_or_conjugate_pairs():
    # Pseudo-Hermiticity forces the eigenvalue SET to be closed under
    # complex conjugation: every non-real eigenvalue has a partner at
    # its complex conjugate (matched here via nearest-neighbor distance,
    # not a naive sort, since floating-point noise can break exact ties
    # between a conjugate pair's real parts).
    ens = rmt.ensembles.PTSymmetricEnsemble(p=10, q=10, g=1.0, beta=2, seed=3)
    spectrum = ens.sample(n_samples=1)
    eigs = spectrum.eigenvalues[0]
    non_real = eigs[np.abs(eigs.imag) > 1e-6]
    for lam in non_real:
        nearest_dist = np.min(np.abs(eigs - np.conj(lam)))
        assert nearest_dist < 1e-8


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        rmt.ensembles.PTSymmetricEnsemble(p=5, q=5, g=-1.0, seed=0)
    with pytest.raises(ValueError):
        rmt.ensembles.PTSymmetricEnsemble(p=5, q=5, g=1.0, beta=4, seed=0)


def test_reproducibility():
    ens_a = rmt.ensembles.PTSymmetricEnsemble(p=6, q=6, g=0.5, seed=42)
    ens_b = rmt.ensembles.PTSymmetricEnsemble(p=6, q=6, g=0.5, seed=42)
    spec_a = ens_a.sample(n_samples=3)
    spec_b = ens_b.sample(n_samples=3)
    np.testing.assert_allclose(spec_a.eigenvalues, spec_b.eigenvalues)


# --- exceptional-point detection ---


def test_find_exceptional_points_finds_real_count_drops():
    eps = rmt.stats.find_exceptional_points(p=6, q=6, beta=2, seed=0, g_max=3.0, n_grid=3000)
    assert len(eps) > 0
    assert np.all(eps > 0.0)
    assert np.all(eps < 3.0)


def test_eigenvector_condition_number_spikes_near_exceptional_point():
    # The defining structural signature distinguishing a genuine
    # exceptional point (eigenvector coalescence) from an incidental
    # level crossing (eigenvectors stay independent): the right-
    # eigenvector matrix's condition number should grow sharply as g
    # approaches a detected transition.
    eps = rmt.stats.find_exceptional_points(p=6, q=6, beta=2, seed=0, g_max=3.0, n_grid=3000)
    g_star = eps[0]
    cond_far = rmt.stats.eigenvector_condition_number(6, 6, 2, 0, g=max(g_star - 0.01, 0.0))
    cond_near = rmt.stats.eigenvector_condition_number(6, 6, 2, 0, g=g_star + 5e-4)
    assert cond_near > 2 * cond_far


def test_mean_exceptional_point_count_grows_with_system_size():
    # Ensemble-level statistic: mean EP count over many independent
    # realizations should grow with system size, roughly linearly in
    # min(p, q) -- verified during development (p=q=4,8,12 gave ~4.2,
    # 8.7, 13.1 over g in [0, 2]).
    mean_small, err_small = rmt.stats.mean_exceptional_point_count(p=4, q=4, beta=2, g_max=2.0, n_realizations=15, seed=0)
    mean_large, err_large = rmt.stats.mean_exceptional_point_count(p=12, q=12, beta=2, g_max=2.0, n_realizations=15, seed=1)
    assert mean_small > 0
    assert mean_large > 2 * mean_small
    assert err_small >= 0 and err_large >= 0


def test_fixed_realization_reconstruction_only_rescales_coupling_block():
    # The trick find_exceptional_points/eigenvector_condition_number rely
    # on: rebuilding the ensemble with a different g but the same seed
    # reproduces the same A, D blocks, only rescaling the coupling block.
    from physicskit.rmt.stats.pt_symmetric import _fixed_realization_matrix

    p, q = 5, 5
    h1 = _fixed_realization_matrix(p, q, 2, 42, g=0.1)
    h2 = _fixed_realization_matrix(p, q, 2, 42, g=0.5)
    np.testing.assert_allclose(h1[:p, :p], h2[:p, :p])
    np.testing.assert_allclose(h1[p:, p:], h2[p:, p:])
    np.testing.assert_allclose(h1[:p, p:] / 0.1, h2[:p, p:] / 0.5)


# --- intermediate (semi-Poisson) spacing statistics ---


def test_semi_poisson_is_normalized_with_unit_mean_spacing():
    from scipy.integrate import quad

    norm, _ = quad(rmt.stats.semi_poisson_pdf, 0, np.inf)
    mean, _ = quad(lambda s: s * rmt.stats.semi_poisson_pdf(np.array([s]))[0], 0, np.inf)
    assert norm == pytest.approx(1.0, abs=1e-8)
    assert mean == pytest.approx(1.0, abs=1e-8)


def test_semi_poisson_cdf_is_derivative_consistent_with_pdf():
    s = np.linspace(0.01, 5.0, 50)
    h = 1e-6
    finite_diff = (rmt.stats.semi_poisson_cdf(s + h) - rmt.stats.semi_poisson_cdf(s - h)) / (2 * h)
    np.testing.assert_allclose(finite_diff, rmt.stats.semi_poisson_pdf(s), atol=1e-4)


def test_real_axis_spacings_matches_semi_poisson_better_than_poisson_or_goe():
    # Verified during development: at small-to-moderate coupling g, the
    # surviving real eigenvalues' spacing statistic is closest to
    # semi-Poisson, not pure Poisson or pure GOE -- the "intermediate
    # statistics" signature the design plan calls for.
    ens = rmt.ensembles.PTSymmetricEnsemble(p=25, q=25, g=0.3, beta=2, seed=0)
    spectrum = cached_sample(ens, n_samples=200)
    spacings = rmt.stats.real_axis_spacings(spectrum.eigenvalues)

    from scipy.stats import kstest

    ks_semi_poisson = kstest(spacings, rmt.stats.semi_poisson_cdf).statistic
    ks_poisson = kstest(spacings, lambda s: 1.0 - np.exp(-s)).statistic
    ks_goe = kstest(spacings, lambda s: rmt.stats.wigner_surmise_cdf(s, beta=1)).statistic

    assert ks_semi_poisson < ks_poisson
    assert ks_semi_poisson < ks_goe


def test_real_axis_spacings_skips_samples_with_too_few_real_eigenvalues():
    # All-complex sample (no real eigenvalues at all) should not crash
    # and should be skipped, not contribute spurious spacings.
    eigenvalues = np.array(
        [
            [1.0 + 2.0j, 1.0 - 2.0j, 3.0 + 1.0j, 3.0 - 1.0j],
            [0.0 + 0j, 1.0 + 0j, 2.0 + 0j, 3.0 + 0j],
        ]
    )
    spacings = rmt.stats.real_axis_spacings(eigenvalues)
    np.testing.assert_allclose(spacings, [1.0, 1.0, 1.0])
