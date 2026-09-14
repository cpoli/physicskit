"""Tests for COE, CUE, CSE and the sine kernel benchmark.

Circular ensembles are structurally different from the Gaussian/Wishart
families: COE is NOT "eigenvalues of a Haar-random orthogonal matrix"
and CSE is NOT "eigenvalues of a Haar-random symplectic matrix" (see the
warning in the ``physicskit.rmt.ensembles.circular`` module docstring for why).
These tests check both the structural correctness of each construction
(unitarity, symmetry/self-duality, exact degeneracy for CSE) and the
resulting spacing statistics against the already-validated Wigner
surmise -- an independent cross-family check, since spacing statistics
are universal in beta across the Gaussian and circular families.
"""

import numpy as np
import pytest
from cache_utils import cached_sample

import physicskit.rmt as rmt
from physicskit.rmt.utils.haar import haar_unitary, symplectic_form

CIRCULAR = [
    ("COE", rmt.ensembles.COE, 1),
    ("CUE", rmt.ensembles.CUE, 2),
    ("CSE", rmt.ensembles.CSE, 4),
]


def test_haar_unitary_is_unitary():
    rng = np.random.default_rng(0)
    u = haar_unitary(150, rng)
    identity_error = np.abs(u.conj().T @ u - np.eye(150)).max()
    assert identity_error < 1e-9


def test_haar_unitary_eigenvalues_on_unit_circle():
    rng = np.random.default_rng(1)
    u = haar_unitary(150, rng)
    eigs = np.linalg.eigvals(u)
    assert np.abs(np.abs(eigs) - 1.0).max() < 1e-9


@pytest.mark.parametrize("name,cls,beta", CIRCULAR)
def test_beta_is_correct(name, cls, beta):
    ens = cls(n=30, seed=0)
    assert ens.beta == beta


@pytest.mark.parametrize("name,cls,beta", CIRCULAR)
def test_phases_in_valid_range(name, cls, beta):
    ens = cls(n=100, seed=2)
    spectrum = cached_sample(ens, n_samples=5)
    assert np.all(spectrum.eigenvalues >= 0.0)
    assert np.all(spectrum.eigenvalues < 2 * np.pi + 1e-9)


@pytest.mark.parametrize("name,cls,beta", CIRCULAR)
def test_reproducibility(name, cls, beta):
    ens_a = cls(n=80, seed=42)
    ens_b = cls(n=80, seed=42)
    spec_a = ens_a.sample(n_samples=3)
    spec_b = ens_b.sample(n_samples=3)
    np.testing.assert_allclose(spec_a.eigenvalues, spec_b.eigenvalues)


@pytest.mark.parametrize("name,cls,beta", CIRCULAR)
def test_mean_spacing_is_exactly_unity(name, cls, beta):
    # Unlike Gaussian/Wishart, circular ensembles have EXACT rotational
    # invariance at any finite n, so mean spacing should be unity to
    # floating-point precision, not just asymptotically.
    ens = cls(n=200, seed=3)
    spectrum = cached_sample(ens, n_samples=15)
    spacings = rmt.stats.circular_spacings(spectrum)
    assert spacings.mean() == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("name,cls,beta", CIRCULAR)
def test_spacing_matches_wigner_surmise_at_matching_beta(name, cls, beta):
    ens = cls(n=250, seed=4)
    spectrum = cached_sample(ens, n_samples=25)
    spacings = rmt.stats.circular_spacings(spectrum)
    benchmark = rmt.validation.WignerSurmise(beta=beta)
    result = benchmark.validate(spacings, seed=4)
    assert result.ks_statistic < 0.02


def test_coe_matrix_is_symmetric():
    rng = np.random.default_rng(5)
    v = haar_unitary(100, rng)
    u = v.T @ v
    assert np.abs(u - u.T).max() < 1e-9
    identity_error = np.abs(u.conj().T @ u - np.eye(100)).max()
    assert identity_error < 1e-9  # still unitary


def test_cse_dual_matrix_is_self_dual_unitary():
    rng = np.random.default_rng(6)
    n2 = 60
    v = haar_unitary(n2, rng)
    z = symplectic_form(n2)
    v_dual = (-z) @ v.T @ z
    u = v_dual @ v
    identity_error = np.abs(u.conj().T @ u - np.eye(n2)).max()
    assert identity_error < 1e-9
    # Self-duality: U^R == U, where U^R = Z U^T Z^{-1}
    u_dual = (-z) @ u.T @ z
    assert np.abs(u_dual - u).max() < 1e-8


def test_cse_eigenvalues_are_doubly_degenerate():
    # inspect the raw (pre-dedup) construction directly
    rng = np.random.default_rng(7)
    from physicskit.rmt.utils.haar import haar_unitary, symplectic_form

    n2 = 80
    v = haar_unitary(n2, rng)
    z = symplectic_form(n2)
    v_dual = (-z) @ v.T @ z
    u = v_dual @ v
    eigs = np.linalg.eigvals(u)
    theta = np.sort(np.angle(eigs) % (2 * np.pi))
    pair_gaps = np.diff(theta)[0::2]  # gaps WITHIN each claimed degenerate pair
    assert np.max(np.abs(pair_gaps)) < 1e-6


def test_sine_kernel_matches_cue_correlations():
    ens = rmt.ensembles.CUE(n=400, seed=8)
    spectrum = cached_sample(ens, n_samples=60)
    benchmark = rmt.validation.SineKernel()
    result = benchmark.validate(spectrum)
    assert result.rmse < 0.05


def test_sine_kernel_rejects_non_cue_beta():
    ens = rmt.ensembles.COE(n=100, seed=0)
    spectrum = cached_sample(ens, n_samples=5)
    benchmark = rmt.validation.SineKernel()
    with pytest.raises(ValueError):
        benchmark.validate(spectrum)


def test_sine_kernel_zero_at_origin_and_one_at_large_separation():
    r = np.array([0.0, 0.01, 2.0, 5.0])
    r2 = rmt.stats.sine_kernel_r2(r)
    assert r2[0] == pytest.approx(0.0, abs=1e-9)
    assert r2[1] < 0.01  # strong level repulsion near r=0
    assert r2[3] == pytest.approx(1.0, abs=1e-2)


# --- Pfaffian kernel building blocks (S, D, I) ---
# See physicskit.rmt/stats/correlations.py module docstring: only these three
# individually-exact building blocks are implemented, NOT their
# combination into a beta=1/4 two-point correlation function (an
# attempted reconstruction was checked and ruled out structurally --
# see that docstring for the Taylor-expansion argument).


def test_kernel_s_matches_sine_kernel_component():
    r = np.array([0.0, 0.3, 1.0, 2.5])
    np.testing.assert_allclose(rmt.stats.kernel_s(r) ** 2, 1.0 - rmt.stats.sine_kernel_r2(r), atol=1e-12)


def test_kernel_d_is_derivative_of_kernel_s():
    r = np.linspace(0.05, 3.0, 20)
    h = 1e-6
    finite_diff = (rmt.stats.kernel_s(r + h) - rmt.stats.kernel_s(r - h)) / (2 * h)
    np.testing.assert_allclose(rmt.stats.kernel_d(r), finite_diff, atol=1e-4)


def test_kernel_i_is_antiderivative_of_kernel_s():
    from scipy.integrate import quad

    for r in [0.3, 1.0, 2.0]:
        expected, _ = quad(lambda t: rmt.stats.kernel_s(np.array([t]))[0], 0, r)
        assert rmt.stats.kernel_i(np.array([r]))[0] == pytest.approx(expected, abs=1e-6)


def test_kernel_d_is_odd_and_kernel_s_is_even():
    r = np.array([0.2, 0.7, 1.5])
    np.testing.assert_allclose(rmt.stats.kernel_s(-r), rmt.stats.kernel_s(r))
    np.testing.assert_allclose(rmt.stats.kernel_d(-r), -rmt.stats.kernel_d(r))


# --- general k-point determinantal correlation function (CUE) ---


def test_k_point_correlation_reduces_to_sine_kernel_r2_at_k_2():
    r = 0.7
    value = rmt.stats.k_point_correlation(np.array([0.0, r]))
    assert value == pytest.approx(rmt.stats.sine_kernel_r2(np.array([r]))[0])


def test_k_point_correlation_vanishes_at_coincident_points():
    # Level repulsion at all orders: a determinant with two identical
    # rows (coincident positions) is exactly zero.
    value = rmt.stats.k_point_correlation(np.array([0.0, 0.0, 1.5]))
    assert value == pytest.approx(0.0, abs=1e-12)


def test_k_point_correlation_factorizes_at_large_separation():
    # Cluster decomposition: R_3(x1, x2, x3) -> R_2(x1, x2) as x3 -> infinity.
    r = 0.7
    r2 = rmt.stats.sine_kernel_r2(np.array([r]))[0]
    r3_near = rmt.stats.k_point_correlation(np.array([0.0, r, 5.0]))
    r3_far = rmt.stats.k_point_correlation(np.array([0.0, r, 500.0]))
    assert abs(r3_far - r2) < abs(r3_near - r2)
    assert r3_far == pytest.approx(r2, abs=1e-4)


def test_level_repulsion_strength_orders_cse_cue_coe():
    # Robust, exponent-free qualitative check (see correlations.py
    # module docstring for why the exact beta=1/4 exponents/kernels are
    # not asserted here): near a fixed small separation, the two-point
    # correlation (level repulsion) should order CSE << CUE << COE,
    # matching beta=4 > beta=2 > beta=1 repulsion strength.
    n = 300
    r_probe = 0.1
    values = {}
    ensembles = [("COE", rmt.ensembles.COE), ("CUE", rmt.ensembles.CUE), ("CSE", rmt.ensembles.CSE)]
    for name, cls in ensembles:
        ens = cls(n=n, seed=0)
        spectrum = cached_sample(ens, n_samples=60)
        centers, empirical = rmt.stats.pair_correlation_estimate(spectrum, r_max=1.0, n_bins=50)
        idx = np.argmin(np.abs(centers - r_probe))
        values[name] = empirical[idx]
    assert values["CSE"] < values["CUE"] < values["COE"]
