import numpy as np
import pytest

from physicskit.astro.cosmic_web import (
    first_caustic_time,
    lagrangian_grid,
    random_displacement_potential,
    zeldovich_displacement,
    zeldovich_hessian_eigenvalues,
    zeldovich_position,
)

# For a single plane-wave potential Phi(q) = A*cos(kx*x + ky*y + phase), the
# Hessian is H = -A*cos(theta)*[[kx^2, kx*ky], [kx*ky, ky^2]] at every q,
# i.e. a rank-1 (outer-product) matrix scaled by c = -A*cos(theta). Its two
# eigenvalues are exactly {0, c*|k|^2}, so the *largest* eigenvalue at any
# given q is max(0, c*|k|^2) -- and this is maximized over q (giving the
# analytic first-caustic time 1/(A*|k|^2)) exactly where cos(theta) = -1,
# i.e. where kx*x + ky*y + phase = pi (mod 2*pi).


def test_single_mode_hessian_eigenvalues_exact_at_compression_maximum():
    """At the specific q where the mode's phase argument equals pi (its
    point of maximal compression), the Hessian eigenvalues are exactly
    (0, A*|k|^2) -- a fully deterministic, hand-derivable check."""
    kx, ky, A = 2 * np.pi, 0.0, 1.5
    k_vectors = np.array([[kx, ky]])
    amplitudes = np.array([A])
    phases = np.array([np.pi])  # theta = kx*0 + ky*0 + pi = pi at q=(0,0)

    q_star = np.array([[0.0, 0.0]])
    eigs = zeldovich_hessian_eigenvalues(q_star, k_vectors, amplitudes, phases)

    k_sq = kx**2 + ky**2
    assert eigs.shape == (1, 2)
    assert eigs[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert eigs[0, 1] == pytest.approx(A * k_sq, rel=1e-10)


def test_first_caustic_time_matches_single_mode_analytic_value_exactly():
    """first_caustic_time at that same single deterministic point should
    equal the closed-form D_collapse = 1/(A*|k|^2) exactly."""
    kx, ky, A = 2 * np.pi, 0.0, 1.5
    k_vectors = np.array([[kx, ky]])
    amplitudes = np.array([A])
    phases = np.array([np.pi])

    q_star = np.array([[0.0, 0.0]])
    D_collapse = first_caustic_time(q_star, k_vectors, amplitudes, phases)
    analytic = 1.0 / (A * (kx**2 + ky**2))
    assert D_collapse == pytest.approx(analytic, rel=1e-10)


def test_first_caustic_time_converges_with_grid_resolution():
    """On a *generic* grid (the compression maximum does not sit exactly on
    a grid point), first_caustic_time only finds the nearest sampled point
    to the true maximum, so it overestimates D_collapse -- but the error
    should shrink as the grid is refined and the sampling gets closer to
    the true maximizing point."""
    L = 1.0
    kx, ky, A = 2 * np.pi / L, 0.0, 1.0
    k_vectors = np.array([[kx, ky]])
    amplitudes = np.array([A])
    phases = np.array([0.7])  # arbitrary offset, not aligned with any grid spacing
    analytic = 1.0 / (A * (kx**2 + ky**2))

    q_coarse = lagrangian_grid(20, L)
    q_fine = lagrangian_grid(400, L)

    D_coarse = first_caustic_time(q_coarse, k_vectors, amplitudes, phases)
    D_fine = first_caustic_time(q_fine, k_vectors, amplitudes, phases)

    error_coarse = D_coarse - analytic
    error_fine = D_fine - analytic

    # The grid-based estimate can only ever undershoot the true max
    # eigenvalue, so it can only ever overshoot D_collapse.
    assert error_coarse >= 0.0
    assert error_fine >= 0.0
    assert error_fine < error_coarse
    assert error_fine < 1e-5


def test_zeldovich_position_at_zero_growth_returns_lagrangian_position():
    k_vectors, amplitudes, phases = random_displacement_potential(6, k_min=1.0, k_max=4.0, amplitude_scale=0.2, seed=3)
    q = lagrangian_grid(12, 2.0)
    x0 = zeldovich_position(q, 0.0, k_vectors, amplitudes, phases)
    assert x0 == pytest.approx(q, abs=1e-12)


def test_zeldovich_displacement_matches_finite_difference_gradient():
    """Sanity check independent of the eigenvalue-formula correctness:
    the analytic gradient should match a numerical finite-difference
    gradient of Phi evaluated by hand from the same plane-wave sum."""
    rng = np.random.default_rng(42)
    n_modes = 4
    k_vectors = rng.uniform(-3.0, 3.0, size=(n_modes, 2))
    amplitudes = rng.uniform(0.1, 1.0, size=n_modes)
    phases = rng.uniform(0.0, 2 * np.pi, size=n_modes)

    def phi(point):
        theta = k_vectors @ point + phases
        return float(np.sum(amplitudes * np.cos(theta)))

    q_points = rng.uniform(-2.0, 2.0, size=(5, 2))
    analytic_grad = zeldovich_displacement(q_points, k_vectors, amplitudes, phases)

    h = 1e-6
    for i, point in enumerate(q_points):
        numeric_grad = np.zeros(2)
        for d in range(2):
            step = np.zeros(2)
            step[d] = h
            numeric_grad[d] = (phi(point + step) - phi(point - step)) / (2 * h)
        assert analytic_grad[i] == pytest.approx(numeric_grad, abs=1e-5)


def test_shapes_and_no_nans():
    k_vectors, amplitudes, phases = random_displacement_potential(5, k_min=1.0, k_max=3.0, amplitude_scale=0.1, seed=7)
    q = lagrangian_grid(10, 2.0)
    n = q.shape[0]

    assert q.shape == (100, 2)
    assert k_vectors.shape == (5, 2)
    assert amplitudes.shape == (5,)
    assert phases.shape == (5,)

    grad = zeldovich_displacement(q, k_vectors, amplitudes, phases)
    x = zeldovich_position(q, 0.5, k_vectors, amplitudes, phases)
    eigs = zeldovich_hessian_eigenvalues(q, k_vectors, amplitudes, phases)
    D_collapse = first_caustic_time(q, k_vectors, amplitudes, phases)

    assert grad.shape == (n, 2)
    assert x.shape == (n, 2)
    assert eigs.shape == (n, 2)
    assert np.all(eigs[:, 0] <= eigs[:, 1])
    assert isinstance(D_collapse, float)
    assert D_collapse > 0.0

    assert not np.any(np.isnan(grad))
    assert not np.any(np.isnan(x))
    assert not np.any(np.isnan(eigs))
    assert not np.isnan(D_collapse)


def test_random_displacement_potential_invalid_arguments_raise():
    with pytest.raises(ValueError):
        random_displacement_potential(4, k_min=1.0, k_max=2.0, amplitude_scale=0.1, ndim=3)
    with pytest.raises(ValueError):
        random_displacement_potential(0, k_min=1.0, k_max=2.0, amplitude_scale=0.1)
    with pytest.raises(ValueError):
        random_displacement_potential(4, k_min=2.0, k_max=1.0, amplitude_scale=0.1)


def test_lagrangian_grid_shape_and_bounds():
    q = lagrangian_grid(7, 3.0)
    assert q.shape == (49, 2)
    assert np.all(q >= 0.0)
    assert np.all(q < 3.0)
