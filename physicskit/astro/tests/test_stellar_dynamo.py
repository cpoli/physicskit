import numpy as np
import pytest

from physicskit.astro.stellar_dynamo import (
    _spectral_grid_2d,
    alpha_omega_growth_rate,
    alpha_omega_wave_frequency,
    convective_roll_rhs,
    dominant_mode_growth_rate,
    kinetic_energy,
    simulate_alpha_omega_dynamo,
    simulate_stellar_convection,
)

# ---------------------------------------------------------------------------
# Part 2: alpha-omega mean-field dynamo -- the core dispersion-relation check
# ---------------------------------------------------------------------------


def test_alpha_omega_growth_rate_matches_simulation_for_unstable_mode():
    """The single most important correctness guarantee in this module:
    seed a pure Fourier mode, evolve it, and check the numerically
    observed growth rate against the closed-form dispersion relation."""
    Lx = 2 * np.pi
    nx = 64
    x = np.linspace(0, Lx, nx, endpoint=False)
    k0 = 1.0  # m = 1: k0 = 2*pi*m/Lx with Lx = 2*pi
    alpha, shear, eta = 1.0, 5.0, 0.05

    A0 = 1e-3 * np.cos(k0 * x)
    B0 = np.zeros_like(A0)

    sigma_predicted = alpha_omega_growth_rate(alpha, shear, k0, eta)
    assert sigma_predicted > 0  # this test is specifically the unstable/growing case

    dt = 0.01
    n_steps = 600
    save_every = 5
    times, A_snapshots, B_snapshots = simulate_alpha_omega_dynamo(A0, B0, alpha, shear, eta, Lx, dt, n_steps, save_every)

    sigma_observed = dominant_mode_growth_rate(B_snapshots, dt_save=dt * save_every)
    assert sigma_observed == pytest.approx(sigma_predicted, rel=0.02)

    # Cross-check against A too, and against the module's independent
    # frequency formula (same magnitude as the growth rate, by construction
    # of the dispersion relation s^2 = i*alpha*shear*k).
    sigma_observed_A = dominant_mode_growth_rate(A_snapshots, dt_save=dt * save_every)
    assert sigma_observed_A == pytest.approx(sigma_predicted, rel=0.02)
    omega_freq = alpha_omega_wave_frequency(alpha, shear, k0)
    assert omega_freq == pytest.approx(np.sqrt(alpha * shear * k0 / 2.0), rel=1e-10)


def test_alpha_omega_dispersion_relation_holds_for_several_wavenumbers():
    """The closed-form growth rate and wave-frequency magnitude must agree
    (they're the real and imaginary parts of the same complex root)."""
    alpha, shear, eta = 0.8, 3.0, 0.1
    k = np.array([0.5, 1.0, 2.0, 3.0])
    sigma = alpha_omega_growth_rate(alpha, shear, k, eta)
    freq = alpha_omega_wave_frequency(alpha, shear, k)
    assert freq == pytest.approx(sigma + eta * k**2, rel=1e-10)


def test_alpha_omega_dynamo_decays_in_stable_regime():
    """Large diffusivity relative to alpha*shear*k should make the
    dispersion relation predict decay, and the simulated field energy
    should actually decay."""
    Lx = 2 * np.pi
    nx = 64
    x = np.linspace(0, Lx, nx, endpoint=False)
    k0 = 1.0
    alpha, shear, eta = 1.0, 1.0, 2.0

    sigma_predicted = alpha_omega_growth_rate(alpha, shear, k0, eta)
    assert sigma_predicted < 0  # this test is specifically the stable/decaying case

    A0 = 1e-3 * np.cos(k0 * x)
    B0 = np.zeros_like(A0)
    dt = 0.01
    n_steps = 600
    save_every = 5
    times, A_snapshots, B_snapshots = simulate_alpha_omega_dynamo(A0, B0, alpha, shear, eta, Lx, dt, n_steps, save_every)

    initial_energy = np.sum(A_snapshots[0] ** 2 + B_snapshots[0] ** 2)
    final_energy = np.sum(A_snapshots[-1] ** 2 + B_snapshots[-1] ** 2)
    assert final_energy < initial_energy


def test_simulate_alpha_omega_dynamo_shapes_and_finiteness():
    Lx = 2 * np.pi
    nx = 40
    x = np.linspace(0, Lx, nx, endpoint=False)
    A0 = 1e-3 * np.cos(2 * x)
    B0 = np.zeros_like(A0)
    times, A_snapshots, B_snapshots = simulate_alpha_omega_dynamo(A0, B0, alpha=1.0, shear=2.0, eta=0.1, Lx=Lx, dt=0.02, n_steps=30, save_every=3)
    n_saved = 30 // 3 + 1
    assert times.shape == (n_saved,)
    assert A_snapshots.shape == (n_saved, nx)
    assert B_snapshots.shape == (n_saved, nx)
    assert np.all(np.isfinite(A_snapshots))
    assert np.all(np.isfinite(B_snapshots))
    assert A_snapshots[0] == pytest.approx(A0)
    assert B_snapshots[0] == pytest.approx(B0)


def test_alpha_omega_growth_rate_vectorizes_over_k():
    k = np.array([0.0, 1.0, 2.0, 4.0])
    sigma = alpha_omega_growth_rate(alpha=1.0, shear=2.0, k=k, eta=0.1)
    assert sigma.shape == (4,)
    # k=0: no alpha-omega drive at all, growth rate is exactly zero (no diffusion either, at k=0).
    assert sigma[0] == pytest.approx(0.0)


def test_dominant_mode_growth_rate_recovers_known_slope():
    t = np.arange(20) * 0.1
    sigma_true = 1.7
    A_snapshots = np.exp(sigma_true * t)[:, None] * np.ones((20, 6))
    sigma_fit = dominant_mode_growth_rate(A_snapshots, dt_save=0.1)
    assert sigma_fit == pytest.approx(sigma_true, rel=1e-6)


# ---------------------------------------------------------------------------
# Part 1: 2D convective rolls
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1])
def test_convection_grows_and_stays_finite(seed):
    """Small random noise should grow into sustained convective motion
    (kinetic energy at the end of the run meaningfully larger than at the
    start) without ever producing NaN/Inf, for at least two random seeds."""
    nx = ny = 32
    Lx = Ly = 2 * np.pi
    times, omega_snapshots, T_snapshots = simulate_stellar_convection(nx, ny, Lx, Ly, n_steps=2000, seed=seed)

    assert np.all(np.isfinite(omega_snapshots))
    assert np.all(np.isfinite(T_snapshots))

    _, _, KX, KY, K2 = _spectral_grid_2d(nx, ny, Lx, Ly)
    KE_initial = kinetic_energy(omega_snapshots[0], KX, KY, K2)
    KE_final = kinetic_energy(omega_snapshots[-1], KX, KY, K2)
    assert KE_final > 100.0 * KE_initial


def test_convective_roll_rhs_shapes():
    nx = ny = 32
    Lx = Ly = 2 * np.pi
    _, Y, KX, KY, K2 = _spectral_grid_2d(nx, ny, Lx, Ly)
    rng = np.random.default_rng(0)
    omega = 1e-3 * rng.standard_normal((ny, nx))
    T = 1e-3 * rng.standard_normal((ny, nx))
    domega_dt, dT_dt = convective_roll_rhs(omega, T, Y, KX, KY, K2, nu=0.05, kappa=0.05, g_alpha=1.0, beta=1.0, Ly=Ly, tau_relax=0.5)
    assert domega_dt.shape == (ny, nx)
    assert dT_dt.shape == (ny, nx)
    assert np.all(np.isfinite(domega_dt))
    assert np.all(np.isfinite(dT_dt))


def test_convective_roll_rhs_relaxation_pulls_uniform_zero_field_toward_teq():
    """With zero velocity and zero perturbation temperature, the only
    surviving term in dT'/dt is the Newtonian-cooling relaxation toward
    Teq(y); check its sign and magnitude directly against the formula."""
    nx = ny = 16
    Lx = Ly = 2 * np.pi
    _, Y, KX, KY, K2 = _spectral_grid_2d(nx, ny, Lx, Ly)
    omega = np.zeros((ny, nx))
    T = np.zeros((ny, nx))
    beta, tau_relax = 2.0, 0.5
    _, dT_dt = convective_roll_rhs(omega, T, Y, KX, KY, K2, nu=0.05, kappa=0.05, g_alpha=1.0, beta=beta, Ly=Ly, tau_relax=tau_relax)
    Teq = -beta * np.cos(2 * np.pi * Y / Ly)
    expected = Teq / tau_relax
    assert dT_dt == pytest.approx(expected, abs=1e-10)


def test_simulate_stellar_convection_shapes():
    nx, ny = 24, 32
    Lx, Ly = 2 * np.pi, 3.0
    n_steps, save_every = 20, 5
    times, omega_snapshots, T_snapshots = simulate_stellar_convection(nx, ny, Lx, Ly, n_steps=n_steps, save_every=save_every, seed=0)
    n_saved = n_steps // save_every + 1
    assert times.shape == (n_saved,)
    assert omega_snapshots.shape == (n_saved, ny, nx)
    assert T_snapshots.shape == (n_saved, ny, nx)


def test_kinetic_energy_of_zero_vorticity_is_zero():
    nx = ny = 32
    Lx = Ly = 2 * np.pi
    _, _, KX, KY, K2 = _spectral_grid_2d(nx, ny, Lx, Ly)
    assert kinetic_energy(np.zeros((ny, nx)), KX, KY, K2) == 0.0


def test_kinetic_energy_positive_for_nonzero_vorticity():
    nx = ny = 32
    Lx = Ly = 2 * np.pi
    X, Y, KX, KY, K2 = _spectral_grid_2d(nx, ny, Lx, Ly)
    omega = np.sin(X) * np.cos(Y)
    assert kinetic_energy(omega, KX, KY, K2) > 0.0
