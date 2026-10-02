import numpy as np
import pytest

from physicskit.fluids.core.grid import poisson_solve_streamfunction, spectral_grid, velocity_from_streamfunction
from physicskit.fluids.exceptions import InvalidParameterError
from physicskit.fluids.systems.convection import (
    RayleighBenard2D,
    RayleighBenardWalls2D,
    rayleigh_benard_critical,
    rayleigh_benard_growth_rate_free,
    rayleigh_benard_marginal_rayleigh,
)
from physicskit.fluids.systems.shallow_water import (
    dam_break_exact,
    geostrophic_adjustment_steady,
    rotating_shallow_water_1d,
    shallow_water_1d,
)
from physicskit.fluids.systems.turbulence import ForcedTurbulence2D, ForcedTurbulence3D, kolmogorov_kraichnan_spectrum

# --- forced turbulence ---------------------------------------------------


def test_spectral_energy_matches_real_space_velocity():
    flow = ForcedTurbulence2D(n=64, kf=10.0, seed=0)
    flow.run(t_max=1.0, dt=0.005, t_average=0.5)
    _, _, KX, KY, K2 = spectral_grid(64, 2 * np.pi)
    u, v = velocity_from_streamfunction(poisson_solve_streamfunction(flow.vorticity(), K2), KX, KY)
    assert flow.kinetic_energy() == pytest.approx(0.5 * np.mean(u**2 + v**2), rel=1e-10)
    assert flow.spectrum()[1].sum() == pytest.approx(flow.kinetic_energy(), rel=1e-10)


def test_energy_grows_at_injection_rate_before_dissipation_acts():
    flow = ForcedTurbulence2D(n=64, kf=10.0, epsilon=2.0, drag=0.0, seed=0)
    out = flow.run(t_max=1.0, dt=0.002, t_average=0.1, sample_every=5)
    slope = np.polyfit(out["t"], out["energy"], 1)[0]
    assert slope == pytest.approx(2.0, rel=0.1)


def test_inverse_cascade_spectrum_has_kolmogorov_slope():
    out = ForcedTurbulence2D(n=128, kf=20.0, seed=0).run(t_max=16.0, dt=0.004, t_average=8.0)
    k, E = out["k"], out["E"]
    band = (k >= 3) & (k <= 12)
    slope = np.polyfit(np.log(k[band]), np.log(E[band]), 1)[0]
    assert slope == pytest.approx(-5.0 / 3.0, abs=0.2)


def test_3d_energy_budget_injection_minus_dissipation():
    # the dealiased nonlinear term only moves energy between scales, so dE/dt = epsilon - D
    flow = ForcedTurbulence3D(n=24, epsilon=1.0, seed=3)
    out = flow.run(t_max=1.0, dt=0.005, t_average=0.1, sample_every=1)
    dEdt = np.gradient(out["energy"], out["t"])
    np.testing.assert_allclose(dEdt[5:-5], 1.0 - out["dissipation"][5:-5], atol=0.02)
    assert flow.max_divergence() < 1e-10
    assert flow.spectrum()[1].sum() == pytest.approx(flow.kinetic_energy())


def test_kolmogorov_kraichnan_spectrum_scaling():
    assert kolmogorov_kraichnan_spectrum(8.0, epsilon=8.0, C=1.0) == pytest.approx(4.0 / 32.0)


def test_forcing_outside_dealiased_range_rejected():
    with pytest.raises(InvalidParameterError):
        ForcedTurbulence2D(n=32, kf=12.0)


# --- Rayleigh-Benard -----------------------------------------------------


def test_free_slip_critical_values_are_exact():
    Ra_c, a_c = rayleigh_benard_critical("free")
    assert Ra_c == pytest.approx(27 * np.pi**4 / 4)
    assert rayleigh_benard_marginal_rayleigh(a_c, "free") == pytest.approx(Ra_c)


def test_rigid_critical_rayleigh_number():
    # Chandrasekhar (1961), Table III: Ra_c = 1707.762, a_c = 3.117
    Ra_c, a_c = rayleigh_benard_critical("rigid")
    assert Ra_c == pytest.approx(1707.762, abs=0.01)
    assert a_c == pytest.approx(3.117, abs=2e-3)


def test_growth_rate_vanishes_at_marginal_rayleigh():
    a = 2.7
    assert rayleigh_benard_growth_rate_free(rayleigh_benard_marginal_rayleigh(a, "free"), a, Pr=7.0) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("Ra", [500.0, 1200.0])
def test_simulated_linear_growth_rate_matches_exact(Ra):
    rb = RayleighBenard2D(Ra, nx=16, nz=16)
    rb.seed_mode(amplitude=1e-6)
    out = rb.run(t_max=0.6, dt=2e-3, sample_every=5)
    late = out["t"] > 0.3  # after the second, faster-decaying root has died out
    rate = np.polyfit(out["t"][late], np.log(np.abs(out["theta_mode"][late])), 1)[0]
    assert rate == pytest.approx(rayleigh_benard_growth_rate_free(Ra, np.pi / np.sqrt(2)), rel=5e-3)


def test_conduction_below_onset_and_heat_transport_above():
    below = RayleighBenard2D(600.0, nx=16, nz=16)
    below.seed_mode(1e-2)
    assert below.run(t_max=3.0, dt=2e-3)["nusselt"][-1] == pytest.approx(1.0, abs=1e-3)
    above = RayleighBenard2D(2000.0, nx=32, nz=32)
    above.seed_mode(1e-2)
    out = above.run(t_max=4.0, dt=2e-3)
    assert out["nusselt"][-1] > 2.0
    assert np.allclose(above.theta, -above.theta[above._mirror, :])


def _growth_rate(rb, t_max, t_skip):
    rb.seed_mode(1e-6)
    out = rb.run(t_max=t_max, dt=2e-3, sample_every=5)
    late = out["t"] > t_skip
    return np.polyfit(out["t"][late], np.log(out["theta_mode"][late]), 1)[0]


def test_chebyshev_solver_reproduces_exact_stress_free_growth_rate():
    a = np.pi / np.sqrt(2)
    rate = _growth_rate(RayleighBenardWalls2D(1000.0, boundaries="free", nx=16, nz=16), 0.8, 0.3)
    assert rate == pytest.approx(rayleigh_benard_growth_rate_free(1000.0, a), rel=1e-3)


def test_simulated_rigid_wall_onset_is_1708():
    Ra_values = [1650.0, 1770.0]
    rates = [_growth_rate(RayleighBenardWalls2D(Ra, nx=16, nz=20), 1.5, 0.5) for Ra in Ra_values]
    assert rates[0] < 0 < rates[1]
    assert np.interp(0.0, rates, Ra_values) == pytest.approx(1707.76, rel=2e-3)


def test_rigid_walls_carry_heat_above_onset():
    rb = RayleighBenardWalls2D(3000.0, nx=32, nz=24)
    rb.seed_mode(1e-2)
    out = rb.run(t_max=6.0, dt=2e-3, sample_every=100)
    assert out["nusselt"][-1] > 1.3
    f = rb.fields()
    # no-slip: both velocity components vanish at the walls
    assert np.abs(f["u"][:, [0, -1]]).max() < 1e-8 and np.abs(f["w"][:, [0, -1]]).max() < 1e-8


def test_walls_solver_rejects_unknown_boundaries():
    with pytest.raises(InvalidParameterError):
        RayleighBenardWalls2D(1000.0, boundaries="slippery")


# --- shallow water -------------------------------------------------------


@pytest.mark.parametrize("h_right", [0.0, 0.2])
def test_dam_break_matches_ritter_stoker(h_right):
    x = np.linspace(-10, 10, 801)
    out = shallow_water_1d(np.where(x < 0, 1.0, h_right), np.zeros_like(x), x, t_max=1.0)
    h_exact, _ = dam_break_exact(x, 1.0, 1.0, h_right)
    assert np.mean(np.abs(out["h"] - h_exact)) < 2e-3


def test_stoker_state_satisfies_rankine_hugoniot():
    g, hL, hR = 9.81, 1.0, 0.3
    x = np.linspace(-1, 10, 20001)
    h, u = dam_break_exact(x, 1.0, hL, hR, g)
    jump = np.argmax(np.abs(np.diff(h)))
    hm, um, s = h[jump], u[jump], x[jump + 1]
    # mass and momentum flux continuity across the moving bore
    assert hm * (um - s) == pytest.approx(hR * (0 - s), rel=2e-3)
    assert hm * um * (um - s) + 0.5 * g * hm**2 == pytest.approx(0.5 * g * hR**2, rel=2e-3)


def test_ritter_wet_front_speed():
    h, _ = dam_break_exact(np.array([1.99, 2.01]) * np.sqrt(9.81), 1.0, 1.0)
    assert h[0] > 0 and h[1] == 0


def test_geostrophic_front_has_rossby_radius_width():
    L = 200.0
    x = np.linspace(-L / 2, L / 2, 2048, endpoint=False)
    eta0 = np.where(np.abs(x) < L / 4, 1.0, -1.0)
    out = geostrophic_adjustment_steady(eta0, x, g=4.0, H=1.0, f=1.0)
    assert float(out["L_d"]) == pytest.approx(2.0)
    near = (np.abs(x + L / 4) < 20) & (np.abs(x + L / 4) > 0.5)
    exact = np.sign(x + L / 4) * (1 - np.exp(-np.abs(x + L / 4) / 2.0))
    np.testing.assert_allclose(out["eta"][near], exact[near], atol=0.02)


def test_rotating_shallow_water_conserves_potential_vorticity_and_averages_to_balance():
    L = 200.0
    x = np.linspace(-L / 2, L / 2, 1024, endpoint=False)
    eta0 = np.exp(-(x**2) / 50.0)
    g, H, f = 1.0, 1.0, 1.0
    times = np.linspace(0.0, 400.0, 2001)
    out = rotating_shallow_water_1d(eta0, x, times, g, H, f)
    k = 2 * np.pi * np.fft.fftfreq(x.size, d=x[1] - x[0])
    pv = np.real(np.fft.ifft(1j * k * np.fft.fft(out["v"], axis=1), axis=1)) - f * out["eta"] / H
    np.testing.assert_allclose(pv, -f * eta0 / H + 0 * pv, atol=1e-10)
    steady = geostrophic_adjustment_steady(eta0, x, g, H, f)
    np.testing.assert_allclose(out["eta"][1000:].mean(axis=0), steady["eta"], atol=5e-3)
