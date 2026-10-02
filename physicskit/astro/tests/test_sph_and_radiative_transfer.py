import numpy as np
import pytest

from physicskit.astro.radiative_transfer import (
    GreyAtmosphere,
    eddington_limb_darkening,
    eddington_temperature,
    emergent_intensity,
)
from physicskit.astro.sph import SPH1D, cubic_spline_kernel
from physicskit.fluids.systems.compressible_flow import exact_riemann_solution


def test_kernel_normalization_and_support():
    r = np.linspace(-2.5, 2.5, 50001)
    for h in (0.3, 1.0):
        assert np.trapezoid(cubic_spline_kernel(r, h), r) == pytest.approx(1.0, rel=1e-6)
    assert cubic_spline_kernel(np.array([2.0, 3.0]), 1.0).tolist() == [0.0, 0.0]


def test_uniform_gas_has_uniform_density_and_stays_at_rest():
    x = np.linspace(-1, 1, 201)
    gas = SPH1D(x, np.zeros_like(x), m=0.01, u=np.ones_like(x), fixed=np.abs(x) > 0.95)
    inner = np.abs(x) < 0.8
    np.testing.assert_allclose(gas.rho[inner], 1.0, rtol=5e-3)  # kernel-sum discretization error
    gas.run(0.1)
    assert np.max(np.abs(gas.v[inner])) < 1e-3


def test_sod_shock_tube_matches_exact_riemann_solution():
    gas = SPH1D.sod_shock_tube(n_left=400)
    gas.run(0.2)
    exact = exact_riemann_solution(gas.x, 0.2)
    inner = np.abs(gas.x) < 0.45
    assert np.mean(np.abs(gas.rho[inner] - exact["rho"][inner])) < 0.01
    assert np.mean(np.abs(gas.pressure[inner] - exact["p"][inner])) < 0.01
    plateau = (gas.x > 0.05) & (gas.x < 0.15)  # between contact and shock
    assert gas.v[plateau].mean() == pytest.approx(float(exact["u_star"]), rel=0.02)
    assert gas.energy_history[-1] == pytest.approx(gas.energy_history[0], rel=1e-3)


def test_exact_riemann_solver_reproduces_toro_tables():
    # Toro (2009), Table 4.3: tests 1-3
    for left, right, p_star, u_star in [
        ((1.0, 0.0, 1.0), (0.125, 0.0, 0.1), 0.30313, 0.92745),
        ((1.0, -2.0, 0.4), (1.0, 2.0, 0.4), 0.00189, 0.0),
        ((1.0, 0.0, 1000.0), (1.0, 0.0, 0.01), 460.894, 19.5975),
    ]:
        out = exact_riemann_solution(np.array([0.0]), 0.1, left, right)
        assert float(out["p_star"]) == pytest.approx(p_star, rel=2e-4, abs=1e-5)
        assert float(out["u_star"]) == pytest.approx(u_star, rel=2e-4, abs=1e-5)


def test_eddington_law():
    assert eddington_temperature(2 / 3) == pytest.approx(1.0)
    assert eddington_limb_darkening(0.0) == pytest.approx(0.4)


def test_grey_atmosphere_hopf_constants():
    atm = GreyAtmosphere(n_streams=64)
    assert atm.hopf(0.0) == pytest.approx(1 / np.sqrt(3), rel=1e-10)
    assert atm.Q == pytest.approx(0.710446, abs=2e-5)  # Hopf's constant
    # Chandrasekhar, Radiative Transfer, Table XIII: I(0, 1) = 1.2591 F/pi... in units of F/pi
    assert atm.emergent_intensity(1.0) == pytest.approx(1.2591, abs=2e-4)
    assert atm.limb_darkening(0.0) == pytest.approx(0.3439, abs=2e-4)


def test_grey_atmosphere_carries_unit_flux_and_matches_formal_solution():
    atm = GreyAtmosphere(n_streams=32)
    mu = np.linspace(0, 1, 20001)
    assert 2 * np.trapezoid(atm.emergent_intensity(mu) * mu, mu) == pytest.approx(1.0, abs=2e-4)
    mus = np.array([0.1, 0.5, 1.0])
    np.testing.assert_allclose(emergent_intensity(lambda t: 0.75 * (t + atm.hopf(t)), mus), atm.emergent_intensity(mus), rtol=1e-5)


def test_one_stream_is_eddington_like():
    atm = GreyAtmosphere(n_streams=1)
    assert atm.Q == pytest.approx(1 / np.sqrt(3))
    assert atm.L.size == 0
