import numpy as np
import pytest

from physicskit.fluids.exceptions import InvalidParameterError
from physicskit.fluids.systems.instabilities import (
    kelvin_helmholtz_growth_rate,
    kelvin_helmholtz_ic,
    rayleigh_taylor_growth_rate,
    rayleigh_taylor_ic,
    simulate_kelvin_helmholtz,
    simulate_rayleigh_taylor,
)
from physicskit.fluids.systems.navier_stokes import simulate_vorticity_streamfunction


def test_kelvin_helmholtz_growth_rate_is_linear_in_k_and_delta_u():
    assert kelvin_helmholtz_growth_rate(k=2.0, delta_u=3.0) == pytest.approx(3.0)
    assert kelvin_helmholtz_growth_rate(k=4.0, delta_u=3.0) == pytest.approx(6.0)


@pytest.mark.slow
def test_kelvin_helmholtz_shear_layer_perturbation_grows():
    n = 96
    omega0 = kelvin_helmholtz_ic(n, 2 * np.pi, shear_width=0.1, perturbation_amplitude=0.05)
    amps = []
    omega = omega0.copy()
    for _ in range(3):
        result = simulate_vorticity_streamfunction(omega, nu=0.001, dt=0.0025, steps=100, length=2 * np.pi)
        omega = result["omega"]
        dev = omega - omega.mean(axis=0, keepdims=True)
        amps.append(np.sqrt(np.mean(dev**2)))
    assert amps[-1] > 2 * amps[0]


def test_simulate_kelvin_helmholtz_matches_direct_vorticity_streamfunction_call():
    n, length = 64, 2 * np.pi
    omega0 = kelvin_helmholtz_ic(n, length, shear_width=0.1, perturbation_amplitude=0.05)
    result_wrapper = simulate_kelvin_helmholtz(omega0, nu=0.001, dt=0.0025, steps=50, length=length)
    result_direct = simulate_vorticity_streamfunction(omega0, nu=0.001, dt=0.0025, steps=50, length=length)
    assert result_wrapper["omega"] == pytest.approx(result_direct["omega"])


@pytest.mark.slow
def test_simulate_kelvin_helmholtz_shear_layer_rolls_up():
    n, length = 96, 2 * np.pi
    omega0 = kelvin_helmholtz_ic(n, length, shear_width=0.1, perturbation_amplitude=0.05)
    omega = omega0.copy()
    amps = []
    for _ in range(3):
        result = simulate_kelvin_helmholtz(omega, nu=0.001, dt=0.0025, steps=100, length=length)
        omega = result["omega"]
        dev = omega - omega.mean(axis=0, keepdims=True)
        amps.append(np.sqrt(np.mean(dev**2)))
    assert amps[-1] > 2 * amps[0]


def test_simulate_kelvin_helmholtz_rejects_nonpositive_viscosity():
    omega0 = kelvin_helmholtz_ic(32, 2 * np.pi, shear_width=0.1, perturbation_amplitude=0.05)
    with pytest.raises(InvalidParameterError):
        simulate_kelvin_helmholtz(omega0, nu=-0.001, dt=0.0025, steps=1, length=2 * np.pi)


def test_rayleigh_taylor_growth_rate_matches_sqrt_agk():
    A, g, k = 0.3, 9.8, 2.0
    assert rayleigh_taylor_growth_rate(k, A, g) == pytest.approx(np.sqrt(A * g * k))


def test_rayleigh_taylor_growth_rate_rejects_invalid_atwood_number():
    with pytest.raises(InvalidParameterError):
        rayleigh_taylor_growth_rate(k=1.0, atwood_number=1.5, g=1.0)


def test_rayleigh_taylor_ic_starts_at_rest_with_heavy_fluid_on_top():
    omega0, buoyancy0 = rayleigh_taylor_ic(64, 2 * np.pi, atwood_number=0.3)
    assert np.all(omega0 == 0.0)
    top_mean = buoyancy0[:, -1].mean()
    bottom_mean = buoyancy0[:, 0].mean()
    assert bottom_mean > 0 > top_mean  # bottom is light (positive b), top is heavy


def test_rayleigh_taylor_interface_perturbation_grows_into_the_nonlinear_regime():
    n, length = 64, 2 * np.pi
    omega0, buoyancy0 = rayleigh_taylor_ic(n, length, atwood_number=0.3, perturbation_amplitude=0.01)
    omega, buoyancy = omega0.copy(), buoyancy0.copy()
    amps = []
    for _ in range(10):
        result = simulate_rayleigh_taylor(omega, buoyancy, nu=0.002, kappa=0.002, g=1.0, dt=0.01, steps=50, length=length)
        omega, buoyancy = result["omega"], result["buoyancy"]
        dev = buoyancy - buoyancy.mean(axis=0, keepdims=True)
        amps.append(np.sqrt(np.mean(dev**2)))
    assert amps[-1] > 3 * amps[0]


def test_simulate_rayleigh_taylor_rejects_nonpositive_parameters():
    omega0, buoyancy0 = rayleigh_taylor_ic(32, 2 * np.pi, atwood_number=0.2)
    with pytest.raises(InvalidParameterError):
        simulate_rayleigh_taylor(omega0, buoyancy0, nu=-1.0, kappa=0.001, g=1.0, dt=0.01, steps=1, length=2 * np.pi)
