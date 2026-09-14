import numpy as np
import pytest

from physicskit.fluids.core.grid import spectral_grid
from physicskit.fluids.exceptions import InvalidParameterError
from physicskit.fluids.systems.navier_stokes import NavierStokes2D, simulate_vorticity_streamfunction


def test_viscous_decay_reduces_peak_vorticity():
    n, length = 48, 2 * np.pi
    X, Y, KX, KY, K2 = spectral_grid(n, length)
    omega0 = np.sin(X) * np.sin(Y)
    result = simulate_vorticity_streamfunction(omega0, nu=0.05, dt=0.01, steps=50, length=length)
    assert np.max(np.abs(result["omega"])) < np.max(np.abs(omega0))


def test_navier_stokes_2d_class_matches_functional_api():
    n, length, nu, dt, steps = 32, 2 * np.pi, 0.05, 0.01, 10
    X, Y, KX, KY, K2 = spectral_grid(n, length)
    omega0 = np.sin(X) * np.sin(Y)
    solver = NavierStokes2D(n=n, length=length, nu=nu)
    result_class = solver.simulate(omega0, dt=dt, steps=steps)
    result_func = simulate_vorticity_streamfunction(omega0, nu=nu, dt=dt, steps=steps, length=length)
    np.testing.assert_allclose(result_class["omega"], result_func["omega"], atol=1e-10)


def test_velocity_field_is_divergence_free():
    """The streamfunction formulation guarantees div(u) = 0 identically;
    check it numerically to catch any sign/index error in the spectral derivatives."""
    n, length = 48, 2 * np.pi
    solver = NavierStokes2D(n=n, length=length, nu=0.01)
    omega0 = np.sin(2 * solver.X) * np.cos(solver.Y) + np.cos(solver.X)
    u, v = solver.velocity(omega0)
    u_hat, v_hat = np.fft.fft2(u), np.fft.fft2(v)
    divergence = np.real(np.fft.ifft2(1j * solver.KX * u_hat + 1j * solver.KY * v_hat))
    assert np.max(np.abs(divergence)) < 1e-8


def test_navier_stokes_rejects_nonpositive_viscosity():
    with pytest.raises(InvalidParameterError):
        NavierStokes2D(n=32, length=2 * np.pi, nu=0.0)
    with pytest.raises(InvalidParameterError):
        simulate_vorticity_streamfunction(np.zeros((16, 16)), nu=-0.1, dt=0.01, steps=1, length=2 * np.pi)


def test_spectral_grid_rejects_too_coarse_resolution():
    with pytest.raises(InvalidParameterError):
        spectral_grid(n=4, length=1.0)
