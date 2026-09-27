"""Heat equation solvers against the spreading Gaussian and single-mode decay."""

import numpy as np
import pytest

from physicskit.fluids.exceptions import InvalidParameterError
from physicskit.fluids.systems.heat_equation import (
    gaussian_heat_solution,
    heat_equation,
    heat_equation_spectral,
    laplacian_matrix,
)


@pytest.mark.parametrize("method", ["explicit", "crank_nicolson"])
def test_gaussian_decays_in_1d(method):
    """Peak falls as sigma0 / sqrt(sigma0^2 + 2 alpha t) and the profile stays Gaussian."""
    x = np.linspace(-20, 20, 801)
    dx = x[1] - x[0]
    alpha, sigma0 = 0.5, 1.0
    dt = 0.4 * dx**2 / alpha if method == "explicit" else 0.01
    t, u = heat_equation(np.exp(-(x**2) / 2), dx, alpha, t_max=4.0, dt=dt, method=method, n_frames=4)
    for tk, uk in zip(t, u):
        assert uk == pytest.approx(gaussian_heat_solution(x, tk, alpha, sigma0), abs=2e-4)
    assert u[-1].max() == pytest.approx(1 / np.sqrt(1 + 2 * alpha * t[-1]), rel=1e-3)


def test_gaussian_decays_in_2d_with_d_over_2_power():
    """In d dimensions the peak falls as (sigma0^2 / sigma^2)^{d/2}; the total heat is conserved."""
    n, L = 128, 40.0  # wide enough that the periodic images are negligible
    h = L / n
    x = (np.arange(n) - n // 2) * h
    X, Y = np.meshgrid(x, x, indexing="ij")
    u0 = np.exp(-(X**2 + Y**2) / 2)
    alpha = 0.7
    times = [0.0, 1.0, 3.0]
    u = heat_equation_spectral(u0, h, alpha, times)
    for tk, uk in zip(times, u):
        assert uk == pytest.approx(gaussian_heat_solution((X, Y), tk, alpha, 1.0), abs=1e-10)
        assert uk.sum() == pytest.approx(u0.sum(), rel=1e-12)
    assert u[-1].max() == pytest.approx(1 / (1 + 2 * alpha * 3.0), rel=1e-10)
    _, cn = heat_equation(u0, h, alpha, t_max=3.0, dt=0.02, bc="periodic")
    assert np.max(np.abs(cn[-1] - u[-1])) < 2e-3


def test_crank_nicolson_is_second_order_in_time():
    x = np.linspace(0, 1, 51)
    u0 = np.sin(np.pi * x) + 0.5 * np.sin(3 * np.pi * x)
    ref = heat_equation(u0, x[1] - x[0], 0.1, t_max=0.5, dt=1e-4)[1][-1]
    errs = [np.max(np.abs(heat_equation(u0, x[1] - x[0], 0.1, t_max=0.5, dt=dt)[1][-1] - ref)) for dt in (0.02, 0.01)]
    assert errs[0] / errs[1] == pytest.approx(4.0, rel=0.05)


def test_neumann_walls_conserve_heat_and_dirichlet_walls_hold_values():
    x = np.linspace(0, 1, 101)
    u0 = np.exp(-((x - 0.3) ** 2) / 0.005)
    _, u = heat_equation(u0, 0.01, 1.0, t_max=2.0, dt=0.01, bc="neumann")
    # trapezoid-weighted mass is exactly conserved by the mirrored-ghost stencil
    w = np.ones_like(x)
    w[[0, -1]] = 0.5
    assert np.sum(w * u[-1]) == pytest.approx(np.sum(w * u0), rel=1e-10)
    assert np.ptp(u[-1]) < 1e-6  # relaxed to uniform
    u0 = np.linspace(2.0, 5.0, 101) + np.sin(7 * np.pi * x)
    _, u = heat_equation(u0, 0.01, 1.0, t_max=2.0, dt=0.01, bc="dirichlet")
    assert u[-1][[0, -1]] == pytest.approx([2.0, 5.0])
    assert u[-1] == pytest.approx(np.linspace(2.0, 5.0, 101), abs=1e-6)  # steady linear profile


def test_explicit_scheme_rejects_unstable_step():
    with pytest.raises(InvalidParameterError):
        heat_equation(np.zeros(11), 0.1, 1.0, t_max=1.0, dt=0.006, method="explicit")
    heat_equation(np.zeros(11), 0.1, 1.0, t_max=0.05, dt=0.005, method="explicit")  # r = 1/2 is allowed
    with pytest.raises(InvalidParameterError):
        heat_equation(np.zeros((11, 11)), 0.1, 1.0, t_max=1.0, dt=0.004, method="explicit")  # 2D bound is 1/4
    with pytest.raises(InvalidParameterError):
        heat_equation(np.zeros(11), 0.1, 1.0, t_max=1.0, dt=0.001, method="leapfrog")
    with pytest.raises(InvalidParameterError):
        laplacian_matrix((5,), 1.0, bc="robin")
