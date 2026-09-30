"""D2Q9 BGK lattice Boltzmann against Poiseuille flow and the viscosity law nu = c_s^2 (tau - 1/2)."""

import numpy as np
import pytest

from physicskit.fluids.exceptions import InvalidParameterError
from physicskit.fluids.systems.lattice_boltzmann import (
    D2Q9_VELOCITIES,
    D2Q9_WEIGHTS,
    LatticeBoltzmannD2Q9,
    lbm_equilibrium,
    lbm_relaxation_time,
    lbm_viscosity,
)
from physicskit.fluids.systems.viscous_flow import poiseuille_flow_velocity

MAGIC_TAU = 0.5 + np.sqrt(3.0 / 16.0)


def _channel(tau, ny=34, g=1e-6):
    """Body-force-driven channel: solid rows 0 and ny-1, walls halfway at y = 0.5 and ny - 1.5."""
    solid = np.zeros((1, ny), dtype=bool)
    solid[:, [0, -1]] = True
    lb = LatticeBoltzmannD2Q9(1, ny, tau, body_force=(g, 0.0), solid=solid)
    H = ny - 2
    lb.step(int(3 * H**2 / lb.viscosity))  # several viscous diffusion times H^2 / nu
    y = np.arange(ny) - 0.5  # distance from the lower wall
    exact = poiseuille_flow_velocity(y[1:-1], dpdx=-g, mu=lb.viscosity, h=H)  # rho = 1
    return lb.velocity[0][0, 1:-1], exact, lb


@pytest.mark.parametrize("tau", [0.6, 0.8, 1.0, 1.5])
def test_poiseuille_profile(tau):
    """Parabolic u(y) = g y (H - y) / (2 nu), to the known O(tau^2) bounce-back slip error."""
    u, exact, lb = _channel(tau)
    assert u == pytest.approx(exact, abs=5e-3 * exact.max())
    assert np.max(np.abs(lb.velocity[1])) < 1e-9 * exact.max()


def test_poiseuille_exact_at_magic_relaxation_time():
    """With halfway bounce-back, BGK reproduces the parabola to round-off at tau = 1/2 + sqrt(3/16)
    (He, Zou, Luo and Dembo, J. Stat. Phys. 87, 115 (1997))."""
    u, exact, _ = _channel(MAGIC_TAU)
    assert u == pytest.approx(exact, rel=1e-8)


@pytest.mark.parametrize("tau", [0.6, 0.9, 1.4])
def test_viscosity_from_taylor_green_decay(tau):
    """A Taylor-Green vortex decays as exp(-2 nu k^2 t), so its kinetic energy decays at
    4 nu k^2 with nu = c_s^2 (tau - 1/2). The lattice adds an O(k^2) error to the effective
    viscosity, about 1% at tau = 1.4 for a 32-node wavelength, hence 64 nodes here."""
    n = 64
    k = 2 * np.pi / n
    x = np.arange(n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    u0 = 1e-3
    ux = u0 * np.sin(k * X) * np.cos(k * Y)
    uy = -u0 * np.cos(k * X) * np.sin(k * Y)
    lb = LatticeBoltzmannD2Q9(n, n, tau, velocity=(ux, uy))
    nu = lbm_viscosity(tau)
    steps = int(0.5 / (2 * nu * k**2))
    lb.step(20)  # let the non-equilibrium part settle
    E0 = sum(np.sum(c**2) for c in lb.velocity)
    lb.step(steps)
    E1 = sum(np.sum(c**2) for c in lb.velocity)
    nu_measured = -np.log(E1 / E0) / (4 * k**2 * steps)
    assert nu_measured == pytest.approx(nu, rel=0.01)


def test_viscosity_from_poiseuille_centerline():
    """u_max = g H^2 / (8 nu) inverted for nu reproduces (tau - 1/2)/3. The channel is wide
    because the bounce-back slip error, which biases u_max, falls as 1/H^2."""
    for tau in (0.7, 1.2):
        u, _, lb = _channel(tau, ny=82)
        H, g = 80, 1e-6
        center = 0.5 * (u[H // 2 - 1] + u[H // 2])  # the centerline falls between two nodes
        u_max = center / (1 - (0.5 / (H / 2)) ** 2)  # remove the half-node offset from the parabola's peak
        assert g * H**2 / (8 * u_max) == pytest.approx(lbm_viscosity(tau), rel=5e-3)


def test_mass_and_equilibrium_moments():
    rng = np.random.default_rng(0)
    rho = 1 + 0.01 * rng.random((6, 5))
    ux, uy = 0.05 * rng.standard_normal((2, 6, 5))
    feq = lbm_equilibrium(rho, ux, uy)
    assert feq.sum(axis=0) == pytest.approx(rho)
    assert np.tensordot(D2Q9_VELOCITIES[:, 0], feq, axes=1) == pytest.approx(rho * ux)
    assert np.tensordot(D2Q9_VELOCITIES[:, 1], feq, axes=1) == pytest.approx(rho * uy)
    # second moment: Pi_xy = rho ux uy (the c_s^2 delta part vanishes off-diagonal)
    exy = D2Q9_VELOCITIES[:, 0] * D2Q9_VELOCITIES[:, 1]
    assert np.tensordot(exy, feq, axes=1) == pytest.approx(rho * ux * uy)
    assert D2Q9_WEIGHTS.sum() == pytest.approx(1.0)
    solid = np.zeros((20, 12), dtype=bool)
    solid[8:11, 4:7] = True
    lb = LatticeBoltzmannD2Q9(20, 12, 0.7, body_force=(1e-5, 0.0), solid=solid)
    m0 = lb.density[~solid].sum()
    lb.step(500)
    assert lb.density[~solid].sum() == pytest.approx(m0, rel=1e-12)
    assert lb.time == 500


def test_validation():
    assert lbm_relaxation_time(lbm_viscosity(0.9)) == pytest.approx(0.9)
    with pytest.raises(InvalidParameterError):
        lbm_viscosity(0.5)
    with pytest.raises(InvalidParameterError):
        lbm_relaxation_time(0.0)
    with pytest.raises(InvalidParameterError):
        LatticeBoltzmannD2Q9(4, 4, 1.0, solid=np.zeros((3, 3), dtype=bool))
