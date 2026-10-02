import numpy as np
import pytest
from matplotlib.path import Path

from physicskit.relativity.chapters.kerr import KerrBlackHole
from physicskit.relativity.chapters.kerr_newman import KerrNewmanBlackHole


def test_horizons_reduce_to_known_limits():
    assert KerrNewmanBlackHole().outer_horizon_radius == pytest.approx(2.0)
    assert KerrNewmanBlackHole(Q=0.6).outer_horizon_radius == pytest.approx(1.8)  # RN: M + sqrt(M^2 - Q^2)
    assert KerrNewmanBlackHole(a=0.6).outer_horizon_radius == pytest.approx(KerrBlackHole(a=0.6).outer_horizon_radius)
    bh = KerrNewmanBlackHole(a=0.3, Q=0.4)
    assert bh.delta(bh.outer_horizon_radius) == pytest.approx(0.0, abs=1e-12)
    assert bh.delta(bh.inner_horizon_radius) == pytest.approx(0.0, abs=1e-12)


def test_naked_singularity_rejected():
    with pytest.raises(ValueError):
        KerrNewmanBlackHole(a=0.8, Q=0.7)


def test_reissner_nordstrom_photon_sphere_and_shadow():
    bh = KerrNewmanBlackHole(Q=0.5)
    r = bh.photon_sphere_radius_nonrotating()
    # dV/dr = 0 for V = Delta / r^4  <=>  r^2 - 3 M r + 2 Q^2 = 0
    assert r * r - 3 * r + 2 * 0.25 == pytest.approx(0.0, abs=1e-12)
    assert bh.shadow_radius_nonrotating() == pytest.approx(r * r / np.sqrt(bh.delta(r)))
    assert KerrNewmanBlackHole().shadow_radius_nonrotating() == pytest.approx(np.sqrt(27))


@pytest.mark.parametrize("a", [0.3, 0.9])
def test_kerr_limit_equatorial_photon_orbits(a):
    pro, retro = KerrNewmanBlackHole(a=a).equatorial_photon_orbits()
    assert pro == pytest.approx(2 * (1 + np.cos(2 / 3 * np.arccos(-a))))
    assert retro == pytest.approx(2 * (1 + np.cos(2 / 3 * np.arccos(a))))


def test_small_spin_shadow_approaches_reissner_nordstrom_disc():
    bh = KerrNewmanBlackHole(a=1e-3, Q=0.5)
    assert bh.shadow_area() == pytest.approx(np.pi * KerrNewmanBlackHole(Q=0.5).shadow_radius_nonrotating() ** 2, rel=1e-4)


def test_charge_shrinks_the_shadow():
    areas = [KerrNewmanBlackHole(a=0.5, Q=q).shadow_area() for q in (0.0, 0.4, 0.8)]
    assert areas[0] > areas[1] > areas[2]


def test_photon_orbit_constants_satisfy_R_and_dR_zero():
    bh = KerrNewmanBlackHole(a=0.7, Q=0.4)
    r = 2.6
    xi, eta = bh.photon_orbit_constants(r)
    D = bh.delta(r)
    A = r * r + bh.a**2 - bh.a * xi
    assert A * A - D * (eta + (xi - bh.a) ** 2) == pytest.approx(0.0, abs=1e-9)
    assert 4 * r * A - (2 * r - 2) * (eta + (xi - bh.a) ** 2) == pytest.approx(0.0, abs=1e-9)


def test_spherical_photon_orbit_keeps_its_radius_and_oscillates_in_theta():
    bh = KerrNewmanBlackHole(a=0.7, Q=0.4)
    r1, r2 = bh.equatorial_photon_orbits()
    r = 0.5 * (r1 + r2)
    xi, eta = bh.photon_orbit_constants(r)
    g = bh.geodesic(r, np.pi / 2, 1.0, xi, eta, n_steps=3000, step=0.002)
    np.testing.assert_allclose(g["r"], r, rtol=1e-9)
    assert g["theta"].min() < 0.5 and g["theta"].max() > np.pi - 0.5


def test_massive_bound_orbit_conserves_constants_and_stays_bound():
    bh = KerrNewmanBlackHole(a=0.5, Q=0.3)
    g = bh.geodesic(12.0, np.pi / 2, 0.97, 3.9, 2.0, mu2=1.0, n_steps=40000, step=0.002, record_every=50)
    assert g["outcome"] == 2  # neither captured nor escaped
    assert 4.0 < g["r"].min() and g["r"].max() < 40.0


def test_ray_traced_shadow_matches_analytic_boundary():
    bh = KerrNewmanBlackHole(a=0.9, Q=0.3)
    x = np.linspace(-7, 7, 29)
    img = bh.ray_traced_shadow(x, x)
    al, be = bh.shadow_boundary()
    X, Y = np.meshgrid(x, x)
    inside = Path(np.column_stack((al, be))).contains_points(np.column_stack((X.ravel(), Y.ravel()))).reshape(X.shape)
    mismatch = (img == 1) != inside
    assert mismatch.mean() < 0.03
