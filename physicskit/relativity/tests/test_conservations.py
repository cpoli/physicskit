import numpy as np

from physicskit.relativity.chapters.kerr import KerrBlackHole
from physicskit.relativity.chapters.schwarzschild import SchwarzschildBlackHole


def test_circular_orbit_energy_and_angular_momentum_conserved():
    bh = SchwarzschildBlackHole(M=1.0)
    y0 = bh.circular_orbit_initial_state(r0=10.0)
    traj = bh.integrate_geodesic(y0, dtau=0.02, n_steps=3000)
    E0, L0 = traj["energy"][0], traj["angular_momentum"][0]
    assert np.allclose(traj["energy"], E0, rtol=1e-5, atol=1e-5)
    assert np.allclose(traj["angular_momentum"], L0, rtol=1e-5, atol=1e-5)


def test_eccentric_orbit_energy_and_angular_momentum_conserved():
    bh = SchwarzschildBlackHole(M=1.0)
    y0 = bh.eccentric_orbit_initial_state(r0=20.0, eccentricity_boost=0.15)
    traj = bh.integrate_geodesic(y0, dtau=0.02, n_steps=20000)
    E0, L0 = traj["energy"][0], traj["angular_momentum"][0]
    assert np.allclose(traj["energy"], E0, rtol=1e-5, atol=1e-5)
    assert np.allclose(traj["angular_momentum"], L0, rtol=1e-5, atol=1e-5)
    # a genuinely eccentric orbit should show radial oscillation
    assert traj["r"].max() - traj["r"].min() > 1.0


def test_null_geodesic_energy_and_angular_momentum_conserved():
    bh = SchwarzschildBlackHole(M=1.0)
    y0 = bh.null_geodesic_initial_state(r0=500.0, impact_parameter=20.0, ingoing=True)
    traj = bh.integrate_geodesic(y0, dtau=0.5, n_steps=5000)
    E0, L0 = traj["energy"][0], traj["angular_momentum"][0]
    assert np.allclose(traj["energy"], E0, rtol=1e-5, atol=1e-5)
    assert np.allclose(traj["angular_momentum"], L0, rtol=1e-5, atol=1e-5)


def test_kerr_equatorial_circular_orbit_four_velocity_normalized():
    bh = KerrBlackHole(M=1.0, a=0.7)
    E, L = bh.circular_orbit_conserved_quantities(r0=10.0, prograde=True)
    result = bh.integrate_equatorial_geodesic(r0=10.0, E=E, L=L, mu2=1.0, dtau=0.02, n_steps=3000)
    assert np.allclose(result["norm"], -1.0, atol=1e-5)


def test_kerr_equatorial_eccentric_orbit_four_velocity_normalized():
    bh = KerrBlackHole(M=1.0, a=0.7)
    E, L = bh.circular_orbit_conserved_quantities(r0=10.0, prograde=True)
    result = bh.integrate_equatorial_geodesic(r0=10.0, E=E, L=L * 0.98, mu2=1.0, dtau=0.02, n_steps=6000)
    assert np.allclose(result["norm"], -1.0, atol=1e-4)
    assert result["r"].max() - result["r"].min() > 0.5


def test_kerr_equatorial_null_geodesic_four_velocity_normalized():
    bh = KerrBlackHole(M=1.0, a=0.5)
    result = bh.integrate_equatorial_geodesic(r0=500.0, E=1.0, L=15.0, mu2=0.0, dtau=0.5, n_steps=5000)
    assert np.allclose(result["norm"], 0.0, atol=1e-4)
