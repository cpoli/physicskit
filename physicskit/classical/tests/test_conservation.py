"""Verify |H(t) - H(0)| / |H(0)| < 1e-6 over 10**5 integration steps for
every conservative system in physicskit.classical, using each system's default
symplectic integrator (Yoshida4 for separable Hamiltonians,
implicit-midpoint for non-separable Lagrangian/rigid-body systems).

Each system's timestep below was chosen (see the module-level
``N_STEPS``/tolerance) so that the discretization error -- which for a
2nd/4th-order symplectic method scales as a fixed power of dt -- comfortably
clears the 1e-6 relative-drift bar over exactly 10**5 steps; see
individual test docstrings for the physical timescale each dt resolves.
"""

from __future__ import annotations

import numpy as np
import pytest

from physicskit.classical.systems.chains import FPUTChain, HarmonicChain, SineGordonChain
from physicskit.classical.systems.hamiltonian import HenonHeilesSystem, PendulumSwarm
from physicskit.classical.systems.lagrangian import BeadOnRotatingHoop, CoupledOscillators, DoublePendulum, ElasticPendulum
from physicskit.classical.systems.newtonian import FoucaultPendulum, KeplerSystem, ProjectileMotion
from physicskit.classical.systems.rotations import EulerTop, HeavySymmetricTop
from physicskit.classical.utils.conservation import lrl_drift, relative_energy_drift

N_STEPS = 100_000
TOL = 1e-6


def _assert_conserved(system, dt, method, n_steps=N_STEPS, tol=TOL):
    result = system.integrate((0.0, n_steps * dt), dt=dt, method=method)
    assert len(result.t) - 1 == n_steps
    drift = relative_energy_drift(result.energy)
    assert np.max(drift) < tol, f"energy drift {np.max(drift):.3e} exceeds tolerance {tol:.1e}"
    return result


def test_projectile_motion_conserves_energy():
    """Free fall with an oblique launch velocity (the "cannonball"
    problem), unconstrained over 1e5 steps (~100s of flight, well past
    where a real cannonball would have hit the ground -- gravity alone
    is still exactly conservative for arbitrarily long unconstrained
    free fall)."""
    system = ProjectileMotion.from_launch(speed=20.0, angle_deg=45.0, g=9.81)
    _assert_conserved(system, dt=1e-3, method="yoshida4")


def test_projectile_motion_matches_analytic_parabola():
    """Over the ~2s a real cannonball is actually airborne, a constant
    force is exactly separable, so the symplectic integrator should
    reproduce Galileo's closed-form parabola to floating-point
    precision, not merely to some small tolerance."""
    system = ProjectileMotion.from_launch(speed=20.0, angle_deg=45.0, g=9.81)
    result = system.integrate((0.0, 2.0), dt=1e-3, method="yoshida4")
    x_analytic, y_analytic = ProjectileMotion.analytic_trajectory(20.0, 45.0, 9.81, result.t)
    assert np.max(np.abs(result.q[:, 0] - x_analytic)) < 1e-9
    assert np.max(np.abs(result.q[:, 1] - y_analytic)) < 1e-9


def test_projectile_motion_with_drag_dissipates_energy_and_rejects_symplectic_methods():
    """Quadratic air drag makes the system genuinely non-conservative:
    energy must decrease monotonically, and requesting a symplectic
    method that can't see the (velocity-dependent) drag force must be
    refused rather than silently ignoring it."""
    system = ProjectileMotion.from_launch(speed=20.0, angle_deg=45.0, g=9.81, drag_coeff=0.05)
    with pytest.raises(ValueError):
        system.integrate((0.0, 1.0), dt=1e-3, method="yoshida4")
    result = system.integrate((0.0, 2.0), dt=1e-3, method="rk4")
    assert np.all(np.diff(result.energy) <= 1e-9)
    assert result.energy[-1] < result.energy[0]


def test_kepler_pure_conserves_energy_and_lrl():
    """Pure 1/r Kepler orbit: energy AND the LRL vector should both be
    conserved (no precession without a perturbation)."""
    system = KeplerSystem.from_orbital_elements(a=1.0, e=0.5, k=1.0, mu=1.0)
    lrl0 = system.lrl_vector()
    result = _assert_conserved(system, dt=1e-3, method="yoshida4")
    drift = lrl_drift(system, result.q, result.p)
    assert np.max(drift) / np.linalg.norm(lrl0) < 1e-4


def test_kepler_perturbed_conserves_energy():
    """With the post-Newtonian 1/r**3 correction switched on, the orbit
    precesses (LRL direction rotates) but total energy remains conserved."""
    system = KeplerSystem.from_orbital_elements(a=1.0, e=0.3, k=1.0, mu=1.0, c_pn=0.01)
    _assert_conserved(system, dt=1e-3, method="yoshida4")


def test_foucault_pendulum_conserves_energy():
    """The Coriolis term is velocity-perpendicular and does no work, so
    energy is exactly conserved even as the swing plane precesses;
    implicit midpoint preserves this quadratic invariant to machine
    precision for this linear system."""
    system = FoucaultPendulum.from_deflection(amplitude=1.0, latitude_deg=48.85)
    _assert_conserved(system, dt=1e-3, method="implicit_midpoint")


def test_foucault_pendulum_matches_analytic_precession():
    """Compare the numerical trajectory against the closed-form
    small-oscillation solution, and check the swing plane is unwound
    (stays near y=0) in the frame corotating at ``omega_z``."""
    system = FoucaultPendulum.from_deflection(amplitude=1.0, latitude_deg=48.85)
    result = system.integrate((0.0, 200.0), dt=1e-3, method="implicit_midpoint")
    x_analytic, y_analytic = FoucaultPendulum.analytic_solution([1.0, 0.0], [0.0, 0.0], system.omega0, system.omega_z, result.t)
    assert np.max(np.abs(result.y[:, 0] - x_analytic)) < 1e-5
    assert np.max(np.abs(result.y[:, 1] - y_analytic)) < 1e-5

    x_rot, y_rot = FoucaultPendulum.to_corotating_frame(result.y[:, 0], result.y[:, 1], result.t, system.omega_z)
    assert np.max(np.abs(y_rot)) < 1e-3


def test_double_pendulum_conserves_energy():
    """Chaotic, non-separable system; validated via the implicit-midpoint
    symplectic integrator on the Legendre-transformed Hamiltonian."""
    system = DoublePendulum([2.0, 1.0], [0.5, -0.3])
    _assert_conserved(system, dt=3e-5, method="implicit_midpoint")


def test_bead_on_rotating_hoop_conserves_energy():
    system = BeadOnRotatingHoop(0.3, 0.0, omega=3.0)
    _assert_conserved(system, dt=2.2e-5, method="implicit_midpoint")


def test_coupled_oscillators_conserve_energy():
    system = CoupledOscillators([0.1, -0.2, 0.15], [0.0, 0.0, 0.0], n=3)
    _assert_conserved(system, dt=1e-4, method="implicit_midpoint")


def test_henon_heiles_conserves_energy():
    system = HenonHeilesSystem(np.array([0.0, 0.3]), np.array([0.3, 0.0]))
    _assert_conserved(system, dt=5e-3, method="yoshida4")


def test_elastic_pendulum_conserves_energy():
    """2-DOF nonlinear (stretch, swing) Lagrangian system; validated via
    implicit midpoint on the Legendre-transformed Hamiltonian, same as
    DoublePendulum."""
    system = ElasticPendulum([0.1, 0.05], [0.0, 0.0])
    _assert_conserved(system, dt=2e-5, method="implicit_midpoint")


def test_pendulum_swarm_conserves_energy_and_liouville_area():
    """Every particle in the swarm independently conserves energy over
    the full 1e5-step run, and -- checked separately over a shorter
    window, before the patch shears into a filament thin enough that a
    convex-hull area estimate starts overstating it -- the occupied
    phase-space area (a proxy for Liouville's theorem) is preserved."""
    system = PendulumSwarm.from_box(1.0, 0.0, 0.05, 0.05, n=200, seed=1)
    _assert_conserved(system, dt=1e-3, method="yoshida4")

    system2 = PendulumSwarm.from_box(1.0, 0.0, 0.05, 0.05, n=200, seed=1)
    area0 = system2.phase_space_area()
    system2.integrate((0.0, 5.0), dt=1e-3, method="yoshida4")
    area1 = system2.phase_space_area()
    assert abs(area1 - area0) / area0 < 0.05


def test_harmonic_chain_conserves_energy():
    rng = np.random.default_rng(0)
    n = 16
    system = HarmonicChain(rng.normal(0, 0.1, n), np.zeros(n), k=1.0)
    _assert_conserved(system, dt=1e-2, method="yoshida4")


def test_fput_chain_conserves_energy():
    """N=32 Fermi-Pasta-Ulam-Tsingou beta-lattice."""
    system = FPUTChain(n=32, beta=0.7, mode=1, amplitude=0.5)
    _assert_conserved(system, dt=5e-2, method="yoshida4")


def test_fput_chain_shows_mode_recurrence():
    """The energy initially placed in the first normal mode should
    remain the dominant mode throughout (the FPUT non-ergodicity
    signature), not spread evenly across all 32 modes."""
    system = FPUTChain(n=32, beta=0.7, mode=1, amplitude=0.5)
    e0 = system.energy()
    result = system.integrate((0.0, 5000.0), dt=5e-2, method="yoshida4")
    energies = system.modal_energies(result.q[-1], result.p[-1])
    assert energies[0] / e0 > 0.5


def test_sine_gordon_chain_conserves_energy():
    q0, p0 = SineGordonChain.kink(100, center=50, width=1.0, velocity=0.2)
    system = SineGordonChain(n=100, q0=q0, p0=p0)
    _assert_conserved(system, dt=1e-2, method="yoshida4")


def test_euler_top_conserves_energy_and_angular_momentum():
    """Torque-free asymmetric top: both energy and |L|^2 are quadratic
    invariants exactly preserved by implicit midpoint."""
    system = EulerTop([0.01, 1.0, 0.01], I1=1.0, I2=2.0, I3=3.0)
    L0 = system.angular_momentum_squared()
    _assert_conserved(system, dt=1e-3, method="implicit_midpoint")
    L1 = system.angular_momentum_squared()
    assert abs(L1 - L0) / L0 < 1e-6


def test_euler_top_intermediate_axis_instability():
    """Spinning (almost) exactly about the intermediate-inertia axis
    (I1 < I2 < I3) must show the tumbling instability: the other two
    body-frame components grow far beyond their tiny initial values."""
    system = EulerTop([0.01, 1.0, 0.01], I1=1.0, I2=2.0, I3=3.0)
    result = system.integrate((0.0, 100.0), dt=1e-3, method="implicit_midpoint")
    assert np.max(np.abs(result.y[:, 0])) > 0.5  # w1 tumbles far above its 0.01 start


def test_euler_top_stable_axis_no_tumbling():
    """Spinning about the smallest-inertia axis is stable: the small
    transverse perturbations stay small for the whole integration."""
    system = EulerTop([1.0, 0.01, 0.01], I1=1.0, I2=2.0, I3=3.0)
    result = system.integrate((0.0, 100.0), dt=1e-3, method="implicit_midpoint")
    assert np.max(np.abs(result.y[:, 1])) < 0.1
    assert np.max(np.abs(result.y[:, 2])) < 0.1


def test_heavy_symmetric_top_conserves_energy():
    system = HeavySymmetricTop([0.0, 0.5, 0.0], [0.0, 0.0, 20.0], I1=1.0, I3=0.5, M=1.0, l=1.0, g=9.81)
    _assert_conserved(system, dt=5e-5, method="implicit_midpoint")


@pytest.mark.parametrize(
    "system_factory,dt",
    [
        (lambda: KeplerSystem.from_orbital_elements(a=1.0, e=0.5), 1e-3),
        (lambda: HenonHeilesSystem(np.array([0.0, 0.3]), np.array([0.3, 0.0])), 5e-3),
    ],
)
def test_verlet_and_yoshida4_agree_to_leading_order(system_factory, dt):
    """Sanity check that the 2nd-order Verlet and 4th-order Yoshida4
    backends integrate the same separable system to consistent, small
    energy drift (Yoshida4 should be at least as good as Verlet)."""
    sys_v = system_factory()
    sys_y = system_factory()
    res_v = sys_v.integrate((0.0, 2000 * dt), dt=dt, method="verlet")
    res_y = sys_y.integrate((0.0, 2000 * dt), dt=dt, method="yoshida4")
    drift_v = np.max(relative_energy_drift(res_v.energy))
    drift_y = np.max(relative_energy_drift(res_y.energy))
    assert drift_y <= drift_v * 10  # Yoshida4 is not meaningfully worse than Verlet
