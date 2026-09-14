"""Every dynamical demo produces a genuine frame-by-frame animation (not a
single static snapshot), and the underlying physics is spot-checked against
known behavior (spreading, tunneling, splitting, entanglement growth, ...).
"""

import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from physicskit.quantum._compat import trapz
from physicskit.quantum.chapters.entanglement import IsingEntangler
from physicskit.quantum.chapters.harmonic_spin import HarmonicOscillator
from physicskit.quantum.chapters.hydrogen_am import HydrogenOrbital, orbital_superposition_density
from physicskit.quantum.chapters.potentials import DoubleWellSimulator, FiniteSquareWell
from physicskit.quantum.chapters.spin import RabiProblem, SternGerlach
from physicskit.quantum.chapters.wave_packets import GaussianDispersion, propagate_double_slit
from physicskit.quantum.visualizers.bloch_sphere import animate_bloch_sphere, state_to_bloch_trajectory
from physicskit.quantum.visualizers.entanglement import animate_entanglement_growth
from physicskit.quantum.visualizers.orbitals import animate_orbital_beating
from physicskit.quantum.visualizers.wavefunctions import animate_density, animate_density_2d


def _save_and_check(anim: FuncAnimation, tmp_path, name: str):
    out = tmp_path / name
    anim.save(out, writer=PillowWriter(fps=10))
    assert out.exists() and out.stat().st_size > 0


# 1. Double-slit: genuine 2D time propagation through a two-gap wall.
def test_double_slit_propagation_animates(tmp_path):
    x = np.linspace(-15, 15, 96)
    y = np.linspace(-12, 12, 96)
    X, Y, frames, times = propagate_double_slit(
        x,
        y,
        x0=-8.0,
        k0=8.0,
        sigma_x=0.7,
        sigma_y=4.0,
        wall_x=0.0,
        n_steps=160,
        save_every=20,
        dt=1e-4,
    )
    assert frames.shape[0] == len(times) >= 5
    norm0 = float(np.sum(np.abs(frames[0]) ** 2)) * (x[1] - x[0]) * (y[1] - y[0])
    norm_end = float(np.sum(np.abs(frames[-1]) ** 2)) * (x[1] - x[0]) * (y[1] - y[0])
    assert abs(norm0 - norm_end) < 1e-2  # probability conserved through the slits

    anim = animate_density_2d(X, Y, frames, times=times)
    _save_and_check(anim, tmp_path, "double_slit.gif")


# 2. Quantum tunnelling: a wavepacket incident on a barrier splits into
# reflected + transmitted pieces.
def test_barrier_tunnelling_animates(tmp_path):
    barrier = FiniteSquareWell(V0=-6.0, width=1.0)
    x, frames, times = barrier.wavepacket_scattering(
        x_extent=25.0,
        n_points=1024,
        x0=-10.0,
        sigma0=1.5,
        k0=None,
        dt=1e-3,
        n_steps=5000,
        save_every=500,
    )
    assert frames.shape[0] == len(times) >= 5

    dx = x[1] - x[0]
    final_density = np.abs(frames[-1]) ** 2
    transmitted = float(np.sum(final_density[x > 0.5]) * dx)
    reflected = float(np.sum(final_density[x < -0.5]) * dx)
    assert transmitted > 0.01  # some probability tunnels through
    assert reflected > 0.01  # some probability reflects
    assert abs((transmitted + reflected) - 1.0) < 0.2

    anim = animate_density(x, frames, times=times)
    _save_and_check(anim, tmp_path, "barrier_tunnelling.gif")


# 3. Hydrogen orbitals: a two-eigenstate superposition beats in density,
# unlike either stationary orbital alone.
def test_hydrogen_orbital_beating_animates():
    orb_a = HydrogenOrbital(2, 0, 0)
    orb_b = HydrogenOrbital(3, 1, 0)
    period = 2 * np.pi / (orb_b.energy - orb_a.energy)
    times = np.linspace(0, abs(period), 4)

    fig = animate_orbital_beating(orb_a, orb_b, times, n_points=12)
    assert len(fig.frames) == len(times)
    assert fig.data[0].value.size > 0
    assert any(m["buttons"] for m in fig.layout.updatemenus)  # a Play button exists

    r = np.linspace(0.1, 15, 20)
    theta = np.full_like(r, np.pi / 3)
    phi = np.zeros_like(r)
    density_0 = orbital_superposition_density(orb_a, orb_b, r, theta, phi, 0.0)
    density_half = orbital_superposition_density(orb_a, orb_b, r, theta, phi, abs(period) / 2)
    assert np.max(np.abs(density_0 - density_half)) > 1e-6  # genuinely time-dependent


# 4. Harmonic-oscillator eigenstate superposition: plumbing onto animate_density.
def test_harmonic_superposition_animates(tmp_path):
    ho = HarmonicOscillator()
    x = np.linspace(-10, 10, 400)
    times = np.linspace(0, 2 * np.pi / ho.omega, 12)
    frames = ho.superposition_trajectory([0, 1, 2], [1.0, 1.0j, 0.5], x, times)

    for psi in frames:
        assert abs(trapz(np.abs(psi) ** 2, x) - 1.0) < 1e-6

    anim = animate_density(x, frames, times=times)
    _save_and_check(anim, tmp_path, "harmonic_superposition.gif")


# 5. Free wave-packet dispersion: plumbing onto animate_density.
def test_wave_packet_dispersion_animates(tmp_path):
    gd = GaussianDispersion(x0=0.0, sigma0=1.0, k0=3.0)
    x = np.linspace(-30, 30, 600)
    times = np.linspace(0, 6.0, 10)
    frames = gd.trajectory(x, times)

    widths = [gd.width(t) for t in times]
    assert widths[-1] > widths[0]  # the packet spreads

    anim = animate_density(x, frames, times=times)
    _save_and_check(anim, tmp_path, "wave_packet_dispersion.gif")


# 6. Scattering off a finite well: resonance/reflection dynamics, sharing
# FiniteSquareWell.wavepacket_scattering with the barrier case above.
def test_well_scattering_animates(tmp_path):
    well = FiniteSquareWell(V0=20.0, width=2.0)
    x, frames, times = well.wavepacket_scattering(
        x_extent=25.0,
        n_points=1024,
        x0=-10.0,
        sigma0=1.5,
        k0=None,
        dt=1e-3,
        n_steps=5000,
        save_every=500,
    )
    assert frames.shape[0] == len(times) >= 5

    anim = animate_density(x, frames, times=times)
    _save_and_check(anim, tmp_path, "well_scattering.gif")


# 7. Stern-Gerlach: the joint spatial density splits into two lobes.
def test_stern_gerlach_splitting_animates(tmp_path):
    sg = SternGerlach(mu=2.0, grad_B=1.0, sigma0=1.0, sigma_x=2.0, k0=4.0)
    x = np.linspace(-15, 15, 60)
    y = np.linspace(-15, 15, 80)
    times = np.array([0.0, 1.0, 2.5])
    stack = sg.joint_density_stack(x, y, times)
    assert stack.shape == (len(times), len(x), len(y))

    def y_spread(density_xy):
        marginal_y = density_xy.sum(axis=0)
        marginal_y /= marginal_y.sum()
        mean_y = np.sum(y * marginal_y)
        return np.sqrt(np.sum((y - mean_y) ** 2 * marginal_y))

    assert y_spread(stack[-1]) > y_spread(stack[0])  # splits apart over time

    anim = animate_density_2d(x, y, stack, times=times, phase_colored=False)
    _save_and_check(anim, tmp_path, "stern_gerlach.gif")


# 8. Rabi oscillations on the Bloch sphere.
def test_rabi_bloch_sphere_animates(tmp_path):
    rabi = RabiProblem(omega0=1.0, omega_d=1.0, Omega=0.5)  # on resonance
    t_pi = np.pi / rabi.Omega  # a resonant pi-pulse fully inverts the population
    assert rabi.excited_state_population(np.array([t_pi]))[0] > 0.999

    times = np.linspace(0, t_pi, 20)
    states = rabi.state_trajectory(times)
    assert np.allclose(np.sum(np.abs(states) ** 2, axis=1), 1.0, atol=1e-10)

    trajectory = state_to_bloch_trajectory(states)
    assert np.allclose(np.linalg.norm(trajectory, axis=1), 1.0, atol=1e-8)

    anim = animate_bloch_sphere(trajectory, times=times)
    _save_and_check(anim, tmp_path, "rabi_bloch.gif")


# 9. Entangling Ising coupling: concurrence grows from 0 toward 1.
def test_entanglement_growth_animates(tmp_path):
    ising = IsingEntangler(J=1.0)
    assert abs(ising.concurrence(ising.initial_state())) < 1e-12  # starts unentangled

    times = np.linspace(0, np.pi / (2 * ising.J), 30)
    concurrence = ising.concurrence_trajectory(times)
    assert concurrence[0] < 1e-6
    assert concurrence.max() > 0.99  # reaches (near-)maximal entanglement

    anim = animate_entanglement_growth(times, concurrence)
    _save_and_check(anim, tmp_path, "entanglement_growth.gif")


# 10. Double-well tunneling oscillation: almost pure plumbing onto animate_density.
def test_double_well_tunneling_animates(tmp_path):
    dw = DoubleWellSimulator(lam=0.3, a=1.5)
    result = dw.solve(n_states=2)
    times = np.linspace(0, result.tunneling_period, 16)
    frames = dw.tunneling_wavefunction(result, times, side="left")
    assert frames.shape == (len(times), len(dw.x))

    left_prob = dw.left_well_probability(result, times, side="left")
    assert left_prob[0] > 0.9  # starts localized on the left
    assert left_prob[np.argmin(np.abs(times - result.tunneling_period / 2))] < 0.5  # tunnels to the right

    anim = animate_density(dw.x, frames, times=times)
    _save_and_check(anim, tmp_path, "double_well.gif")
