"""Regenerate the README's per-subpackage teaser figures.

    python docs/make_readme_subpackage_figures.py              # all subpackages
    python docs/make_readme_subpackage_figures.py astro chaos  # just these

Writes docs/source/_static/images/readme_<subpackage>.png, three panels each,
at the same size as readme_hero.png (see make_readme_figure.py). README.md
embeds them by their raw.githubusercontent.com URLs so they also render on PyPI.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

OUT_DIR = Path(__file__).parent / "source" / "_static" / "images"


def _replace_with_3d(ax):
    """Swap a 2D axes for a 3D one in the same grid slot."""
    fig, spec = ax.figure, ax.get_subplotspec()
    ax.remove()
    return fig.add_subplot(spec, projection="3d")


def hofstadter_butterfly(ax, q_max=61):
    """Harper-Hofstadter band edges vs. rational flux p/q per plaquette."""
    from math import gcd

    from physicskit.condensed.models import harper_hofstadter_hamiltonian

    for q in range(2, q_max + 1):
        for p in range(1, q):
            if gcd(p, q) != 1:
                continue
            # Sample the magnetic Brillouin zone and fill each band between its extremes.
            bands = np.array(
                [np.linalg.eigvalsh(harper_hofstadter_hamiltonian(k1, k2, p, q)) for k1 in np.linspace(0, np.pi, 3) for k2 in np.linspace(0, np.pi / q, 3)]
            )
            ax.vlines(np.full(q, p / q), bands.min(axis=0), bands.max(axis=0), color="black", lw=0.6)
            ax.plot(np.full(bands.size, p / q), bands.ravel(), ",", color="black")  # exponentially narrow bands
    ax.set_xlabel("flux quanta per plaquette")
    ax.set_ylabel("energy")
    ax.set_title("The Hofstadter butterfly")


def black_hole_shadow(ax, n_pixels=200):
    """Ray-traced image of a Schwarzschild black hole and its accretion disk."""
    from physicskit.relativity.visualizers.shadow_render import plot_black_hole_shadow, render_black_hole_image

    plot_black_hole_shadow(render_black_hole_image(M=1.0, ny=n_pixels, nx=n_pixels, inclination=1.3), ax=ax)
    ax.set_title("A black hole's shadow, ray-traced")


def kelvin_helmholtz(ax, n=128):
    """Roll-up of a perturbed shear layer into vortices (2D Navier-Stokes)."""
    from physicskit.fluids.systems.instabilities import kelvin_helmholtz_ic
    from physicskit.fluids.systems.navier_stokes import NavierStokes2D
    from physicskit.fluids.visualizers.flow_fields import plot_vorticity_field

    solver = NavierStokes2D(n=n, length=2 * np.pi, nu=0.0005)
    omega = solver.simulate(kelvin_helmholtz_ic(n, 2 * np.pi, shear_width=0.15, perturbation_amplitude=0.1), dt=0.002, steps=3000)["omega"]
    plot_vorticity_field(solver.X, solver.Y, omega, ax=ax)
    ax.set_title("Kelvin-Helmholtz roll-up of a shear layer")


def astro(axes):
    from physicskit.astro.galactic_dynamics import circular_velocity, nfw_enclosed_mass
    from physicskit.astro.nbody import NBodySystem, figure_eight_initial_conditions
    from physicskit.astro.stellar_structure import lane_emden
    from physicskit.astro.visualizers import plot_nbody_trajectories

    ax1, ax2, ax3 = axes
    system = NBodySystem(*figure_eight_initial_conditions())
    plot_nbody_trajectories(system.simulate(0.002, 3163), ax=ax1)  # one period, T = 6.326
    ax1.set_title("The figure-eight three-body choreography")

    for n in (0.0, 1.0, 1.5, 3.0, 4.0):
        xi, theta = lane_emden(n)
        ax2.plot(xi, theta, label=f"n = {n}")
    ax2.axhline(0.0, color="0.7", lw=0.8)
    ax2.set_xlabel(r"$\xi$")
    ax2.set_ylabel(r"$\theta(\xi)$")
    ax2.set_title("Lane-Emden polytropes: stellar structure")
    ax2.legend(fontsize=8)

    def disk_mass(r):
        x = np.asarray(r, dtype=float) / 3.0
        return 60.0 * (1.0 - (1.0 + x) * np.exp(-x))

    def halo_mass(r):
        return nfw_enclosed_mass(r, 0.03, 12.0)

    r = np.linspace(0.3, 32.0, 300)
    r_obs = np.linspace(1.0, 30.0, 25)
    v_obs = circular_velocity(r_obs, lambda x: disk_mass(x) + halo_mass(x))
    v_obs = v_obs * (1 + np.random.default_rng(1970).normal(0.0, 0.03, r_obs.size))
    ax3.plot(r, circular_velocity(r, lambda x: disk_mass(x) + halo_mass(x)), color="steelblue", label="disk + dark halo")
    ax3.errorbar(r_obs, v_obs, yerr=0.03 * v_obs, fmt="o", color="black", ms=3, label="observed")
    ax3.plot(r, circular_velocity(r, disk_mass), "--", color="firebrick", label="visible disk alone")
    ax3.plot(r, circular_velocity(r, halo_mass), ":", color="darkorchid", label="dark halo")
    ax3.set_xlabel("radius r")
    ax3.set_ylabel(r"$v_c$")
    ax3.set_title("Flat rotation curves: dark matter")
    ax3.legend(fontsize=8, loc="lower right")


def chaos(axes):
    from physicskit.chaos.systems.billiards import BunimovichStadium
    from physicskit.chaos.systems.continuous import Lorenz
    from physicskit.chaos.systems.maps import StandardMap
    from physicskit.chaos.visualizers.phase_space import plot_billiard_trajectory

    ax1, ax2, ax3 = axes
    stadium = BunimovichStadium(radius=1.0, straight_length=2.0)
    plot_billiard_trajectory(stadium, pos=np.array([0.1, 0.2]), vel=np.array([0.5, 0.9]), n_bounces=100, ax=ax1)
    ax1.set_title("Bunimovich stadium: a chaotic billiard")

    _, states = Lorenz(sigma=10.0, rho=28.0, beta=8.0 / 3.0).trajectory(n_steps=20000, dt=0.01)
    ax2.plot(states[500:, 0], states[500:, 2], lw=0.3, color="darkorange")
    ax2.set_xlabel("x")
    ax2.set_ylabel("z")
    ax2.set_title("The Lorenz attractor")

    standard_map = StandardMap(k=1.2)
    for p0 in np.linspace(0.05, np.pi - 0.05, 40):
        orbit = standard_map.trajectory(np.array([0.0, p0]), n_iter=400)
        ax3.plot(orbit[:, 0], orbit[:, 1], ",", color="black", alpha=0.6)
    ax3.set_xlim(0.0, 2.0 * np.pi)
    ax3.set_ylim(0.0, 2.0 * np.pi)
    ax3.set_xlabel(r"$\theta$")
    ax3.set_ylabel(r"$p$")
    ax3.set_title("Chirikov's standard map: KAM islands in a chaotic sea")


def classical(axes):
    from physicskit.classical.systems.chains import FPUTChain
    from physicskit.classical.systems.hamiltonian import HenonHeilesSystem
    from physicskit.classical.systems.rotations import EulerTop
    from physicskit.classical.visualizers.phase_space import plot_poincare_section

    ax1, ax2, ax3 = axes
    chain = FPUTChain(n=32, beta=1.0, mode=1, amplitude=1.5)
    result = chain.integrate((0, 9500), dt=0.02, method="yoshida4")
    modal = np.array([chain.modal_energies(q, p) for q, p in zip(result.q[::200], result.p[::200], strict=True)])
    for k in range(5):
        ax1.plot(result.t[::200], modal[:, k], label=f"mode {k + 1}")
    ax1.set_xlabel("t")
    ax1.set_ylabel(r"$E_k(t)$")
    ax1.set_title("The Fermi-Pasta-Ulam-Tsingou recurrence")
    ax1.legend(fontsize=8)

    rng = np.random.default_rng(0)
    E = 0.1167
    for x0 in np.linspace(-0.3, 0.3, 12):
        px0 = np.sqrt(2 * E - x0**2) * rng.choice([-1, 1]) * 0.6
        py0 = np.sqrt(max(2 * E - x0**2 - px0**2, 1e-6))
        res = HenonHeilesSystem(np.array([x0, 0.0]), np.array([px0, py0])).integrate((0, 2000), dt=0.01, method="yoshida4")
        plot_poincare_section(res.q[:, 0], res.p[:, 0], res.q[:, 1], res.p[:, 1], value=0.0, direction=1, ax=ax2, color="steelblue")
    ax2.set_xlabel("x")
    ax2.set_ylabel(r"$p_x$")
    ax2.set_title(f"Hénon-Heiles Poincaré section, E = {E}")

    tumbling = EulerTop(omega0=[0.01, 1.0, 0.01], I1=1.0, I2=2.0, I3=3.0).integrate((0, 60), dt=1e-3, method="implicit_midpoint")
    for i in range(3):
        ax3.plot(tumbling.t, tumbling.y[:, i], label=rf"$\omega_{i + 1}$")
    ax3.set_xlabel("t")
    ax3.set_title("Dzhanibekov effect: spin about the middle axis flips")
    ax3.legend(fontsize=8)


def condensed(axes):
    from physicskit.condensed.models import haldane_lattice_hamiltonian, ssh_lattice_hamiltonian
    from physicskit.condensed.tight_binding import build_finite_cluster, build_ribbon

    ax1, ax2, ax3 = axes
    hofstadter_butterfly(ax1)

    ribbon = build_ribbon(haldane_lattice_hamiltonian(t=1.0, t2=0.2, phi=np.pi / 2, M=0.0), open_direction=1, n_cells=20)
    k = np.linspace(0, 2 * np.pi, 200)
    ax2.plot(k, [np.linalg.eigvalsh(ribbon([kk])) for kk in k], color="C0", lw=0.8)
    ax2.set_xlabel(r"$k_\parallel$")
    ax2.set_ylabel("energy")
    ax2.set_title("Haldane ribbon: chiral edge states cross the gap")

    v_values = np.linspace(0.0, 2.0, 101)
    spectra = np.array([np.linalg.eigvalsh(build_finite_cluster(ssh_lattice_hamiltonian(v=v, w=1.0), n_cells=20)[0]) for v in v_values])
    ax3.plot(v_values, spectra, ",", color="black")
    ax3.plot(v_values, spectra, color="black", lw=0.3)
    ax3.axvline(1.0, color="gray", ls="--", lw=0.8)
    ax3.set_xlabel("intra-cell hopping v / w")
    ax3.set_ylabel("energy")
    ax3.set_title("SSH chain: zero-energy edge modes for v < w")


def fields(axes):
    from physicskit.fields import (
        courant_limit_2d,
        fdtd_2d_tmz_evolve,
        gpe_imprint_vortex,
        harmonic_trap_grid,
        kdv_evolve_frames,
        oscillating_dipole_source,
        plot_bec_phase,
    )

    ax1, ax2, ax3 = axes
    x = np.linspace(-30.0, 30.0, 1024, endpoint=False)
    frames, times = kdv_evolve_frames(6.0 * np.exp(-(((x + 15) / 2.0) ** 2)), x, dt=0.0002, steps_per_frame=8000 // 120, n_frames=120)
    ax1.imshow(frames, extent=(x.min(), x.max(), times.min(), times.max()), origin="lower", aspect="auto", cmap="viridis")
    ax1.set_xlabel("x")
    ax1.set_ylabel("t")
    ax1.set_title("KdV: a bump fissions into a soliton train")

    n = 100
    dt = 0.5 * courant_limit_2d(1e-3, 1e-3)
    zeros = np.zeros((n, n))
    frames, _ = fdtd_2d_tmz_evolve(
        zeros, zeros.copy(), zeros.copy(), np.ones((n, n)), np.ones((n, n)), steps=140, dt=dt, dx=1e-3, dy=1e-3,
        source=oscillating_dipole_source(n // 2, n // 2, amplitude=1.0, freq=3e10), snapshot_stride=2,
    )  # fmt: skip
    vmax = np.abs(frames[-1]).max() * 0.5
    ax2.imshow(frames[-1].T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=(0, 100, 0, 100))
    ax2.set_xlabel("x (mm)")
    ax2.set_ylabel("y (mm)")
    ax2.set_title("FDTD: a dipole radiating on the Yee grid")

    X, Y, *_ = harmonic_trap_grid(64, 12.0)
    psi = gpe_imprint_vortex(np.exp(-0.25 * (X**2 + Y**2)).astype(complex), X, Y, [(-1.0, 0.0)]) * ((X - 1.0) - 1j * Y)
    plot_bec_phase(X, Y, psi, ax=ax3)
    ax3.set_title("BEC phase: a vortex-antivortex pair")


def fluids(axes):
    from physicskit.fluids.systems.potential_flow import flow_past_cylinder
    from physicskit.fluids.systems.vortex_dynamics import PointVortexSystem, von_karman_vortex_street
    from physicskit.fluids.visualizers.flow_fields import plot_streamlines

    ax1, ax2, ax3 = axes
    kelvin_helmholtz(ax1)

    positions0, circulations = von_karman_vortex_street(n_pairs=6, spacing_l=1.0)
    _, trajectory = PointVortexSystem(positions=positions0, circulations=circulations).trajectory(dt=0.01, n_steps=400)
    for i, gamma in enumerate(circulations):
        color = "firebrick" if gamma > 0 else "steelblue"
        ax2.plot(trajectory[:, i, 0], trajectory[:, i, 1], color=color, lw=1.0)
        ax2.plot(*trajectory[-1, i], "o", color=color)
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("A von Kármán vortex street drifting")

    flow = flow_past_cylinder(U_inf=1.0, radius=1.0, circulation=-3.0)
    grid = np.linspace(-3, 3, 300)
    X, Y = np.meshgrid(grid, grid, indexing="ij")
    u, v = flow.velocity(X, Y)
    inside = X**2 + Y**2 < 1.0
    u[inside], v[inside] = np.nan, np.nan
    plot_streamlines(X, Y, u, v, ax=ax3)
    ax3.add_patch(plt.Circle((0, 0), 1.0, color="0.3"))
    ax3.set_aspect("equal")
    ax3.set_title("Kutta-Joukowski lift: a spinning cylinder")


def optics(axes):
    from physicskit.optics.gaussian import GaussianBeam, laguerre_gaussian_mode
    from physicskit.optics.quantum_optics import compute_wigner_function, fock_state
    from physicskit.optics.wave import double_slit_aperture, fraunhofer_diffraction, intensity

    ax1, ax2, ax3 = axes
    N, dx, wavelength, z, width, separation = 512, 0.002, 0.5e-3, 500.0, 0.01, 0.08
    I = intensity(fraunhofer_diffraction(double_slit_aperture((N, N), dx=dx, width=width, separation=separation), wavelength=wavelength, z=z, dx=dx))
    x = (np.arange(N) - N // 2) * (wavelength * z / (N * dx))
    ax1.plot(x, I[N // 2] / I[N // 2].max(), label="double slit")
    ax1.plot(x, np.sinc(width * x / (wavelength * z)) ** 2, "--", color="gray", label="single-slit envelope")
    ax1.set_xlim(-12, 12)
    ax1.set_xlabel("screen position x (mm)")
    ax1.set_ylabel("intensity")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_title("Young's double slit: far-field fringes")

    beam = GaussianBeam(wavelength=0.5e-3, w0=0.5, z0=0.0)
    coords = np.linspace(-2, 2, 300)
    X, Y = np.meshgrid(coords, coords)
    u = laguerre_gaussian_mode(np.hypot(X, Y), np.arctan2(Y, X), 0.0, beam, l=3, p=0)
    brightness = np.abs(u) ** 2 / np.max(np.abs(u) ** 2)
    ax2.imshow(np.angle(u), extent=[-2, 2, -2, 2], cmap="twilight", alpha=np.clip(1.5 * brightness**0.5, 0, 1))
    ax2.set_facecolor("black")
    ax2.set_title("Laguerre-Gauss vortex beam, l = 3 (phase)")

    x = np.linspace(-4, 4, 121)
    W = compute_wigner_function(fock_state(2, 10), x, x)
    vmax = np.abs(W).max()
    ax3.contourf(x, x, W.T, levels=40, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax3.set_aspect("equal")
    ax3.set_xlabel("x")
    ax3.set_ylabel("p")
    ax3.set_title("Wigner function of the Fock state |2⟩")


def particle(axes):
    from scipy.optimize import minimize

    from physicskit.particle.decays import two_body_decay
    from physicskit.particle.kinematics import FourVector, boost_generic, invariant_mass
    from physicskit.particle.nuclear import binding_energy_per_nucleon, semf_binding_energy
    from physicskit.particle.scattering import rutherford_dsigma_domega
    from physicskit.particle.visualizers import plot_differential_cross_section

    ax1, ax2, ax3 = axes
    rng = np.random.default_rng(2012)
    bins = np.linspace(105, 160, 111)
    centres = 0.5 * (bins[1:] + bins[:-1])
    signal = []
    for _ in range(1200):
        g1, g2 = two_body_decay(125.1, 0.0, 0.0, rng.uniform(-1, 1), rng.uniform(0, 2 * np.pi))
        beta = rng.normal(0, 0.35, 3)
        beta *= min(1.0, 0.9 / np.linalg.norm(beta))
        photons = []
        for g in (boost_generic(g1, beta), boost_generic(g2, beta)):
            E = g.E * (1 + 0.012 * rng.standard_normal())
            photons.append(FourVector(E, *(E * g.p_vec / g.p_mag)))
        signal.append(invariant_mass(photons))
    slope = 0.035
    background = bins[0] - np.log(1 - rng.uniform(size=240000) * (1 - np.exp(-slope * (bins[-1] - bins[0])))) / slope
    counts = np.histogram(np.concatenate([signal, background]), bins=bins)[0].astype(float)
    sideband, x = np.abs(centres - 125.1) > 5, centres - bins[0]
    fit = minimize(
        lambda q: np.sum(np.exp(q[0] - q[1] * x[sideband]) - counts[sideband] * (q[0] - q[1] * x[sideband])), [np.log(counts[0]), 0.03], method="Nelder-Mead"
    )
    ax1.errorbar(centres, counts, yerr=np.sqrt(counts), fmt="o", color="k", ms=2, lw=0.8, label="simulated data")
    ax1.plot(centres, np.exp(fit.x[0] - fit.x[1] * x), color="steelblue", label="background fit")
    ax1.set_xlabel(r"$m_{\gamma\gamma}$ (GeV)")
    ax1.set_ylabel("events / 0.5 GeV")
    ax1.set_title(r"The Higgs boson in $H \to \gamma\gamma$")
    ax1.legend(fontsize=8)

    A_values = np.arange(12, 240)
    Z_best = [np.arange(1, A)[np.argmax([semf_binding_energy(Z, A) for Z in range(1, A)])] for A in A_values]
    ax2.plot(A_values, [binding_energy_per_nucleon(Z, A) for Z, A in zip(Z_best, A_values, strict=True)], color="firebrick")
    ax2.set_xlabel("mass number A")
    ax2.set_ylabel("binding energy per nucleon (MeV)")
    ax2.set_title("Weizsäcker's semi-empirical mass formula")

    theta = np.linspace(0.02, np.pi - 0.02, 400)
    plot_differential_cross_section(theta, rutherford_dsigma_domega(theta, 2, 79, 5.0), ax=ax3)
    ax3.set_title(r"Rutherford scattering of 5 MeV $\alpha$ on gold")


def plasma(axes):
    import physicskit as pk

    ax1, ax2, ax3 = axes
    L = 2 * np.pi / 0.3
    result = pk.plasma.pic_simulate(*pk.plasma.two_stream_ic(20000, L=L, v_drift=3.0, v_th=0.5, seed=0), L=L, ng=64, dt=0.05, steps=400)
    pk.plasma.plot_phase_space(result["x"], result["v"], ax=ax1)
    ax1.set_title("Two-stream instability: phase-space vortex (PIC)")

    R, Z = np.linspace(0.5, 1.5, 61), np.linspace(-0.5, 0.5, 61)
    pk.plasma.plot_flux_surfaces(R, Z, pk.plasma.solve_grad_shafranov(R, Z, 1.0, -2.0), ax=ax2)
    ax2.set_aspect("equal")
    ax2.set_title("Grad-Shafranov: nested tokamak flux surfaces")

    def B_func(z):
        return 1.0 + 4.0 * (z / 0.05) ** 2

    z_hist, _ = pk.plasma.magnetic_mirror_bounce(z0=0.0, v_par0=2e4, v_perp0=8e4, m=pk.plasma.MP, B_func=B_func, steps=6000)
    B_hist = B_func(z_hist)
    mu = pk.plasma.magnetic_moment(8e4, pk.plasma.MP, B_func(0.0))
    r_L = pk.plasma.larmor_radius(np.sqrt(np.clip(2 * mu * B_hist / pk.plasma.MP, 0.0, None)), pk.plasma.QE, pk.plasma.MP, B_hist)
    phase = np.cumsum(pk.plasma.cyclotron_frequency(pk.plasma.QE, pk.plasma.MP, B_hist)) * 1e-10
    ax3 = _replace_with_3d(ax3)
    pk.plasma.plot_particle_orbit_3d(np.column_stack([r_L * np.cos(phase), r_L * np.sin(phase), z_hist]), ax=ax3)
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_zticklabels([])
    ax3.set_title("A proton bouncing in a magnetic mirror")


def quantum(axes):
    from physicskit.quantum.chapters.hydrogen_am import HydrogenOrbital
    from physicskit.quantum.chapters.potentials import FiniteSquareWell
    from physicskit.quantum.chapters.wave_packets import propagate_double_slit

    ax1, ax2, ax3 = axes
    x, y = np.linspace(-20, 20, 160), np.linspace(-15, 15, 160)
    X, Y, frames, _ = propagate_double_slit(
        x, y, x0=-10.0, k0=8.0, sigma_x=0.7, sigma_y=4.0, wall_x=0.0, slit_separation=3.0, slit_width=0.8, dt=1e-4, n_steps=20000, save_every=20000
    )
    ax1.pcolormesh(X, Y, np.abs(frames[-1]), shading="auto", cmap="inferno")
    ax1.axvline(0.0, color="cyan", lw=0.8, alpha=0.6)
    ax1.set_aspect("equal")
    ax1.set_title(r"A wave packet through a double slit, $|\psi|$")

    E, width = np.linspace(0.5, 15.0, 150), np.linspace(0.2, 3.0, 150)
    T_map = np.array([FiniteSquareWell(V0=-6.0, width=w).transmission_spectrum(E) for w in width])
    ax2.pcolormesh(E, width, T_map, shading="auto", cmap="inferno", vmin=0, vmax=1)
    ax2.axvline(6.0, color="cyan", ls=":", lw=1)
    ax2.set_xlabel("incident energy E (barrier height 6)")
    ax2.set_ylabel("barrier width")
    ax2.set_title("Tunneling and transmission resonances")

    r = np.linspace(1e-6, 30, 2000)
    for n, l, m in ((1, 0, 0), (2, 0, 0), (2, 1, 0), (3, 2, 1)):
        ax3.plot(r, HydrogenOrbital(n, l, m).radial_density(r), label=f"n={n}, l={l}")
    ax3.set_xlabel("r (Bohr radii)")
    ax3.set_ylabel(r"$r^2 |R_{nl}(r)|^2$")
    ax3.set_title("Hydrogen radial probability densities")
    ax3.legend(fontsize=8)


def relativity(axes):
    from physicskit.relativity.chapters.gw_merger import BinaryMerger
    from physicskit.relativity.chapters.schwarzschild import SchwarzschildBlackHole
    from physicskit.relativity.utils import constants as const
    from physicskit.relativity.visualizers.wave_plots import plot_strain_waveform

    ax1, ax2, ax3 = axes
    bh = SchwarzschildBlackHole(M=1.0)
    for b in (4.0, 5.0, bh.critical_impact_parameter, 6.0, 10.0, 20.0):
        traj = bh.integrate_geodesic(bh.null_geodesic_initial_state(r0=200.0, impact_parameter=b, ingoing=True), dtau=0.05, n_steps=20000)
        ax1.plot(traj["r"] * np.cos(traj["phi"]), traj["r"] * np.sin(traj["phi"]), lw=1, label=f"b = {b:.2f} M")
    ax1.add_patch(plt.Circle((0, 0), bh.horizon_radius, color="black", zorder=5))
    ax1.set_xlim(-25, 25)
    ax1.set_ylim(-25, 25)
    ax1.set_aspect("equal")
    ax1.set_title("Light bending around a black hole")
    ax1.legend(fontsize=7, loc="upper right")

    black_hole_shadow(ax2)

    merger = BinaryMerger(
        m1=const.solar_masses_to_geometrized(36.0), m2=const.solar_masses_to_geometrized(29.0), distance=const.PARSEC_M * 410.0e6, inclination=0.5
    )
    t_seconds = np.linspace(-0.15, 0.03, 4000)
    plot_strain_waveform(t_seconds, *merger.full_waveform(const.seconds_to_geometrized(t_seconds), 0.0), t_merger=0.0, ax=ax3)
    ax3.set_xlabel("time from merger (s)")
    ax3.set_title("GW150914: the binary black hole chirp")


def rmt(axes):
    import physicskit.rmt as rmt

    ax1, ax2, ax3 = axes
    semicircle = rmt.validation.WignerSemicircle()
    centers, counts = rmt.stats.empirical_density(rmt.ensembles.GOE(n=800, seed=2026).sample(n_samples=20), bins=80)
    ax1.bar(centers, counts, width=centers[1] - centers[0], alpha=0.6, color="steelblue", label="GOE eigenvalues")
    x = np.linspace(-2.2, 2.2, 500)
    ax1.plot(x, semicircle.theoretical_pdf(x), "k-", lw=2, label="semicircle law")
    ax1.set_xlabel(r"$\lambda / \sqrt{N\beta}$")
    ax1.set_title("Wigner's semicircle law")
    ax1.legend(fontsize=8)

    s = np.linspace(0, 3.5, 400)
    for label, cls, beta, color in (("GOE", rmt.ensembles.GOE, 1, "C0"), ("GUE", rmt.ensembles.GUE, 2, "C1"), ("GSE", rmt.ensembles.GSE, 4, "C2")):
        spacings = rmt.stats.nearest_neighbor_spacings(cls(n=600, seed=2026).sample(n_samples=10), rmt.stats.semicircle_cdf)
        ax2.hist(spacings, bins=50, range=(0, 3.5), density=True, histtype="step", color=color)
        ax2.plot(s, rmt.validation.WignerSurmise(beta=beta).theoretical_pdf(s), color=color, lw=2, label=rf"{label}, $\beta$ = {beta}")
    ax2.plot(s, np.exp(-s), "k--", lw=1, label="Poisson (integrable)")
    ax2.set_xlabel("level spacing s")
    ax2.set_ylabel("P(s)")
    ax2.set_title("Level repulsion: Dyson's threefold way")
    ax2.legend(fontsize=8)

    eigs = rmt.ensembles.GinUE(n=1000, seed=2026).sample(n_samples=1).rescaled[0]
    theta = np.linspace(0, 2 * np.pi, 300)
    ax3.scatter(eigs.real, eigs.imag, s=3, alpha=0.6, color="darkorange")
    ax3.plot(np.cos(theta), np.sin(theta), "k-", lw=1.5)
    ax3.set_aspect("equal")
    ax3.set_xlabel("Re")
    ax3.set_ylabel("Im")
    ax3.set_title("Ginibre's circular law")


def semiclassical(axes):
    from physicskit.quantum.chapters.harmonic_spin import HarmonicOscillator
    from physicskit.quantum.chapters.potentials import StadiumBilliard2D
    from physicskit.semiclassical.core.gutzwiller import gutzwiller_density_of_states
    from physicskit.semiclassical.core.path_integral import build_phasor_diagram
    from physicskit.semiclassical.systems.scarring import bouncing_ball_orbit_points, scar_enhancement

    ax1, ax2, ax3 = axes
    stadium = StadiumBilliard2D(L=1.0, R=0.5)
    _, wavefunctions, Xs, Ys, mask = stadium.solve(n_points=220, n_states=10)
    scarred = int(np.argmax([scar_enhancement(wf**2, Xs, Ys, mask, x0=0.0, half_width=0.08) for wf in wavefunctions]))
    ax1.pcolormesh(Xs, Ys, np.where(mask, wavefunctions[scarred] ** 2, np.nan), shading="auto", cmap="magma")
    ax1.plot(*bouncing_ball_orbit_points(x0=0.0, R=stadium.R, n_bounces=3), color="cyan", lw=1.5, label="bouncing-ball orbit")
    ax1.set_aspect("equal")
    ax1.set_title("A quantum scar in the stadium")
    ax1.legend(fontsize=8, loc="lower right")

    ho = HarmonicOscillator()
    E = np.linspace(0.2, 6.5, 1200)
    ax2.plot(E, gutzwiller_density_of_states(E, lambda x: 0.5 * ho.m * ho.omega**2 * x**2, ho.m, -20.0, 20.0, hbar=ho.hbar), lw=1)
    for E_n in ho.energy(np.arange(6)):
        ax2.axvline(E_n, color="gray", ls=":", lw=0.8)
    ax2.set_xlabel("E")
    ax2.set_ylabel("g(E)")
    ax2.set_title("Gutzwiller trace formula: levels from orbits")

    paths, _, x_cl, _, partial = build_phasor_diagram(0.0, 1.0, 1.0, 1.0, 1.0, potential="free", n_slices=40, n_paths=400, sigma=0.5, seed=42)
    ax3.plot(partial.real, partial.imag, lw=1.0)
    ax3.plot(partial[-1].real, partial[-1].imag, "o", color="crimson", label="path-integral sum")
    ax3.set_aspect("equal", adjustable="datalim")
    ax3.set_xlabel("Re")
    ax3.set_ylabel("Im")
    ax3.set_title(f"Feynman's sum over {paths.shape[0]} paths: phasor spiral")
    ax3.legend(fontsize=8)


def statphys(axes):
    from physicskit.statphys.chapters.ising_lattice import Ising2D
    from physicskit.statphys.chapters.percolation import Percolation2D
    from physicskit.statphys.utils.partition_function import ising_partition_polynomial, yang_lee_zeros
    from physicskit.statphys.visualizers.lattice_render import plot_percolation_clusters, plot_spin_grid

    ax1, ax2, ax3 = axes
    model = Ising2D(L=128, J=1.0, kB=1.0, seed=1)
    model.sweep(beta=1.0 / model.T_C, algorithm="wolff", n_sweeps=200)
    plot_spin_grid(model.spins, ax=ax1, cmap=ListedColormap(["black", "lightgrey"]))
    ax1.set_title("2D Ising model at its critical point")

    plot_percolation_clusters(Percolation2D(L=128, p=Percolation2D.P_C_SITE, mode="site", seed=1), ax=ax2)
    ax2.set_title("Site percolation clusters at $p_c$")

    theta = np.linspace(0, 2 * np.pi, 200)
    ax3.plot(np.cos(theta), np.sin(theta), color="gray", lw=0.8, ls="--")
    for N, color in ((6, "tab:blue"), (12, "tab:orange"), (18, "tab:red")):
        zeros = yang_lee_zeros(ising_partition_polynomial(N, beta=0.4, J=1.0, periodic=True))
        ax3.scatter(zeros.real, zeros.imag, s=25, color=color, label=f"N = {N}")
    ax3.set_aspect("equal")
    ax3.set_xlabel("Re z")
    ax3.set_ylabel("Im z")
    ax3.set_title("Lee-Yang zeros on the unit circle")
    ax3.legend(fontsize=8)


FIGURES = {
    f.__name__: f
    for f in (
        astro,
        chaos,
        classical,
        condensed,
        fields,
        fluids,
        optics,
        particle,
        plasma,
        quantum,
        relativity,
        rmt,
        semiclassical,
        statphys,
    )
}


def main(names):
    for name in names or FIGURES:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
        FIGURES[name](axes)
        out = OUT_DIR / f"readme_{name}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
