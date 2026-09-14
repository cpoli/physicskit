"""Physics-correctness tests for physicskit.fields."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import find_peaks

from physicskit.fields.electrodynamics import (
    C0,
    courant_limit_1d,
    courant_limit_2d,
    dielectric_slab,
    fdtd_1d,
    fdtd_2d_tmz,
    fdtd_2d_tmz_evolve,
    flux_tube_field_1d,
    oscillating_dipole_source,
    pml_conductivity_profile_2d,
    tmz_cavity_mode,
)
from physicskit.fields.quantum_fields import (
    casimir_energy_1d,
    casimir_mode_frequencies,
    count_vortices,
    gpe_energy,
    gpe_evolve,
    gpe_imprint_vortex,
    gpe_relax,
    harmonic_trap_grid,
)
from physicskit.fields.solitons import (
    kdv_evolve,
    kdv_evolve_frames,
    kdv_soliton,
    nls_bright_soliton,
    nls_evolve,
    nls_evolve_frames,
    sine_gordon_evolve,
    sine_gordon_evolve_frames,
    sine_gordon_kink,
)


class TestFDTDWaveSpeed:
    def test_1d_pulse_propagates_at_c(self):
        N, dx = 800, 1e-3
        dt = 0.99 * courant_limit_1d(dx)
        eta0 = np.sqrt(4 * np.pi * 1e-7 / 8.8541878128e-12)
        x0, sigma_pulse = 100, 25
        Ez0 = np.exp(-((np.arange(N) - x0) ** 2) / (2 * sigma_pulse**2))
        xh = np.arange(N - 1) + 0.5
        Hy0 = np.exp(-((xh - x0) ** 2) / (2 * sigma_pulse**2)) / eta0
        eps_r, mu_r = np.ones(N), np.ones(N)
        steps = 400
        Ez, _ = fdtd_1d(Ez0, Hy0, eps_r, mu_r, steps=steps, dt=dt, dx=dx)
        peak = np.argmax(Ez)
        measured_speed = (peak - x0) * dx / (steps * dt)
        assert measured_speed == pytest.approx(C0, rel=0.01)

    def test_2d_tmz_runs_and_conserves_shape(self):
        Nx, Ny = 40, 40
        Ez0 = np.zeros((Nx, Ny))
        Ez0[Nx // 2, Ny // 2] = 1.0
        Hx0, Hy0 = np.zeros((Nx, Ny)), np.zeros((Nx, Ny))
        eps_r, mu_r = np.ones((Nx, Ny)), np.ones((Nx, Ny))
        dx = dy = 1e-3
        dt = 0.5 * courant_limit_2d(dx, dy)
        Ez, _, _ = fdtd_2d_tmz(Ez0, Hx0, Hy0, eps_r, mu_r, steps=15, dt=dt, dx=dx, dy=dy)
        assert Ez.shape == (Nx, Ny)
        assert np.all(np.isfinite(Ez))


class TestKdVSolitonCollision:
    def test_two_soliton_collision_is_elastic(self):
        N, L = 512, 60.0
        x = np.linspace(-L / 2, L / 2, N, endpoint=False)
        c_fast, c_slow = 9.0, 4.0
        u0 = kdv_soliton(x, c_fast, x0=-20) + kdv_soliton(x, c_slow, x0=-8)
        u = kdv_evolve(u0, x, dt=0.0005, steps=6000)
        peaks, _ = find_peaks(u, height=1.0)
        heights = sorted(u[peaks])
        assert len(heights) == 2
        assert heights[0] == pytest.approx(c_slow / 2, abs=0.02)
        assert heights[1] == pytest.approx(c_fast / 2, abs=0.02)

    def test_mass_is_conserved(self):
        N, L = 512, 60.0
        x = np.linspace(-L / 2, L / 2, N, endpoint=False)
        dx = x[1] - x[0]
        u0 = kdv_soliton(x, 9.0, x0=-20) + kdv_soliton(x, 4.0, x0=-8)
        u = kdv_evolve(u0, x, dt=0.0005, steps=6000)
        assert np.sum(u) * dx == pytest.approx(np.sum(u0) * dx, rel=1e-6)

    def test_single_soliton_shape_preserved(self):
        N, L = 512, 60.0
        x = np.linspace(-L / 2, L / 2, N, endpoint=False)
        c = 4.0
        u0 = kdv_soliton(x, c, x0=-15)
        steps, dt = 2000, 0.001
        u = kdv_evolve(u0, x, dt=dt, steps=steps)
        expected = kdv_soliton(x, c, x0=-15 + c * dt * steps)
        assert np.max(np.abs(u - expected)) < 1e-3


class TestGPEVortexLattice:
    def test_imaginary_time_energy_decreases_monotonically(self):
        n, length, g = 48, 12.0, 4.0
        X, Y, _, _, K2 = harmonic_trap_grid(n, length)
        V = 0.5 * (X**2 + Y**2)
        psi = np.exp(-0.5 * (X**2 + Y**2)).astype(complex)
        energies = []
        for _ in range(8):
            psi = gpe_relax(psi, V, g, dtau=5e-4, steps=250, X=X, Y=Y, K2=K2)
            energies.append(gpe_energy(psi, V, g, X, Y, K2)["total"])
        assert all(energies[i + 1] <= energies[i] + 1e-9 for i in range(len(energies) - 1))

    def test_vortex_state_is_a_stationary_solution_with_quantized_circulation(self):
        # Seeded slightly off a grid vertex: a core sitting exactly on a lattice
        # site makes the discrete plaquette circulation ambiguous (it's shared
        # between four adjacent plaquettes), which is a detection subtlety, not
        # a physical effect.
        n, length, g = 48, 10.0, 4.0
        X, Y, _, _, K2 = harmonic_trap_grid(n, length)
        V = 0.5 * (X**2 + Y**2)
        psi0 = gpe_imprint_vortex(np.exp(-0.5 * (X**2 + Y**2)).astype(complex), X, Y, [(0.03, 0.02)])
        psi = gpe_relax(psi0, V, g, dtau=5e-4, steps=4000, X=X, Y=Y, K2=K2)
        winding = count_vortices(psi)
        assert np.sum(np.abs(winding)) == 1
        # a nonzero winding number topologically requires a density zero somewhere.
        assert np.min(np.abs(psi) ** 2) < 0.01 * np.max(np.abs(psi) ** 2)

    def test_rotation_favors_the_vortex_state_above_critical_frequency(self):
        n, length, g = 64, 12.0, 4.0
        X, Y, _, _, K2 = harmonic_trap_grid(n, length)
        V = 0.5 * (X**2 + Y**2)
        psi_vf = gpe_relax(np.exp(-0.5 * (X**2 + Y**2)).astype(complex), V, g, dtau=5e-4, steps=4000, X=X, Y=Y, K2=K2)
        psi0_v = gpe_imprint_vortex(np.exp(-0.5 * (X**2 + Y**2)).astype(complex), X, Y, [(0.0, 0.0)])
        psi_v = gpe_relax(psi0_v, V, g, dtau=5e-4, steps=4000, X=X, Y=Y, K2=K2)
        E0 = gpe_energy(psi_vf, V, g, X, Y, K2)
        E1 = gpe_energy(psi_v, V, g, X, Y, K2)
        omega_c = (E1["total"] - E0["total"]) / (E1["angular_momentum"] - E0["angular_momentum"])
        assert 0.0 < omega_c < 1.0
        # below Omega_c the vortex-free state has lower rotating-frame energy; above, the vortex does.
        below = E0["total"] - 0.5 * omega_c * E0["angular_momentum"] < E1["total"] - 0.5 * omega_c * E1["angular_momentum"]
        above = E1["total"] - 1.5 * omega_c * E1["angular_momentum"] < E0["total"] - 1.5 * omega_c * E0["angular_momentum"]
        assert below and above


class TestNLSSolitons:
    def test_bright_soliton_does_not_disperse(self):
        N, L = 1024, 80.0
        x = np.linspace(-L / 2, L / 2, N, endpoint=False)
        psi0 = nls_bright_soliton(x, t=0.0, A=1.0)
        psi = nls_evolve(psi0, x, dt=0.001, steps=2000, g=1.0)
        assert np.max(np.abs(np.abs(psi) - np.abs(psi0))) < 1e-3

    def test_norm_conserved(self):
        N, L = 1024, 80.0
        x = np.linspace(-L / 2, L / 2, N, endpoint=False)
        dx = x[1] - x[0]
        psi0 = nls_bright_soliton(x, t=0.0, A=1.0)
        psi = nls_evolve(psi0, x, dt=0.001, steps=2000, g=1.0)
        assert np.sum(np.abs(psi) ** 2) * dx == pytest.approx(np.sum(np.abs(psi0) ** 2) * dx, rel=1e-9)


class TestSineGordonKink:
    def test_kink_propagates_at_prescribed_velocity(self):
        N, L, v, x0 = 4000, 200.0, 0.5, -50.0
        x = np.linspace(-L / 2, L / 2, N)
        dx = x[1] - x[0]
        dt = 0.4 * dx
        u_prev = sine_gordon_kink(x, -dt, v, x0)
        u0 = sine_gordon_kink(x, 0.0, v, x0)
        steps = 1000
        u, _ = sine_gordon_evolve(u0, u_prev, x, dt, steps)
        expected = sine_gordon_kink(x, steps * dt, v, x0)
        assert np.max(np.abs(u - expected)) < 0.01


class TestEvolveFramesChaining:
    """The *_evolve_frames wrappers must reproduce plain repeated evolve() calls exactly:
    they are pure chunking/bookkeeping around the existing, already-tested physics."""

    def test_kdv_frames_match_direct_evolve(self):
        N, L, c = 256, 40.0, 4.0
        x = np.linspace(-L / 2, L / 2, N, endpoint=False)
        u0 = kdv_soliton(x, c, x0=-10)
        dt, steps_per_frame, n_frames = 0.001, 100, 4
        frames, times = kdv_evolve_frames(u0, x, dt, steps_per_frame, n_frames)
        assert frames.shape == (n_frames + 1, N)
        direct = kdv_evolve(u0, x, dt, steps_per_frame * n_frames)
        assert np.max(np.abs(frames[-1] - direct)) < 1e-10
        assert times[-1] == pytest.approx(steps_per_frame * n_frames * dt)

    def test_nls_frames_match_direct_evolve(self):
        N, L = 256, 40.0
        x = np.linspace(-L / 2, L / 2, N, endpoint=False)
        psi0 = nls_bright_soliton(x, t=0.0, A=1.0)
        dt, steps_per_frame, n_frames = 0.001, 100, 4
        frames, times = nls_evolve_frames(psi0, x, dt, steps_per_frame, n_frames, g=1.0)
        assert frames.shape == (n_frames + 1, N)
        direct = nls_evolve(psi0, x, dt, steps_per_frame * n_frames, g=1.0)
        assert np.max(np.abs(frames[-1] - direct)) < 1e-10

    def test_sine_gordon_frames_match_direct_evolve(self):
        N, L, v, x0 = 800, 100.0, 0.4, -20.0
        x = np.linspace(-L / 2, L / 2, N)
        dt = 0.4 * (x[1] - x[0])
        u_prev = sine_gordon_kink(x, -dt, v, x0)
        u0 = sine_gordon_kink(x, 0.0, v, x0)
        steps_per_frame, n_frames = 50, 4
        frames, times = sine_gordon_evolve_frames(u0, u_prev, x, dt, steps_per_frame, n_frames)
        assert frames.shape == (n_frames + 1, N)
        direct, _ = sine_gordon_evolve(u0, u_prev, x, dt, steps_per_frame * n_frames)
        assert np.max(np.abs(frames[-1] - direct)) < 1e-10


class TestDipoleRadiationCausality:
    def test_wavefront_respects_causality(self):
        """An oscillating dipole source can't be felt anywhere before the wavefront,
        traveling at c0, has had time to arrive -- the basic physical signature of radiation."""
        N = 120
        dx = dy = 2e-3
        dt = 0.5 * courant_limit_2d(dx, dy)
        eps_r = mu_r = np.ones((N, N))
        Ez0 = Hx0 = Hy0 = np.zeros((N, N))
        sigma = pml_conductivity_profile_2d((N, N), pml_width=15, dx=dx, dy=dy)
        source = oscillating_dipole_source(N // 2, N // 2, amplitude=1.0, freq=3e10)
        r_near, r_far = 10, 40
        snaps, times = fdtd_2d_tmz_evolve(Ez0, Hx0, Hy0, eps_r, mu_r, steps=4000, dt=dt, dx=dx, dy=dy, sigma=sigma, source=source, snapshot_stride=20)
        near_vals = snaps[:, N // 2 + r_near, N // 2]
        far_vals = snaps[:, N // 2 + r_far, N // 2]
        t_near, t_far = r_near * dx / C0, r_far * dx / C0
        idx_near = np.searchsorted(times, t_near * 0.5)
        idx_far = np.searchsorted(times, t_far * 0.5)
        assert idx_near > 0 and idx_far > idx_near  # sanity: the test windows are non-trivial
        assert np.max(np.abs(near_vals[:idx_near])) == 0.0
        assert np.max(np.abs(far_vals[:idx_far])) == 0.0
        # after enough time for the wavefront to pass both probes, the field is excited at both
        assert np.max(np.abs(near_vals)) > 0.0
        assert np.max(np.abs(far_vals)) > 0.0


class TestDielectricSlabReflection:
    def test_normal_incidence_reflection_matches_fresnel(self):
        """Uses the existing (already-tested) fdtd_1d, which already supports a spatially
        varying eps_r -- the same physics fdtd_2d_tmz uses for a dielectric-slab demo,
        checked here against the analytic Fresnel reflection coefficient."""
        N, dx = 3000, 1e-3
        dt = 0.99 * courant_limit_1d(dx)
        eta0 = np.sqrt(4 * np.pi * 1e-7 / 8.8541878128e-12)
        x0, sigma_pulse = 500, 20
        xs = np.arange(N)
        Ez0 = np.exp(-((xs - x0) ** 2) / (2 * sigma_pulse**2))
        xh = np.arange(N - 1) + 0.5
        Hy0 = np.exp(-((xh - x0) ** 2) / (2 * sigma_pulse**2)) / eta0
        eps_r_slab = 4.0
        eps_r = np.ones(N)
        eps_r[1500:] = eps_r_slab
        mu_r = np.ones(N)
        Ez, _ = fdtd_1d(Ez0, Hy0, eps_r, mu_r, steps=2400, dt=dt, dx=dx)
        R_measured = np.max(np.abs(Ez[:400]))
        n1, n2 = 1.0, np.sqrt(eps_r_slab)
        R_analytic = abs((n1 - n2) / (n1 + n2))
        assert R_measured == pytest.approx(R_analytic, rel=0.02)

    def test_dielectric_slab_map_values(self):
        eps_r = dielectric_slab((40, 20), i_start=15, i_end=25, eps_r_slab=4.0)
        assert eps_r[10, 5] == pytest.approx(1.0)
        assert eps_r[20, 5] == pytest.approx(4.0)
        assert eps_r[30, 5] == pytest.approx(1.0)


class TestCavityStandingWaveMode:
    def test_seeded_mode_oscillates_at_analytic_frequency_with_pec_walls(self):
        N = 41
        dx = dy = 1e-3
        dt = 0.5 * courant_limit_2d(dx, dy)
        Ez0, omega = tmz_cavity_mode((N, N), dx, dy, m=1, n=1)
        Hx0 = Hy0 = np.zeros((N, N))
        eps_r = mu_r = np.ones((N, N))
        period = 2 * np.pi / omega
        steps = int(3 * period / dt)
        snaps, times = fdtd_2d_tmz_evolve(Ez0, Hx0, Hy0, eps_r, mu_r, steps=steps, dt=dt, dx=dx, dy=dy, snapshot_stride=1)
        assert np.max(np.abs(snaps[:, 0, :])) == 0.0  # PEC walls: exactly zero, not just small
        assert np.max(np.abs(snaps[:, -1, :])) == 0.0
        probe = snaps[:, N // 2, N // 2]
        spectrum = np.abs(np.fft.rfft(probe - probe.mean()))
        freqs = np.fft.rfftfreq(len(probe), d=dt)
        peak_freq = freqs[np.argmax(spectrum[1:]) + 1]
        measured_omega = 2 * np.pi * peak_freq
        assert measured_omega == pytest.approx(omega, rel=0.02)


class TestFluxTubeConfinement:
    def test_energy_grows_linearly_with_separation(self):
        """The defining signature of confinement (vs. an ordinary field that spreads and
        whose energy is roughly separation-independent): total field energy in the
        confined tube should scale linearly with charge separation."""
        x = np.linspace(-30, 30, 4000)
        dx = x[1] - x[0]
        separations = np.array([6.0, 10.0, 14.0, 18.0, 22.0])
        energies = np.array([np.sum(0.5 * flux_tube_field_1d(x, s) ** 2) * dx for s in separations])
        slope, intercept = np.polyfit(separations, energies, 1)
        predicted = slope * separations + intercept
        assert np.max(np.abs(energies - predicted)) < 1e-6
        assert slope > 0

    def test_field_plateaus_between_charges_and_vanishes_outside(self):
        x = np.linspace(-20, 20, 2000)
        field = flux_tube_field_1d(x, separation=10.0, flux_quantum=1.0)
        assert abs(field[np.argmin(np.abs(x))] - 1.0) < 0.05
        assert abs(field[np.argmin(np.abs(x - 15))]) < 0.05


class TestCasimirEffect:
    def test_mode_frequencies_are_evenly_spaced(self):
        omega = casimir_mode_frequencies(d=2.0, c=1.0, n_max=10)
        spacing = np.diff(omega)
        assert np.allclose(spacing, np.pi / 2.0)

    def test_regularized_energy_converges_to_known_closed_form(self):
        d = 3.0
        exact = -np.pi / (24 * d)
        coarse_error = abs(casimir_energy_1d(d, cutoff=3e-3 * d) - exact)
        fine_error = abs(casimir_energy_1d(d, cutoff=7e-4 * d) - exact)
        assert coarse_error < 1e-6
        assert fine_error < coarse_error

    def test_energy_magnitude_shrinks_as_plates_separate(self):
        # weaker (less negative) zero-point energy at larger separation --
        # the discrete spectrum crowds toward the continuum as d grows.
        assert casimir_energy_1d(1.0) < casimir_energy_1d(2.0) < casimir_energy_1d(4.0) < 0.0


class TestGPERealTimeEvolution:
    def test_norm_is_conserved_under_real_time_propagation(self):
        n, length, g = 40, 12.0, 2.0
        X, Y, _, _, K2 = harmonic_trap_grid(n, length)
        dx = X[1, 0] - X[0, 0]
        V = 0.5 * (X**2 + Y**2)
        psi0 = np.exp(-0.5 * (X**2 + Y**2)).astype(complex)
        psi0 *= 1.0 / np.sqrt(np.sum(np.abs(psi0) ** 2) * dx * dx)
        snaps, _ = gpe_evolve(psi0, V, g, dt=1e-3, steps=400, K2=K2, snapshot_stride=50)
        norms = np.sum(np.abs(snaps) ** 2, axis=(1, 2)) * dx * dx
        assert np.max(np.abs(norms - norms[0])) < 1e-9

    def test_off_center_vortex_precesses_around_the_trap(self):
        """A vortex imprinted off-center on a relaxed (vortex-free) ground state is not a
        stationary GPE solution -- unlike an imaginary-time relaxation (which would just
        expel it), real-time propagation makes it precess around the trap center at
        roughly constant radius, the standard single-vortex precession phenomenon."""
        n, length, g = 64, 14.0, 4.0
        X, Y, _, _, K2 = harmonic_trap_grid(n, length)
        V = 0.5 * (X**2 + Y**2)
        psi_gs = gpe_relax(np.exp(-0.5 * (X**2 + Y**2)).astype(complex), V, g, dtau=5e-4, steps=4000, X=X, Y=Y, K2=K2)
        psi_v = gpe_imprint_vortex(psi_gs, X, Y, [(1.2, 0.03)])
        snaps, times = gpe_evolve(psi_v, V, g, dt=2e-3, steps=6000, K2=K2, snapshot_stride=300)

        def core_position(psi, margin=8, threshold=0.0005):
            winding = count_vortices(psi, density_threshold=threshold)
            winding[:margin, :] = 0
            winding[-margin:, :] = 0
            winding[:, :margin] = 0
            winding[:, -margin:] = 0
            idxs = np.argwhere(np.abs(winding) > 0)
            return (float(X[idxs[0, 0], idxs[0, 1]]), float(Y[idxs[0, 0], idxs[0, 1]])) if len(idxs) else None

        positions = [core_position(snap) for snap in snaps]
        found = [p for p in positions if p is not None]
        assert len(found) >= len(positions) - 3  # the detector should find the core in almost every frame
        angles = np.unwrap([np.arctan2(y, x) for x, y in found])
        radii = [np.hypot(x, y) for x, y in found]
        assert abs(angles[-1] - angles[0]) > np.deg2rad(90)  # it has visibly gone around part of an orbit
        assert all(0.3 < r < 3.0 for r in radii)  # stays in a bounded annulus -- neither collapsed nor ejected

    def test_self_focusing_nls_concentrates_before_grid_breaks_down(self):
        """Attractive interactions (g<0 in this module's +g|psi|^2 convention) with a tall,
        narrow packet cause self-focusing: peak density grows and the packet narrows over
        time. True collapse is a singularity this finite grid cannot resolve, so the test
        (like the function's documented usage) stops well before that breakdown."""
        n, length = 128, 20.0
        X, Y, _, _, K2 = harmonic_trap_grid(n, length)
        dx = X[1, 0] - X[0, 0]
        V = np.zeros((n, n))
        psi0 = 3.0 * np.exp(-0.5 * (X**2 + Y**2)).astype(complex)
        norm0 = np.sum(np.abs(psi0) ** 2) * dx * dx
        snaps, _ = gpe_evolve(psi0, V, g=-1.0, dt=2e-4, steps=1400, K2=K2, snapshot_stride=200)
        norms = np.sum(np.abs(snaps) ** 2, axis=(1, 2)) * dx * dx
        peaks = np.max(np.abs(snaps) ** 2, axis=(1, 2))
        assert np.max(np.abs(norms - norm0)) / norm0 < 1e-6
        assert np.all(np.diff(peaks) > 0)  # monotonically concentrating
        assert peaks[-1] > 10 * peaks[0]  # substantial concentration within the simulated window
