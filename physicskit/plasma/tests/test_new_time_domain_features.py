"""Physics-correctness and animation-smoke tests for the newly added time-domain
plasma simulations: magnetic reconnection and Weibel filamentation
(:mod:`physicskit.plasma.instabilities`), Langmuir waves
(:mod:`physicskit.plasma.kinetic`), ion-acoustic solitons and Alfven waves
(:mod:`physicskit.plasma.waves`), drift-wave turbulence
(:mod:`physicskit.plasma.turbulence`), wakefield particle acceleration
(:mod:`physicskit.plasma.acceleration`), and the two-stream phase-space
animation (:mod:`physicskit.plasma.visualizers`).
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.animation import PillowWriter
from scipy.signal import find_peaks

from physicskit.plasma.acceleration import simulate_wakefield_acceleration
from physicskit.plasma.instabilities import (
    reconnection_field_from_flux,
    reconnection_harris_ic,
    simulate_reconnection,
    simulate_weibel_filamentation,
    weibel_fastest_growing_mode,
    weibel_growth_rate,
)
from physicskit.plasma.kinetic import langmuir_wave_ic, pic_simulate, two_stream_ic
from physicskit.plasma.turbulence import drift_wave_noise_ic, simulate_hasegawa_mima
from physicskit.plasma.visualizers import (
    animate_alfven_wave,
    animate_drift_wave_turbulence,
    animate_ion_acoustic_soliton,
    animate_langmuir_wave,
    animate_reconnection,
    animate_two_stream_phase_space,
    animate_wakefield_acceleration,
    animate_weibel_filamentation,
)
from physicskit.plasma.waves import alfven_wave_pulse_ic, ion_acoustic_soliton_profile, simulate_alfven_wave


class TestReconnection:
    def test_resistivity_dissipates_magnetic_energy_faster_at_higher_eta(self):
        nx, ny, Lx, Ly = 48, 48, 20.0, 20.0
        dx, dy = Lx / nx, Ly / ny
        psi0 = reconnection_harris_ic(nx, ny, Lx, Ly, sheet_width=1.0, perturbation_amplitude=0.2)

        def magnetic_energy(psi):
            Bx, By = reconnection_field_from_flux(psi, dx, dy)
            return np.sum(Bx**2 + By**2) * dx * dy

        e0 = magnetic_energy(psi0)
        r_low = simulate_reconnection(psi0, eta=0.01, v0=0.05, dt=0.01, steps=200, Lx=Lx, Ly=Ly)
        r_high = simulate_reconnection(psi0, eta=0.05, v0=0.05, dt=0.01, steps=200, Lx=Lx, Ly=Ly)
        e_low = magnetic_energy(r_low["psi"])
        e_high = magnetic_energy(r_high["psi"])
        assert e_high < e_low < e0

    @pytest.mark.slow
    def test_animation_saves_to_gif(self, tmp_path):
        psi0 = reconnection_harris_ic(32, 32, Lx=20.0, Ly=20.0, sheet_width=1.0, perturbation_amplitude=0.2)
        anim = animate_reconnection(psi0, eta=0.02, v0=0.05, dt=0.02, steps_per_frame=5, n_frames=5, Lx=20.0, Ly=20.0)
        out = tmp_path / "reconnection.gif"
        anim.save(out, writer=PillowWriter(fps=10))
        assert out.exists() and out.stat().st_size > 0


class TestWeibelFilamentation:
    # Davidson (1983): gamma = sqrt(2/pi) k v_T|| / A * (A - 1 - k^2 c^2 / wpe^2).
    wpe, aniso, c, v_th = 1.0, 4.0, 1.0, 0.1

    def rate(self, k):
        return weibel_growth_rate(k, self.wpe, self.aniso, c=self.c, v_th_par=self.v_th)

    def test_growth_rate_vanishes_at_zero_k(self):
        assert self.rate(0.0) == 0.0

    def test_growth_rate_peaks_at_closed_form_wavenumber(self):
        k_star = (self.wpe / self.c) * np.sqrt((self.aniso - 1.0) / 3.0)
        k = np.linspace(0.0, 2.0, 20001)
        assert k[np.argmax(self.rate(k))] == pytest.approx(k_star, abs=1e-4)
        gamma_max = np.sqrt(2.0 / np.pi) * (2.0 / (3.0 * np.sqrt(3.0))) * self.v_th * (self.aniso - 1.0) ** 1.5 / self.aniso
        assert self.rate(k_star) == pytest.approx(gamma_max)

    def test_growth_rate_vanishes_at_and_above_cutoff(self):
        k_max = (self.wpe / self.c) * np.sqrt(self.aniso - 1.0)
        assert self.rate(0.999 * k_max) > 0.0
        assert self.rate(k_max) == pytest.approx(0.0, abs=1e-15)
        assert self.rate(2.0 * k_max) == 0.0

    def test_growth_rate_cutoff_in_si_units(self):
        wpe, c = 1e10, 3e8
        k_max = (wpe / c) * np.sqrt(self.aniso - 1.0)
        assert weibel_growth_rate(0.5 * k_max, wpe, self.aniso, c=c, v_th_par=1e6) > 0.0
        assert weibel_growth_rate(2.0 * k_max, wpe, self.aniso, c=c, v_th_par=1e6) == 0.0

    def test_growth_rate_matches_exact_vlasov_root_at_weak_anisotropy(self):
        from scipy.optimize import brentq
        from scipy.special import wofz

        aniso, k = 1.05, 0.1

        def dispersion(gamma):
            zeta = 1j * gamma / (np.sqrt(2.0) * k * self.v_th)
            zeta_Z = zeta * 1j * np.sqrt(np.pi) * wofz(zeta)
            return (k**2 + gamma**2 - (aniso - 1.0) - aniso * zeta_Z).real

        exact = brentq(dispersion, 1e-12, 1.0)
        approx = weibel_growth_rate(k, 1.0, aniso, c=1.0, v_th_par=self.v_th)
        assert approx == pytest.approx(exact, rel=0.05)

    def test_fastest_growing_mode_matches_closed_form(self):
        k0, gamma_max = weibel_fastest_growing_mode(wpe=1.0, temperature_anisotropy=4.0, c=1.0, v_th_par=0.1)
        assert k0 == pytest.approx(1.0)
        assert gamma_max == pytest.approx(np.sqrt(2.0 / np.pi) * 0.1 / 4.0 * 2.0)

    def test_fastest_growing_mode_is_zero_for_isotropic_temperature(self):
        assert weibel_fastest_growing_mode(wpe=1.0, temperature_anisotropy=1.0, v_th_par=0.1) == (0.0, 0.0)

    def test_current_filaments_grow_in_time(self):
        x = np.linspace(0, 20.0, 256, endpoint=False)
        t = np.array([0.0, 200.0])
        J = simulate_weibel_filamentation(x, t, wpe=1.0, temperature_anisotropy=4.0, n_modes=8, seed=0)
        assert np.std(J[1]) > 100 * np.std(J[0])

    def test_simulated_modes_grow_at_the_linear_rate(self):
        x = np.linspace(0, 60.0, 512, endpoint=False)
        t = np.array([0.0, 50.0])
        J = simulate_weibel_filamentation(x, t, wpe=1.0, temperature_anisotropy=1.5, n_modes=10, seed=1, v_th_par=0.2)
        spectrum = np.abs(np.fft.rfft(J, axis=1))
        k = 2.0 * np.pi * np.fft.rfftfreq(x.size, d=x[1] - x[0])
        seeded = spectrum[0] > 1e-6 * spectrum[0].max()
        assert seeded.sum() >= 5
        measured = np.log(spectrum[1, seeded] / spectrum[0, seeded]) / t[1]
        expected = weibel_growth_rate(k[seeded], 1.0, 1.5, c=1.0, v_th_par=0.2)
        np.testing.assert_allclose(measured, expected, rtol=1e-8)

    def test_domain_shorter_than_any_unstable_wavelength_raises(self):
        x = np.linspace(0, 1.0, 32, endpoint=False)
        with pytest.raises(ValueError, match="no Fourier mode"):
            simulate_weibel_filamentation(x, np.array([0.0]), wpe=1.0, temperature_anisotropy=1.5)

    @pytest.mark.slow
    def test_animation_saves_to_gif(self, tmp_path):
        x = np.linspace(0, 20.0, 128, endpoint=False)
        t = np.linspace(0, 5.0, 6)
        anim = animate_weibel_filamentation(x, t, wpe=1.0, temperature_anisotropy=4.0, n_modes=6)
        out = tmp_path / "weibel.gif"
        anim.save(out, writer=PillowWriter(fps=10))
        assert out.exists() and out.stat().st_size > 0


class TestLangmuirWave:
    def test_field_energy_oscillates_without_substantial_decay(self):
        k, L = 2 * np.pi / 4.0, 4.0
        x0, v0 = langmuir_wave_ic(20000, L=L, k_mode=k, alpha=0.05, v_th=0.05, seed=1)
        result = pic_simulate(x0, v0, L=L, ng=64, dt=0.05, steps=400)
        field_energy = result["field_energy"]
        peaks, _ = find_peaks(field_energy)
        assert len(peaks) >= 4
        peak_heights = field_energy[peaks]
        assert peak_heights[-1] > 0.5 * peak_heights[0]

    @pytest.mark.slow
    def test_animation_saves_to_gif(self, tmp_path):
        k, L = 2 * np.pi / 4.0, 4.0
        x0, v0 = langmuir_wave_ic(4000, L=L, k_mode=k, alpha=0.05, v_th=0.05, seed=0)
        anim = animate_langmuir_wave(x0, v0, L=L, ng=32, dt=0.05, steps_per_frame=4, n_frames=5)
        out = tmp_path / "langmuir.gif"
        anim.save(out, writer=PillowWriter(fps=10))
        assert out.exists() and out.stat().st_size > 0


class TestTwoStreamAnimation:
    @pytest.mark.slow
    def test_animation_saves_to_gif(self, tmp_path):
        x0, v0 = two_stream_ic(2000, L=10.0, v_drift=3.0, v_th=0.5, seed=0)
        anim = animate_two_stream_phase_space(x0, v0, L=10.0, ng=32, dt=0.05, steps_per_frame=5, n_frames=5)
        out = tmp_path / "two_stream.gif"
        anim.save(out, writer=PillowWriter(fps=10))
        assert out.exists() and out.stat().st_size > 0


class TestIonAcousticSoliton:
    def test_soliton_propagates_at_its_speed_and_keeps_amplitude(self):
        N, L = 512, 60.0
        x = np.linspace(-L / 2, L / 2, N, endpoint=False)
        u0 = ion_acoustic_soliton_profile(x, speed=4.0, x0=-15.0)
        from physicskit.plasma.waves import ion_acoustic_soliton_evolve

        dt, steps = 0.0005, 4000
        u = ion_acoustic_soliton_evolve(u0, x, dt, steps)
        shift = x[np.argmax(u)] - x[np.argmax(u0)]
        assert shift == pytest.approx(4.0 * steps * dt, abs=0.5)
        assert u.max() == pytest.approx(u0.max(), abs=0.05)

    @pytest.mark.slow
    def test_animation_saves_to_gif(self, tmp_path):
        x = np.linspace(-30, 30, 256, endpoint=False)
        u0 = ion_acoustic_soliton_profile(x, speed=4.0, x0=-15.0)
        anim = animate_ion_acoustic_soliton(u0, x, dt=0.0005, steps_per_frame=200, n_frames=5)
        out = tmp_path / "ion_acoustic.gif"
        anim.save(out, writer=PillowWriter(fps=10))
        assert out.exists() and out.stat().st_size > 0


class TestAlfvenWave:
    def test_pulse_splits_and_propagates_at_alfven_speed(self):
        x = np.linspace(-20, 20, 256, endpoint=False)
        By0, vy0 = alfven_wave_pulse_ic(x, x0=0.0, width=1.0, amplitude=0.1)
        B0, rho0, mu0 = 1.0, 1.0, 1.0
        vA = B0 / np.sqrt(mu0 * rho0)
        dt, steps = 0.002, 2000
        result = simulate_alfven_wave(By0, vy0, x, dt, steps, B0, rho0, mu0=mu0)
        By = result["By"]
        right_half = By[x > 0]
        left_half = By[x < 0]
        peak_right = x[x > 0][np.argmax(right_half)]
        peak_left = x[x < 0][np.argmax(left_half)]
        expected_shift = vA * steps * dt
        assert peak_right == pytest.approx(expected_shift, abs=1.0)
        assert peak_left == pytest.approx(-expected_shift, abs=1.0)

    @pytest.mark.slow
    def test_animation_saves_to_gif(self, tmp_path):
        x = np.linspace(-20, 20, 128, endpoint=False)
        By0, vy0 = alfven_wave_pulse_ic(x, x0=0.0, width=1.0, amplitude=0.1)
        anim = animate_alfven_wave(By0, vy0, x, dt=0.002, steps_per_frame=100, n_frames=5, B0=1.0, rho0=1.0)
        out = tmp_path / "alfven.gif"
        anim.save(out, writer=PillowWriter(fps=10))
        assert out.exists() and out.stat().st_size > 0


class TestDriftWaveTurbulence:
    def test_higher_dissipation_decays_faster(self):
        n, length = 48, 2 * np.pi
        phi0 = drift_wave_noise_ic(n, length, amplitude=0.05, seed=2)
        dt, steps = 0.02, 300
        r_low = simulate_hasegawa_mima(phi0, dt, steps, length, nu=0.02)
        r_high = simulate_hasegawa_mima(phi0, dt, steps, length, nu=0.06)
        rms_low = np.sqrt(np.mean(r_low["phi"] ** 2))
        rms_high = np.sqrt(np.mean(r_high["phi"] ** 2))
        assert np.isfinite(rms_low) and np.isfinite(rms_high)
        assert rms_high < rms_low

    @pytest.mark.slow
    def test_animation_saves_to_gif(self, tmp_path):
        phi0 = drift_wave_noise_ic(48, 2 * np.pi, amplitude=0.05, seed=0)
        anim = animate_drift_wave_turbulence(phi0, dt=0.02, steps_per_frame=10, n_frames=5, length=2 * np.pi)
        out = tmp_path / "drift_wave.gif"
        anim.save(out, writer=PillowWriter(fps=10))
        assert out.exists() and out.stat().st_size > 0


class TestWakefieldAcceleration:
    def test_phase_locked_particle_gains_energy_matching_work_energy_theorem(self):
        result = simulate_wakefield_acceleration(x0=0.0, v0=0.9, q=1.0, m=1.0, E0=0.05, k=1.0, v_phase=1.0, dt=0.001, steps=20000)
        ke = result["kinetic_energy"]
        assert ke[-1] > ke[0]
        v, t, x = result["v"], result["t"], result["x"]
        dt = t[1] - t[0]
        from physicskit.plasma.acceleration import wakefield_e_field

        work = np.sum(wakefield_e_field(x[:-1], t[:-1], 0.05, 1.0, 1.0) * v[:-1] * dt)
        assert (ke[-1] - ke[0]) == pytest.approx(work, rel=0.1)

    @pytest.mark.slow
    def test_animation_saves_to_gif(self, tmp_path):
        anim = animate_wakefield_acceleration(x0=0.0, v0=0.9, q=1.0, m=1.0, E0=0.05, k=1.0, v_phase=1.0, dt=0.01, steps=400, frame_stride=40)
        out = tmp_path / "wakefield.gif"
        anim.save(out, writer=PillowWriter(fps=10))
        assert out.exists() and out.stat().st_size > 0
