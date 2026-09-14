"""Physics-correctness tests for physicskit.plasma."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import find_peaks

from physicskit.plasma.kinetic import (
    deposit_number_density,
    landau_damping_ic,
    landau_damping_rate,
    pic_simulate,
    solve_poisson_1d,
    two_stream_ic,
)
from physicskit.plasma.mhd import (
    alfven_speed,
    magnetosonic_speeds,
    petschek_rate,
    solovev_particular_solution,
    solve_grad_shafranov,
    sweet_parker_rate,
)
from physicskit.plasma.single_particle import (
    ME,
    MP,
    QE,
    boris_integrate,
    curvature_drift,
    cyclotron_frequency,
    exb_drift,
    grad_b_drift,
    larmor_radius,
    magnetic_mirror_bounce,
    magnetic_moment,
)
from physicskit.plasma.waves import cold_plasma_dispersion, rl_parameters, stix_parameters


class TestBorisPusher:
    def test_conserves_kinetic_energy_in_pure_b_field(self):
        B = np.array([0.0, 0.0, 1.0])
        E = np.zeros(3)
        omega_c = cyclotron_frequency(QE, MP, 1.0)
        dt = (2 * np.pi / omega_c) / 500
        pos_hist, vel_hist = boris_integrate(np.zeros(3), np.array([1e5, 0.0, 0.0]), QE, MP, E, B, dt, steps=5000)
        speeds = np.linalg.norm(vel_hist, axis=1)
        assert speeds == pytest.approx(1e5, rel=1e-9)

    def test_gyroradius_matches_larmor_formula(self):
        B = np.array([0.0, 0.0, 1.0])
        E = np.zeros(3)
        v0 = 1e5
        omega_c = cyclotron_frequency(QE, MP, 1.0)
        dt = (2 * np.pi / omega_c) / 2000
        pos_hist, _ = boris_integrate(np.zeros(3), np.array([v0, 0.0, 0.0]), QE, MP, E, B, dt, steps=2000)
        # the particle starts on its own gyro-circle, so its farthest excursion
        # from the starting point is the orbit *diameter*, 2 * r_L
        diameter_from_orbit = np.max(np.linalg.norm(pos_hist, axis=1))
        assert diameter_from_orbit == pytest.approx(2 * larmor_radius(v0, QE, MP, 1.0), rel=1e-3)

    def test_exb_drift_matches_analytic_average_velocity(self):
        E = np.array([0.0, 1e3, 0.0])
        B = np.array([0.0, 0.0, 1.0])
        omega_c = cyclotron_frequency(QE, MP, 1.0)
        dt = (2 * np.pi / omega_c) / 1000
        pos_hist, vel_hist = boris_integrate(np.zeros(3), np.zeros(3), QE, MP, E, B, dt, steps=10000)
        drift = exb_drift(E, B)
        measured = np.mean(vel_hist[-1000:], axis=0)
        assert measured == pytest.approx(drift, abs=5.0)

    def test_magnetic_mirror_reflects_confined_particle(self):
        B_func = lambda z: 1.0 + 4.0 * (z / 0.05) ** 2
        z_hist, v_par_hist = magnetic_mirror_bounce(z0=0.0, v_par0=2e4, v_perp0=8e4, m=MP, B_func=B_func, steps=6000)
        assert v_par_hist[0] > 0
        assert v_par_hist[-1] < 0


class TestGuidingCenterDrifts:
    def test_grad_b_and_curvature_drift_are_perpendicular_to_b(self):
        B = np.array([0.0, 0.0, 1.0])
        grad_B = np.array([0.1, 0.0, 0.0])
        drift = grad_b_drift(1e5, QE, MP, B, grad_B)
        assert drift @ B == pytest.approx(0.0, abs=1e-12)

        R_c = np.array([1.0, 0.0, 0.0])
        drift_c = curvature_drift(1e5, QE, MP, B, R_c)
        assert drift_c @ B == pytest.approx(0.0, abs=1e-12)

    def test_electron_and_ion_grad_b_drifts_are_antiparallel(self):
        B = np.array([0.0, 0.0, 1.0])
        grad_B = np.array([0.1, 0.0, 0.0])
        drift_ion = grad_b_drift(1e5, QE, MP, B, grad_B)
        drift_electron = grad_b_drift(1e5, -QE, ME, B, grad_B)
        assert drift_ion @ drift_electron < 0

    def test_magnetic_moment_matches_definition(self):
        assert magnetic_moment(v_perp=2e5, m=MP, B=2.0) == pytest.approx(0.5 * MP * 2e5**2 / 2.0)


class TestMHDWaveSpeeds:
    def test_alfven_speed_matches_formula(self):
        B, rho = 1.5, 2e-6
        mu0 = 4 * np.pi * 1e-7
        assert alfven_speed(B, rho) == pytest.approx(B / np.sqrt(mu0 * rho))

    def test_magnetosonic_parallel_limits_reduce_to_max_and_min(self):
        vA, cs = 8e5, 1e5
        v_fast, v_slow = magnetosonic_speeds(vA, cs, theta=0.0)
        assert v_fast == pytest.approx(max(vA, cs))
        assert v_slow == pytest.approx(min(vA, cs))

    def test_magnetosonic_perpendicular_limit(self):
        vA, cs = 8e5, 1e5
        v_fast, v_slow = magnetosonic_speeds(vA, cs, theta=np.pi / 2)
        assert v_fast == pytest.approx(np.sqrt(vA**2 + cs**2))
        assert v_slow == pytest.approx(0.0, abs=1e-6)


class TestGradShafranov:
    def test_sor_solution_matches_solovev_analytic(self):
        R = np.linspace(0.5, 1.5, 61)
        Z = np.linspace(-0.5, 0.5, 61)
        c1, c2 = 1.0, -2.0
        psi = solve_grad_shafranov(R, Z, c1, c2)
        RR, ZZ = np.meshgrid(R, Z, indexing="ij")
        psi_exact = solovev_particular_solution(RR, ZZ, c1, c2)
        assert np.max(np.abs(psi - psi_exact)) < 1e-3


class TestReconnection:
    def test_petschek_faster_than_sweet_parker_at_high_lundquist_number(self):
        S = 1e10
        assert petschek_rate(S) > sweet_parker_rate(S)


class TestColdPlasmaWaves:
    def test_parallel_propagation_gives_r_and_l_waves(self):
        electrons = (1e19, -QE, ME)
        ions = (1e19, QE, MP)
        S, D, P = stix_parameters(omega=2e9, B=1.0, species=[electrons, ions])
        R, L = rl_parameters(S, D)
        n2_plus, n2_minus = cold_plasma_dispersion(theta=0.0, S=S, D=D, P=P)
        assert sorted([n2_plus, n2_minus]) == pytest.approx(sorted([R, L]))

    def test_perpendicular_propagation_gives_o_and_x_modes(self):
        electrons = (1e19, -QE, ME)
        ions = (1e19, QE, MP)
        S, D, P = stix_parameters(omega=2e9, B=1.0, species=[electrons, ions])
        R, L = rl_parameters(S, D)
        n2_plus, n2_minus = cold_plasma_dispersion(theta=np.pi / 2, S=S, D=D, P=P)
        assert sorted([n2_plus, n2_minus]) == pytest.approx(sorted([P, R * L / S]))


class TestKineticPIC:
    def test_cic_deposit_conserves_total_charge(self):
        x = np.array([0.1, 2.4, 4.9, 7.7, 9.99])
        rho = deposit_number_density(x, L=10.0, ng=20, n0=3.0)
        assert np.sum(rho) * (10.0 / 20) == pytest.approx(3.0 * 10.0)

    def test_poisson_solver_matches_analytic_sinusoid(self):
        ng = 64
        L = 2 * np.pi
        x_grid = np.linspace(0, L, ng, endpoint=False)
        rho = np.sin(x_grid)
        E = solve_poisson_1d(rho, L)
        assert E == pytest.approx(-np.cos(x_grid), abs=1e-10)

    def test_two_stream_ic_shapes_and_symmetric_mean_velocity(self):
        x, v = two_stream_ic(2000, L=10.0, v_drift=3.0, v_th=0.5, seed=0)
        assert x.shape == (2000,)
        assert v.shape == (2000,)
        assert np.mean(v) == pytest.approx(0.0, abs=0.5)

    def test_landau_damping_matches_analytic_decay_rate(self):
        k, v_th = 0.5, 1.0
        L = 2 * np.pi / k
        x0, v0 = landau_damping_ic(100000, L=L, k_mode=k, alpha=0.05, v_th=v_th, seed=1)
        result = pic_simulate(x0, v0, L=L, ng=64, dt=0.05, steps=250)
        field_energy, t = result["field_energy"], result["t"]

        peak_idx, _ = find_peaks(field_energy)
        t_peaks, e_peaks = t[peak_idx], field_energy[peak_idx]
        above_noise_floor = e_peaks > 1e-4
        t_peaks, e_peaks = t_peaks[above_noise_floor][:6], e_peaks[above_noise_floor][:6]

        gamma_fit = 0.5 * np.polyfit(t_peaks, np.log(e_peaks), 1)[0]
        gamma_analytic = landau_damping_rate(k, v_th)
        assert gamma_fit == pytest.approx(gamma_analytic, rel=0.5)
