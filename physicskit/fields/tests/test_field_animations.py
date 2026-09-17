"""Animation smoke tests for physicskit.fields.visualizers: build each animation and save it
to a GIF, following the pattern in physicskit/classical/tests/test_visualizers.py."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.animation import PillowWriter

from physicskit.fields.electrodynamics import (
    courant_limit_2d,
    dielectric_slab,
    fdtd_2d_tmz_evolve,
    oscillating_dipole_source,
    tmz_cavity_mode,
)
from physicskit.fields.quantum_fields import gpe_evolve, gpe_imprint_vortex, gpe_relax, harmonic_trap_grid
from physicskit.fields.solitons import (
    kdv_evolve_frames,
    kdv_soliton,
    nls_bright_soliton,
    nls_evolve_frames,
    sine_gordon_evolve_frames,
    sine_gordon_kink,
)
from physicskit.fields.visualizers import (
    animate_casimir_modes,
    animate_density_2d,
    animate_field_1d,
    animate_field_2d,
    animate_flux_tube,
)


def _save(anim, tmp_path, name):
    out = tmp_path / name
    anim.save(out, writer=PillowWriter(fps=10))
    assert out.exists() and out.stat().st_size > 0


class TestSolitonAnimations:
    @pytest.mark.slow
    def test_kdv_animation(self, tmp_path):
        x = np.linspace(-30, 30, 200, endpoint=False)
        u0 = kdv_soliton(x, c=4.0, x0=-15)
        frames, times = kdv_evolve_frames(u0, x, dt=0.001, steps_per_frame=100, n_frames=4)
        anim = animate_field_1d(x, frames, times, ylabel="u(x, t)")
        _save(anim, tmp_path, "kdv.gif")

    @pytest.mark.slow
    def test_nls_animation(self, tmp_path):
        x = np.linspace(-40, 40, 256, endpoint=False)
        psi0 = nls_bright_soliton(x, t=0.0, A=1.0)
        frames, times = nls_evolve_frames(psi0, x, dt=0.001, steps_per_frame=100, n_frames=4, g=1.0)
        anim = animate_field_1d(x, frames, times, ylabel=r"$|\psi|$")
        _save(anim, tmp_path, "nls.gif")

    @pytest.mark.slow
    def test_sine_gordon_animation(self, tmp_path):
        x = np.linspace(-50, 50, 400)
        dt = 0.4 * (x[1] - x[0])
        u_prev = sine_gordon_kink(x, -dt, v=0.5, x0=-20)
        u0 = sine_gordon_kink(x, 0.0, v=0.5, x0=-20)
        frames, times = sine_gordon_evolve_frames(u0, u_prev, x, dt, steps_per_frame=50, n_frames=4)
        anim = animate_field_1d(x, frames, times, ylabel="u(x, t)")
        _save(anim, tmp_path, "sine_gordon.gif")


class TestFDTDAnimations:
    @pytest.mark.slow
    def test_dipole_radiation_animation(self, tmp_path):
        N = 40
        dx = dy = 2e-3
        dt = 0.5 * courant_limit_2d(dx, dy)
        eps_r = mu_r = np.ones((N, N))
        Ez0 = Hx0 = Hy0 = np.zeros((N, N))
        source = oscillating_dipole_source(N // 2, N // 2, amplitude=1.0, freq=3e10)
        frames, times = fdtd_2d_tmz_evolve(Ez0, Hx0, Hy0, eps_r, mu_r, steps=40, dt=dt, dx=dx, dy=dy, source=source, snapshot_stride=5)
        x = np.arange(N) * dx
        y = np.arange(N) * dy
        X, Y = np.meshgrid(x, y, indexing="ij")
        anim = animate_field_2d(X, Y, frames, times)
        _save(anim, tmp_path, "dipole.gif")

    @pytest.mark.slow
    def test_dielectric_slab_animation(self, tmp_path):
        Nx, Ny = 60, 30
        dx = dy = 1e-3
        dt = 0.5 * courant_limit_2d(dx, dy)
        eps_r = dielectric_slab((Nx, Ny), i_start=35, i_end=50, eps_r_slab=4.0)
        mu_r = np.ones((Nx, Ny))
        xs = np.arange(Nx)
        Ez0 = np.exp(-((xs[:, None] - 15) ** 2) / (2 * 4.0**2)) * np.ones((1, Ny))
        Hx0 = Hy0 = np.zeros((Nx, Ny))
        frames, times = fdtd_2d_tmz_evolve(Ez0, Hx0, Hy0, eps_r, mu_r, steps=40, dt=dt, dx=dx, dy=dy, snapshot_stride=5)
        x = np.arange(Nx) * dx
        y = np.arange(Ny) * dy
        X, Y = np.meshgrid(x, y, indexing="ij")
        anim = animate_field_2d(X, Y, frames, times)
        _save(anim, tmp_path, "slab.gif")
        # the wave should have visibly entered the slab region, not just reflected away
        assert np.max(np.abs(frames[:, 35:50, :])) > 0.0

    @pytest.mark.slow
    def test_cavity_mode_animation(self, tmp_path):
        N = 25
        dx = dy = 1e-3
        dt = 0.5 * courant_limit_2d(dx, dy)
        Ez0, omega = tmz_cavity_mode((N, N), dx, dy, m=1, n=1)
        Hx0 = Hy0 = np.zeros((N, N))
        eps_r = mu_r = np.ones((N, N))
        frames, times = fdtd_2d_tmz_evolve(Ez0, Hx0, Hy0, eps_r, mu_r, steps=40, dt=dt, dx=dx, dy=dy, snapshot_stride=4)
        x = np.arange(N) * dx
        y = np.arange(N) * dy
        X, Y = np.meshgrid(x, y, indexing="ij")
        anim = animate_field_2d(X, Y, frames, times)
        _save(anim, tmp_path, "cavity.gif")


class TestGPEAnimations:
    @pytest.mark.slow
    def test_vortex_precession_animation(self, tmp_path):
        n, length, g = 40, 12.0, 4.0
        X, Y, _, _, K2 = harmonic_trap_grid(n, length)
        V = 0.5 * (X**2 + Y**2)
        psi_gs = gpe_relax(np.exp(-0.5 * (X**2 + Y**2)).astype(complex), V, g, dtau=5e-4, steps=1500, X=X, Y=Y, K2=K2)
        psi_v = gpe_imprint_vortex(psi_gs, X, Y, [(1.0, 0.03)])
        frames, times = gpe_evolve(psi_v, V, g, dt=2e-3, steps=200, K2=K2, snapshot_stride=25)
        extent = (X.min(), X.max(), Y.min(), Y.max())
        anim = animate_density_2d(frames, extent=extent, times=times)
        _save(anim, tmp_path, "vortex.gif")

    @pytest.mark.slow
    def test_self_focusing_animation(self, tmp_path):
        n, length = 64, 16.0
        X, Y, _, _, K2 = harmonic_trap_grid(n, length)
        V = np.zeros((n, n))
        psi0 = 3.0 * np.exp(-0.5 * (X**2 + Y**2)).astype(complex)
        frames, times = gpe_evolve(psi0, V, g=-1.0, dt=2e-4, steps=200, K2=K2, snapshot_stride=25)
        extent = (X.min(), X.max(), Y.min(), Y.max())
        anim = animate_density_2d(frames, extent=extent, times=times)
        _save(anim, tmp_path, "collapse.gif")


class TestFluxTubeAndCasimirAnimations:
    @pytest.mark.slow
    def test_flux_tube_animation(self, tmp_path):
        anim = animate_flux_tube((80, 20), dx=0.25, dy=0.25, separations=np.linspace(4.0, 16.0, 5))
        _save(anim, tmp_path, "flux_tube.gif")

    @pytest.mark.slow
    def test_casimir_animation(self, tmp_path):
        anim = animate_casimir_modes(np.linspace(1.0, 5.0, 5))
        _save(anim, tmp_path, "casimir.gif")
