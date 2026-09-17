"""Tests for physicskit.classical.visualizers, previously untested (the whole package
had only manual, un-persisted smoke checks during development). These
run headlessly (see conftest.py's Agg backend) and check both "does it
run without raising" and, where practical, that the output is actually
correct (e.g. Poincare-section crossings at known locations), not just
that a figure object came back.
"""

from __future__ import annotations

import numpy as np
from matplotlib.animation import PillowWriter

from physicskit.classical.systems.chains import FPUTChain
from physicskit.classical.systems.hamiltonian import HenonHeilesSystem
from physicskit.classical.systems.lagrangian import DoublePendulum, ElasticPendulum
from physicskit.classical.systems.newtonian import KeplerSystem
from physicskit.classical.systems.rotations import EulersDisk, EulerTop, Rattleback
from physicskit.classical.visualizers.animations import (
    SideBySideAnimator,
    animate_elastic_pendulum,
    animate_eulers_disk,
    animate_rattleback,
    animate_rigid_body_tumble,
    orbit_trace_animation,
    pendulum_animation,
)
from physicskit.classical.visualizers.interactive import interactive_orbit, interactive_so3_momentum_sphere
from physicskit.classical.visualizers.modal_analysis import animate_modal_energies, plot_modal_energy_bars
from physicskit.classical.visualizers.phase_space import (
    plot_phase_portrait,
    plot_phase_swarm,
    plot_poincare_section,
    plot_so3_momentum_sphere,
    poincare_section,
)


def test_poincare_section_finds_known_crossings():
    """Synthetic q(t) = sin(t), p(t) = cos(t): q crosses zero going
    upward (direction=+1, p>0) exactly at t = 0, 2*pi, 4*pi, ..."""
    t = np.linspace(0, 4 * np.pi, 20_000)
    q = np.sin(t)
    p = np.cos(t)

    cq, cp = poincare_section(q, p, section_q=q, section_p=p, value=0.0, direction=1)

    assert len(cq) == 2  # crossings at t=0 (endpoint, not detected by the interior scan) is excluded; t=2pi, 4pi found
    assert np.allclose(cq, 0.0, atol=1e-3)
    assert np.allclose(cp, 1.0, atol=1e-3)


def test_plot_phase_portrait_1d_and_2d():
    q1 = np.linspace(0, 1, 50)
    p1 = np.cos(q1)
    ax = plot_phase_portrait(q1, p1)
    assert ax is not None

    q2 = np.column_stack([q1, q1 * 2])
    p2 = np.column_stack([p1, -p1])
    ax2 = plot_phase_portrait(q2, p2, label="test")
    assert len(ax2.lines) >= 2


def test_plot_phase_swarm_and_poincare_wrapper():
    q = np.random.default_rng(0).normal(size=200)
    p = np.random.default_rng(1).normal(size=200)
    ax = plot_phase_swarm(q, p)
    assert ax is not None

    hh = HenonHeilesSystem(np.array([0.0, 0.3]), np.array([0.3, 0.0]))
    result = hh.integrate((0, 100), dt=0.01, method="yoshida4")
    ax2 = plot_poincare_section(result.q[:, 0], result.p[:, 0], result.q[:, 1], result.p[:, 1], value=0.0, direction=1)
    assert ax2 is not None


def test_plot_so3_momentum_sphere():
    top = EulerTop([0.01, 1.0, 0.01], I1=1.0, I2=2.0, I3=3.0)
    result = top.integrate((0, 10), dt=1e-3, method="implicit_midpoint")
    ax = plot_so3_momentum_sphere(result.y[:, :3], I1=1.0, I2=2.0, I3=3.0)
    assert ax is not None


def test_modal_energy_bars_and_animation(tmp_path):
    chain = FPUTChain(n=16, beta=0.5, mode=1, amplitude=0.5)
    result = chain.integrate((0, 5), dt=0.05, method="yoshida4")  # short: this only needs a few frames

    ax = plot_modal_energy_bars(chain, result.q[0], result.p[0])
    assert len(ax.patches) == 16

    anim = animate_modal_energies(chain, result, stride=10)
    out = tmp_path / "modal_energy.gif"
    anim.save(out, writer=PillowWriter(fps=10))
    assert out.exists() and out.stat().st_size > 0


def test_side_by_side_animator_pendulum_and_orbit(tmp_path):
    dp = DoublePendulum([2.0, 1.0], [0.5, -0.3])
    result = dp.integrate((0, 0.02), dt=1e-4, method="implicit_midpoint")  # short: just needs a few frames
    anim = pendulum_animation(result, dp.positions, mode="phase", stride=10)
    assert isinstance(anim, SideBySideAnimator)
    fig = anim.build()
    assert fig is not None
    out1 = tmp_path / "pendulum.gif"
    anim.save(out1, writer=PillowWriter(fps=10))
    assert out1.exists() and out1.stat().st_size > 0

    kep = KeplerSystem.from_orbital_elements(a=1.0, e=0.4)
    result2 = kep.integrate((0, 0.5), dt=1e-3, method="yoshida4")
    anim2 = orbit_trace_animation(result2, mode="energy", stride=50)
    out2 = tmp_path / "orbit.gif"
    anim2.save(out2, writer=PillowWriter(fps=10))
    assert out2.exists() and out2.stat().st_size > 0


def test_animate_elastic_pendulum_saves_gif(tmp_path):
    system = ElasticPendulum([0.1, 0.05], [0.0, 0.0])
    # short: animate_elastic_pendulum has no stride and renders one frame
    # per sample, so this only needs enough samples for a valid animation
    result = system.integrate((0, 0.1), dt=1e-3, method="implicit_midpoint")
    anim = animate_elastic_pendulum(system, result)
    out = tmp_path / "elastic_pendulum.gif"
    anim.save(out, writer=PillowWriter(fps=10))
    assert out.exists() and out.stat().st_size > 0


def test_animate_rigid_body_tumble_saves_gif(tmp_path):
    top = EulerTop([0.01, 1.0, 0.01], I1=1.0, I2=2.0, I3=3.0)
    result = top.integrate((0, 2.0), dt=1e-3, method="implicit_midpoint")
    anim = animate_rigid_body_tumble(top, result, stride=20)
    out = tmp_path / "rigid_body_tumble.gif"
    anim.save(out, writer=PillowWriter(fps=10))
    assert out.exists() and out.stat().st_size > 0


def test_animate_eulers_disk_saves_gif(tmp_path):
    disk = EulersDisk(0.5, decay_rate=0.02, precession_const=1.0)
    # short: animate_eulers_disk has no stride and renders one frame per
    # sample (collapse time here is t_f=6.25, so 0.3 stays far from it)
    result = disk.integrate((0.0, 0.15), dt=1e-3, method="rk4")
    anim = animate_eulers_disk(disk, result)
    out = tmp_path / "eulers_disk.gif"
    anim.save(out, writer=PillowWriter(fps=10))
    assert out.exists() and out.stat().st_size > 0


def test_animate_rattleback_saves_gif(tmp_path):
    system = Rattleback([0.01, 0.01, 3.0])
    result = system.integrate((0.0, 8.0), dt=5e-3, method="rk4")
    anim = animate_rattleback(system, result, stride=20)
    out = tmp_path / "rattleback.gif"
    anim.save(out, writer=PillowWriter(fps=10))
    assert out.exists() and out.stat().st_size > 0


def test_interactive_visualizers_return_populated_figures():
    top = EulerTop([0.01, 1.0, 0.01], I1=1.0, I2=2.0, I3=3.0)
    result = top.integrate((0, 10), dt=1e-3, method="implicit_midpoint")
    fig = interactive_so3_momentum_sphere(result.y[:, :3], I1=1.0, I2=2.0, I3=3.0)
    assert len(fig.data) == 3  # sphere surface + trajectory + start marker

    kep = KeplerSystem.from_orbital_elements(a=1.0, e=0.3, c_pn=0.01)
    result2 = kep.integrate((0, 50), dt=1e-3, method="yoshida4")
    fig2 = interactive_orbit(result2, system=kep)
    assert len(fig2.data) == 4  # orbit + focus + 2 LRL arrows
