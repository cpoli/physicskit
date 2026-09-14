"""Smoke tests for physicskit.particle.visualizers: every animation
builds and saves a nonempty GIF (see physicskit.classical.tests.test_visualizers
for the pattern this follows)."""

from __future__ import annotations

import numpy as np
from matplotlib.animation import PillowWriter

from physicskit.particle.collider import charged_track_points, parton_shower, simple_shower
from physicskit.particle.confinement import string_break_chain
from physicskit.particle.decays import bateman_decay_chain, sample_michel_electron_energies
from physicskit.particle.electroweak import higgs_field_rollover
from physicskit.particle.kinematics import FourVector
from physicskit.particle.visualizers import (
    animate_cp_asymmetry,
    animate_decay_chain_bars,
    animate_detector_event,
    animate_higgs_rollover,
    animate_michel_histogram,
    animate_neutrino_oscillation,
    animate_particle_cascade,
    animate_parton_shower,
    animate_qed_angular_distribution,
    animate_string_breaking,
    plot_decay_chain,
    plot_differential_cross_section,
)


def _save(anim, tmp_path, name):
    out = tmp_path / name
    anim.save(out, writer=PillowWriter(fps=10))
    assert out.exists() and out.stat().st_size > 0


def test_static_plots_still_work():
    from physicskit.particle.scattering import rutherford_dsigma_domega

    t = np.linspace(0, 10, 20)
    N = bateman_decay_chain(1000.0, [0.5, 0.2], t)
    fig, ax = plot_decay_chain(t, N)
    assert fig is not None and ax is not None

    theta = np.linspace(0.1, np.pi - 0.1, 20)
    vals = rutherford_dsigma_domega(theta, Z1=2, Z2=79, E_kin=5.0)
    fig2, ax2 = plot_differential_cross_section(theta, vals)
    assert fig2 is not None and ax2 is not None


def test_animate_particle_cascade_saves_gif(tmp_path):
    root = simple_shower(50.0, E_threshold=6.0, rng=np.random.default_rng(0))
    anim = animate_particle_cascade(root)
    _save(anim, tmp_path, "cascade.gif")


def test_animate_parton_shower_saves_gif(tmp_path):
    root = parton_shower(60.0, E_threshold=6.0, rng=np.random.default_rng(1))
    anim = animate_parton_shower(root, n_jets=2)
    _save(anim, tmp_path, "parton_shower.gif")


def test_animate_string_breaking_saves_gif(tmp_path):
    sim = string_break_chain(np.linspace(0, 20, 60), v=0.3, kappa=1.0, m_q=1.0, n_breaks=3)
    anim = animate_string_breaking(sim)
    _save(anim, tmp_path, "string_breaking.gif")


def test_animate_higgs_rollover_saves_gif(tmp_path):
    t = np.linspace(0, 40, 150)
    phi, _ = higgs_field_rollover(1e-3, 0.0, a=1.0, b=1.0, t_eval=t, damping=0.06)
    anim = animate_higgs_rollover(phi, a=1.0, b=1.0)
    _save(anim, tmp_path, "higgs_rollover.gif")


def test_animate_decay_chain_bars_saves_gif(tmp_path):
    t = np.linspace(0, 10, 40)
    N = bateman_decay_chain(1000.0, [0.5, 0.2], t)
    anim = animate_decay_chain_bars(t, N, labels=["Parent", "Daughter"])
    _save(anim, tmp_path, "decay_chain_bars.gif")


def test_animate_neutrino_oscillation_saves_gif(tmp_path):
    L = np.linspace(0, 1000, 60)
    anim = animate_neutrino_oscillation(L, E=1.0, theta=0.6, delta_m2=2.5e-3)
    _save(anim, tmp_path, "neutrino_oscillation.gif")


def test_animate_qed_angular_distribution_saves_gif(tmp_path):
    sqrt_s = np.linspace(5.0, 20.0, 15)
    anim = animate_qed_angular_distribution(sqrt_s)
    _save(anim, tmp_path, "qed_angular.gif")


def test_animate_michel_histogram_saves_gif(tmp_path):
    xs = sample_michel_electron_energies(600, rng=np.random.default_rng(0))
    anim = animate_michel_histogram(xs, batch_size=100)
    _save(anim, tmp_path, "michel_histogram.gif")


def test_animate_cp_asymmetry_saves_gif(tmp_path):
    t = np.linspace(0, 20, 80)
    anim = animate_cp_asymmetry(t, delta_m=0.5, gamma_s=1.0, gamma_l=0.1, epsilon=0.003 + 0.001j)
    _save(anim, tmp_path, "cp_asymmetry.gif")


def test_animate_detector_event_saves_gif(tmp_path):
    ps = [FourVector(6.0, 3.0, 0.0, 4.0), FourVector(5.0, 0.0, 3.0, 3.0), FourVector(4.0, -2.0, -1.0, 3.0)]
    charges = [1.0, -1.0, 0.0]
    anim = animate_detector_event(ps, charges, B=0.5, n_frames=30)
    _save(anim, tmp_path, "detector_event.gif")


def test_charged_track_points_used_by_event_display_is_consistent():
    p = FourVector(5.0, 3.0, 0.0, 3.0)
    pts = charged_track_points(p, charge=1.0, B=0.5, path_length=2.0)
    assert pts.shape[1] == 2
