"""Fast coverage of physicskit.optics.visualizers: the ax=None/ax=<given>
branch of every function, plus the log_scale=True branches of
plot_diffraction_pattern and animate_diffraction_propagation. See
test_diffraction_animation.py for the real anim.save()-to-disk smoke test
(marked slow)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from physicskit.optics.gaussian import GaussianBeam
from physicskit.optics.ray import OpticalElement, OpticalSystem, free_space, thin_lens
from physicskit.optics.visualizers import animate_diffraction_propagation, plot_beam_envelope, plot_diffraction_pattern, plot_ray_trace
from physicskit.optics.wave import double_slit_aperture


def test_plot_ray_trace_given_ax():
    system = OpticalSystem(
        [
            OpticalElement(free_space(1.0), name="d1", length=1.0),
            OpticalElement(thin_lens(1.0), name="lens"),
        ]
    )
    fig, ax = plot_ray_trace(system, y0=0.5, theta0=0.0)
    assert isinstance(fig, plt.Figure)
    _, ax_given = plot_ray_trace(system, y0=0.5, theta0=0.0, ax=ax)
    assert ax_given is ax


def test_plot_beam_envelope_given_ax():
    beam = GaussianBeam(wavelength=0.5e-3, w0=0.1, z0=0.0)
    fig, ax = plot_beam_envelope(beam, z_range=(-5, 5), n_points=20)
    assert isinstance(fig, plt.Figure)
    _, ax_given = plot_beam_envelope(beam, z_range=(-5, 5), n_points=20, ax=ax)
    assert ax_given is ax


def test_plot_diffraction_pattern_log_scale_and_given_ax():
    dx = 1e-3
    U = double_slit_aperture((16, 16), dx=dx, width=2e-3, separation=1e-2).astype(complex)
    fig, ax = plot_diffraction_pattern(U, dx=dx, log_scale=True)
    assert isinstance(fig, plt.Figure)
    assert ax.get_ylabel() == "y"
    _, ax_given = plot_diffraction_pattern(U, dx=dx, ax=ax)
    assert ax_given is ax


def test_animate_diffraction_propagation_log_scale_and_given_ax_builds_and_draws_a_frame():
    dx = 1e-3
    ap = double_slit_aperture((16, 16), dx=dx, width=2e-3, separation=1e-2).astype(complex)
    z_values = np.linspace(0.01, 1.0, 3)

    fig, ax = plt.subplots()
    anim = animate_diffraction_propagation(ap, wavelength=0.5e-3, z_values=z_values, dx=dx, log_scale=True, ax=ax)
    assert anim._fig is fig
    anim._draw_frame(1)
    assert ax.get_title() == f"z = {z_values[1]:.4g}"
