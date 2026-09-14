"""Animation smoke + sanity tests for physicskit.optics.visualizers.animate_diffraction_propagation."""

from __future__ import annotations

import numpy as np
from matplotlib.animation import PillowWriter

from physicskit.optics.visualizers import animate_diffraction_propagation
from physicskit.optics.wave import angular_spectrum_propagate, double_slit_aperture, intensity


def test_diffraction_propagation_animation(tmp_path):
    dx = 1e-3
    ap = double_slit_aperture((64, 64), dx=dx, width=2e-3, separation=1e-2).astype(complex)
    z_values = np.linspace(0.01, 2.0, 6)
    anim = animate_diffraction_propagation(ap, wavelength=0.5e-3, z_values=z_values, dx=dx)
    out = tmp_path / "diffraction.gif"
    anim.save(out, writer=PillowWriter(fps=10))
    assert out.exists() and out.stat().st_size > 0


def test_pattern_widens_as_z_increases_matching_angular_spectrum_directly():
    """The animation just steps angular_spectrum_propagate over z -- sanity-check that the
    interference pattern's spatial extent (its second moment) actually grows with z, as
    expected of a diverging diffraction pattern developing from the near to the far field."""
    dx = 1e-3
    ap = double_slit_aperture((64, 64), dx=dx, width=2e-3, separation=1e-2).astype(complex)
    z_small, z_large = 0.05, 3.0
    x = (np.arange(64) - 32) * dx
    X, _ = np.meshgrid(x, x)

    def spread(z):
        I = intensity(angular_spectrum_propagate(ap, wavelength=0.5e-3, z=z, dx=dx))
        return np.sqrt(np.sum(I * X**2) / np.sum(I))

    assert spread(z_large) > spread(z_small)
