"""Animation smoke tests for physicskit.fluids.visualizers.flow_fields, previously untested
(no visualizer in physicskit.fluids had any test coverage before this). These run headlessly
(see conftest.py's Agg backend) and check that the Kelvin-Helmholtz and Rayleigh-Taylor
animations build and save correctly.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.animation import PillowWriter

from physicskit.fluids.systems.instabilities import kelvin_helmholtz_ic, rayleigh_taylor_ic
from physicskit.fluids.visualizers.flow_fields import animate_kelvin_helmholtz, animate_rayleigh_taylor


@pytest.mark.slow
def test_animate_kelvin_helmholtz_saves_to_gif(tmp_path):
    n, length = 64, 2 * np.pi
    omega0 = kelvin_helmholtz_ic(n, length, shear_width=0.1, perturbation_amplitude=0.05)
    anim = animate_kelvin_helmholtz(omega0, nu=0.001, dt=0.0025, steps_per_frame=20, n_frames=5, length=length)
    out = tmp_path / "kelvin_helmholtz.gif"
    anim.save(out, writer=PillowWriter(fps=10))
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.slow
def test_animate_rayleigh_taylor_saves_to_gif(tmp_path):
    n, length = 64, 2 * np.pi
    omega0, buoyancy0 = rayleigh_taylor_ic(n, length, atwood_number=0.3, perturbation_amplitude=0.01)
    anim = animate_rayleigh_taylor(omega0, buoyancy0, nu=0.002, kappa=0.002, g=1.0, dt=0.01, steps_per_frame=20, n_frames=5, length=length)
    out = tmp_path / "rayleigh_taylor.gif"
    anim.save(out, writer=PillowWriter(fps=10))
    assert out.exists() and out.stat().st_size > 0
