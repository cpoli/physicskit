"""Matplotlib- and Plotly-based visualizers for every physicskit.relativity chapter.

- :mod:`physicskit.relativity.visualizers.shadow_render` -- ray-traced black hole shadows
  and lensed, redshift-shaded accretion disks (Schwarzschild and Kerr).
- :mod:`physicskit.relativity.visualizers.spacetime_3d` -- Flamm paraboloid embedding
  diagrams.
- :mod:`physicskit.relativity.visualizers.spacetime_diagrams` -- Kruskal-Szekeres and
  Penrose-Carter conformal diagrams.
- :mod:`physicskit.relativity.visualizers.wave_plots` -- gravitational wave strain and
  spatial ripple-pattern plots.
- :mod:`physicskit.relativity.visualizers.interactive` -- Plotly-based interactive 3D
  orbits and shadow images.
"""

from physicskit.relativity.visualizers.interactive import interactive_orbit_3d, interactive_shadow_image
from physicskit.relativity.visualizers.shadow_render import plot_black_hole_shadow, render_black_hole_image
from physicskit.relativity.visualizers.spacetime_3d import flamm_paraboloid, plot_flamm_paraboloid
from physicskit.relativity.visualizers.spacetime_diagrams import (
    kruskal_coordinates,
    penrose_carter_coordinates,
    plot_kruskal_diagram,
    plot_penrose_diagram,
)
from physicskit.relativity.visualizers.wave_plots import (
    animate_wave_ripple,
    plot_strain_waveform,
    plot_wave_ripple,
    wave_ripple_snapshot,
)

__all__ = [
    "animate_wave_ripple",
    "flamm_paraboloid",
    "interactive_orbit_3d",
    "interactive_shadow_image",
    "kruskal_coordinates",
    "penrose_carter_coordinates",
    "plot_black_hole_shadow",
    "plot_flamm_paraboloid",
    "plot_kruskal_diagram",
    "plot_penrose_diagram",
    "plot_strain_waveform",
    "plot_wave_ripple",
    "render_black_hole_image",
    "wave_ripple_snapshot",
]
