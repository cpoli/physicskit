r"""physicskit.semiclassical: WKB/EBK quantization, Van Vleck/Herman-Kluk semiclassical propagators, the Gutzwiller trace formula, and quantum scarring.

- :mod:`physicskit.semiclassical.core.wkb` -- 1D WKB wavefunctions,
  classical turning points, and Bohr-Sommerfeld (EBK) quantization.
- :mod:`physicskit.semiclassical.core.propagators` -- the Van
  Vleck-Morette semiclassical propagator and the multi-trajectory
  Herman-Kluk frozen-Gaussian propagator, both built on a Numba-jitted
  classical trajectory + monodromy-matrix integrator.
- :mod:`physicskit.semiclassical.core.gutzwiller` -- the (exact, for
  1D bound systems) Gutzwiller trace formula, reconstructing a spectrum
  from the classical action and period of a single periodic orbit.
- :mod:`physicskit.semiclassical.systems.scarring` -- quantum scars on
  the stadium billiard's bouncing-ball orbit family, and a non-periodic
  Husimi phase-space projection.
- :mod:`physicskit.semiclassical.visualizers` -- classical
  trajectories overlaid on Wigner functions, trace-formula spectra, and
  scar maps.
"""

from __future__ import annotations

from physicskit.semiclassical.core.gutzwiller import (
    classical_period,
    gutzwiller_amplitude_from_monodromy,
    gutzwiller_density_of_states,
)
from physicskit.semiclassical.core.propagators import (
    coherent_state_overlap,
    count_caustics,
    frozen_gaussian_1d,
    herman_kluk_prefactor,
    herman_kluk_propagate_wavepacket,
    propagate_trajectory_monodromy_action,
    van_vleck_prefactor,
    van_vleck_propagator_1d,
)
from physicskit.semiclassical.core.wkb import (
    bohr_sommerfeld_energies,
    classical_momentum,
    turning_points,
    wkb_action,
    wkb_wavefunction,
)
from physicskit.semiclassical.systems.scarring import (
    bouncing_ball_energies,
    bouncing_ball_orbit_points,
    husimi_projection_1d,
    scar_enhancement,
)
from physicskit.semiclassical.visualizers import (
    plot_classical_trajectory_on_wigner,
    plot_density_of_states,
    plot_husimi_1d,
    plot_scar_map,
    plot_scar_map_interactive,
    plot_wkb_wavefunction,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # core.wkb
    "classical_momentum",
    "turning_points",
    "wkb_action",
    "bohr_sommerfeld_energies",
    "wkb_wavefunction",
    # core.propagators
    "propagate_trajectory_monodromy_action",
    "van_vleck_prefactor",
    "count_caustics",
    "van_vleck_propagator_1d",
    "frozen_gaussian_1d",
    "coherent_state_overlap",
    "herman_kluk_prefactor",
    "herman_kluk_propagate_wavepacket",
    # core.gutzwiller
    "classical_period",
    "gutzwiller_density_of_states",
    "gutzwiller_amplitude_from_monodromy",
    # systems.scarring
    "bouncing_ball_energies",
    "bouncing_ball_orbit_points",
    "scar_enhancement",
    "husimi_projection_1d",
    # visualizers
    "plot_wkb_wavefunction",
    "plot_classical_trajectory_on_wigner",
    "plot_density_of_states",
    "plot_scar_map",
    "plot_husimi_1d",
    "plot_scar_map_interactive",
]
