"""Matplotlib- and Plotly-based visualizers for every physicskit.statphys chapter.

- :mod:`physicskit.statphys.visualizers.lattice_render` -- spin grid heatmaps, sweep
  animations, thermodynamic diagnostic plots, and percolation cluster and
  cluster-size-distribution plots.
- :mod:`physicskit.statphys.visualizers.vortex_render` -- XY model vector fields and
  topological vortex/antivortex overlays.
- :mod:`physicskit.statphys.visualizers.particle_render` -- Lennard-Jones gas snapshots,
  velocity histograms, and motion animations.
- :mod:`physicskit.statphys.visualizers.rg_render` -- side-by-side block-spin
  renormalization group flow panels.
- :mod:`physicskit.statphys.visualizers.random_walk_render` -- walk trajectories, MSD,
  and central-limit-theorem histograms.
- :mod:`physicskit.statphys.visualizers.sandpile_render` -- sandpile heights and
  avalanche size distributions.
- :mod:`physicskit.statphys.visualizers.urn_render` -- Ehrenfest urn relaxation and
  entropy history.
- :mod:`physicskit.statphys.visualizers.spin_glass_render` -- spin-glass replica-overlap
  distributions.
- :mod:`physicskit.statphys.visualizers.kpz_render` -- RSOS interface profiles and
  width-growth scaling.
- :mod:`physicskit.statphys.visualizers.jarzynski_render` -- nonequilibrium work
  distributions and free-energy estimates.
- :mod:`physicskit.statphys.visualizers.interactive` -- Plotly-based interactive
  counterparts of the temperature sweep, vortex field, and particle
  snapshot plots.
"""

from physicskit.statphys.visualizers.interactive import (
    interactive_particle_snapshot,
    interactive_temperature_sweep,
    interactive_vortex_field,
)
from physicskit.statphys.visualizers.jarzynski_render import plot_work_distribution
from physicskit.statphys.visualizers.kpz_render import plot_interface_profile, plot_width_growth
from physicskit.statphys.visualizers.lattice_render import (
    animate_lattice_sweeps,
    plot_cluster_size_distribution,
    plot_percolation_clusters,
    plot_spin_grid,
    plot_thermodynamics,
)
from physicskit.statphys.visualizers.particle_render import (
    animate_gas,
    plot_particle_snapshot,
    plot_velocity_histogram,
)
from physicskit.statphys.visualizers.random_walk_render import (
    plot_clt_histogram,
    plot_msd,
    plot_trajectories_2d,
)
from physicskit.statphys.visualizers.rg_render import plot_rg_flow
from physicskit.statphys.visualizers.sandpile_render import (
    plot_avalanche_size_distribution,
    plot_sandpile_heights,
)
from physicskit.statphys.visualizers.spin_glass_render import plot_overlap_distribution
from physicskit.statphys.visualizers.urn_render import plot_ehrenfest_history
from physicskit.statphys.visualizers.vortex_render import plot_vortices, plot_xy_vector_field

__all__ = [
    "animate_gas",
    "animate_lattice_sweeps",
    "interactive_particle_snapshot",
    "interactive_temperature_sweep",
    "interactive_vortex_field",
    "plot_avalanche_size_distribution",
    "plot_clt_histogram",
    "plot_cluster_size_distribution",
    "plot_ehrenfest_history",
    "plot_interface_profile",
    "plot_msd",
    "plot_overlap_distribution",
    "plot_particle_snapshot",
    "plot_percolation_clusters",
    "plot_rg_flow",
    "plot_sandpile_heights",
    "plot_spin_grid",
    "plot_thermodynamics",
    "plot_trajectories_2d",
    "plot_velocity_histogram",
    "plot_vortices",
    "plot_width_growth",
    "plot_work_distribution",
    "plot_xy_vector_field",
]
