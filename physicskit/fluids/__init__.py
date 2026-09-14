"""physicskit.fluids: fluid dynamics from inviscid potential flow to compressible shocks.

Six physical regimes, each a module of :mod:`physicskit.fluids.systems`,
sharing the doubly-periodic pseudo-spectral grid and RK4 time-stepping core
in :mod:`physicskit.fluids.core` wherever a grid-based simulation is needed:

- :mod:`~physicskit.fluids.systems.potential_flow` -- inviscid, irrotational
  flow built by superposing elementary solutions (uniform flow,
  source/sink, doublet, point vortex), including flow past a cylinder with
  lift.
- :mod:`~physicskit.fluids.systems.viscous_flow` -- classic exact viscous
  solutions: Couette and Poiseuille flow, Stokes drag, and the Blasius
  boundary layer.
- :mod:`~physicskit.fluids.systems.vortex_dynamics` -- point-vortex N-body
  dynamics via the Biot-Savart law, and the von Karman vortex street.
- :mod:`~physicskit.fluids.systems.instabilities` -- the Kelvin-Helmholtz
  and Rayleigh-Taylor instabilities, each with its linear growth-rate law
  and a full nonlinear simulation.
- :mod:`~physicskit.fluids.systems.compressible_flow` -- the 1D Euler
  equations, Rankine-Hugoniot shock relations, and the Sod shock tube.
- :mod:`~physicskit.fluids.systems.navier_stokes` -- the 2D incompressible
  vorticity-streamfunction solver underlying the instability and
  turbulence-spectrum tools.

Typical usage::

    import physicskit as pk
    import numpy as np

    # Flow past a lifting cylinder:
    flow = pk.fluids.flow_past_cylinder(U_inf=1.0, radius=1.0, circulation=4 * np.pi)
    lift = pk.fluids.kutta_joukowski_lift(rho=1.2, U_inf=1.0, circulation=4 * np.pi)
"""

from physicskit.fluids.exceptions import FluidskitError, InvalidParameterError
from physicskit.fluids.systems.compressible_flow import (
    normal_shock_relations,
    rankine_hugoniot_jump_conditions,
    sod_shock_tube,
)
from physicskit.fluids.systems.instabilities import (
    kelvin_helmholtz_growth_rate,
    kelvin_helmholtz_ic,
    rayleigh_taylor_growth_rate,
    rayleigh_taylor_ic,
    simulate_rayleigh_taylor,
)
from physicskit.fluids.systems.navier_stokes import NavierStokes2D, simulate_vorticity_streamfunction
from physicskit.fluids.systems.potential_flow import (
    PotentialFlow,
    doublet_potential,
    flow_past_cylinder,
    kutta_joukowski_lift,
    point_vortex_potential,
    pressure_coefficient,
    source_potential,
    uniform_flow_potential,
)
from physicskit.fluids.systems.viscous_flow import (
    blasius_boundary_layer_thickness,
    blasius_skin_friction_coefficient,
    blasius_solve,
    couette_flow_velocity,
    poiseuille_flow_rate,
    poiseuille_flow_velocity,
    stokes_drag,
)
from physicskit.fluids.systems.vortex_dynamics import (
    VON_KARMAN_SPACING_RATIO,
    PointVortexSystem,
    point_vortex_velocities,
    von_karman_vortex_street,
)
from physicskit.fluids.utils.dimensionless import (
    froude_number,
    mach_number,
    reynolds_number,
    strouhal_number,
    weber_number,
)
from physicskit.fluids.utils.spectral_analysis import energy_spectrum, kolmogorov_reference_slope
from physicskit.fluids.visualizers.compressible import plot_shock_tube_profiles
from physicskit.fluids.visualizers.flow_fields import plot_streamlines, plot_vorticity_field
from physicskit.fluids.visualizers.potential_flow import plot_pressure_coefficient
from physicskit.fluids.visualizers.spectra import plot_energy_spectrum

__version__ = "0.1.0"

__all__ = [
    "VON_KARMAN_SPACING_RATIO",
    "FluidskitError",
    "InvalidParameterError",
    "NavierStokes2D",
    "PointVortexSystem",
    "PotentialFlow",
    "blasius_boundary_layer_thickness",
    "blasius_skin_friction_coefficient",
    "blasius_solve",
    "couette_flow_velocity",
    "doublet_potential",
    "energy_spectrum",
    "flow_past_cylinder",
    "froude_number",
    "kelvin_helmholtz_growth_rate",
    "kelvin_helmholtz_ic",
    "kolmogorov_reference_slope",
    "kutta_joukowski_lift",
    "mach_number",
    "normal_shock_relations",
    "plot_energy_spectrum",
    "plot_pressure_coefficient",
    "plot_shock_tube_profiles",
    "plot_streamlines",
    "plot_vorticity_field",
    "point_vortex_potential",
    "point_vortex_velocities",
    "poiseuille_flow_rate",
    "poiseuille_flow_velocity",
    "pressure_coefficient",
    "rankine_hugoniot_jump_conditions",
    "rayleigh_taylor_growth_rate",
    "rayleigh_taylor_ic",
    "reynolds_number",
    "simulate_rayleigh_taylor",
    "simulate_vorticity_streamfunction",
    "sod_shock_tube",
    "source_potential",
    "stokes_drag",
    "strouhal_number",
    "uniform_flow_potential",
    "von_karman_vortex_street",
    "weber_number",
    "__version__",
]
