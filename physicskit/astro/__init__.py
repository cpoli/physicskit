"""physicskit.astro: stellar structure, N-body dynamics, orbital mechanics, and galactic dynamics.

Uses **gravitational units with** :math:`G=1` by default throughout
(every function that needs it accepts ``G`` as a keyword argument), the
same convention :mod:`physicskit.relativity` uses for its own
geometrized units.

- :mod:`physicskit.astro.stellar_structure` -- polytropic stellar models
  via the Lane-Emden equation, the Chandrasekhar mass limit, and the
  main-sequence mass-luminosity relation.
- :mod:`physicskit.astro.nbody` -- direct-summation N-body gravitational
  dynamics with a symplectic leapfrog integrator.
- :mod:`physicskit.astro.orbital_mechanics` -- classical orbital elements,
  state-vector conversion, and Hohmann transfers.
- :mod:`physicskit.astro.galactic_dynamics` -- the NFW dark matter halo
  profile and the rotation curves it produces.
- :mod:`physicskit.astro.visualizers` -- N-body, Lane-Emden, and
  rotation-curve plots.
"""

from physicskit.astro.galactic_dynamics import circular_velocity, nfw_density, nfw_enclosed_mass, nfw_potential
from physicskit.astro.nbody import NBodySystem, gravitational_acceleration, leapfrog_step
from physicskit.astro.orbital_mechanics import (
    hohmann_transfer,
    orbital_elements_from_state,
    orbital_period,
    state_from_orbital_elements,
    vis_viva_speed,
)
from physicskit.astro.stellar_structure import (
    PolytropicStar,
    chandrasekhar_mass,
    lane_emden,
    main_sequence_luminosity,
)
from physicskit.astro.visualizers import plot_lane_emden, plot_nbody_trajectories, plot_rotation_curve

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # stellar_structure
    "lane_emden",
    "PolytropicStar",
    "chandrasekhar_mass",
    "main_sequence_luminosity",
    # nbody
    "gravitational_acceleration",
    "leapfrog_step",
    "NBodySystem",
    # orbital_mechanics
    "orbital_period",
    "vis_viva_speed",
    "orbital_elements_from_state",
    "state_from_orbital_elements",
    "hohmann_transfer",
    # galactic_dynamics
    "nfw_density",
    "nfw_enclosed_mass",
    "nfw_potential",
    "circular_velocity",
    # visualizers
    "plot_nbody_trajectories",
    "plot_lane_emden",
    "plot_rotation_curve",
]
