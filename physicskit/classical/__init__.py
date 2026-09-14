"""physicskit.classical: classical_mechanics_kit.

Visual and computational demonstrations of classical mechanics -- from
Newtonian vector dynamics to Lagrangian variational principles,
Hamiltonian phase spaces, coupled lattice chains, and rigid body
rotations -- built on Numba-accelerated symplectic integrators.
"""

__version__ = "0.1.0"

from physicskit.classical.core.base_system import (
    DynamicalSystem,
    HamiltonianSystem,
    LagrangianSystem,
    ODESystem,
    SimulationResult,
)
from physicskit.classical.systems.chains import FPUTChain, HarmonicChain, SineGordonChain
from physicskit.classical.systems.hamiltonian import HenonHeilesSystem, PendulumSwarm, pendulum_action_angle
from physicskit.classical.systems.lagrangian import BeadOnRotatingHoop, CoupledOscillators, DoublePendulum
from physicskit.classical.systems.newtonian import KeplerSystem, ProjectileMotion
from physicskit.classical.systems.rotations import (
    EulerTop,
    HeavySymmetricTop,
    effective_potential_symmetric_top,
    find_theta_equilibrium,
    nutation_frequency,
    precession_frequency,
)

__all__ = [
    "__version__",
    "DynamicalSystem",
    "ODESystem",
    "HamiltonianSystem",
    "LagrangianSystem",
    "SimulationResult",
    "ProjectileMotion",
    "KeplerSystem",
    "DoublePendulum",
    "BeadOnRotatingHoop",
    "CoupledOscillators",
    "HenonHeilesSystem",
    "PendulumSwarm",
    "pendulum_action_angle",
    "HarmonicChain",
    "FPUTChain",
    "SineGordonChain",
    "EulerTop",
    "HeavySymmetricTop",
    "effective_potential_symmetric_top",
    "find_theta_equilibrium",
    "nutation_frequency",
    "precession_frequency",
]
