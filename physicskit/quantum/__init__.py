"""physicskit.quantum: a visual and computational tour of quantum mechanics.

- :mod:`physicskit.quantum.core.eigensolvers` -- the Numerov shooting-method
  bound-state solver, plus ready-made potential wells (infinite, finite,
  asymmetric step, linear-gravitational, double, harmonic).
- :mod:`physicskit.quantum.core.operators` -- spin, ladder, position, and
  momentum operators as explicit matrices, the Pauli matrices, and generic
  commutator/expectation-value/Hermiticity utilities.
- :mod:`physicskit.quantum.core.solvers` -- split-operator time propagation
  for the time-dependent Schrodinger equation, in 1D and 2D.
- :mod:`physicskit.quantum.chapters.wave_packets` -- free and dispersing
  Gaussian wave packets, the double-slit experiment, and quantum revivals.
- :mod:`physicskit.quantum.chapters.potentials` -- asymmetric/step wells,
  the quantum bouncer, the double well, the finite square well, and 2D
  quantum boxes (rectangle, circular dot, stadium billiard).
- :mod:`physicskit.quantum.chapters.hydrogen_am` -- the hydrogen atom's
  exact radial and angular (spherical harmonic) wavefunctions.
- :mod:`physicskit.quantum.chapters.harmonic_spin` -- the quantum harmonic
  oscillator (ladder operators, coherent states) and thermal (mixed) states.
- :mod:`physicskit.quantum.chapters.perturbation` -- the Zeeman and Stark
  effects, and a Floquet-driven infinite square well.
- :mod:`physicskit.quantum.chapters.entanglement` -- Bell states and CHSH
  correlations, and the Aharonov-Bohm ring.

See :mod:`physicskit.semiclassical` for WKB/EBK quantization, Van
Vleck/Herman-Kluk semiclassical propagators, the Gutzwiller trace
formula, and quantum scarring.
"""

__version__ = "0.1.0"

from physicskit.quantum.chapters.entanglement import (
    AharonovBohmRing,
    BellCorrelations,
    IsingEntangler,
    bell_state,
)
from physicskit.quantum.chapters.harmonic_spin import HarmonicOscillator, ThermalState
from physicskit.quantum.chapters.hydrogen_am import (
    HydrogenOrbital,
    orbital_superposition_density,
    orbital_superposition_psi,
    radial_wavefunction,
    spherical_harmonic,
)
from physicskit.quantum.chapters.perturbation import (
    FloquetDrivenBox,
    StarkLevel,
    linear_stark_shift,
    stark_n2_quartet,
    zeeman_spectrum,
    zeeman_splitting,
)
from physicskit.quantum.chapters.potentials import (
    Box2DEigenstate,
    CircularBox2D,
    CircularEigenstate,
    DoubleWellResult,
    DoubleWellSimulator,
    FiniteSquareWell,
    RectangularBox2D,
    ScatteringResult,
    StadiumBilliard2D,
    airy_bouncer_energies,
    airy_wavefunction,
    asymmetric_well_states,
    gravitational_bouncer_states,
)
from physicskit.quantum.chapters.spin import RabiProblem, SternGerlach
from physicskit.quantum.chapters.wave_packets import (
    GaussianDispersion,
    QuantumRevival,
    TwinSlit,
    double_slit_potential,
    free_gaussian_wavepacket,
    propagate_double_slit,
)
from physicskit.quantum.core.eigensolvers import (
    EigenResult,
    NumerovSolver,
    asymmetric_step_well,
    double_well,
    finite_well,
    harmonic_well,
    infinite_well,
    linear_gravitational_well,
)
from physicskit.quantum.core.operators import (
    annihilation_operator,
    anticommutator,
    commutator,
    creation_operator,
    expectation,
    identity2,
    is_hermitian,
    momentum_operator,
    number_operator,
    position_operator,
    sigma_x,
    sigma_y,
    sigma_z,
    spin_operator,
)
from physicskit.quantum.core.solvers import SplitOperatorSolver1D, SplitOperatorSolver2D

__all__ = [
    "__version__",
    # core.eigensolvers
    "EigenResult",
    "NumerovSolver",
    "infinite_well",
    "finite_well",
    "asymmetric_step_well",
    "linear_gravitational_well",
    "double_well",
    "harmonic_well",
    # core.operators
    "sigma_x",
    "sigma_y",
    "sigma_z",
    "identity2",
    "spin_operator",
    "annihilation_operator",
    "creation_operator",
    "number_operator",
    "position_operator",
    "momentum_operator",
    "commutator",
    "anticommutator",
    "expectation",
    "is_hermitian",
    # core.solvers
    "SplitOperatorSolver1D",
    "SplitOperatorSolver2D",
    # chapters.wave_packets
    "free_gaussian_wavepacket",
    "GaussianDispersion",
    "TwinSlit",
    "QuantumRevival",
    "double_slit_potential",
    "propagate_double_slit",
    # chapters.potentials
    "asymmetric_well_states",
    "gravitational_bouncer_states",
    "airy_bouncer_energies",
    "airy_wavefunction",
    "DoubleWellResult",
    "DoubleWellSimulator",
    "ScatteringResult",
    "FiniteSquareWell",
    "Box2DEigenstate",
    "RectangularBox2D",
    "CircularEigenstate",
    "CircularBox2D",
    "StadiumBilliard2D",
    # chapters.hydrogen_am
    "spherical_harmonic",
    "radial_wavefunction",
    "HydrogenOrbital",
    "orbital_superposition_psi",
    "orbital_superposition_density",
    # chapters.harmonic_spin
    "HarmonicOscillator",
    "ThermalState",
    # chapters.perturbation
    "zeeman_splitting",
    "zeeman_spectrum",
    "linear_stark_shift",
    "StarkLevel",
    "stark_n2_quartet",
    "FloquetDrivenBox",
    # chapters.entanglement
    "bell_state",
    "BellCorrelations",
    "AharonovBohmRing",
    "IsingEntangler",
    # chapters.spin
    "SternGerlach",
    "RabiProblem",
]
