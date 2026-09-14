"""physicskit.statphys: interactive, computational statistical mechanics.

``physicskit.statphys`` brings the core breakthroughs of statistical mechanics, lattice
physics, critical phenomena, molecular dynamics, and disordered systems into
a visual, computational Python framework:

- **Lattice models** (:mod:`physicskit.statphys.chapters.ising_lattice`): the 2D Ising
  model with Metropolis and Wolff-cluster dynamics, the q-state Potts model,
  and the XY model's topological Kosterlitz-Thouless transition.
- **Disordered systems** (:mod:`physicskit.statphys.chapters.spin_glass`): the
  Edwards-Anderson Ising spin glass, with quenched bond disorder,
  frustration, and the replica-overlap order parameter.
- **Molecular dynamics** (:mod:`physicskit.statphys.chapters.molecular_dynamics`): a
  Velocity-Verlet-integrated Lennard-Jones gas that relaxes toward the
  Maxwell-Boltzmann distribution, with live tracking of Boltzmann's
  H-function.
- **Diffusion** (:mod:`physicskit.statphys.chapters.random_walk`): random walk
  ensembles illustrating the Einstein relation and the central limit
  theorem.
- **Critical phenomena** (:mod:`physicskit.statphys.chapters.percolation`,
  :mod:`physicskit.statphys.chapters.renormalization`, :mod:`physicskit.statphys.chapters.sandpile`):
  site/bond percolation with Hoshen-Kopelman cluster labeling, Kadanoff
  block-spin renormalization group coarse-graining, and the
  self-organized-critical Bak-Tang-Wiesenfeld sandpile.
- **Irreversibility** (:mod:`physicskit.statphys.chapters.ehrenfest_urn`): the
  Ehrenfest urn model, the simplest system reconciling reversible
  microscopic dynamics with an observed thermodynamic arrow of time.
- **Replica symmetry breaking** (:mod:`physicskit.statphys.chapters.spin_glass`): the
  infinite-range Sherrington-Kirkpatrick spin glass and its numerically
  observable replica-overlap distribution.
- **Interface growth** (:mod:`physicskit.statphys.chapters.kpz_growth`): the
  Kardar-Parisi-Zhang universality class via Restricted Solid-On-Solid
  deposition.
- **Nonequilibrium work** (:mod:`physicskit.statphys.chapters.nonequilibrium_work`):
  Jarzynski's equality, recovering exact equilibrium free energies from
  irreversible, finite-speed work measurements.

Examples
--------
>>> from physicskit.statphys.chapters import Ising2D
>>> model = Ising2D(L=32, seed=0)
>>> round(float(model.T_C), 3)
2.269
>>> model.sweep(beta=1.0 / model.T_C, n_sweeps=100)
"""

from physicskit.statphys.chapters import (
    BlockSpinRG,
    BTWSandpile,
    EdwardsAndersonSpinGlass2D,
    EhrenfestUrn,
    Ising2D,
    JarzynskiHarmonicTrap,
    KPZInterface,
    LennardJonesGas,
    Percolation2D,
    PottsModel2D,
    RandomWalk,
    SherringtonKirkpatrick,
    XYModel2D,
)

__version__ = "0.2.0"

__all__ = [
    "BTWSandpile",
    "BlockSpinRG",
    "EdwardsAndersonSpinGlass2D",
    "EhrenfestUrn",
    "Ising2D",
    "JarzynskiHarmonicTrap",
    "KPZInterface",
    "LennardJonesGas",
    "Percolation2D",
    "PottsModel2D",
    "RandomWalk",
    "SherringtonKirkpatrick",
    "XYModel2D",
    "__version__",
]
