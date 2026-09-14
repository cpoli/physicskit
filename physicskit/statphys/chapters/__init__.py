"""Physics chapters: user-facing model classes built on :mod:`physicskit.statphys.core`.

- :mod:`physicskit.statphys.chapters.ising_lattice` -- Ising, Potts, and XY lattice spin models.
- :mod:`physicskit.statphys.chapters.spin_glass` -- the Edwards-Anderson Ising spin glass and the
  infinite-range Sherrington-Kirkpatrick model.
- :mod:`physicskit.statphys.chapters.molecular_dynamics` -- Lennard-Jones gas and the H-theorem.
- :mod:`physicskit.statphys.chapters.random_walk` -- random walks, diffusion, and the CLT.
- :mod:`physicskit.statphys.chapters.percolation` -- site/bond percolation and cluster geometry.
- :mod:`physicskit.statphys.chapters.renormalization` -- Kadanoff block-spin coarse-graining.
- :mod:`physicskit.statphys.chapters.sandpile` -- the Bak-Tang-Wiesenfeld self-organized-critical sandpile.
- :mod:`physicskit.statphys.chapters.ehrenfest_urn` -- the Ehrenfest urn model of statistical irreversibility.
- :mod:`physicskit.statphys.chapters.kpz_growth` -- the Kardar-Parisi-Zhang universality class via
  Restricted Solid-On-Solid interface growth.
- :mod:`physicskit.statphys.chapters.nonequilibrium_work` -- Jarzynski's equality for a dragged
  harmonic trap.
"""

from physicskit.statphys.chapters.ehrenfest_urn import EhrenfestUrn
from physicskit.statphys.chapters.ising_lattice import Ising2D, PottsModel2D, XYModel2D
from physicskit.statphys.chapters.kpz_growth import KPZInterface
from physicskit.statphys.chapters.molecular_dynamics import LennardJonesGas
from physicskit.statphys.chapters.nonequilibrium_work import JarzynskiHarmonicTrap
from physicskit.statphys.chapters.percolation import Percolation2D
from physicskit.statphys.chapters.random_walk import RandomWalk
from physicskit.statphys.chapters.renormalization import BlockSpinRG
from physicskit.statphys.chapters.sandpile import BTWSandpile
from physicskit.statphys.chapters.spin_glass import EdwardsAndersonSpinGlass2D, SherringtonKirkpatrick

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
]
