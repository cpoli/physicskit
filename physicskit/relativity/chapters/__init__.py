"""Physics chapters: user-facing model classes built on :mod:`physicskit.relativity.core`.

- :mod:`physicskit.relativity.chapters.schwarzschild` -- orbits, precession, the photon
  sphere, light deflection, the Shapiro delay, disk redshift, and tidal
  forces.
- :mod:`physicskit.relativity.chapters.kerr` -- frame dragging, the ergosphere, the ISCO,
  and the Penrose process.
- :mod:`physicskit.relativity.chapters.lensing` -- Einstein rings, multiple images, and
  microlensing magnification.
- :mod:`physicskit.relativity.chapters.gw_merger` -- binary black hole inspiral chirps,
  ringdown, and Hulse-Taylor-style orbital decay.
- :mod:`physicskit.relativity.chapters.cosmology` -- FLRW cosmic expansion and redshift.
- :mod:`physicskit.relativity.chapters.neutron_star` -- the Tolman-Oppenheimer-Volkoff
  equations and the neutron star maximum mass.
- :mod:`physicskit.relativity.chapters.timekeeping` -- relativistic satellite clock
  corrections (GPS).
"""

from physicskit.relativity.chapters.cosmology import FLRWCosmology
from physicskit.relativity.chapters.gw_merger import BinaryMerger
from physicskit.relativity.chapters.kerr import KerrBlackHole
from physicskit.relativity.chapters.lensing import PointMassLens
from physicskit.relativity.chapters.neutron_star import NeutronStar
from physicskit.relativity.chapters.schwarzschild import SchwarzschildBlackHole

__all__ = [
    "BinaryMerger",
    "FLRWCosmology",
    "KerrBlackHole",
    "NeutronStar",
    "PointMassLens",
    "SchwarzschildBlackHole",
]
