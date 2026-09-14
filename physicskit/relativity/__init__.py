"""physicskit.relativity: fast, visually captivating numerical General Relativity.

``physicskit.relativity`` brings curved spacetime geometry, black hole physics,
ray-traced gravitational lensing, and gravitational wave dynamics into an
interactive, computational Python framework, in geometrized units
(:math:`G = c = 1`; see :mod:`physicskit.relativity.utils.constants`):

- **Black hole physics** (:mod:`physicskit.relativity.chapters.schwarzschild`,
  :mod:`physicskit.relativity.chapters.kerr`): timelike orbits, perihelion precession, the
  innermost stable circular orbit (ISCO), the photon sphere, light
  deflection and the Shapiro delay, disk redshift, tidal forces, frame
  dragging, the ergosphere, and the Penrose process.
- **Spacetime geometry** (:mod:`physicskit.relativity.core.tensors`): a numerical
  differential geometry engine computing Christoffel symbols and curvature
  tensors for arbitrary metrics (Schwarzschild, Kerr, Reissner-Nordstrom,
  FLRW, the Alcubierre warp drive).
- **Gravitational lensing** (:mod:`physicskit.relativity.core.raytracer`,
  :mod:`physicskit.relativity.core.kerr_raytracer`, :mod:`physicskit.relativity.visualizers.shadow_render`,
  :mod:`physicskit.relativity.chapters.lensing`): Numba-accelerated backward photon
  ray-tracers (Schwarzschild and Kerr) rendering black hole shadows and
  redshift-shaded lensed accretion disks, plus point-source Einstein rings,
  multiple images, and microlensing magnification.
- **Gravitational waves** (:mod:`physicskit.relativity.chapters.gw_merger`): binary black
  hole inspiral chirps and ringdown, and Hulse-Taylor-style orbital decay.
- **Cosmology & compact objects** (:mod:`physicskit.relativity.chapters.cosmology`,
  :mod:`physicskit.relativity.chapters.neutron_star`): the FLRW Friedmann equations for
  cosmic expansion, and the Tolman-Oppenheimer-Volkoff equations for
  neutron star structure and the maximum neutron star mass.
- **Relativity in everyday technology** (:mod:`physicskit.relativity.chapters.timekeeping`):
  the GPS satellite relativistic clock correction.

Examples
--------
>>> from physicskit.relativity.chapters import SchwarzschildBlackHole
>>> bh = SchwarzschildBlackHole(M=1.0)
>>> bh.isco_radius
6.0
>>> bh.photon_sphere_radius
3.0
"""

from physicskit.relativity.chapters import (
    BinaryMerger,
    FLRWCosmology,
    KerrBlackHole,
    NeutronStar,
    PointMassLens,
    SchwarzschildBlackHole,
)

__version__ = "0.2.0"

__all__ = [
    "BinaryMerger",
    "FLRWCosmology",
    "KerrBlackHole",
    "NeutronStar",
    "PointMassLens",
    "SchwarzschildBlackHole",
    "__version__",
]
