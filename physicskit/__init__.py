"""physicskit: unified scientific toolkit for computational physics.

Import as ``pk`` by convention::

    import physicskit as pk
    pk.condensed.compute_chern_number(...)
    pk.chaos.systems.maps.LogisticMap(...)
    pk.fields.nls_dark_soliton(...)
    pk.optics.GaussianBeam(wavelength=1.064e-3, w0=0.05)
    pk.particle.FourVector(E=1.0, px=0.0, py=0.0, pz=0.6)
    pk.astro.PolytropicStar(n=1.5, K=1.0, rho_c=1.0)
    pk.plasma.alfven_speed(B=1.0, rho=1e-6)
    pk.fluids.reynolds_number(velocity=1.0, length=0.01, nu=1e-6)
    pk.units.geometrized_units(mass_kg=pk.constants.SOLAR_MASS_KG).scale("length")
"""

from physicskit import (
    astro,
    chaos,
    classical,
    condensed,
    constants,
    fields,
    fluids,
    integrators,
    io,
    optics,
    particle,
    plasma,
    quantum,
    relativity,
    results,
    rmt,
    semiclassical,
    statphys,
    units,
)

__version__ = "0.4.0"

__all__ = [
    "astro",
    "chaos",
    "classical",
    "condensed",
    "constants",
    "fields",
    "fluids",
    "integrators",
    "io",
    "optics",
    "particle",
    "plasma",
    "quantum",
    "relativity",
    "results",
    "rmt",
    "semiclassical",
    "statphys",
    "units",
]
