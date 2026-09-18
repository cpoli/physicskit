Examples
========

This gallery walks through ``physicskit.astro``'s worked examples: orbital
mechanics, N-body dynamics, stellar structure, galactic dynamics and dark
matter, cosmic-web structure formation, and the stellar dynamo -- one
section per corresponding subpackage module.

See also the narrative tutorials:

- :doc:`/tutorials/polytropic_stellar_models`
- :doc:`/tutorials/nbody_orbits_and_galactic_rotation_curves`

Each script in this gallery is self-contained and can be run directly with
``python examples/astro/<section>/<script>.py``. Every script also carries
an RST module docstring as its title/description and uses ``# %%`` markers
to split narrative text from code, which is exactly what Sphinx-Gallery
renders into the pages below -- the script *is* the source of truth for what
you see, not a copy of it.

Sections
--------

- **orbital_mechanics** -- Kepler's laws, Newton's inverse-square gravity,
  Gauss's orbit-determination problem, and the Hohmann transfer.
- **nbody** -- Poincare's sensitive dependence in the three-body problem,
  the figure-eight choreography, the virial theorem, and long-term
  symplectic stability.
- **stellar_structure** -- Lane-Emden polytropes, the Chandrasekhar mass
  limit, the Eddington mass-luminosity relation, and the pp-chain/CNO-cycle
  energy release behind stellar nucleosynthesis.
- **galactic_dynamics** -- Oort's constants, flat rotation curves and the
  dark-matter mass discrepancy, the NFW halo profile, and Chandrasekhar's
  dynamical friction.
- **cosmic_web** -- the Zel'dovich approximation: a uniform particle grid
  collapsing into the filaments, sheets, and nodes of the cosmic web.
- **stellar_dynamo** -- 2D Boussinesq convective rolls, and the linearized
  alpha-omega mean-field dynamo wave that reproduces the solar butterfly
  diagram.
