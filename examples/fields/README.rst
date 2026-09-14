Examples
========

This gallery walks through ``physicskit.fields``: electrodynamics on a Yee
grid, soliton-bearing nonlinear wave equations, and Gross-Pitaevskii
Bose-Einstein condensates -- each script reproducing the signature
observable of a specific breakthrough in :doc:`/history/fields_breakthroughs`.
Fluid dynamics has since moved to its own package and gallery; see
:mod:`physicskit.fluids` and :doc:`/history/fluid_breakthroughs`.

Each script in this gallery is self-contained and can be run directly with
``python examples/fields/<section>/<script>.py``. Every script also
carries an RST module docstring as its title/description and uses ``# %%``
markers to split narrative text from code, which is exactly what
Sphinx-Gallery renders into the pages below -- the script *is* the source
of truth for what you see, not a copy of it.

Sections
--------

- **solitons** -- the Korteweg-de Vries, nonlinear Schrodinger, and
  Sine-Gordon equations: Russell's shape-preserving wave of translation,
  the elastic KdV soliton-soliton collision, Zabusky and Kruskal's fission
  of a generic pulse into a soliton train, free-particle wavepacket
  spreading under the linear Schrodinger equation, a topologically
  protected Sine-Gordon kink, an exact kink-antikink collision from the
  inverse scattering transform, and a dispersion-free optical soliton --
  several of these animated frame by frame.
- **electrodynamics** -- Maxwell's equations on a Yee grid: measuring the
  vacuum speed of light from a propagating FDTD pulse, a point source
  radiating outward in 2D, Berenger's Perfectly Matched Layer absorbing
  boundary compared against a hard wall, a Hertzian oscillating dipole
  antenna radiating outward, a wave crossing a dielectric slab, and a
  standing TM cavity mode.
- **quantum_fields** -- the Gross-Pitaevskii equation for a trapped BEC:
  detecting a single quantum of circulation around an imprinted vortex, a
  bound vortex-antivortex pair with canceling net winding (the
  Kosterlitz-Thouless building block), relaxing to the interacting
  mean-field ground state, the critical rotation frequency for vortex
  nucleation, a vortex genuinely precessing in real time, self-focusing
  wave collapse of an attractive condensate, and the Casimir vacuum force
  between confining "plates."
- **gauge_confinement** -- a simplified 1+1D toy model of quark
  confinement: a flux tube stretching between two charges, reproducing
  the linearly growing confinement energy of Wilson's 1974 lattice gauge
  theory.
