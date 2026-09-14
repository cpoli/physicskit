Examples
========

This gallery walks through every physical regime in ``physicskit.fluids``:
inviscid potential flow, classic viscous exact solutions, point-vortex
N-body dynamics, the Kelvin-Helmholtz and Rayleigh-Taylor instabilities,
1D compressible shocks, and the 2D incompressible Navier-Stokes solver that
underlies the instability and turbulence-spectrum tools. Each script
reproduces the signature observable of a specific breakthrough in
:doc:`/history/fluid_breakthroughs`.

Each script in this gallery is self-contained and can be run directly with
``python examples/fluids/<section>/<script>.py``. Every script also carries
an RST module docstring as its title/description and uses ``# %%`` markers
to split narrative text from code, which is exactly what Sphinx-Gallery
renders into the pages below -- the script *is* the source of truth for what
you see, not a copy of it.

Sections
--------

- **potential_flow** -- inviscid, irrotational flow built by superposing
  elementary solutions: source, sink, and doublet streamline patterns,
  d'Alembert's zero-drag paradox for a circulation-free cylinder, and the
  classic lifting-cylinder problem with the Kutta-Joukowski lift compared
  against a direct pressure-integral calculation.
- **viscous_flow** -- classic exact viscous solutions: Couette and
  Poiseuille flow profiles side by side, Stokes drag on a settling sphere,
  the Reynolds number as the single dimensionless ratio governing flow
  regime regardless of size or fluid, and the Blasius boundary layer solved
  by shooting.
- **vortex_dynamics** -- point-vortex N-body dynamics via the Biot-Savart
  law: a co-rotating vortex pair versus a translating vortex dipole, and a
  von Karman vortex street holding its staggered shape at the stable
  spacing ratio.
- **instabilities** -- the Kelvin-Helmholtz shear instability rolling a
  rippled interface into vortex cores, and the Rayleigh-Taylor instability
  growing a heavy-over-light interface into mushroom plumes, each checked
  against its linear growth-rate law.
- **compressible_flow** -- the 1D Euler equations: normal shock relations
  across a range of Mach numbers, and the classic Sod shock tube resolving
  a rarefaction fan, contact discontinuity, and shock in one Riemann problem.
- **navier_stokes** -- the 2D incompressible vorticity-streamfunction
  solver: viscous decay of a periodic vortex patch, and a decaying
  turbulence simulation whose energy spectrum is checked against
  Kolmogorov's -5/3 law.
