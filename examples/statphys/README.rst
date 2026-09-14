Examples
========

Runnable demonstrations of the central results in each
``physicskit.statphys`` chapter: lattice phase transitions, molecular
dynamics, random walks, and self-organized criticality.

Each script in this gallery is self-contained and can be run directly with
``python examples/statphys/<section>/<script>.py``. Every script also carries
an RST module docstring as its title/description and uses ``# %%`` markers to
split narrative text from code, which is exactly what Sphinx-Gallery renders
into the pages below -- the script *is* the source of truth for what you see,
not a copy of it.

Sections
--------

- **ising** -- the 2D Ising model: Onsager's exact phase transition,
  finite-size scaling to extract critical exponents, and critical slowing
  down (Metropolis versus the Wolff cluster algorithm).
- **xy_model** -- the Kosterlitz-Thouless transition via vortex-antivortex
  unbinding.
- **potts_model** -- first- versus second-order transitions in the q-state
  Potts model.
- **spin_glass** -- frustration and the Edwards-Anderson order parameter.
- **percolation** -- percolation as a purely geometric phase transition.
- **renormalization** -- Kadanoff block-spin renormalization group flow.
- **quantum_statistics** -- Bose-Einstein and Fermi-Dirac distributions and
  Bose-Einstein condensation.
- **molecular_dynamics** -- Boltzmann's H-theorem: relaxation to the
  Maxwell-Boltzmann distribution.
- **random_walk** -- diffusion, the Einstein relation, and the central limit
  theorem.
- **ehrenfest_urn** -- the Ehrenfest urn: reversibility, recurrence, and the
  arrow of time.
- **sandpile** -- self-organized criticality in the Bak-Tang-Wiesenfeld
  sandpile.
- **interactive** -- interactive Plotly dashboards.
