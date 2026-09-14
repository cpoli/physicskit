Examples
========

Runnable scripts demonstrating the conceptual breakthroughs behind
``physicskit.condensed``, from Bloch's band theory through topological
superconductors -- see :doc:`/history/condensed_breakthroughs` for the full
chronology each script illustrates.

Each script in this gallery is self-contained and can be run directly with
``python examples/condensed/<section>/<script>.py``. Every script also
carries an RST module docstring as its title/description and uses ``# %%``
markers to split narrative text from code, which is exactly what
Sphinx-Gallery renders into the pages below -- the script *is* the source
of truth for what you see, not a copy of it.

Sections
--------

- **tight_binding** -- the generic Bloch-Hamiltonian machinery: recovering
  the exact tight-binding dispersion from a real-space chain, a
  Hofstadter-like spectrum from Peierls-substituted Landau levels, and a
  two-orbital Slater-Koster LCAO band structure.
- **correlated** -- interacting-electron models: the BCS/Bogoliubov-de
  Gennes quasiparticle gap, Mott suppression of itinerant motion in the 1D
  Hubbard model, and the short-range antiferromagnetic spin correlations
  that model develops at half filling and strong coupling -- the arena any
  theory of cuprate superconductivity has to live in.
- **laughlin** -- the fractional quantum Hall effect: Metropolis-sampling
  Laughlin's trial wavefunction directly via its plasma analogy, and the
  correlation hole and incompressible-droplet density profile that result.
- **topology** -- topological band theory: SSH edge states and the Zak
  phase, exactly quantized Chern numbers (TKNN) on both the Haldane model
  and the lattice (Harper-Hofstadter) route to the integer quantum Hall
  effect, the Haldane model's zero-net-flux Chern insulator and its chiral
  edge states, unpaired Majorana zero modes in the Kitaev chain, graphene's
  massless Dirac cone, the Kane-Mele/BHZ :math:`\mathbb{Z}_2` topological
  insulators, Weyl semimetal Fermi arcs, and the tenfold-way classification
  tying the Chern, :math:`\mathbb{Z}_2`, and Majorana invariants together.
