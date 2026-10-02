Examples
========

Runnable, self-contained scripts demonstrating every module of
``physicskit.semiclassical`` -- WKB/Bohr-Sommerfeld quantization,
semiclassical propagators, Feynman paths, the Gutzwiller trace formula and
its relatives, the Ehrenfest time, and quantum scars.

See also the narrative tutorials:

- :doc:`/tutorials/semiclassical_wavepacket_propagation`
- :doc:`/tutorials/gutzwiller_periodic_orbit_spectrum`

Each script in this gallery is self-contained and can be run directly with
``python examples/semiclassical/<section>/<script>.py``. Every script also
carries an RST module docstring as its title/description and uses ``# %%``
markers to split narrative text from code, which is exactly what
Sphinx-Gallery renders into the pages below -- the script *is* the source
of truth for what you see, not a copy of it.

Sections
--------

- **wkb** -- classical momentum, turning points, WKB wavefunctions, and
  Bohr-Sommerfeld (EBK) quantization, checked against the exact harmonic
  oscillator spectrum and eigenstates; Weyl's law, Einstein's torus
  quantization, and the Langer correction.
- **propagators** -- the Van Vleck-Morette single-trajectory propagator
  (with its Maslov-index caustic count), Heller's thawed Gaussian and its
  time-dependent spectroscopy, and the Herman-Kluk multi-trajectory
  frozen-Gaussian propagator, all checked against exact quantum evolution.
- **gutzwiller** -- the exact 1D Gutzwiller trace formula, reconstructing
  a spectrum from a single classical periodic orbit's action and period,
  and the general isolated-orbit stability amplitude; the Berry-Tabor and
  Balian-Bloch level densities of billiards; and Bogomolny's transfer
  operator.
- **path_integral** -- Feynman's sum over paths and the emergence of the
  classical path as :math:`\hbar\to0`.
- **ehrenfest** -- the logarithmic Ehrenfest time of a chaotic wavepacket.
- **scarring** -- quantum scars on the stadium billiard's bouncing-ball
  orbit family, a Husimi phase-space view of a scarred eigenstate, and
  Berry's random-wave conjecture for the eigenstates that are not
  scarred.
