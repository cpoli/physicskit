Examples
========

Runnable, self-contained scripts demonstrating every module of
``physicskit.semiclassical`` -- WKB/Bohr-Sommerfeld quantization,
semiclassical propagators, the Gutzwiller trace formula, and quantum
scars.

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
  oscillator spectrum and eigenstates.
- **propagators** -- the Van Vleck-Morette single-trajectory propagator
  (with its Maslov-index caustic count) and the Herman-Kluk
  multi-trajectory frozen-Gaussian propagator, both checked against exact
  quantum evolution.
- **gutzwiller** -- the exact 1D Gutzwiller trace formula, reconstructing
  a spectrum from a single classical periodic orbit's action and period,
  and the general isolated-orbit stability amplitude.
- **scarring** -- quantum scars on the stadium billiard's bouncing-ball
  orbit family, and a Husimi phase-space view of a scarred eigenstate.
