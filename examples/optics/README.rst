Examples
========

Runnable, self-contained scripts reproducing the key conceptual
breakthroughs behind ``physicskit.optics`` -- from Huygens' 1690 wave
construction through Allen et al.'s 1992 discovery of the orbital angular
momentum of light (see :doc:`/history/optics_breakthroughs` for the full
chronology).

Each script in this gallery is self-contained and can be run directly with
``python examples/optics/<section>/<script>.py``. Every script also carries
an RST module docstring as its title/description and uses ``# %%`` markers to
split narrative text from code, which is exactly what Sphinx-Gallery renders
into the pages below -- the script *is* the source of truth for what you see,
not a copy of it.

Sections
--------

- **diffraction** -- scalar wave optics from Huygens through Airy: Huygens'
  wavelet construction, Young's double-slit fringes, Fresnel's near-field
  diffraction theory and the Poisson/Arago spot, and Airy's circular-aperture
  pattern and the diffraction limit.
- **ray_optics** -- paraxial ABCD matrix optics: Gauss's composite-system
  matrix product, the two-mirror resonator stability behind Maiman's ruby
  laser cavity, and Kao's graded-index (GRIN) fiber guiding.
- **gaussian_beams** -- complex-beam-parameter wave optics: Kogelnik and
  Li's ABCD transformation of a Gaussian beam through a thin lens,
  Siegman's :math:`M^2` beam-quality factor, and Allen et al.'s
  Laguerre-Gaussian orbital-angular-momentum vortex.
- **quantum_optics** -- light in the truncated Fock basis: Wigner's
  phase-space negativity, Glauber's coherent states, the Jaynes-Cummings
  model's vacuum Rabi oscillation and collapse-and-revival,
  Kimble/Dagenais/Mandel's photon antibunching, and Slusher's observation
  of squeezed light.
