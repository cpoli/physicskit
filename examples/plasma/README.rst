Examples
========

This gallery walks through every public feature of ``physicskit.plasma``:
single-particle motion in electromagnetic fields, magnetohydrodynamics,
cold-plasma waves, particle-in-cell kinetic theory, drift-wave
turbulence, and particle acceleration in a plasma wakefield.

Each script in this gallery is self-contained and can be run directly with
``python examples/plasma/<section>/<script>.py``. Every script also
carries an RST module docstring as its title/description and uses ``# %%``
markers to split narrative text from code, which is exactly what
Sphinx-Gallery renders into the pages below -- the script *is* the source of
truth for what you see, not a copy of it.

Sections
--------

- **single_particle** -- charged-particle motion in electromagnetic fields:
  the guiding-center drifts (E-cross-B, grad-B, curvature) and magnetic
  mirror bounce motion of the adiabatic-invariant picture, the
  energy-conserving Boris particle pusher, and the exact gyro-orbit
  recovering the guiding-center E-cross-B drift on time-average -- the
  single-particle foundation gyrokinetic turbulence theory builds on.
- **mhd** -- the plasma as a single conducting fluid: Alfven and
  magnetosonic wave speeds (including an animated Alfven pulse splitting
  and propagating along the field), the Grad-Shafranov equilibrium behind
  every tokamak's nested flux surfaces and safety-factor profile, and the
  Sweet-Parker vs. Petschek models of magnetic reconnection, with a
  time-evolving X-point reconnection simulation animated alongside the
  Sweet-Parker scaling laws.
- **waves** -- cold-plasma wave theory: Langmuir's plasma frequency across
  astrophysical and laboratory densities, Stix's dielectric tensor with
  the Clemmow-Mullaly-Allis (CMA) diagram organizing every cold-plasma wave
  mode, and the ion-acoustic soliton propagating without change of shape
  under the Washimi-Taniuti KdV reduction.
- **kinetic** -- particle-in-cell (PIC) kinetic theory: Landau damping
  reproduced from first principles with no collision term anywhere in the
  equations, the two-stream instability as its mirror-image kinetic growth
  mechanism (now with an animated phase-space vortex), the Weibel/filamentation
  instability growing current filaments from temperature anisotropy, and
  a Langmuir wave ringing at the plasma frequency instead of damping away.
- **turbulence** -- the reduced Hasegawa-Mima model for magnetized-plasma
  drift-wave turbulence: small-amplitude potential noise self-organizing
  into long-lived coherent vortices.
- **acceleration** -- a test charge surfing a prescribed traveling plasma
  wakefield, the qualitative picture behind Tajima and Dawson's laser
  wakefield accelerator proposal.
