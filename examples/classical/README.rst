Examples
========

This gallery walks through every public feature of ``physicskit.classical``:
Newtonian, Lagrangian, and Hamiltonian formulations of the same mechanics,
coupled chains and their solitons, rigid-body rotations, the symplectic
integrators underneath all of it, and the interactive/animated visualizers
built on top.

Each script in this gallery is self-contained and can be run directly with
``python examples/classical/<section>/<script>.py``. Every script also
carries an RST module docstring as its title/description and uses ``# %%``
markers to split narrative text from code, which is exactly what
Sphinx-Gallery renders into the pages below -- the script *is* the source of
truth for what you see, not a copy of it.

Sections
--------

- **newtonian** -- direct force-and-acceleration integration: the oblique
  cannonball problem, Kepler orbits with perihelion precession and a
  power-law perturbation, angular-momentum conservation as a diagnostic
  distinct from energy conservation, and Noether's theorem's manifest
  (rotational) versus hidden (Kepler's Laplace-Runge-Lenz) symmetries side
  by side.
- **lagrangian** -- systems built from a Lagrangian: the chaotic double
  pendulum, a bead on a rotating hoop (a pitchfork bifurcation), the normal
  modes of coupled oscillators, and how to write a custom system with
  ``LagrangianEngine`` from scratch.
- **hamiltonian** -- phase-space methods: Poincare sections and KAM torus
  breakdown in the Henon-Heiles system, Liouville's theorem illustrated with
  a swarm of pendulums, action-angle variables for the pendulum, and
  Hamilton's unified ``(q, p)`` canonical phase space shared by a librating
  and a rotating pendulum.
- **rotations** -- rigid-body dynamics: the intermediate axis theorem
  (Dzhanibekov effect), both as a stability argument and as a literal
  tumbling 3D rigid body, and the heavy symmetric top's precession and
  nutation.
- **chains** -- coupled degrees of freedom: the harmonic chain's exact normal
  modes, the Fermi-Pasta-Ulam-Tsingou recurrence, and Sine-Gordon kink
  propagation and kink-antikink breathers.
- **integrators** -- why the choice of integrator matters: symplectic
  (Yoshida4) versus RK4 long-horizon energy drift, and automatic timestep
  selection with ``estimate_dt``.
- **visualizers** -- live and interactive displays: ``SideBySideAnimator``
  pairing a physical-space animation with phase/energy diagnostics, and
  interactive Plotly views of the SO(3) momentum sphere and orbits.
