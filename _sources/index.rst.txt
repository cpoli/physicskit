physicskit
==========

.. include:: /_generated/vars.rst

**physicskit** is a unified scientific toolkit for computational physics,
spanning |num_subpackages| domains -- from stellar dynamics to quantum
entanglement -- under one NumPy-based API. It's built for physics students
working through a textbook problem, researchers prototyping a new model,
and educators building a demonstration: every subpackage is called,
tested, and visualized the same way, so moving from classical mechanics to
condensed matter to general relativity means picking up new physics, not a
new set of conventions. Units stay native to the field instead:
:math:`G=1` for orbits, :math:`\hbar=1` for quantum states, :math:`k_B=1`
for statistical mechanics, matching how the papers you're checking against
actually write it, rather than forcing everything through SI.

Every subpackage is grounded in the physics it implements, not just coded
against it: public functions carry runnable, CI-checked examples, and each
subpackage's :doc:`history </history/index>` page traces the breakthroughs
behind it -- from Huygens' 1690 wave construction to the 2017 GW170817
neutron-star chirp -- each one linked to the code that reproduces it. The
goal is a toolkit you can trust to move between domains without
re-deriving the plumbing every time:

- :mod:`physicskit.astro` -- stellar structure, N-body dynamics, orbital mechanics, galactic dynamics
- :mod:`physicskit.chaos` -- chaotic dynamical systems and 2D quantum billiards
- :mod:`physicskit.classical` -- classical (Newtonian/Lagrangian/Hamiltonian) mechanics
- :mod:`physicskit.condensed` -- tight-binding models, topological band theory, superconductivity
- :mod:`physicskit.fields` -- electrodynamics (FDTD), solitons, BEC vortex lattices
- :mod:`physicskit.fluids` -- potential flow, viscous flow, vortex dynamics, instabilities, compressible flow, Navier-Stokes
- :mod:`physicskit.optics` -- ray/wave/Gaussian-beam optics, Wigner functions, Jaynes-Cummings dynamics
- :mod:`physicskit.particle` -- relativistic kinematics, two-body decays, scattering, nuclear physics
- :mod:`physicskit.plasma` -- single-particle motion, magnetohydrodynamics, cold-plasma waves, kinetic theory
- :mod:`physicskit.quantum` -- quantum mechanics: wave packets, potentials, entanglement
- :mod:`physicskit.relativity` -- numerical general relativity: black holes, lensing, gravitational waves
- :mod:`physicskit.rmt` -- random matrix theory, organized around Dyson's threefold way
- :mod:`physicskit.semiclassical` -- WKB/EBK quantization, semiclassical propagators, the Gutzwiller trace formula, and quantum scarring
- :mod:`physicskit.statphys` -- statistical mechanics: lattice models, molecular dynamics, criticality

Conventionally imported as ``pk``:

.. code-block:: python

   import physicskit as pk
   import numpy as np

   H = lambda k1, k2: pk.condensed.haldane_model(k1, k2, phi=np.pi / 2)
   chern_numbers = pk.condensed.compute_chern_number(H, grid_size=30)
   print(chern_numbers)  # [1, -1]

.. toctree::
   :maxdepth: 2
   :caption: Subpackages
   :hidden:

   subpackages/index

.. toctree::
   :maxdepth: 2
   :caption: History
   :hidden:

   history/index

.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API
   :hidden:

   api/index

