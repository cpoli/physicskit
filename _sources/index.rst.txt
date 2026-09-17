physicskit
==========

**physicskit** is a unified scientific toolkit for computational physics:

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

Pick a subpackage
-------------------

.. tab-set::

   .. tab-item:: History

      .. grid:: 1 2 3 3
         :gutter: 2

         .. grid-item-card:: physicskit.astro
            :link: /history/astro_breakthroughs
            :link-type: doc

            Stellar structure, N-body dynamics, orbital mechanics, galactic dynamics.

         .. grid-item-card:: physicskit.chaos
            :link: /history/chaos_breakthroughs
            :link-type: doc

            Chaotic dynamical systems and 2D quantum billiards.

         .. grid-item-card:: physicskit.classical
            :link: /history/classical_breakthroughs
            :link-type: doc

            Classical (Newtonian/Lagrangian/Hamiltonian) mechanics.

         .. grid-item-card:: physicskit.condensed
            :link: /history/condensed_breakthroughs
            :link-type: doc

            Tight-binding models, topological band theory, superconductivity.

         .. grid-item-card:: physicskit.fields
            :link: /history/fields_breakthroughs
            :link-type: doc

            Electrodynamics (FDTD), solitons, BEC vortex lattices.

         .. grid-item-card:: physicskit.fluids
            :link: /history/fluid_breakthroughs
            :link-type: doc

            Potential flow, viscous flow, vortex dynamics, instabilities, compressible flow, Navier-Stokes.

         .. grid-item-card:: physicskit.optics
            :link: /history/optics_breakthroughs
            :link-type: doc

            Ray/wave/Gaussian-beam optics and quantum optics.

         .. grid-item-card:: physicskit.particle
            :link: /history/particle_breakthroughs
            :link-type: doc

            Relativistic kinematics, decays, scattering, nuclear physics.

         .. grid-item-card:: physicskit.plasma
            :link: /history/plasma_breakthroughs
            :link-type: doc

            Single-particle motion, magnetohydrodynamics, cold-plasma waves, kinetic theory.

         .. grid-item-card:: physicskit.quantum
            :link: /history/quantum_breakthroughs
            :link-type: doc

            Quantum mechanics: wave packets, potentials, entanglement.

         .. grid-item-card:: physicskit.relativity
            :link: /history/relativity_breakthroughs
            :link-type: doc

            Numerical general relativity: black holes, lensing, gravitational waves.

         .. grid-item-card:: physicskit.rmt
            :link: /history/rmt_breakthroughs
            :link-type: doc

            Random matrix theory, organized around Dyson's threefold way.

         .. grid-item-card:: physicskit.semiclassical
            :link: /history/semiclassical_breakthroughs
            :link-type: doc

            WKB/EBK quantization, semiclassical propagators, the Gutzwiller trace formula, and quantum scarring.

         .. grid-item-card:: physicskit.statphys
            :link: /history/statphys_breakthroughs
            :link-type: doc

            Statistical mechanics: lattice models, molecular dynamics, criticality.

   .. tab-item:: Examples

      .. grid:: 1 2 3 3
         :gutter: 2

         .. grid-item-card:: physicskit.astro
            :link: /tutorials/polytropic_stellar_models
            :link-type: doc

            Stellar structure, N-body dynamics, orbital mechanics, galactic dynamics.

         .. grid-item-card:: physicskit.chaos
            :link: /api/gallery/chaos/index
            :link-type: doc

            Chaotic dynamical systems and 2D quantum billiards.

         .. grid-item-card:: physicskit.classical
            :link: /api/gallery/classical/index
            :link-type: doc

            Classical (Newtonian/Lagrangian/Hamiltonian) mechanics.

         .. grid-item-card:: physicskit.condensed
            :link: /api/gallery/condensed/index
            :link-type: doc

            Tight-binding models, topological band theory, superconductivity.

         .. grid-item-card:: physicskit.fields
            :link: /tutorials/fdtd_waveguide_simulation
            :link-type: doc

            Electrodynamics (FDTD), solitons, BEC vortex lattices.

         .. grid-item-card:: physicskit.fluids
            :link: /api/gallery/fluids/index
            :link-type: doc

            Potential flow, viscous flow, vortex dynamics, instabilities, compressible flow, Navier-Stokes.

         .. grid-item-card:: physicskit.optics
            :link: /api/gallery/optics/index
            :link-type: doc

            Ray/wave/Gaussian-beam optics and quantum optics.

         .. grid-item-card:: physicskit.particle
            :link: /tutorials/relativistic_two_body_decay
            :link-type: doc

            Relativistic kinematics, decays, scattering, nuclear physics.

         .. grid-item-card:: physicskit.plasma
            :link: /api/gallery/plasma/index
            :link-type: doc

            Single-particle motion, magnetohydrodynamics, cold-plasma waves, kinetic theory.

         .. grid-item-card:: physicskit.quantum
            :link: /api/gallery/quantum/index
            :link-type: doc

            Quantum mechanics: wave packets, potentials, entanglement.

         .. grid-item-card:: physicskit.relativity
            :link: /api/gallery/relativity/index
            :link-type: doc

            Numerical general relativity: black holes, lensing, gravitational waves.

         .. grid-item-card:: physicskit.rmt
            :link: /api/gallery/rmt/index
            :link-type: doc

            Random matrix theory, organized around Dyson's threefold way.

         .. grid-item-card:: physicskit.semiclassical
            :link: /examples/semiclassical
            :link-type: doc

            WKB/EBK quantization, semiclassical propagators, the Gutzwiller trace formula, and quantum scarring.

         .. grid-item-card:: physicskit.statphys
            :link: /tutorials/statphys_deep_dives
            :link-type: doc

            Statistical mechanics: lattice models, molecular dynamics, criticality.

   .. tab-item:: API

      .. grid:: 1 2 3 3
         :gutter: 2

         .. grid-item-card:: physicskit.astro
            :link: /api/astro
            :link-type: doc

            Stellar structure, N-body dynamics, orbital mechanics, galactic dynamics.

         .. grid-item-card:: physicskit.chaos
            :link: /api/chaos
            :link-type: doc

            Chaotic dynamical systems and 2D quantum billiards.

         .. grid-item-card:: physicskit.classical
            :link: /api/classical
            :link-type: doc

            Classical (Newtonian/Lagrangian/Hamiltonian) mechanics.

         .. grid-item-card:: physicskit.condensed
            :link: /api/condensed
            :link-type: doc

            Tight-binding models, topological band theory, superconductivity.

         .. grid-item-card:: physicskit.fields
            :link: /api/fields
            :link-type: doc

            Electrodynamics (FDTD), solitons, BEC vortex lattices.

         .. grid-item-card:: physicskit.fluids
            :link: /api/fluids
            :link-type: doc

            Potential flow, viscous flow, vortex dynamics, instabilities, compressible flow, Navier-Stokes.

         .. grid-item-card:: physicskit.optics
            :link: /api/optics
            :link-type: doc

            Ray/wave/Gaussian-beam optics and quantum optics.

         .. grid-item-card:: physicskit.particle
            :link: /api/particle
            :link-type: doc

            Relativistic kinematics, decays, scattering, nuclear physics.

         .. grid-item-card:: physicskit.plasma
            :link: /api/plasma
            :link-type: doc

            Single-particle motion, magnetohydrodynamics, cold-plasma waves, kinetic theory.

         .. grid-item-card:: physicskit.quantum
            :link: /api/quantum
            :link-type: doc

            Quantum mechanics: wave packets, potentials, entanglement.

         .. grid-item-card:: physicskit.relativity
            :link: /api/relativity
            :link-type: doc

            Numerical general relativity: black holes, lensing, gravitational waves.

         .. grid-item-card:: physicskit.rmt
            :link: /api/rmt
            :link-type: doc

            Random matrix theory, organized around Dyson's threefold way.

         .. grid-item-card:: physicskit.semiclassical
            :link: /api/semiclassical
            :link-type: doc

            WKB/EBK quantization, semiclassical propagators, the Gutzwiller trace formula, and quantum scarring.

         .. grid-item-card:: physicskit.statphys
            :link: /api/statphys
            :link-type: doc

            Statistical mechanics: lattice models, molecular dynamics, criticality.


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

