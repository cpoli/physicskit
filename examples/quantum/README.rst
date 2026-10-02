Examples
========

Runnable, self-contained scripts demonstrating every chapter of
``physicskit.quantum`` -- from arbitrary-potential bound states through
wave-packet dynamics, entanglement, and Floquet driving.

Each script in this gallery is self-contained and can be run directly with
``python examples/quantum/<section>/<script>.py``. Every script also carries
an RST module docstring as its title/description and uses ``# %%`` markers to
split narrative text from code, which is exactly what Sphinx-Gallery renders
into the pages below -- the script *is* the source of truth for what you see,
not a copy of it.

Sections
--------

- **potentials** -- bound and scattering states in model potentials: the
  matrix Numerov solver for step and finite square wells and the
  gravitational "quantum bouncer" (checked against exact Airy-function
  zeros), real-time animated barrier tunneling and well scattering,
  double-well tunneling (including an animated complex-valued version),
  2D quantum boxes, and the Ramsauer-Townsend transmission resonance.
- **wave_packets** -- time-dependent dynamics: de Broglie's traveling matter
  wave, dispersion (with an animated view of the spreading), twin-slit
  interference and quantum revivals, Ehrenfest's theorem in an anharmonic
  well, and a phase-colored tunneling animation.
- **harmonic_oscillator** -- the harmonic oscillator suite of eigenstates and
  coherent/squeezed states, an animated Fock-state superposition, its Wigner
  phase-space quasi-probability distribution, and the raw ladder-operator
  algebra the whole suite is built on.
- **hydrogen** -- the hydrogen atom's orbitals in three dimensions, including
  an animated two-eigenstate superposition beating at the Bohr frequency.
- **entanglement** -- Bloch-sphere spin dynamics (Larmor precession and an
  animated Rabi oscillation), Bell correlations, the topological
  Aharonov-Bohm effect, and the dynamical generation of entanglement between
  two Ising-coupled qubits.
- **measurement** -- projective measurement and the Born rule, the
  position-momentum uncertainty bound traced to its operator-algebra root,
  and two measurement/interference demonstrations propagated genuinely in
  time: the double-slit experiment and Stern-Gerlach beam splitting.
- **perturbation** -- Stark/Zeeman splitting from perturbation theory and
  Floquet driving.
- **atoms_molecules** -- the Hartree-Fock self-consistent field for
  helium and the hydrogen molecule.
- **scattering** -- 3D potential scattering: Rutherford's formula from the
  Born approximation and partial-wave phase shifts of the hard sphere and
  square well.
- **relativistic** -- the Dirac equation's hydrogen fine structure and
  Klein's paradox for Dirac and Klein-Gordon particles.
- **open_systems** -- a qubit coupled to its environment: Lindblad
  :math:`T_1`/:math:`T_2` decay and the quantum-trajectory (Monte Carlo
  wavefunction) unravelling of the same dynamics.
- **quantum_computing** -- a state-vector circuit simulator: the
  Deutsch-Jozsa algorithm, the quantum Fourier transform and Shor's period
  finding, and Grover search.
- **tensor_networks** -- matrix product states and TEBD: a quench checked
  against exact time evolution, and the ground state of a 60-site
  Heisenberg chain.
