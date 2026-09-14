# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `physicskit.relativity.utils.constants.check_geometrized_mass` and
  `check_geometrized_distance`: heuristic runtime warnings that catch the
  most common `physicskit.relativity` unit mistake -- passing a raw
  physical value (solar masses, Mpc) where a geometrized length is
  required. Wired into `physicskit.relativity.visualizers.wave_plots`,
  where an unconverted `BinaryMerger` silently produced a non-oscillating
  "chirp" animation with no error at all. Deliberately *not* wired into
  `BinaryMerger.__init__` itself, since many legitimate uses (unit tests
  of a formula's mathematical properties) pass small toy numbers with no
  claim of physical realism.

## [0.1.0] - 2026-09-09

### Added

Initial public release. 14 physics subpackages, each with model classes,
matplotlib/plotly visualizers, and a test suite verifying results against
closed-form or historically known values:

- `physicskit.astro` — stellar structure, N-body dynamics, orbital
  mechanics, galactic dynamics, stellar convection/dynamo.
- `physicskit.chaos` — chaotic dynamical systems, 2D quantum billiards,
  the quantum kicked rotor.
- `physicskit.classical` — Newtonian, Lagrangian, and Hamiltonian
  mechanics; rigid-body and lattice/chain dynamics.
- `physicskit.condensed` — tight-binding models, topological band theory
  (Chern numbers, edge states), correlated-electron superconductivity.
- `physicskit.fields` — FDTD electrodynamics, KdV/NLS/Sine-Gordon
  solitons, BEC vortex lattices.
- `physicskit.fluids` — potential flow, viscous exact solutions,
  point-vortex dynamics, Kelvin-Helmholtz/Rayleigh-Taylor instabilities,
  compressible shocks, 2D incompressible Navier-Stokes.
- `physicskit.optics` — ray, wave, and Gaussian-beam optics; quantum
  optics.
- `physicskit.particle` — relativistic kinematics, decays, scattering,
  nuclear physics.
- `physicskit.plasma` — single-particle motion, magnetohydrodynamics,
  cold-plasma waves, kinetic (particle-in-cell) theory, magnetic
  reconnection.
- `physicskit.quantum` — wave packets, potentials, hydrogen, entanglement,
  measurement.
- `physicskit.relativity` — numerical general relativity: Schwarzschild
  and Kerr black holes, gravitational lensing, gravitational-wave
  inspiral-merger-ringdown waveforms, cosmology, neutron stars.
- `physicskit.rmt` — random matrix theory, organized around Dyson's
  threefold way.
- `physicskit.semiclassical` — WKB/EBK quantization, semiclassical
  propagators, the Gutzwiller trace formula, quantum scarring.
- `physicskit.statphys` — lattice models (Ising/Potts/XY), molecular
  dynamics, criticality, disordered systems (spin glasses).

Extensive Sphinx documentation, including a per-subpackage chronology of
the field's foundational breakthroughs (`docs/source/history/`) linked to
the corresponding implementation.

[Unreleased]: https://github.com/physicskit/physicskit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/physicskit/physicskit/releases/tag/v0.1.0
