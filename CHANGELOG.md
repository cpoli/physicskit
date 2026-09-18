# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-09-18

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
- Non-visualizer test coverage raised from 96.7% to 99.94% (1,597 -> 1,720
  tests), with new tests across `quantum`, `statphys`, `condensed`,
  `optics`, `rmt`, `plasma`, `fluids`, `fields`, and `relativity`.
- `physicskit/py.typed` at the package root, so external type checkers
  (mypy/pyright) recognize the whole package as typed under PEP 561 --
  previously only the nested `physicskit/rmt/py.typed` existed, which had
  no effect outside that one subpackage. Added the `Typing :: Typed`
  classifier to match.
- Docs are now hosted on GitHub Pages (`gh-pages` branch) instead of Read
  the Docs, which was never actually set up (its badge showed "unknown"
  indefinitely). README/`pyproject.toml`/`CITATION.cff` links updated
  accordingly.
- CI now uploads coverage to Codecov on one matrix cell (ubuntu/py3.12);
  README also gained a manually-maintained static coverage badge.

### Fixed

- `physicskit.quantum.chapters.potentials.airy_wavefunction` had a sign
  error in its Airy-function argument (`x/length - a_n` instead of
  `x/length + a_n`), so the returned wavefunction didn't actually satisfy
  the `psi(0)=0` hard-floor boundary condition -- its overlap with the
  correct Numerov ground state was only ~0.39, not ~1. Found while writing
  tests for the coverage push above.
- GitHub links across README/`CITATION.cff`/`CHANGELOG.md`/`pyproject.toml`/
  docs `conf.py` pointed at the nonexistent `physicskit/physicskit` org
  instead of the actual `cpoli/physicskit` repo.
- `pyproject.toml`'s `version` was a separate hardcoded string, already
  drifted from `physicskit/__init__.py`'s `__version__` (docs `conf.py`'s
  `release` was hardcoded to the wrong value, `"1.0.0"`). `pyproject.toml`
  now sources the version dynamically via `[tool.setuptools.dynamic]`, and
  `conf.py` reads it from the installed package's metadata.

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

[Unreleased]: https://github.com/cpoli/physicskit/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/cpoli/physicskit/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/cpoli/physicskit/releases/tag/v0.1.0
