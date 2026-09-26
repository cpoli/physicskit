# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `physicskit.chaos.ForcedVanDerPol`, the forced Van der Pol oscillator
  `x'' - mu(1 - x^2)x' + x = A cos(omega t)`, with a parallel
  `stroboscopic_map` that samples many trajectories once per forcing
  period. Its defaults (`mu=10`, `omega=2.5`, `A = 0.58*omega*mu`) are in
  the Cartwright-Littlewood regime where stable subharmonics of period
  3T and 5T coexist.
- Gallery example "Cartwright-Littlewood: Chaos in the Forced Van der Pol
  Oscillator" (`plot_cartwright_littlewood_forced_van_der_pol.py`),
  showing the two coexisting periodic motions, their interleaved basins,
  and an orbit on the chaotic "bad" set tracked along the basin boundary.
- Chaos history entry "1945 -- 1949 -- Cartwright, Littlewood, and
  Levinson: Chaos in the Forced Van der Pol Oscillator", covering the 1927
  van der Pol-van der Mark observation, the 1938 radar question, the 1945
  result, Levinson's 1949 model and its route to Smale's horseshoe. The
  Smale entry now points back to it.

## [0.2.0] - 2026-09-25

### Fixed

- The shared `physicskit.integrators` (`rk4`, `leapfrog`/`velocity_verlet`,
  `yoshida4`, `dopri5`), `physicskit.chaos.visualizers.basins`, and
  `physicskit.semiclassical.core.propagators` kernels that take an njit
  callback as an argument were `@njit(cache=True)`. Their compiled
  signature includes the callback's dispatcher type, which is fresh for
  every factory-built system instance, so each run appended new,
  never-reused entries to the on-disk cache (`__pycache__` grew without
  bound, the cache never hit, and some Numba versions raise
  `ReferenceError: underlying object has vanished` when re-saving such an
  index). These kernels are no longer disk-cached; the callbacks
  themselves still are.
- `physicskit.integrators.dopri5_integrate` silently returned a trajectory
  that stopped short of `t_end` when `max_steps` ran out. It now emits a
  `RuntimeWarning` naming where it stopped.
- `physicskit.astro.lane_emden`'s default `xi_max=10` cut off every
  polytrope whose surface lies beyond it (e.g. `n=4`, `xi_1 = 14.97`;
  `n=4.5`, `xi_1 = 31.84`), contrary to its docstring, so
  `PolytropicStar` reported the wrong radius and mass for those indices
  with no error. The default is now `xi_max=1000` (enough for every
  `n <= 4.9`), `PolytropicStar` raises `ValueError` for `n >= 5` (infinite
  radius), and its mass uses a second-order surface derivative (tabulated
  `-xi_1^2 theta'(xi_1)` now matches to ~1e-6 instead of ~5e-4).
- `physicskit.relativity.FLRWCosmology.luminosity_distance_mpc` used the
  flat-space `D_L = (1+z) D_C` for every cosmology, although the class
  supports curvature; open/closed universes now use the transverse
  comoving distance (e.g. an open `Omega_m=0.3` dust universe at `z=1`:
  5872 Mpc, matching Mattig's formula, instead of 5590 Mpc).
- `physicskit.rmt.validation.check_universality` ignored its
  `ks_threshold` argument, and `UniversalityResult` had no pass/fail
  verdict despite the docstring promising one. `UniversalityResult` now
  stores `ks_threshold` and exposes a `passed` property.
- `physicskit.semiclassical.core.path_integral.feynman_phasor_partial_sums`
  now raises `ValueError` when `paths` and `actions` disagree in length,
  instead of ignoring `paths` entirely.
- `physicskit.astro.figure_eight_initial_conditions` had a dropped digit in
  the published Chenciner-Montgomery position (`0.9700436` for
  `0.97000436`), so the choreography missed periodicity by ~6e-4 per
  period instead of ~1e-5.
- `physicskit.fields.nls_dark_soliton` used the width `sqrt(rho0/2)`; the
  defocusing NLS it documents requires `sqrt(rho0)` (matching its own
  stated healing length `1/sqrt(rho0)`). The old profile was not a
  solution (O(1) PDE residual).
- `physicskit.optics.compute_wigner_function` mirrored the momentum axis:
  a coherent state `|alpha>` peaked at `p = -sqrt(2) Im(alpha)`. Fock and
  other p-symmetric states were unaffected.
- `physicskit.quantum.chapters.perturbation.FloquetDrivenBox.floquet_quasienergies`
  stepped every time slice at `t=0`, so the drive
  `V0 (x - L/2) cos(omega t)` was frozen into a static tilt and the
  "Floquet operator" was not one. It now passes the actual time.
- `physicskit.quantum.chapters.harmonic_spin.HarmonicOscillator.squeezed_vacuum_wavefunction`
  with `phi != 0` put a chirp on the `phi=0` Gaussian (shearing, not
  rotating, the uncertainty ellipse; at `r=0` it wasn't even the vacuum).
  It now returns the exact `S(r e^{2i phi})|0>` wavefunction, with
  `Delta x^2 = (hbar/2m omega)(e^{-2r} cos^2 phi + e^{2r} sin^2 phi)`.
- `physicskit.quantum.chapters.wave_packets.TwinSlit.amplitude`'s
  single-slit envelope was missing a factor `k^2` (dimensionally wrong and
  effectively flat); it is now the Gaussian-aperture far field
  `exp(-(k w sin theta)^2 / 4)`.
- `DoubleWellSimulator.localized_state` / `tunneling_wavefunction` relied
  on the eigensolver's arbitrary eigenvector sign to decide which side is
  "left"; the doublet's relative sign is now fixed explicitly.
- `physicskit.semiclassical.core.propagators.herman_kluk_prefactor` used
  the Gaussian width `gamma` where the standard formula needs `2*gamma`
  for this module's `e^{-gamma x^2}` frozen Gaussians, so Herman-Kluk
  propagation did not conserve the norm even for a harmonic oscillator
  (1.16 or 0.86 instead of 1 at `t=0.9`, depending on `gamma`).
- `physicskit.rmt` Tracy-Widom distributions were tabulated only down to
  `s=-6`, where `F_4` is still ~2e-3, so `tracy_widom_cdf(beta=4)` jumped
  from 0 to 0.002 there. The grid now starts at `-8`; the means and
  variances of `F_1`, `F_2`, `F_4` match the literature to four digits.
- `physicskit.condensed.bhz_hamiltonian` / `bhz_ribbon_hamiltonian` scaled
  the particle-hole term as `-D(2 - cos kx - cos ky)`, half the standard
  Qi-Hughes-Zhang lattice form `-2D(...)` that matches `B`'s convention
  (no effect at the default `D=0`).
- `physicskit.constants.GAS_CONSTANT` is computed as `AVOGADRO * K_B`
  (exact by SI definition) rather than taken from `scipy.constants.R`,
  which older SciPy releases store truncated.

### Changed

These fix wrong signs or values, but change results callers may depend on:

- `physicskit.fields.fdtd_1d` now uses Maxwell's curl signs
  (`dHy/dt = +dEz/dx / mu`, as `fdtd_2d_tmz` already did). Previously the
  returned `Hy` was the negative of the physical field, and a right-moving
  pulse had to be launched with `Hy = +Ez/eta`; it is now
  `Hy = -Ez/eta` (Poynting vector along `+x`).
- `physicskit.fluids.kutta_joukowski_lift` returns the signed lift
  `-rho U Gamma` (along `+y`), consistent with the module's
  counterclockwise-positive `Gamma`; it returned `+rho U Gamma`, the
  opposite of the force from integrating the cylinder's own surface
  pressure. Clockwise circulation (`Gamma < 0`) now gives upward lift.
- `physicskit.fluids.von_karman_vortex_street` alternated circulation
  signs *within* each row, which is not a Karman street (the pattern
  drifted sideways instead of along itself). The upper row is now all
  clockwise (-1) and the lower row all counterclockwise (+1); the street
  translates at the classical `(Gamma/2l) tanh(pi h/l)`.
- `physicskit.fluids.energy_spectrum` shells now sum to the
  domain-averaged kinetic energy, as documented; they were `n**2` times
  larger (the grid-summed total).
- `physicskit.chaos.QuantumBakersMap.hbar` is now the reduced Planck
  constant `1/(2*pi*dim)` rather than `h = 1/dim`. Using `h` as `hbar`
  put coherent states' momentum at `p0/(2*pi)` and made them 2.5x too
  wide; `husimi_function` now takes the true `hbar` with width
  `sqrt(hbar)` (unchanged for `QuantumKickedRotor`).
- `physicskit.particle.charged_track_points` now bends positive charges
  clockwise for `B > 0` along `+z`, as `q v x B` requires; it bent them
  counterclockwise.
- `physicskit.relativity.KerrBlackHole.circular_orbit_conserved_quantities`
  returns `L < 0` for retrograde orbits. It returned `|L|`, which fed back
  into `integrate_equatorial_geodesic` gives an eccentric orbit (r = 12 ->
  26) instead of a circular one.
- `physicskit.fluids.rankine_hugoniot_jump_conditions` gained a
  `gamma=1.4` keyword; the energy residual hardcoded `gamma = 1.4`, so
  shocks from `normal_shock_relations(gamma=...)` with other `gamma`
  showed a spurious residual.
- `physicskit.statphys.LennardJonesGas.h_function` now includes the 2D
  Jacobian, `H = int f(v) ln[f(v) / (2 pi v)] dv`, i.e. Boltzmann's
  `int f ln f d^2v` for an isotropic gas. It computed `int f ln f dv` over
  the speed histogram, which Maxwell-Boltzmann does not minimize at fixed
  energy (a half-Gaussian speed distribution scores lower), contrary to
  its docstring. Returned values shift accordingly (an equilibrium gas
  now gives `-1 - ln(2 pi k_B T / m)`).

### Documentation

- `airy_bouncer_energies`' formula had `alpha^2` where `alpha` belongs in
  the length scale (the code was right).
- `weibel_fastest_growing_mode`'s Returns section described a third value
  (a cutoff wavenumber) that isn't returned; its unused `c` and the
  `anisotropy <= 1` -> `(0.0, 0.0)` behavior are now documented.
- `laughlin_metropolis_sweep` now explains that it draws from Numba's
  internal RNG and how to seed it (`seed_numba_random`); its doctests and
  gallery example previously created an unused `np.random.default_rng`,
  implying a reproducibility they didn't have.
- Documented previously missing or description-less parameters: `seed`
  across the `rmt` ensembles, bare `n`/`rng` entries in `rmt`,
  `SideBySideAnimator`'s `figsize`/`left_title`, and the unused
  `length` (`drift_wave_noise_ic`) and `b` (`landau_susceptibility`)
  parameters.
- Corrected docstrings whose formulas or claims were wrong while the code
  was right: `first_caustic_time` (an upper, not lower, bound);
  `grin_medium` (the `n0` factors come from the rod's entrance/exit
  faces); `SineGordonChain.kink` (static width `sqrt(k)`, not
  `sqrt(k/m)`; `p0` assumes `m=1`); the BHZ topological range
  (`0 < M/B < 8`, not `M/B > 0`); the Harris-sheet field sign;
  `SYKEnsemble` (energies are `2**(q/2)` times Maldacena-Stanford's,
  owing to the `{gamma_a, gamma_b} = 2 delta_ab` normalization);
  `SingleRingTheorem` (a uniform annulus is a special case, not the
  theorem's general radial law); `EARTH_RADIUS_M` (mean, not equatorial,
  radius).
- `langmuir_wave_ic` and its gallery example claimed the velocity kick
  follows a "linear-theory phase relationship" and prevents the Landau
  damping `landau_damping_ic` suffers; the kick just enlarges and
  phase-shifts the same standing wave, and damping is set by `k lambda_D`
  for both.
- `integrated_autocorrelation_time` said samples are independent every
  `2 tau_int` sweeps; with its `tau_int = 1 + 2 sum rho` normalization
  (`tau_int = 1` for uncorrelated data) the spacing is `tau_int`.
- `bec_condensate_fraction` attributed its `1 - (T/T_c)^{3/2}` law to a
  trapped gas; it is the box result (a 3D harmonic trap gives exponent 3).
- `binder_cumulant` claimed `U_4` lies in `[0, 2/3]` for any symmetric
  distribution; heavier-than-Gaussian tails make it negative.
- `Percolation2D` called the site threshold `p_c ~ 0.592746` exact; only
  the bond threshold is.
- The Heisenberg-uncertainty and lifting-cylinder gallery examples were
  updated for the squeezed-state and Kutta-Joukowski fixes above, the
  Maxwell and PML examples for the 1D FDTD sign fix, the von Karman
  example for the corrected vortex signs, and the H-theorem example for
  the `h_function` Jacobian.

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

[Unreleased]: https://github.com/cpoli/physicskit/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/cpoli/physicskit/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/cpoli/physicskit/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/cpoli/physicskit/releases/tag/v0.1.0
