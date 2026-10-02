# Roadmap

What physicskit does not do yet, in priority order. physicskit is a
teaching toolkit, so the measure is not feature parity with research codes
(QuTiP, PlasmaPy, EinsteinPy, Dedalus) but whether each domain covers the
canonical topics of a course in that field, tells their history, and
checks that every example still shows the right physics.

Coverage is already broad: 14 domains, over 300 breakthroughs in
`docs/source/history/`, 392 gallery examples, and 20 tutorials. Most
of what is left is about releases rather than new domains: item 1
covers it.

Each item lists why it matters, the current state, and a sketch of the
approach. The project conventions apply to all of them: NumPy-style
docstrings with doctests, a cited source formula, closed-form tests,
and, for each new breakthrough, a history entry with its own gallery
example(s), titled after it and linked from no other entry.

## 1. A DOI for each release

**Why.** The 0.3.0 entry in `CITATION.cff` has no DOI.

**Now.** Releases are automated (see Done), and Zenodo archives each
GitHub Release once the repository is enabled there.

**Approach.** Enable the repository at
<https://zenodo.org/account/settings/github>, cut the next release, and
add its concept DOI (the one that always resolves to the latest version)
to `CITATION.cff` (`doi:`) and as a badge in the README.

## Done

Finished items move here with the release they shipped in, as in
CHANGELOG.md.

### Unreleased

**A blocking type check** (formerly item 2). `mypy` checks the whole
package with `check_untyped_defs`, `strict_equality` and
`warn_unreachable` on for all library code, and CI fails on any error;
the advisory job is gone. Tests are checked at the signature level only.
The 165 errors this turned up in library code were mostly `None`
defaults without `Optional`, njit callback slots typed as `None`, and
lists passed where an `ndarray` was annotated. CLAUDE.md now gives the CI
Python range as 3.10-3.14.

**Automated releases and docs deployment** (formerly part of item 1).
Pushing a `vX.Y.Z` tag runs `.github/workflows/release.yml`: it checks the
tag against `physicskit.__version__` and the `CITATION.cff` version,
builds the wheel and sdist, publishes to PyPI with trusted publishing,
creates the GitHub Release from the CHANGELOG section and rebuilds
`gh-pages`. `docs.yml` deploys `gh-pages` on every push to `main`, so the
hosted docs follow `main` between releases. The steps are in
CONTRIBUTING.md under "Releasing".

**One gallery convention for `rmt`** (formerly item 1). The 20 `rmt`
examples are now `plot_*.py` scripts titled after their breakthroughs, in
five sections (`gaussian_ensembles`, `spacings`, `edge_statistics`,
`non_hermitian`, `wishart`) instead of one `paper_replications/` folder.
They no longer save `*_replication.png` files, the committed copies are
gone, and `filename_pattern` in `docs/source/conf.py` is plain `/plot_`.
`rmt.validation.sine_kernel.CorrelationValidationResult` now has a
docstring.

**Gallery examples that check their physics** (formerly item 1). All 392
examples end with a `Check` cell asserting the closed form or known value
they illustrate, and CONTRIBUTING.md requires one of new examples. Writing
the checks turned up two library bugs, now fixed
(`exact_deflection_angle` was low by `1.5 b / r_far`; `two_stream_ic` set
the beams in opposite halves of the box), and 16 examples that did not
show their physics (see CHANGELOG). Two questions were left open:
`Percolation2D.fractal_dimension` (`log M / log R_g` of a single cluster)
overestimates `d_f`, up to 2.26 at `L = 128`, and `weibel_growth_rate`
(`gamma^2 = omega_pe^2 (A - 1) - k^2 c^2`) peaks at `k = 0`, where the
kinetic and fluid Weibel rates both vanish; only its cutoff is checked.

**Semiclassical history to 15 breakthroughs** (formerly item 1). The page
went from 7 entries to 17, each with its own example: Weyl's law, Einstein's
torus quantization, the Langer correction (`langer_corrected_wkb`), Feynman's
path integral, Balian and Bloch's level density
(`balian_bloch_level_density`), Heller's thawed Gaussians
(`thawed_gaussian_propagate`), Berry's random waves, the Ehrenfest time,
Heller's time-dependent spectroscopy, and Bogomolny's transfer operator
(`core.bogomolny`). Miller's classical S-matrix and Davis and Heller's
dynamical tunnelling were not done.

**Missing canonical topics** (formerly item 4). Each topic has a history
entry and a gallery example that asserts the physics it shows.
- **statphys**: Swendsen-Wang updates (`Ising2D.sweep(algorithm="swendsen-wang")`),
  Wang-Landau sampling (`WangLandauIsing`, checked against the exact 4x4
  density of states), and path-integral Monte Carlo of a particle in a
  harmonic or quartic well (`PathIntegralParticle`). The 1D Bose gas was
  not done.
- **chaos**: `Kuramoto` with the closed-form `r = sqrt(1 - 2γ/K)`, and the
  Hopf bifurcation (`HopfNormalForm`, `Brusselator`).
- **fluids**: Rayleigh-Bénard convection, with the onset simulated at
  657.5 between stress-free walls (`RayleighBenard2D`, on the periodic
  pseudo-spectral grid) and at 1708 between rigid walls
  (`RayleighBenardWalls2D`, Fourier-Chebyshev); the shallow-water equations
  (dam break against Ritter and Stoker, Rossby adjustment); and K41 as the
  k^(-5/3) spectrum of forced 2D (`ForcedTurbulence2D`, inverse cascade)
  and 3D (`ForcedTurbulence3D`, 64^3, forward cascade) turbulence.
- **optics**: a new characteristic-matrix module, `optics.thin_films`
  (the existing transfer matrices were ray ABCD matrices only), for
  antireflection coatings, Fabry-Pérot cavities and 1D photonic band gaps,
  and `optics.nonlinear` for second-harmonic generation and phase matching.
- **astro**: 1D SPH on Sod's tube (`SPH1D`, against the new
  `fluids.exact_riemann_solution`), and the grey atmosphere (`GreyAtmosphere`)
  with the Eddington approximation, the Hopf function and limb darkening.
  The collapsing polytrope was not done.
- **relativity**: `KerrNewmanBlackHole`, with photon orbits, the analytic
  and ray-traced shadow, and charged-particle geodesics.
- **quantum**: TEBD on matrix product states (`MPS`, `tebd`), checked
  against `condensed.spin_chains`, and `QuantumCircuit` with Deutsch-Jozsa,
  the quantum Fourier transform with Shor period finding, and Grover search.
