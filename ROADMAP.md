# Roadmap

What physicskit does not do yet, in priority order. physicskit is a
teaching toolkit, so the measure is not feature parity with research codes
(QuTiP, PlasmaPy, EinsteinPy, Dedalus) but whether each domain covers the
canonical topics of a course in that field, tells their history, and
checks that every example still shows the right physics.

Coverage is already broad: 14 domains, over 300 breakthroughs in
`docs/source/history/`, 392 gallery examples, and 20 tutorials. Most of
what is left is not new physics. The checks the project relies on are not
running yet (item 1), and the release workflow has never published
(item 2). These come before any new content.

Each item lists why it matters, the current state, a sketch of the
approach, and when it counts as done. Each one can be taken on by itself,
in its own prompt. Items 1, 2, 4, 5 and 6 all edit `.github/workflows/`,
so they should not run in parallel. Items marked **Needs the maintainer**
include a step in a PyPI, Zenodo or journal account that only the
maintainer can do. The project conventions apply to all of them:
NumPy-style docstrings with doctests, a cited source formula, closed-form
tests, and, for each new breakthrough, a history entry with its own
gallery example(s), titled after it and linked from no other entry.

## 1. Run the gallery in CI again

**Why.** The `Check` cell at the end of every gallery example only runs
when the docs are built. The `Docs` workflow has been disabled by hand
since 2026-09-17, when a build took 75 to 110 minutes. Since then no PR
and no push to `main` has run the gallery. So the project's main promise,
that every example still shows the right physics, is not being checked.

**Now.** `docs/source/sg_execution_times.rst` gives 35 minutes for all
392 examples on a local machine. The faster build is written but not
committed: `docs.yml` caches the numba cache and the sphinx-gallery
output, runs the full gallery on pushes to `main` and nightly, and on a PR
executes only the examples `docs/select_examples.py` selects (passed to
`docs/source/conf.py` as `PHYSICSKIT_GALLERY_PATTERN`). CONTRIBUTING.md
and CHANGELOG.md are updated to match. The workflow is still
`disabled_manually`, so none of this has run in CI yet.

**Approach.** Commit the changes and run `gh workflow enable Docs`. Then
open a PR that breaks one example's `Check` to time the PR path and make
sure a failed `assert` fails the job, and watch the first full build on
`main`.

**Done when.** The `Docs` workflow is enabled. A PR that breaks one
example's `Check` fails within 15 minutes. The full build on `main`
passes and redeploys `gh-pages`.

## 2. A release workflow that publishes

**Why.** The `v0.4.0` run of `release.yml` failed at "Publish to PyPI"
with `invalid-publisher`: PyPI has no trusted publisher matching
`cpoli/physicskit`, `release.yml`, environment `pypi`. Because of that,
the two jobs after it ("Rebuild gh-pages" and "Create the GitHub
Release") were skipped. 0.4.0 reached PyPI and GitHub by hand, so the
workflow has never completed a release. 0.2.0 and 0.3.0 have no GitHub
Release at all, and therefore no Zenodo DOI.

**Now.** The workflow side is done (not yet committed). The GitHub
Release and `gh-pages` jobs need only the build, not PyPI, and running
the workflow by hand publishes a `X.Y.Z.dev<run id>` build to TestPyPI as
a dry run.

**Approach.** **Needs the maintainer:** add the trusted publisher on
PyPI (project physicskit → Publishing: owner `cpoli`, repository
`physicskit`, workflow `release.yml`, environment `pypi`), and the same
on TestPyPI with environment `testpypi`. Then run
`gh workflow run release.yml` and check that the dev build reaches
TestPyPI before the next real tag.

**Done when.** One tag push publishes to PyPI, creates the GitHub
Release, gets a Zenodo DOI and redeploys `gh-pages`, with no manual
step.

## 3. Release the unreleased work as 0.4.1

**Why.** Since 0.4.0, `main` and the working tree hold work that no
release has shipped: the `weibel_growth_rate` and
`Percolation2D.fractal_dimension` fixes, the Node 24 CI, the faster
`Docs` and `Release` workflows (items 1 and 2), the conda-forge install
line and badge in the README, sitemap and OpenGraph metadata for the docs
(`sphinx-sitemap` and `sphinxext-opengraph` in the `docs` extra), and
links to the sibling toolkits on the docs front page. The `[Unreleased]`
section of CHANGELOG.md records the two fixes and the two workflows, but
not the CI move or the README and docs metadata. All README badges are
now on badgen.

**Approach.** Write the missing CHANGELOG entries. Check that the
conda-forge feedstock is at 0.4.0, and merge the bot PR if it is behind.
Then cut 0.4.1, which also tests item 2 if that is done first.

**Done when.** 0.4.1 is on PyPI and conda-forge with its own Zenodo DOI
in `CITATION.cff`.

## 4. The paper

**Why.** `paper/` (a SciPost manuscript with validation against QuTiP,
Astropy and SciPy, benchmarks, and figures) is committed, but nothing
checks that it still runs against the code it describes. It also has an
open `TODO` (acknowledgements, funding).

**Approach.** Add a workflow that runs `paper/validation/run_all.py` on
changes to `paper/**` or `physicskit/**` (QuTiP and Astropy installed
from `paper/environment.yml`). Record in the manuscript the release it
describes, with that release's version DOI. **Needs the maintainer:**
write the acknowledgements and submit.

**Done when.** The validation runs in CI, and the manuscript cites a
tagged release.

## 5. Test Windows and the minimum dependency versions

**Why.** Many students use Windows, and CI only runs on Linux and macOS.
`pyproject.toml` promises `numpy>=1.24`, `scipy>=1.10` and `numba>=0.58`,
but CI only installs the latest versions, so those lower bounds have
never been tested.

**Approach.** Add `windows-latest` on one Python version. Add a job that
installs the oldest versions allowed (`uv pip install --resolution
lowest-direct`) on Python 3.10. Fix what fails, or raise the lower
bounds to what actually works.

**Done when.** Both jobs pass and block merging.

## 6. Benchmarks that catch regressions

**Why.** `benchmarks.yml` uploads `benchmark.json` on every push, but
nothing ever reads it, so a slowdown goes unnoticed.

**Approach.** Store the results from `main`, for example with
`benchmark-action/github-action-benchmark` on a `gh-pages` subpath, and
compare each PR against them. Warn above a 1.5x slowdown and fail above
3x. Shared runners are noisy, so keep the thresholds loose.

**Done when.** A PR that doubles the time of a benchmark gets a comment
or a failing check.

## 7. A tutorial for `classical`

**Why.** `classical` is the first course of every physics degree, but no
tutorial is about it. It only appears in `units_and_conventions.rst`.

**Approach.** Write one tutorial in `docs/source/tutorials/`: from a
Lagrangian to `integrate()`, energy conservation with symplectic and
non-symplectic integrators, and normal modes. Use existing
`physicskit.classical` systems only. Each step asserts its result.

**Done when.** The tutorial is linked from `tutorials/index.rst` and
builds without errors.

## 8. A tutorial for `quantum`

**Why.** `quantum` has the largest history page (38 entries), but its
only tutorial is the Gutzwiller one, which is semiclassical.

**Approach.** Write one tutorial that goes from bound states in a well,
to time evolution of a wavepacket, to a small `QuantumCircuit`. Use
existing API only, and assert each result.

**Done when.** As in item 7.

## 9. Measure "canonical topics"

**Why.** The roadmap's measure of success, coverage of the canonical
topics of a course, is not written down anywhere. So nobody can tell
what is missing, and the topic items above were picked by intuition.

**Approach.** Add one page per domain (or one table) that maps the
chapters of a standard textbook (Goldstein or Taylor for `classical`,
Griffiths for `quantum`, Kittel for `condensed`, Chen for `plasma`, and
so on) to modules and gallery examples. Mark each chapter as covered,
partly covered, or missing. Do not write any new code in this item. The
"missing" rows become new roadmap items.

**Done when.** All 14 domains are mapped, and this roadmap lists the
gaps the map shows.

## 10. Run the examples in the browser

**Why.** A teaching gallery that students can only run after installing
numba locally loses most of them at the first step.

**Approach.** Turn on sphinx-gallery's JupyterLite or Binder buttons.
numba does not run under Pyodide, so Binder, with an environment built
from `pyproject.toml`, is probably the only option. Check one example
from each domain before switching it on everywhere.

**Done when.** Every example page has a working "launch" button.

## 11. Remaining topics

These were in scope for earlier items and were not done. Each is a
separate prompt, with its own history entry and example.
- **semiclassical**: Miller's classical S-matrix (1970).
- **semiclassical**: Davis and Heller's dynamical tunnelling (1981).
- **statphys**: the Lieb-Liniger 1D Bose gas (1963), with the Bethe-ansatz
  ground-state energy and the Tonks-Girardeau limit.
- **astro**: the collapsing polytrope (free-fall time, Larson-Penston or
  homologous collapse).

## Done

Finished items move here with the release they shipped in, as in
CHANGELOG.md.

### Unreleased

**A correct `Percolation2D.fractal_dimension`** (formerly item 3). It
returned `log M / log R_g` for a single cluster, up to 2.26 at `L = 128`.
It now fits the slope of `log M(R)` against `log R` for the spanning
cluster at `p_c` (the sandbox method), averaged over the `n_trials`
realizations that span, and gives 1.88 at `L = 256` against the exact
`91/48 ≈ 1.896`. The signature is unchanged. The percolation example's
`Check` asserts it, and the CHANGELOG records the fix.

**A correct `weibel_growth_rate`** (formerly item 4). It returned
`gamma^2 = omega_pe^2 (A - 1) - k^2 c^2`, which peaks at `k = 0`. It now
returns the small-growth kinetic limit for a bi-Maxwellian (Davidson
1983), `gamma = sqrt(2/pi) k v_th_par / A * (A - 1 - k^2 c^2 / omega_pe^2)`,
which is zero at `k = 0`, peaks at `k c / omega_pe = sqrt((A - 1) / 3)`
and keeps the old cutoff. It takes a new required `v_th_par`.
`simulate_weibel_filamentation` used the SI speed of light on a grid of a
few units, so no filaments formed; it now takes `c` and `v_th_par`. The
example's `Check` asserts the zero, the peak, the cutoff, and the
measured growth rate of the dominant mode against the function.

**CI off deprecated runners and actions** (formerly item 8). All four
workflows use the Node 24 majors of `checkout`, `setup-python`,
`upload-artifact`/`download-artifact` and `codecov-action`. `ci.yml` and
`benchmarks.yml` have `concurrency` groups that cancel superseded runs,
and pytest runs with `--dist worksteal`, so CI takes about 13 minutes
instead of 28. `ubuntu-latest` stays unpinned: a run on Ubuntu 26.04
(branch `ci/ubuntu-26-check`, 2026-10-21) passed with no annotations.

### 0.4.0

**A DOI for each release** (formerly item 1). The repository is enabled
on Zenodo. The 0.4.0 GitHub Release has its own version DOI,
`10.5281/zenodo.23116492`, which is in `CITATION.cff` and on a README
badge. 0.2.0 and 0.3.0 had no GitHub Release and have no DOI.


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
`gh-pages`. The steps are in CONTRIBUTING.md under "Releasing". The
PyPI step has not yet succeeded, and the docs workflow was later disabled
(see items 1 and 2).

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
show their physics (see CHANGELOG). Two questions were left open, both
fixed since (see Unreleased): `Percolation2D.fractal_dimension`
overestimated `d_f`, and `weibel_growth_rate` peaked at `k = 0`.

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
