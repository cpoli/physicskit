# Contributing to physicskit

Thanks for considering a contribution. physicskit is organized as one
subpackage per physics domain (`physicskit/<name>/`), each with its own
`chapters/` (model classes), `core/` (numba-accelerated kernels, where
needed), `visualizers/` (matplotlib/plotly plotting), and `tests/`
directory. New physics belongs in the subpackage it fits best; a genuinely
new domain gets its own subpackage following the same layout.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Before opening a PR

```bash
ruff check .                 # lint
ruff format .                # format
MPLBACKEND=Agg pytest -q     # unit tests
MPLBACKEND=Agg pytest --doctest-modules physicskit \
    --ignore-glob="*/tests/*" \
    --ignore=physicskit/chaos/visualizers/viewer3d.py   # docstring examples
mypy                         # type check (blocking)
```

All of these run in CI (`.github/workflows/ci.yml`) on every PR, across
Python 3.10-3.14 on Linux and macOS — see "Type checking" below for
what `mypy` covers.

If your change could affect performance (an integrator, a hot loop, a
solver), compare the benchmarks before and after on your own machine —
see [`benchmarks/README.md`](benchmarks/README.md):

```bash
pip install -e ".[bench]"
pytest benchmarks --benchmark-autosave    # on main
pytest benchmarks --benchmark-compare     # on your branch
```

If you touch anything under `docs/` or add/modify an example in
`examples/`, also build the docs locally before opening a PR (this
re-executes every changed `examples/*/plot_*.py` script via
sphinx-gallery, which is the only way to catch a broken example):

```bash
pip install -e ".[docs]"
cd docs && make html
```

Every gallery example must check the physics it shows. End the script
with a `Check` cell (`# %%`, then `# Check` / `# -----`) holding one or
more `assert`s against the closed form or known value the example
illustrates: an exact invariant, a textbook constant, a scaling exponent,
a conserved quantity. Set each tolerance from the physics and the run's
own statistics, not from the last printed number, and say in a comment
what is being checked. The docs build fails on a failed `assert`, so an
example that stops showing its physics breaks CI instead of going
unnoticed. Keep the asserts at the end so they don't interrupt the
narrative; if a loop overwrites a value the check needs, collect it in
the loop rather than recomputing it.

## Code style

- Follow the existing style in the subpackage you're editing — physics
  notation (`t`, `M`, `N`, single-letter variables) is intentional and
  allowed (see the `ruff` ignores in `pyproject.toml` for the specific,
  deliberate exceptions).
- Every public function/class gets a NumPy-style docstring with a
  `Parameters`, `Returns`, and (where it adds real value beyond what the
  signature already says) an `Examples` section with a runnable doctest.
  Doctests are checked in CI — a docstring example that doesn't actually
  run is worse than no example.
- Prefer closed-form / analytically-verifiable results in tests
  (`pytest.approx` against a known formula) over snapshot-testing plot
  output.
- New physics should cite where the formula comes from, either in the
  docstring or as a comment, so a reader can verify it independently.

## Type checking

`mypy` with no arguments checks the whole package (`[tool.mypy] files`
in `pyproject.toml`), and a failure fails CI. The settings, all in
`pyproject.toml`:

- **Library code** is checked with `check_untyped_defs`,
  `strict_equality` and `warn_unreachable`, so the bodies of unannotated
  functions are checked too.
- **`units`, `results` and `io`** are also held to near-`--strict`
  settings.
- **Tests** (`physicskit/**/tests/`) are checked at the signature level
  only: their bodies use private matplotlib and numba attributes and
  deliberately loose argument types.

New code must type-check. Where a NumPy or matplotlib stub is wrong,
prefer a narrow `# type: ignore[code]` with a short reason over loosening
the annotation.

## Tests

Tests live alongside each subpackage (`physicskit/<name>/tests/`), not in
a top-level `tests/` directory. A new chapter/model needs:

- At least one test that checks a closed-form / analytically-known result,
  not just "it runs without crashing."
- A visualizer needs only a smoke test (it returns the right type/shape,
  and — for animations — that `anim.save()` to a temp file succeeds).

## History entries

`docs/source/history/*_breakthroughs.rst` is a curated chronology per
domain, not a general-purpose list of "interesting physics history." Each
entry exists because it connects to something this package actually
implements. Keep it that way:

- A new entry must cite the concrete class/method it connects to
  (`:meth:`.../:class:`...` cross-reference) and belongs in the same PR
  as the implementation it describes, not added on its own.
- One entry per PR. If you're adding several related pieces of physics,
  open separate PRs (or ask first) rather than batching a set of history
  entries together.
- Changes to `docs/source/history/**` require a maintainer review
  (enforced via `.github/CODEOWNERS`) even if the rest of the PR is
  otherwise approved.

## Stability and deprecation policy

physicskit follows [Semantic Versioning](https://semver.org/). It is
pre-1.0, so the rules below are what contributors should follow now, and
they become a guarantee to users at 1.0.

### What counts as public API

- Names exported in a module's `__all__`, or documented in the API
  reference, tutorials or the example gallery. This includes
  `physicskit.constants`, `physicskit.integrators`, `physicskit.units`,
  `physicskit.results` and `physicskit.io`, and each subpackage's
  top-level namespace.
- **The unit convention of each subpackage** (see the "Units and
  conventions" docs page). Silently switching a function from
  `G = c = 1` to SI changes every number a caller gets, which is as
  breaking as renaming it.
- The on-disk format written by `physicskit.io.save`.

Anything whose name starts with an underscore is private, as are
`tests/` directories and anything undocumented. It can change at any
time.

### Deprecating something

1. Keep the old name or behaviour working and have it emit a
   `DeprecationWarning` (or `FutureWarning` when a default *value* or
   *result* is going to change) that says what to use instead and in
   which release the old form goes away. Use `stacklevel=2` so the
   warning points at the caller.
2. Add a test that asserts the warning (`pytest.warns`) and that the
   old path still gives the right answer.
3. Add a bullet under `### Deprecated` in `CHANGELOG.md`.
4. Remove it no sooner than **one minor release** later while pre-1.0
   (deprecated in 0.3, removed in 0.4 at the earliest), and no sooner
   than **two minor releases** later after 1.0. Removals are listed
   under `### Removed`.

### Changes that don't need a deprecation cycle

- **Physics fixes.** If a function returns a wrong value (a sign error,
  a missing factor of 2), fix it immediately. A wrong answer shouldn't
  stay available for another release. Because callers may depend on the
  old number, list the fix in its own "these change results" block
  under `### Changed` in the CHANGELOG, with the old and new behaviour,
  as the 0.2.0 entry does.
- **Numerical details.** Results are reproducible within documented
  tolerances, not bit-for-bit, across releases. A change of integrator
  internals, default grid resolution or RNG stream is allowed if the
  documented accuracy still holds. Mention it in the CHANGELOG when it
  visibly changes outputs, such as a seeded example's printed numbers.
- **Additions:** new functions, new keyword arguments with defaults that
  keep the old behaviour, and new fields at the end of result
  dataclasses.

### Saved files

`physicskit.io` writes a `FORMAT_VERSION` into every file. A new release
must still read files written by every earlier format version, and must
refuse (with a clear error), not misread, files from a newer one. Bump
`FORMAT_VERSION` whenever the layout changes, and add a round-trip test
that loads a file written in the old layout.

### Supported Python versions

Supported versions are the ones in the CI test matrix (currently
3.10-3.14). Dropping one happens in a minor release, is announced in the
CHANGELOG, and updates `requires-python`, the classifiers and the CI
matrix together.

### Before 1.0

1.0 will be tagged once the public API above has been through at least
one release without a breaking change.

## Releasing

Releases are cut from `main` by pushing a tag; `.github/workflows/release.yml`
does the rest.

1. Bump `__version__` in `physicskit/__init__.py`, and `version` and
   `date-released` in `CITATION.cff`.
2. Rename `## [Unreleased]` in CHANGELOG.md to `## [X.Y.Z] - YYYY-MM-DD`,
   open a new empty `[Unreleased]` section above it, and update the
   compare links at the bottom.
3. Commit, then `git tag vX.Y.Z && git push origin main vX.Y.Z`.

The workflow fails before publishing anything if the tag, `__version__`
and `CITATION.cff` disagree. It then publishes to PyPI (trusted
publishing, no token), creates the GitHub Release with that CHANGELOG
section as its notes, and rebuilds `gh-pages`. Zenodo archives each
GitHub Release with its own DOI. Between releases, `docs.yml` redeploys
`gh-pages` on every push to `main`.

## Reporting bugs / requesting features

Open a GitHub issue. For a physics bug specifically, include the formula
or reference you expected the code to match, and (if possible) the
numeric discrepancy — "this doesn't look right" is much slower to act on
than "the ringdown frequency should be within a few % of GW150914's
measured ~250 Hz and I'm getting X."

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
