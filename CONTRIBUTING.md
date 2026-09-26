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
mypy                         # type check the typed core (blocking)
```

All of these run in CI (`.github/workflows/ci.yml`) on every PR, across
Python 3.10-3.14 on Linux and macOS. `mypy` on the whole package also
runs, but only as an advisory job — see "Type checking" below.

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

Type checking is adopted module by module, configured in
`pyproject.toml`:

- **Typed core (blocking).** `mypy` with no arguments checks the modules
  listed under `[tool.mypy] files` — `constants`, `integrators`, `units`,
  `results` and `io` — with `check_untyped_defs` on, and `units`,
  `results` and `io` held to near-`--strict` settings. A failure here
  fails CI. Errors in other modules that the typed core merely imports
  are silenced by a `follow_imports = "silent"` override.
- **Whole package (advisory).** `mypy physicskit` checks everything. It
  is not yet clean (mostly matplotlib/numpy stub gaps around animation
  objects and array-typed arguments), so CI runs it as a non-blocking
  job.

New code should type-check cleanly. To promote a module to the typed
core, fix its errors, then add it to `files` and to the first
`[[tool.mypy.overrides]]` block (and remove its subpackage from the
`follow_imports = "silent"` list if the whole subpackage is now clean).
Fixing pre-existing errors in code you're not otherwise touching is
welcome but not required.

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

## Reporting bugs / requesting features

Open a GitHub issue. For a physics bug specifically, include the formula
or reference you expected the code to match, and (if possible) the
numeric discrepancy — "this doesn't look right" is much slower to act on
than "the ringdown frequency should be within a few % of GW150914's
measured ~250 Hz and I'm getting X."

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
