# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A unified scientific toolkit for computational physics: 14 domain subpackages (`astro`, `chaos`, `classical`, `condensed`, `fields`, `fluids`, `optics`, `particle`, `plasma`, `quantum`, `relativity`, `rmt`, `semiclassical`, `statphys`) sharing common ODE integrators, physical constants, and a consistent NumPy-based API. Conventionally imported as `pk`.

## Commands

```bash
# setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install

# lint / format
ruff check .
ruff format .

# unit tests -- everything (parallelized across cores via pytest-xdist,
# ~75s vs ~250s serial on a 10-core machine), or a single subpackage / file / test
MPLBACKEND=Agg pytest -q -n auto
pytest physicskit/rmt/tests
pytest physicskit/classical/tests/test_conservation.py::test_energy_conserved -q

# docstring examples (every subpackage's doctests; pyvista's 3D viewer is
# optional and excluded rather than installed as a hard test dependency)
MPLBACKEND=Agg pytest --doctest-modules physicskit \
    --ignore-glob="*/tests/*" \
    --ignore=physicskit/chaos/visualizers/viewer3d.py

# type check (advisory only in CI, not blocking -- see below)
mypy physicskit

# docs (re-executes examples/*/plot_*.py via sphinx-gallery -- the only way
# to catch a broken example; do this if you touch docs/ or examples/)
pip install -e ".[docs]"
cd docs && make html
```

All of lint, unit tests, doctests, and (advisory) mypy run in CI (`.github/workflows/ci.yml`) across Python 3.10-3.12 on Linux and macOS.

## Architecture

**One subpackage per physics domain**, each living at `physicskit/<name>/` with its own `tests/` directory (`physicskit/<name>/tests/`, not a top-level `tests/`). New physics belongs in the subpackage it fits best; a genuinely new domain gets its own subpackage. Internal layout varies by subpackage size/age rather than following one rigid template — some are flat (`astro`), others split into `core/` (base classes, integrators), `systems/` or `chapters/` (concrete models), `utils/`, and `visualizers/` (`chaos`, `classical`). Read the target subpackage's own `__init__.py` and existing modules before assuming a layout.

**Two separate integrator layers — do not confuse them:**
- `physicskit.integrators` (top-level, shared) — `rk4`, `leapfrog`/`velocity_verlet`, `yoshida4`, adaptive `dopri5`. Callbacks use the convention `f(state_or_pos, t, params) -> ndarray`, letting one compiled callback be reused across systems with different parameter values.
- Several subpackages instead define their own `core/integrators.py` with a domain-specific closure-based calling convention (e.g. `physicskit.classical.core.integrators` uses `force_func(q, t)` plus a per-instance `mass_inv`; `physicskit.statphys.core.md_engine` has its own periodic-boundary, force-caching variant). These are intentional, documented divergences, not inconsistency to "fix."

**The numba first-class-function pattern**, used throughout (see `physicskit/classical/core/base_system.py` for the canonical explanation): integrator step/loop functions are themselves `@njit`-compiled and call the supplied force/derivative callback *from inside nopython code*. Numba can only do this when the callback is itself a genuine `@njit` dispatcher — a plain bound Python method (e.g. `self.derivatives`) cannot be typed and will fail. Every concrete system therefore builds its own standalone njit callback (conventionally `self._force_njit` and/or `self._deriv_njit`), produced by a module-level factory function that closes over the system's numeric parameters, and base classes always integrate using those attributes, never a bound method. Follow this pattern when adding a new system rather than passing `self.method` directly to an integrator.

Where subpackages share base classes (`classical`, and similarly-shaped subpackages), the common flow is: `core/base_system.py` defines abstract `DynamicalSystem`/`HamiltonianSystem`/`LagrangianSystem` classes and the `SimulationResult` dataclass returned by every `integrate()` call; concrete models in `systems/`/`chapters/` implement them; `visualizers/` consumes `SimulationResult` for plotting/animation.

`physicskit.constants` is the single source of truth for *actual SI values* (from `scipy.constants`, CODATA/2019 SI). Individual subpackages deliberately compute in their own natural unit system instead (`astro`: G=1; `relativity`: G=c=1; `quantum`: ħ=1; `statphys`: k_B=1) — this is documented per-subpackage and is by design, not an inconsistency. Only reach for a `constants` conversion helper (energy↔temperature, energy↔eV, G=1 unit-system derivation) where one is explicitly provided; converting a ħ=1 result to SI needs a domain-specific mass/length scale choice with no generic helper, by design.

## Conventions

- Physics single-letter variable names (`t`, `M`, `N`, etc.) are intentional and preserved — see the deliberate `ruff` ignores in `pyproject.toml` (`E741` and others) rather than "fixing" them.
- Every public function/class needs a NumPy-style docstring (`Parameters`, `Returns`, and an `Examples` section with a runnable doctest where it adds real value). Doctests are checked in CI — an example that doesn't actually execute correctly is worse than no example.
- New physics should cite its source formula (docstring or comment) so it can be independently verified.
- Tests prefer closed-form/analytically-verifiable assertions (`pytest.approx` against a known formula) over snapshot-testing plot output; a visualizer needs only a smoke test (right return type/shape; for animations, that `anim.save()` to a temp file succeeds).
- `mypy` is configured but not yet fully clean (mostly matplotlib/numpy stub gaps around animation objects and array-typed arguments) and runs advisory/non-blocking in CI. New code should type-check where practical; fixing unrelated pre-existing errors is not required.
- `docs/source/history/` documents each subpackage's foundational physics breakthroughs linked to the corresponding implementation — worth checking when adding a major new model to understand the expected historical framing.

# Output Guidelines
- **Be Concise:** Provide direct code and answers first. Omit setup text, fluff, and conversational responses.
- **Code Generation:** Output only updated code blocks, functions, or unified diffs. Never output full files unless instructed.
- **Explanations:** Limit inline code comments. Explain high-level logic in 1–2 bullet points after the code block.
- **Format:** Use bulleted lists and tables for comparisons instead of paragraph blocks.
