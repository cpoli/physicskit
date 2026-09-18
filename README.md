# physicskit

| | |
|:--|:-:|
| Package | [![PyPI version](https://img.shields.io/pypi/v/physicskit)](https://pypi.org/project/physicskit/) [![Python versions](https://img.shields.io/pypi/pyversions/physicskit)](https://pypi.org/project/physicskit/) |
| Quality | [![License](https://img.shields.io/github/license/cpoli/physicskit)](https://github.com/cpoli/physicskit/blob/main/LICENSE) [![CI](https://github.com/cpoli/physicskit/actions/workflows/ci.yml/badge.svg)](https://github.com/cpoli/physicskit/actions/workflows/ci.yml) [![Coverage](https://img.shields.io/codecov/c/github/cpoli/physicskit)](https://codecov.io/gh/cpoli/physicskit) [![Coverage (manual)](https://img.shields.io/badge/coverage-96%25-brightgreen)](#coverage) |
| Documentation | [![Docs](https://img.shields.io/badge/docs-cpoli.github.io%2Fphysicskit-blue)](https://cpoli.github.io/physicskit/) |
| Code style | [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) |
| Downloads | [![Downloads](https://static.pepy.tech/badge/physicskit)](https://pepy.tech/project/physicskit) [![Downloads/Month](https://static.pepy.tech/badge/physicskit/month)](https://pepy.tech/project/physicskit) |
| Community | [![GitHub Stars](https://img.shields.io/github/stars/cpoli/physicskit?style=social)](https://github.com/cpoli/physicskit) [![GitHub Forks](https://img.shields.io/github/forks/cpoli/physicskit?style=social)](https://github.com/cpoli/physicskit) [![Contributors](https://img.shields.io/github/contributors/cpoli/physicskit)](https://github.com/cpoli/physicskit/graphs/contributors) [![Last Commit](https://img.shields.io/github/last-commit/cpoli/physicskit)](https://github.com/cpoli/physicskit/commits/main) |

A unified scientific toolkit for computational physics, spanning the
field end to end: the quantum mechanics of a single hydrogen atom, the
numerical relativity of colliding black holes, turbulent fluid
instabilities, topological superconductors, magnetically confined
plasmas, and chaotic quantum billiards -- with each domain's docs
tracing the field's own foundational breakthroughs in chronological,
pedagogical order, every historical milestone linked directly to the
runnable code that reproduces it. 14 domain subpackages, one consistent
NumPy-based API, sharing common ODE integrators and physical constants
throughout. Conventionally imported as `pk`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start

```python
import physicskit as pk
import numpy as np

H = lambda k1, k2: pk.condensed.haldane_model(k1, k2, phi=np.pi / 2)
chern_numbers = pk.condensed.compute_chern_number(H, grid_size=30)
print(chern_numbers)  # [1, -1]
```

Every subpackage below links to its worked examples; see
[Subpackages](#subpackages) for the full list, or browse the docs at
<https://cpoli.github.io/physicskit/>.

## Subpackages

Domain subpackages, each with runnable examples linked below:

- [`physicskit.astro`](https://cpoli.github.io/physicskit/api/gallery/astro/) -- stellar structure (Tolman-Oppenheimer-Volkoff, polytropes), N-body dynamics, orbital mechanics, galactic rotation curves, stellar convection and the magnetic dynamo.
- [`physicskit.chaos`](https://cpoli.github.io/physicskit/api/gallery/chaos/) -- chaotic dynamical systems and 2D quantum billiards.
- [`physicskit.classical`](https://cpoli.github.io/physicskit/api/gallery/classical/) -- classical mechanics: Newtonian, Lagrangian, Hamiltonian, lattice, and rigid-body dynamics.
- [`physicskit.condensed`](https://cpoli.github.io/physicskit/api/gallery/condensed/) -- tight-binding models, topological band theory (Chern numbers, edge states), correlated-electron superconductivity.
- [`physicskit.fields`](https://cpoli.github.io/physicskit/api/gallery/fields/) -- FDTD electrodynamics, KdV/NLS/Sine-Gordon solitons, BEC vortex lattices.
- [`physicskit.fluids`](https://cpoli.github.io/physicskit/api/gallery/fluids/) -- potential flow, viscous exact solutions, point-vortex dynamics, Kelvin-Helmholtz and Rayleigh-Taylor instabilities, compressible shocks, and the 2D incompressible Navier-Stokes solver underlying them.
- [`physicskit.optics`](https://cpoli.github.io/physicskit/api/gallery/optics/) -- ray, wave, and Gaussian-beam optics; quantum optics (squeezed states, Wigner functions).
- [`physicskit.particle`](https://cpoli.github.io/physicskit/api/gallery/particle/) -- relativistic kinematics, particle decays and scattering, nuclear physics.
- [`physicskit.plasma`](https://cpoli.github.io/physicskit/api/gallery/plasma/) -- Boris-pusher single-particle motion, guiding-center drifts, Grad-Shafranov MHD equilibrium, magnetic reconnection, cold-plasma wave dispersion, particle-in-cell Vlasov-Poisson kinetics.
- [`physicskit.quantum`](https://cpoli.github.io/physicskit/api/gallery/quantum/) -- quantum mechanics: wave packets, potentials, hydrogen, entanglement, measurement.
- [`physicskit.relativity`](https://cpoli.github.io/physicskit/api/gallery/relativity/) -- numerical general relativity: black holes, lensing, gravitational waves, cosmology.
- [`physicskit.rmt`](https://cpoli.github.io/physicskit/api/gallery/rmt/) -- random matrix theory, organized around Dyson's threefold way.
- [`physicskit.semiclassical`](https://cpoli.github.io/physicskit/api/gallery/semiclassical/) -- WKB/EBK quantization, Van Vleck/Herman-Kluk semiclassical propagators, the Gutzwiller trace formula, and quantum scarring.
- [`physicskit.statphys`](https://cpoli.github.io/physicskit/api/gallery/statphys/) -- statistical mechanics: lattice models, molecular dynamics, criticality, disordered systems.

Shared infrastructure, used across the subpackages above rather than
standalone toolkits:

- `physicskit.constants` -- SI physical constants shared across subpackages, plus a few well-defined unit conversions (energy/temperature, eV/joules, gravitational G=1 unit systems).
- `physicskit.integrators` -- shared numerical ODE integrators (RK4, leapfrog, Yoshida4, adaptive Dormand-Prince) used across the other subpackages.

## Test

Tests live alongside each subpackage, at `physicskit/<name>/tests/`.

```bash
pytest                                              # everything
pytest physicskit/rmt/tests                         # a single subpackage

# docstring examples, across every subpackage (pyvista's 3D viewer is
# optional and skipped rather than installed as a hard test dependency):
MPLBACKEND=Agg pytest --doctest-modules physicskit \
    --ignore-glob="*/tests/*" \
    --ignore=physicskit/chaos/visualizers/viewer3d.py
```

Both commands, plus `ruff check`/`ruff format --check`, run in CI on
every PR (`.github/workflows/ci.yml`) across Python 3.10-3.12 on Linux and
macOS. See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR.

### Coverage

```bash
MPLBACKEND=Agg pytest -q -n auto --cov=physicskit --cov-report=term
```

1,720 tests, 96% line coverage overall. Per-subpackage coverage:

| Subpackage | Coverage | | Subpackage | Coverage |
|:--|--:|---|:--|--:|
| `astro` | 100% | | `plasma` | 99% |
| `chaos` | 87% | | `quantum` | 96% |
| `classical` | 100% | | `relativity` | 99% |
| `condensed` | 100% | | `rmt` | 100% |
| `fields` | 100% | | `semiclassical` | 99% |
| `fluids` | 100% | | `statphys` | 86% |
| `integrators` | 100% | | `constants` | 100% |
| `optics` | 100% | | | |
| `particle` | 100% | | | |

Every subpackage is at 100% coverage outside `visualizers/` modules (99.9%
in aggregate — six rare bootstrap-loop edge cases remain uncovered across
`statphys.chapters.percolation` and `rmt.stats`). `chaos`, `quantum`, and
`statphys` still sit lower overall because their `visualizers/` modules
are smoke-tested only (correct return type/shape, or that `anim.save()`
succeeds) rather than covered line-by-line, per the testing convention in
[CLAUDE.md](CLAUDE.md). `@njit`-compiled lines are excluded from coverage
entirely (`pyproject.toml`, `[tool.coverage.report]`) since
`coverage.py` cannot trace into numba-compiled native code.

## Docs

Built docs are hosted at <https://cpoli.github.io/physicskit/>, served from
the `gh-pages` branch. To build locally:

```bash
pip install -e ".[docs]"
cd docs && make html
```

See `docs/source/history/` for a chronology of each field's foundational
breakthroughs, linked to the corresponding implementation at each step.

## Citation

If you use physicskit in your research, please cite it — see
[CITATION.cff](CITATION.cff).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Please note that this project
follows the [Contributor Covenant](CODE_OF_CONDUCT.md).

## License

MIT -- see [LICENSE](LICENSE).
