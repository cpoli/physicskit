# physicskit

| | |
|:--|:-:|
| Package | [![PyPI version](https://img.shields.io/pypi/v/physicskit)](https://pypi.org/project/physicskit/) [![Python versions](https://img.shields.io/pypi/pyversions/physicskit)](https://pypi.org/project/physicskit/) |
| Quality | [![License](https://img.shields.io/github/license/physicskit/physicskit)](https://github.com/physicskit/physicskit/blob/main/LICENSE) [![CI](https://github.com/physicskit/physicskit/actions/workflows/ci.yml/badge.svg)](https://github.com/physicskit/physicskit/actions/workflows/ci.yml) [![Coverage](https://img.shields.io/codecov/c/github/physicskit/physicskit)](https://codecov.io/gh/physicskit/physicskit) |
| Documentation | [![Docs](https://readthedocs.org/projects/physicskit/badge/?version=latest)](https://physicskit.readthedocs.io) |
| Code style | [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) |
| Downloads | [![Downloads](https://static.pepy.tech/badge/physicskit)](https://pepy.tech/project/physicskit) [![Downloads/Month](https://static.pepy.tech/badge/physicskit/month)](https://pepy.tech/project/physicskit) |
| Community | [![GitHub Stars](https://img.shields.io/github/stars/physicskit/physicskit?style=social)](https://github.com/physicskit/physicskit) [![GitHub Forks](https://img.shields.io/github/forks/physicskit/physicskit?style=social)](https://github.com/physicskit/physicskit) [![Contributors](https://img.shields.io/github/contributors/physicskit/physicskit)](https://github.com/physicskit/physicskit/graphs/contributors) [![Last Commit](https://img.shields.io/github/last-commit/physicskit/physicskit)](https://github.com/physicskit/physicskit/commits/main) |

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
<https://physicskit.readthedocs.io>.

## Subpackages

Domain subpackages, each with runnable examples linked below:

- [`physicskit.astro`](https://physicskit.readthedocs.io/en/latest/examples/astro.html) -- stellar structure (Tolman-Oppenheimer-Volkoff, polytropes), N-body dynamics, orbital mechanics, galactic rotation curves, stellar convection and the magnetic dynamo.
- [`physicskit.chaos`](https://physicskit.readthedocs.io/en/latest/examples/chaos.html) -- chaotic dynamical systems and 2D quantum billiards.
- [`physicskit.classical`](https://physicskit.readthedocs.io/en/latest/examples/classical.html) -- classical mechanics: Newtonian, Lagrangian, Hamiltonian, lattice, and rigid-body dynamics.
- [`physicskit.condensed`](https://physicskit.readthedocs.io/en/latest/examples/condensed.html) -- tight-binding models, topological band theory (Chern numbers, edge states), correlated-electron superconductivity.
- [`physicskit.fields`](https://physicskit.readthedocs.io/en/latest/examples/fields.html) -- FDTD electrodynamics, KdV/NLS/Sine-Gordon solitons, BEC vortex lattices.
- [`physicskit.fluids`](https://physicskit.readthedocs.io/en/latest/examples/fluids.html) -- potential flow, viscous exact solutions, point-vortex dynamics, Kelvin-Helmholtz and Rayleigh-Taylor instabilities, compressible shocks, and the 2D incompressible Navier-Stokes solver underlying them.
- [`physicskit.optics`](https://physicskit.readthedocs.io/en/latest/examples/optics.html) -- ray, wave, and Gaussian-beam optics; quantum optics (squeezed states, Wigner functions).
- [`physicskit.particle`](https://physicskit.readthedocs.io/en/latest/examples/particle.html) -- relativistic kinematics, particle decays and scattering, nuclear physics.
- [`physicskit.plasma`](https://physicskit.readthedocs.io/en/latest/examples/plasma.html) -- Boris-pusher single-particle motion, guiding-center drifts, Grad-Shafranov MHD equilibrium, magnetic reconnection, cold-plasma wave dispersion, particle-in-cell Vlasov-Poisson kinetics.
- [`physicskit.quantum`](https://physicskit.readthedocs.io/en/latest/examples/quantum.html) -- quantum mechanics: wave packets, potentials, hydrogen, entanglement, measurement.
- [`physicskit.relativity`](https://physicskit.readthedocs.io/en/latest/examples/relativity.html) -- numerical general relativity: black holes, lensing, gravitational waves, cosmology.
- [`physicskit.rmt`](https://physicskit.readthedocs.io/en/latest/examples/rmt.html) -- random matrix theory, organized around Dyson's threefold way.
- [`physicskit.semiclassical`](https://physicskit.readthedocs.io/en/latest/examples/semiclassical.html) -- WKB/EBK quantization, Van Vleck/Herman-Kluk semiclassical propagators, the Gutzwiller trace formula, and quantum scarring.
- [`physicskit.statphys`](https://physicskit.readthedocs.io/en/latest/examples/statphys.html) -- statistical mechanics: lattice models, molecular dynamics, criticality, disordered systems.

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
every PR (`.github/workflows/ci.yml`) across Python 3.9-3.12 on Linux and
macOS. See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR.

## Docs

Built docs are hosted at <https://physicskit.readthedocs.io>. To build
locally:

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
