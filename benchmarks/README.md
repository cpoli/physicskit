# Benchmarks

Performance benchmarks for the shared integrators (`physicskit.integrators`)
and three representative solvers, one per numerical method:

| Benchmark | What it times |
|:--|:--|
| `test_bench_integrators.py` | `rk4`, `leapfrog`, `yoshida4` and adaptive `dopri5` on a 2D Kepler orbit |
| `test_bench_solvers.py::test_ising_metropolis` | `statphys.Ising2D.sweep` (Monte Carlo, 64x64 lattice) |
| `test_bench_solvers.py::test_navier_stokes_spectral` | `fluids.NavierStokes2D.simulate` (pseudo-spectral PDE, 64x64 grid) |
| `test_bench_solvers.py::test_numerov_eigensolver` | `quantum.NumerovSolver.solve` (1D eigenproblem, 2001 grid points) |

Each benchmark calls its target once before timing, so numba compilation is
not counted. They are not in the default `pytest` run (`testpaths` in
`pyproject.toml` does not include `benchmarks/`).

```bash
pip install -e ".[bench]"

pytest benchmarks                                  # time everything
pytest benchmarks --benchmark-disable              # just check they still run
pytest benchmarks --benchmark-autosave             # save to .benchmarks/
pytest benchmarks --benchmark-compare              # compare against the last save
pytest benchmarks --benchmark-compare --benchmark-compare-fail=mean:10%
```

CI runs them with `--benchmark-disable` on every PR (so they don't rot), and
does a timed run on pushes to `main`, uploading the JSON as a workflow
artifact (`.github/workflows/benchmarks.yml`). Shared CI runners are too
noisy to gate PRs on timing, so compare locally on one machine
(`--benchmark-compare`) when a change could affect performance.
