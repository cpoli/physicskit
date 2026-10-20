# The physicskit paper

Everything behind the manuscript, versioned with the code it describes.

| Directory | Contents |
|---|---|
| `validation/` | Cross-checks against QuTiP, Astropy and SciPy, and the catalogue of closed-form results asserted by the gallery |
| `benchmarks/` | Time against SciPy, QuTiP and Astropy |
| `listings/` | The worked example printed in the manuscript, which asserts its results |
| `figures/` | The script that draws every figure and table of the manuscript |
| `results/` | JSON written by the validation and benchmarks, read by the figures and tables |
| `manuscript/` | LaTeX source (SciPost template) |

## Reproducing

```bash
micromamba create -f paper/environment.yml && micromamba activate physicskit-paper
pip install -e .
python paper/validation/run_all.py        # asserts every cross-check (about 30 s)
python paper/benchmarks/run_benchmarks.py # about 5 min, one thread
python paper/figures/build_all.py         # figures and tables from results/ (about 1 min)
cd paper/manuscript && latexmk -pdf physicskit.tex
```

The validation asserts every comparison, and re-runs the gallery examples
of the closed-form table and the listings, so a failing check stops the
run. The domain figures are drawn by the panel functions of
`docs/make_readme_subpackage_figures.py`, so they match the README.
Benchmark timings depend on the machine. `results/benchmarks.json`
records the platform it ran on.
